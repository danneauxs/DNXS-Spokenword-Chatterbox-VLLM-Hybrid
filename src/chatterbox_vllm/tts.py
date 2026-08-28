from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Tuple, Any, List
import gc
import logging
import time
import os
import queue
import threading

# Enable expandable segments to reduce CUDA memory fragmentation (OOM fix)
# This is suggested by PyTorch OOM error messages
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from vllm import LLM, SamplingParams
from functools import lru_cache

import librosa
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from chatterbox_vllm.models.t3.modules.t3_config import T3Config

from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.voice_encoder import VoiceEncoder
from .models.t3 import SPEECH_TOKEN_OFFSET
from .models.t3.modules.cond_enc import T3Cond, T3CondEnc
from .models.t3.modules.learned_pos_emb import LearnedPositionEmbeddings
from .text_utils import punc_norm, SUPPORTED_LANGUAGES
from modules.punctuation_pauses import min_speech_tokens_for_text, split_t3_sentences

from config import config

# Apply monkey patches for performance optimization
# TEMPORARILY DISABLED - causing hang during model loading
# from .models.s3gen.monkey_patches import apply_monkey_patches
# apply_monkey_patches()

REPO_ID = "ResembleAI/chatterbox"

# Parallel S3Gen configuration (config file with environment variable override)
S3GEN_NUM_WORKERS = int(os.environ.get("S3GEN_NUM_WORKERS", str(config.VLLM_S3GEN_NUM_WORKERS)))
S3GEN_ENABLE_PARALLEL = os.environ.get("S3GEN_ENABLE_PARALLEL", str(config.VLLM_S3GEN_ENABLE_PARALLEL).lower()).lower() in ("true", "1")
SPEECH_VOCAB_SIZE = 6561  # Speech tokens must be < this value


def resolve_t3_weight_file(ckpt_dir: str | Path, variant: str = "english") -> Path:
    """Locate the T3 safetensors file for english or multilingual variants.

    Multilingual V3 weights live outside CHATTERBOX_CKPT_DIR (VE/S3Gen stay there).
    Prefer config.t3_weights_path() when T3_SOURCE matches `variant`; otherwise
    look in ckpt_dir for v3 then v2 filenames so older layouts still load.

    Args:
        ckpt_dir: Directory used for VE/S3Gen or a combined checkpoint folder.
        variant: "english" or "multilingual".

    Returns:
        Path to the T3 safetensors file.

    Raises:
        FileNotFoundError: If no matching T3 checkpoint exists.
    """
    ckpt_dir = Path(ckpt_dir)
    try:
        source = config.resolve_t3_source()
        if config.t3_vllm_variant(source) == variant:
            configured = config.t3_weights_path(source)
            if configured.is_file():
                return configured
    except Exception:
        # Config resolver is best-effort; fall through to ckpt_dir filenames.
        pass

    if variant == "turbo":
        turbo_path = config.t3_weights_path("turbo")
        if turbo_path.is_file():
            return turbo_path
        raise FileNotFoundError(f"Turbo T3 checkpoint not found: {turbo_path}")

    if variant == "english":
        english_path = ckpt_dir / "t3_cfg.safetensors"
        if english_path.is_file():
            return english_path
        raise FileNotFoundError(f"English T3 checkpoint not found: {english_path}")

    for name in ("t3_mtl23ls_v3.safetensors", "t3_mtl23ls_v2.safetensors"):
        candidate = ckpt_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Multilingual T3 checkpoint not found in {ckpt_dir} "
        "(looked for t3_mtl23ls_v3.safetensors and t3_mtl23ls_v2.safetensors). "
        "Set CHATTERBOX_T3_SOURCE/CHATTERBOX_MTL_CKPT_DIR or download V3 T3."
    )

@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        """Transfers model to specified device."""
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath):
        """Loads model weights from a file path."""
        kwargs = torch.load(fpath, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


def compute_conditionals(
    ckpt_dir: str | Path,
    target_device: str = "cuda",
    variant: str = "english",
    audio_prompt_path: Optional[str] = None,
) -> torch.Tensor:
    """Load VE+S3Gen+conds, compute cond_emb, free VE+S3Gen. Returns cond_emb tensor on CPU.

    Phase 0 of the three-phase pipeline. VE and S3Gen are temporarily loaded,
    used to compute the conditional embedding, then freed. Only cond_emb survives.
    Mirrors ChatterboxTTS.get_audio_conditionals() exactly, but standalone so
    Phase 1 (vLLM T3) never has to coexist with VE/S3Gen in VRAM.

    Args:
        ckpt_dir: Directory with VE, S3Gen, and conds.pt (usually models/chatterbox).
        target_device: Device for computation.
        variant: Model variant ("english" or "multilingual").
        audio_prompt_path: Optional voice prompt WAV path.

    Returns:
        cond_emb tensor on CPU.
    """
    ckpt_dir = Path(ckpt_dir)

    # Load T3 cond components (shared between default and voice-prompt paths).
    # T3 weights may live in a separate multilingual dir; VE/S3Gen stay in ckpt_dir.
    t3_config = T3Config()
    turbo = variant == "turbo"
    if turbo:
        t3_config.speech_tokens_dict_size = 6563
        t3_config.speech_cond_prompt_len = 375
        t3_config.use_perceiver_resampler = False
        t3_config.emotion_adv = False
    t3_ckpt_path = resolve_t3_weight_file(ckpt_dir, variant)
    logging.info("Phase 0 loading T3 cond weights from %s", t3_ckpt_path)
    t3_weights = load_file(t3_ckpt_path)

    t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
    t3_speech_emb.load_state_dict({
        k.replace('speech_emb.', ''): v
        for k, v in t3_weights.items()
        if k.startswith('speech_emb.')
    })
    t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

    t3_speech_pos_emb = None
    if not turbo:
        t3_speech_pos_emb = LearnedPositionEmbeddings(
            t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels
        )
        t3_speech_pos_emb.load_state_dict({
            k.replace('speech_pos_emb.', ''): v
            for k, v in t3_weights.items()
            if k.startswith('speech_pos_emb.')
        })
        t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

    t3_cond_enc = T3CondEnc(t3_config)
    t3_cond_enc.load_state_dict({
        k.replace('cond_enc.', ''): v
        for k, v in t3_weights.items()
        if k.startswith('cond_enc.')
    }, strict=False)
    t3_cond_enc = t3_cond_enc.to(device=target_device).eval()

    # Load VE + S3Gen + conds for conditional computation
    ve = VoiceEncoder()
    ve_path = Path(config.t3_checkpoint_dir("turbo")) / "ve.safetensors" if turbo else ckpt_dir / "ve.safetensors"
    ve.load_state_dict(load_file(ve_path))
    ve = ve.to(device=target_device).eval()

    s3gen = S3Gen(use_fp16=False)
    s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen = s3gen.to(device=target_device).eval()

    default_conds = Conditionals.load(ckpt_dir / "conds.pt")
    default_conds.to(device=target_device)

    with torch.inference_mode():
        if audio_prompt_path is None:
            t3_cond_prompt_tokens = default_conds.t3.cond_prompt_speech_tokens
            ve_embed = default_conds.t3.speaker_emb
        else:
            audio_prompt_path = Path(audio_prompt_path)
            s3gen_ref_wav, _sr = librosa.load(str(audio_prompt_path), sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(
                s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR
            )

            # English/MTL use 6s; Turbo T3 was trained with a 15s speech cond prompt.
            enc_len = 15 * S3_SR if turbo else ChatterboxTTS.ENC_COND_LEN
            s3_tokzr = s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                [ref_16k_wav[:enc_len]],
                max_len=t3_config.speech_cond_prompt_len,
            )
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            ve_embed = torch.from_numpy(
                ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
            )
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        t3_cond_prompt_tokens = t3_cond_prompt_tokens.to(device=target_device)
        cond_prompt_speech_emb = t3_speech_emb(t3_cond_prompt_tokens)[0]
        if t3_speech_pos_emb is not None:
            cond_prompt_speech_emb = cond_prompt_speech_emb + t3_speech_pos_emb(
                t3_cond_prompt_tokens
            )
        need = int(t3_config.speech_cond_prompt_len)
        if cond_prompt_speech_emb.shape[0] < need:
            pad = torch.zeros(
                need - cond_prompt_speech_emb.shape[0],
                cond_prompt_speech_emb.shape[-1],
                device=cond_prompt_speech_emb.device,
                dtype=cond_prompt_speech_emb.dtype,
            )
            cond_prompt_speech_emb = torch.cat([cond_prompt_speech_emb, pad], dim=0)
        elif cond_prompt_speech_emb.shape[0] > need:
            cond_prompt_speech_emb = cond_prompt_speech_emb[:need]

        cond_emb = t3_cond_enc(
            T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                cond_prompt_speech_emb=cond_prompt_speech_emb,
                emotion_adv=0.5 * torch.ones(1, 1),
            ).to(device=target_device)
        ).to(device="cpu")

    # Free Phase 0 models — no longer needed
    del ve, s3gen, default_conds
    del t3_speech_emb, t3_speech_pos_emb, t3_cond_enc
    del t3_weights, t3_config
    gc.collect()
    torch.cuda.empty_cache()

    return cond_emb


@dataclass
class S3GenWorkItem:
    """Work item for S3Gen worker threads."""
    index: int                    # For result ordering
    speech_tokens: torch.Tensor   # Input tokens
    ref_dict: dict                # Voice reference (shared across workers)
    diffusion_steps: int          # Number of diffusion steps


@dataclass
class S3GenResult:
    """Result from S3Gen worker thread."""
    index: int                    # For result ordering
    wav: torch.Tensor             # Generated audio (on CPU)
    error: Optional[Exception]    # Error if any


def s3gen_worker(
    worker_id: int,
    s3gen_model: S3Gen,
    work_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event
):
    """Worker thread that processes S3Gen inference requests."""
    while not stop_event.is_set():
        try:
            work_item = work_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill to stop worker
                break

            # Process with error handling
            try:
                with torch.inference_mode():
                    wav, _ = s3gen_model.inference(
                        speech_tokens=work_item.speech_tokens,
                        ref_dict=work_item.ref_dict,
                        n_timesteps=work_item.diffusion_steps,
                    )
                    result = S3GenResult(
                        index=work_item.index,
                        wav=wav.cpu(),  # Move to CPU immediately to free VRAM
                        error=None
                    )
            except Exception as e:
                result = S3GenResult(
                    index=work_item.index,
                    wav=None,
                    error=e
                )

            result_queue.put(result)
            work_queue.task_done()

        except queue.Empty:
            continue


def determine_worker_count(target_workers: int = 2) -> int:
    """Determine safe worker count based on available VRAM."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    available = total - allocated

    required_per_worker = 0.8  # GB per worker (conservative estimate)
    required_total = target_workers * required_per_worker

    if available < required_total:
        print(f"[WARNING] Insufficient VRAM ({available:.2f}GB available, "
              f"{required_total:.2f}GB needed). Falling back to 1 worker.")
        return 1

    return target_workers


class ChatterboxTTS:
    """Defines the ChatterboxTTS class with model components and initialization."""
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, target_device: str, max_model_len: int,
                 t3: LLM, t3_config: T3Config, t3_cond_enc: T3CondEnc,
                 t3_speech_emb: torch.nn.Embedding, t3_speech_pos_emb: LearnedPositionEmbeddings,
                 s3gen: Optional[S3Gen] = None, ve: Optional[VoiceEncoder] = None,
                 default_conds: Optional[Conditionals] = None,
                 variant: str = "english"):
        """Store model components; s3gen/ve/default_conds are None in T3-only (Phase 1) mode."""
        self.target_device = target_device
        self.max_model_len = max_model_len
        self.t3 = t3
        self.t3_config = t3_config
        self.t3_cond_enc = t3_cond_enc
        self.t3_speech_emb = t3_speech_emb
        self.t3_speech_pos_emb = t3_speech_pos_emb

        self.s3gen = s3gen
        self.ve = ve
        self.default_conds = default_conds
        self.variant = variant

    @property
    def sr(self) -> int:
        """Sample rate of synthesized audio"""
        return S3GEN_SR

    @classmethod
    def from_local(cls, ckpt_dir: str | Path, target_device: str = "cuda", 
                   max_model_len: int = 1000, compile: bool = False,
                   max_batch_size: int = 10,
                   variant: str = "english",

                   # Original Chatterbox defaults this to False. I don't see a substantial performance difference when running with FP16.
                   s3gen_use_fp16: bool = False,

                   # Phase 1 mode: skip VoiceEncoder/S3Gen/default conds (~0.5GB VRAM saved).
                   # Use compute_conditionals() beforehand and pass cond_emb to generate_speech_tokens().
                   load_t3_only: bool = False,
                   **kwargs) -> 'ChatterboxTTS':
        """Load ChatterboxTTS from a local checkpoint dir; load_t3_only=True loads just the vLLM T3 stack."""
        ckpt_dir = Path(ckpt_dir)

        t3_config = T3Config()
        if variant == "turbo":
            t3_config.speech_tokens_dict_size = 6563
            t3_config.speech_cond_prompt_len = 375
            t3_config.use_perceiver_resampler = False
            t3_config.emotion_adv = False

        # Load *just* the necessary weights to perform inference with T3CondEnc
        t3_ckpt_path = resolve_t3_weight_file(ckpt_dir, variant)
        print(f"[vLLM] Loading T3 cond weights from {t3_ckpt_path}")
        t3_weights = load_file(t3_ckpt_path)

        t3_enc = T3CondEnc(t3_config)
        t3_enc.load_state_dict({ k.replace('cond_enc.', ''):v for k,v in t3_weights.items() if k.startswith('cond_enc.') }, strict=False)
        t3_enc = t3_enc.to(device=target_device).eval()

        t3_speech_emb = torch.nn.Embedding(t3_config.speech_tokens_dict_size, t3_config.n_channels)
        t3_speech_emb.load_state_dict({ k.replace('speech_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_emb.') })
        t3_speech_emb = t3_speech_emb.to(device=target_device).eval()

        if variant == "turbo":
            t3_speech_pos_emb = None
        else:
            t3_speech_pos_emb = LearnedPositionEmbeddings(t3_config.max_speech_tokens + 2 + 2, t3_config.n_channels)
            t3_speech_pos_emb.load_state_dict({ k.replace('speech_pos_emb.', ''):v for k,v in t3_weights.items() if k.startswith('speech_pos_emb.') })
            t3_speech_pos_emb = t3_speech_pos_emb.to(device=target_device).eval()

        # File-size-based VRAM heuristic for vLLM gpu_memory_utilization.
        # Measure the resolved T3 safetensors on disk; bf16 on GPU is ~half of fp32 on disk.
        # Deliberately avoids torch.cuda.memory_allocated() for deterministic behavior.
        model_gb = t3_ckpt_path.stat().st_size / (1024**3) * 0.5
        # System reserve + activation overhead (larger when S3Gen/VE share the card)
        reserve_gb = 0.5 + (0.0 if load_t3_only else 1.5)
        vllm_memory_needed = (model_gb + reserve_gb) * 1024**3
        vllm_memory_needed += max_batch_size * max_model_len * 1024 * 128

        total_gpu_memory = torch.cuda.get_device_properties(0).total_memory
        vllm_memory_percent = vllm_memory_needed / total_gpu_memory

        print(f"Giving vLLM {vllm_memory_percent * 100:.2f}% of GPU memory ({vllm_memory_needed / 1024**2:.2f} MB)")

        # Guard against runaway KV-cache allocation on 8GB cards
        safe_limit = int(os.environ.get("VLLM_MAX_MODEL_LEN_SAFE", "1200") or 1200)
        if max_model_len > safe_limit:
            logging.info("[vLLM] Clamping max_model_len %s -> %s", max_model_len, safe_limit)
            max_model_len = safe_limit

        # Resolve the vLLM model dir (config.json + tokenizer.json + ONE weights file).
        # Anchored to the project root so this works from any CWD. The top-level
        # ./t3-model is only usable if it isn't polluted with extra *.safetensors
        # (vLLM loads every safetensors file in the dir as a shard).
        project_root = Path(__file__).resolve().parents[2]
        dir_name = "t3-model" if variant == "english" else "t3-model-multilingual"
        try:
            override = config.vllm_t3_model_dir()
        except Exception:
            override = (config.VLLM_ENGLISH_CKPT_DIR if variant == "english"
                        else config.VLLM_MULTILINGUAL_CKPT_DIR)
        candidates = ([Path(override)] if override else []) + [
            project_root / "chatterbox-vllm" / dir_name,
            project_root / dir_name,
        ]
        t3_model_dir = next(
            (d for d in candidates
             if (d / "config.json").exists()
             and len(list(d.glob("*.safetensors"))) == 1),
            candidates[-1],
        )
        print(f"[vLLM] Using model dir: {t3_model_dir}")

        if variant == "english":
            tok_name = "EnTokenizer"
        elif variant == "turbo":
            tok_name = "TurboTokenizer"
        else:
            tok_name = "MtlTokenizer"
        base_vllm_kwargs = {
            "model": str(t3_model_dir),
            "task": "generate",
            "tokenizer": tok_name,
            "tokenizer_mode": "custom",
            "gpu_memory_utilization": vllm_memory_percent,
            "enforce_eager": not compile,
            "max_model_len": max_model_len,
        }
        # Turbo cond embeddings are 376 tokens. vLLM's encoder cache is ~8192
        # tokens, so ~21 Turbo prompts fit. A full-book generate() overflows
        # that cache, splits prefill, and CUDA-asserts in speech_emb.
        if variant == "turbo":
            base_vllm_kwargs["max_num_seqs"] = 8
            base_vllm_kwargs["enable_chunked_prefill"] = False

        # Verified engine config (2026-07-08): V1 engine, in-process. The EngineCore
        # subprocess of the default multiprocessing mode cannot see our custom
        # EnTokenizer/T3VllmModel registrations (TokenizerRegistry is a plain
        # in-process dict), so it must run in this process.
        os.environ["VLLM_USE_V1"] = "1"
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        t3 = LLM(**{**base_vllm_kwargs, **kwargs})

        if load_t3_only:
            # Phase 1 mode: no VE/S3Gen/default conds; cond_emb comes from compute_conditionals()
            return cls(
                target_device=target_device, max_model_len=max_model_len,
                t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc,
                t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
                variant=variant,
            )

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve = ve.to(device=target_device).eval()

        s3gen = S3Gen(use_fp16=s3gen_use_fp16)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen = s3gen.to(device=target_device)
        if s3gen_use_fp16:
            s3gen = s3gen.half()  # Convert all model weights to FP16
            # Keep tokenizer in FP32 (it doesn't benefit from FP16 and causes dtype issues)
            s3gen.tokenizer = s3gen.tokenizer.float()
        s3gen = s3gen.eval()

        default_conds = Conditionals.load(ckpt_dir / "conds.pt")
        default_conds.to(device=target_device)

        return cls(
            target_device=target_device, max_model_len=max_model_len,
            t3=t3, t3_config=t3_config, t3_cond_enc=t3_enc, t3_speech_emb=t3_speech_emb, t3_speech_pos_emb=t3_speech_pos_emb,
            s3gen=s3gen, ve=ve, default_conds=default_conds,
            variant=variant,
        )

    @classmethod
    def from_pretrained(cls,
                        repo_id: str = REPO_ID,
                        revision: str = "1b475dffa71fb191cb6d5901215eb6f55635a9b6",
                        *args, **kwargs) -> 'ChatterboxTTS':
        """Loads English model components from Hugging Face repository."""
        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / "t3_cfg.safetensors"
        model_safetensors_path = Path.cwd() / "t3-model" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="english", *args, **kwargs)

    @classmethod
    def from_pretrained_multilingual(cls,
                                    repo_id: str = REPO_ID,
                                    revision: str = "05e904af2b5c7f8e482687a9d7336c5c824467d9",
                                    *args, **kwargs) -> 'ChatterboxTTS':
        """Loads multilingual model components from Hugging Face repository."""
        t3_name = config.t3_weights_filename()
        if not t3_name.startswith("t3_mtl"):
            t3_name = "t3_mtl23ls_v3.safetensors"
        for fpath in ["ve.safetensors", t3_name, "s3gen.safetensors", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"]:
            local_path = hf_hub_download(repo_id=repo_id, filename=fpath, revision=revision)

        # Ensure the symlink in './t3-model-multilingual/model.safetensors' points to t3_cfg_path
        t3_cfg_path = Path(local_path).parent / t3_name
        model_safetensors_path = Path.cwd() / "t3-model-multilingual" / "model.safetensors"
        model_safetensors_path.unlink(missing_ok=True)
        model_safetensors_path.symlink_to(t3_cfg_path)

        return cls.from_local(Path(local_path).parent, variant="multilingual", *args, **kwargs)
    
    def get_supported_languages(self) -> dict[str, str]:
        """Return dictionary of supported language codes and names."""
        if self.variant == "multilingual":
            return SUPPORTED_LANGUAGES.copy()
        else:
            return { "en": "English" }

    @lru_cache(maxsize=10)
    def get_audio_conditionals(self, wav_fpath: Optional[str] = None) -> Tuple[dict[str, Any], torch.Tensor]:
        """Retrieves audio conditionals from a WAV file or default settings."""
        if wav_fpath is None:
            s3gen_ref_dict = self.default_conds.gen
            t3_cond_prompt_tokens = self.default_conds.t3.cond_prompt_speech_tokens
            ve_embed = self.default_conds.t3.speaker_emb
        else:
            ## Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

            s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR)

            # Speech cond prompt tokens
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=self.t3_config.speech_cond_prompt_len)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens)

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        cond_prompt_speech_emb = self.t3_speech_emb(t3_cond_prompt_tokens)[0] + self.t3_speech_pos_emb(t3_cond_prompt_tokens)

        cond_emb = self.t3_cond_enc(T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            cond_prompt_speech_emb=cond_prompt_speech_emb,
            emotion_adv=0.5 * torch.ones(1, 1)
        ).to(device=self.target_device)).to(device="cpu")  # Conditionals need to be given to VLLM in CPU

        return s3gen_ref_dict, cond_emb

    def update_exaggeration(self, cond_emb: torch.Tensor, exaggeration: float) -> torch.Tensor:
        """Adjusts condition embedding for text generation based on exaggeration factor."""
        if exaggeration == 0.5 or self.variant == "turbo":
            return cond_emb

        new_cond_emb = cond_emb.clone()
        new_cond_emb[-1] = self.t3_cond_enc.emotion_adv_fc(
            (exaggeration * torch.ones(1, 1)).to(self.target_device)
        ).to('cpu')
        return new_cond_emb

    def generate(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000, # Capped at max_model_len

        # From original Chatterbox HF generation args
        top_p=0.8,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        """Produces audio from text prompts, supporting optional audio prompts."""
        s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        return self.generate_with_conds(
            prompts=prompts,
            s3gen_ref=s3gen_ref,
            cond_emb=cond_emb,
            temperature=temperature,
            language_id=language_id,
            exaggeration=exaggeration,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            *args, **kwargs
        )

    def _parallel_s3gen_inference(
        self,
        batch_results,
        s3gen_ref: dict,
        diffusion_steps: int,
        num_workers: int = 2
    ) -> list:
        """Process S3Gen inference in parallel using worker threads."""

        # Create queues
        work_queue = queue.Queue(maxsize=50)
        result_queue = queue.Queue()
        stop_event = threading.Event()

        # Start worker threads
        workers = []
        for worker_id in range(num_workers):
            worker = threading.Thread(
                target=s3gen_worker,
                args=(worker_id, self.s3gen, work_queue, result_queue, stop_event),
                daemon=False
            )
            worker.start()
            workers.append(worker)

        # Enqueue all work items
        total_items = 0
        for i, batch_result in enumerate(batch_results):
            for output in batch_result.outputs:
                # Preprocess tokens
                speech_tokens = torch.tensor(
                    [token - SPEECH_TOKEN_OFFSET for token in output.token_ids],
                    device="cuda"
                )
                speech_tokens = drop_invalid_tokens(speech_tokens)
                speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]

                work_item = S3GenWorkItem(
                    index=total_items,
                    speech_tokens=speech_tokens,
                    ref_dict=s3gen_ref,
                    diffusion_steps=diffusion_steps
                )
                work_queue.put(work_item)
                total_items += 1

        # Send poison pills to stop workers
        for _ in range(num_workers):
            work_queue.put(None)

        # Collect results (maintain order via index)
        results_dict = {}
        for _ in range(total_items):
            result = result_queue.get()
            if result.error:
                stop_event.set()  # Stop other workers on error
                for worker in workers:
                    worker.join(timeout=5.0)
                raise result.error
            results_dict[result.index] = result.wav

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=10.0)

        # Return results in original order
        return [results_dict[i] for i in range(total_items)]

    def generate_with_conds(
        self,
        prompts: Union[str, list[str]],
        s3gen_ref: dict[str, Any],
        cond_emb: torch.Tensor,
        language_id: Optional[str] = 'en',
        temperature: float = 0.8,
        exaggeration: float = 0.5,
        max_tokens=1000, # Capped at max_model_len

        # Number of diffusion steps to use for S3Gen
        # The original Chatterbox uses 10. 5 is often enough for good quality audio, though some quality loss can be detected.
        # This can be as low as 2 or 3 for faster generation, though the audio quality will degrade substantially.
        diffusion_steps: int = 5,  # OPTIMIZED: Reduced from 10 for 2× speedup

        # From original Chatterbox HF generation args
        top_p=1.0,
        min_p=0.05,
        repetition_penalty=2.0,

        # Supports anything in https://docs.vllm.ai/en/v0.9.2/api/vllm/index.html?h=samplingparams#vllm.SamplingParams
        *args, **kwargs,
    ) -> list[any]:
        """Generates text with given conditions using S3Gen and transformer models."""
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate language_id
        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        # Norm and tokenize text
        prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in prompts]

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            # Use angle brackets to avoid conflicts with other start/stop tokens.
            # This will be parsed and replaced in the tokenizer.
            prompts = [f"<{language_id.lower()}>{p}" for p in prompts]

        # Use torch.no_grad() instead of inference_mode() to avoid interference with inner contexts
        with torch.no_grad():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for text in prompts
                ],
                sampling_params=SamplingParams(
                    temperature=temperature,

                    stop_token_ids=[self.t3_config.stop_speech_token + SPEECH_TOKEN_OFFSET],
                    max_tokens=min(max_tokens, self.max_model_len),
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,

                    *args, **kwargs,
                )
            )
            t3_gen_time = time.time() - start_time
            print(f"[T3] Speech Token Generation time: {t3_gen_time:.2f}s")

            # run torch gc
            torch.cuda.empty_cache()

            # Monitor VRAM after T3
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[VRAM] Allocated after T3: {allocated:.2f} GB")

            # Determine worker count based on VRAM and batch size
            if S3GEN_ENABLE_PARALLEL and len(batch_results) > 1:
                num_workers = determine_worker_count(S3GEN_NUM_WORKERS)
            else:
                num_workers = 1  # Disable parallel for single prompts

            start_time = time.time()

            # Use parallel or sequential path
            if num_workers > 1:
                print(f"[S3Gen] Using {num_workers} parallel workers (FP16 shared model)")
                # DEBUG: Print actual diffusion_steps being used
                print(f"[DEBUG] Using n_timesteps={diffusion_steps}")
                results = self._parallel_s3gen_inference(
                    batch_results, s3gen_ref, diffusion_steps, num_workers
                )
            else:
                print(f"[S3Gen] Using sequential processing (fallback)")
                # Keep existing sequential code as fallback
                results = []
                for i, batch_result in enumerate(batch_results):
                    for output in batch_result.outputs:
                        if i % 5 == 0:
                            print(f"[S3] Processing prompt {i} of {len(batch_results)}")

                        speech_tokens = torch.tensor([token - SPEECH_TOKEN_OFFSET for token in output.token_ids], device="cuda")
                        speech_tokens = drop_invalid_tokens(speech_tokens)
                        speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]

                        # DEBUG: Print actual diffusion_steps being used
                        if i == 0:
                            print(f"[DEBUG] Using n_timesteps={diffusion_steps}")

                        wav, _ = self.s3gen.inference(
                            speech_tokens=speech_tokens,
                            ref_dict=s3gen_ref,
                            n_timesteps=diffusion_steps,
                        )
                        results.append(wav.cpu())

                        # Free CUDA reference but DON'T clear cache (performance optimization)
                        # PyTorch will manage memory efficiently without manual intervention
                        del wav

            # Clear cache only once at the end of the batch (not per-prompt)
            torch.cuda.empty_cache()
            s3gen_gen_time = time.time() - start_time
            print(f"[S3Gen] Waveform Generation time: {s3gen_gen_time:.2f}s")

            return results
        
    def generate_speech_tokens(
        self,
        prompts: Union[str, list[str]],
        audio_prompt_path: Optional[str] = None,
        cond_emb: Optional[torch.Tensor] = None,
        language_id: Optional[str] = 'en',
        exaggeration: float = 0.5,
        temperature: float = 0.8,
        max_tokens=1000,
        top_p=0.8,
        repetition_penalty=2.0,
        use_tqdm=True,

        *args, **kwargs,
    ) -> List[List[int]]:
        """Generate raw speech token lists via vLLM T3 (Phase 1 of the phased pipeline).

        Multi-sentence prompts are generated one sentence at a time with a
        word-count min_tokens floor so speech-stop cannot fire at the first
        period. Token lists are concatenated back into one list per input prompt.

        Args:
            prompts: Input text prompt(s).
            audio_prompt_path: Voice WAV; used only when cond_emb is not given
                (requires the full model, i.e. load_t3_only=False).
            cond_emb: Pre-computed conditioning from compute_conditionals();
                lets this run in T3-only mode with no VE/S3Gen loaded.
            language_id: Language for the multilingual variant.
            exaggeration: Emotion exaggeration; applied to cond_emb via the cond encoder.
            temperature/max_tokens/top_p/repetition_penalty: vLLM sampling params.
            use_tqdm: True for vLLM's own progress bar, False to disable, or a
                tqdm-compatible callable (see vllm.LLM._run_engine) to relay
                real per-request completion progress elsewhere (e.g. a GUI).

        Returns:
            One list of speech token ids per original prompt (sentences concatenated).
        """
        if cond_emb is None:
            _s3gen_ref, cond_emb = self.get_audio_conditionals(audio_prompt_path)

        if isinstance(prompts, str):
            prompts = [prompts]

        if language_id and language_id.lower() not in self.get_supported_languages():
            supported_langs = ", ".join(self.get_supported_languages().keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        cond_emb = self.update_exaggeration(cond_emb, exaggeration)

        sentence_groups: List[List[str]] = []
        for prompt in prompts:
            parts = split_t3_sentences(prompt)
            sentence_groups.append(parts if parts else [prompt])
        flat_source = [sentence for group in sentence_groups for sentence in group]

        # Norm and tokenize text. Turbo GPT2 has no [START]/[STOP] text specials.
        if self.variant == "turbo":
            flat_prompts = [punc_norm(p) for p in flat_source]
            from chatterbox_vllm.models.t3.t3_turbo import SPEECH_TOKEN_OFFSET as _off
        else:
            flat_prompts = ["[START]" + punc_norm(p) + "[STOP]" for p in flat_source]
            _off = SPEECH_TOKEN_OFFSET

        # For multilingual, prepend the language token
        if self.variant == "multilingual":
            flat_prompts = [f"<{language_id.lower()}>{p}" for p in flat_prompts]

        # Strip pipeline-level kwargs that SamplingParams would reject
        for _key in ('diffusion_steps', 'cfg_weight', 'language', 'min_tokens'):
            kwargs.pop(_key, None)

        capped_max = min(max_tokens, self.max_model_len)
        stop_ids = [self.t3_config.stop_speech_token + _off]
        sampling_params = [
            SamplingParams(
                temperature=temperature,
                stop_token_ids=stop_ids,
                max_tokens=capped_max,
                min_tokens=min_speech_tokens_for_text(source, capped_max),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                detokenize=self.variant != "turbo",
                *args,
                **kwargs,
            )
            for source in flat_source
        ]

        with torch.inference_mode():
            start_time = time.time()
            batch_results = self.t3.generate(
                [
                    {
                        "prompt": text,
                        "multi_modal_data": {
                            "conditionals": [cond_emb],
                        },
                    }
                    for text in flat_prompts
                ],
                sampling_params=sampling_params,
                use_tqdm=use_tqdm,
            )
            print(f"[T3] Speech token generation time: {time.time() - start_time:.2f}s")

            flat_speech_tokens = []
            for batch_result in batch_results:
                for output in batch_result.outputs:
                    if not output.token_ids:
                        flat_speech_tokens.append([])
                        continue
                    speech_tokens = torch.tensor([token - _off for token in output.token_ids], device="cpu")
                    speech_tokens = drop_invalid_tokens(speech_tokens)
                    speech_tokens = speech_tokens[(speech_tokens >= 0) & (speech_tokens < SPEECH_VOCAB_SIZE)]
                    flat_speech_tokens.append(speech_tokens.tolist())

            stitched: List[List[int]] = []
            cursor = 0
            for group in sentence_groups:
                combined: List[int] = []
                for _ in group:
                    if cursor < len(flat_speech_tokens):
                        combined.extend(flat_speech_tokens[cursor])
                        cursor += 1
                stitched.append(combined)
            return stitched

    def generate_from_tokens(self, speech_tokens_list: List[List[int]], s3gen_ref: dict, cond_emb: torch.Tensor, diffusion_steps: int = 10) -> List:
        """Decode pre-generated speech tokens to waveforms with S3Gen (Phase 2 helper).

        Args:
            speech_tokens_list: One token list per text segment.
            s3gen_ref: S3Gen reference dict from get_audio_conditionals().
            cond_emb: Voice conditioning embedding (unused by S3Gen; kept for API symmetry).
            diffusion_steps: S3Gen flow-matching steps.

        Returns:
            List of audio waveforms (CPU tensors), one per segment.
        """
        if not speech_tokens_list:
            return []

        results = []
        for i, tokens in enumerate(speech_tokens_list):
            if i % 5 == 0:
                print(f"[S3] Processing segment {i} of {len(speech_tokens_list)}")

            speech_tokens = torch.tensor(tokens, device=self.target_device)
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens[speech_tokens < SPEECH_VOCAB_SIZE]

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=s3gen_ref,
                n_timesteps=diffusion_steps,
            )
            results.append(wav.cpu())

            # Periodically release fragmentation from long decode runs
            if i % 10 == 0:
                torch.cuda.empty_cache()

        return results

    def shutdown(self):
        """Release vLLM engine, worker, cache, and wrapper GPU references.

        vLLM V1's in-process executor does not reliably release its worker and
        KV-cache tensors through its nominal shutdown method alone. This method
        explicitly shuts down the engine core, severs the executor/worker/cache
        references, and then clears the wrapper so Phase 2 can load S3Gen.
        """
        t3 = getattr(self, "t3", None)
        llm_engine = getattr(t3, "llm_engine", None)
        engine_core = getattr(llm_engine, "engine_core", None)
        if engine_core is not None:
            engine_core.shutdown()

            # EngineCore retains the executor, and the executor retains the
            # worker/model/KV cache. Break this ownership chain explicitly.
            engine_core.model_executor = None

        if llm_engine is not None:
            llm_engine.engine_core = None
            llm_engine.model_executor = None

        del t3
        del engine_core
        del llm_engine
        # T3CondEnc and the two speech embeddings are Phase-1 CUDA modules.
        # They are not owned by vLLM's executor, so executor shutdown cannot
        # release them.  Leaving these attributes alive retained about 3 GB
        # after English T3 runs and prevented Parakeet from loading.
        for attr in (
            't3',
            't3_config',
            't3_cond_enc',
            't3_speech_emb',
            't3_speech_pos_emb',
            's3gen',
            've',
            'default_conds',
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        # The cached method can retain voice conditional GPU tensors when this
        # class was used outside T3-only Phase 1.
        self.get_audio_conditionals.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
