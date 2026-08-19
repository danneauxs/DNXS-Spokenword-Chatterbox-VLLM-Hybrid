from __future__ import annotations

import logging
import os
import sys
import queue
from pathlib import Path
from threading import Thread, Event
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from pydub import AudioSegment
from tqdm import tqdm

from config.config import *
from modules.audio_processor import process_audio_with_trimming_and_silence, add_chunk_end_silence
import json

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Add chatterbox-vllm to Python path (but don't shadow standard imports)
vllm_root = project_root / "chatterbox-vllm" / "src"
sys.path.append(str(vllm_root))

# Set environment variable to limit PyTorch CPU threads for better concurrency
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Performance optimizations
PARALLEL_S3GEN_WORKERS = int(os.environ.get("S3GEN_NUM_WORKERS", "2"))
ENABLE_PARALLEL_S3GEN = os.environ.get("S3GEN_ENABLE_PARALLEL", "true").lower() in ("true", "1")

@dataclass
class S3GenWorkItem:
    """Work item for parallel S3Gen processing."""
    index: int
    speech_tokens: torch.Tensor
    chunk_meta: Dict[str, Any]
    s3gen_ref: Optional[Dict] = None
    diffusion_steps: int = 25

@dataclass
class S3GenResult:
    """Result from parallel S3Gen processing."""
    index: int
    audio_tensor: Optional[torch.Tensor]
    chunk_meta: Dict[str, Any]
    error: Optional[Exception] = None

def determine_worker_count(target_workers: int = 2) -> int:
    """Determine safe worker count based on available VRAM."""
    try:
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        available = total - allocated

        required_per_worker = 0.8  # GB per worker
        required_total = target_workers * required_per_worker

        logger.info(f"[VRAM] Target workers: {target_workers}, Available: {available:.2f}GB, Required: {required_total:.2f}GB")

        if available < required_total:
            logger.warning(f"[VRAM] Insufficient VRAM. Using 1 worker instead of {target_workers}.")
            return 1

        return target_workers
    except Exception as e:
        logger.warning(f"[VRAM] Could not check VRAM: {e}. Using 1 worker.")
        return 1

def s3gen_worker(worker_id: int, full_model, work_queue: queue.Queue,
                 result_queue: queue.Queue, stop_event: Event, cuda_stream=None):
    """Worker thread that processes audio generation requests using the full model."""
    processed_count = 0
    while not stop_event.is_set():
        try:
            work_item = work_queue.get(timeout=1.0)
            if work_item is None:  # Poison pill
                logger.info(f"[Audio Worker {worker_id}] Processed {processed_count} chunks, shutting down")
                break

            try:
                # Prepare cond_emb from s3gen_ref (matches sequential code)
                cond_emb = None
                if work_item.s3gen_ref and "cond_emb" in work_item.s3gen_ref:
                    try:
                        cond_emb = torch.tensor(work_item.s3gen_ref["cond_emb"], device=work_item.speech_tokens.device).unsqueeze(0)
                    except Exception:
                        logger.warning(f"[Audio Worker {worker_id}] Failed to load cond_emb for chunk {work_item.index}")

                # Use CUDA stream for parallel GPU execution if provided
                if cuda_stream is not None:
                    with torch.cuda.stream(cuda_stream):
                        with torch.inference_mode():
                            # Check if this is turbo S3Gen
                            if hasattr(full_model, 's3gen') and hasattr(full_model.s3gen, 'meanflow'):
                                # Turbo S3Gen path (2 diffusion steps)
                                # Debug logging (once for first chunk from first worker)
                                if worker_id == 0 and work_item.index == 0:
                                    logger.info(f"[Worker {worker_id}] Turbo mode detected (meanflow=True)")
                                    logger.info(f"[Worker {worker_id}] _cached_s3gen_ref: {full_model._cached_s3gen_ref is not None}")
                                    logger.info(f"[Worker {worker_id}] work_item.s3gen_ref: {work_item.s3gen_ref is not None}")

                                # Use cached ref_dict from model wrapper instead of work_item
                                ref_dict = full_model._cached_s3gen_ref or work_item.s3gen_ref

                                if worker_id == 0 and work_item.index == 0:
                                    logger.info(f"[Worker {worker_id}] final ref_dict: {ref_dict is not None}")

                                if ref_dict is None:
                                    raise RuntimeError(f"No voice conditioning available for turbo S3Gen (chunk {work_item.index})")

                                wav, _ = full_model.s3gen.inference(
                                    speech_tokens=work_item.speech_tokens.squeeze(0),
                                    ref_dict=ref_dict,
                                    n_cfm_timesteps=2,
                                )
                                # wav is already a Tensor, not numpy array
                                audio_tensor = wav.unsqueeze(0).to(full_model.device)
                            else:
                                # Standard ChatterboxTTS path
                                audio_tensor = full_model.generate_from_tokens(
                                    speech_tokens=work_item.speech_tokens
                                )
                        cuda_stream.synchronize()
                        result = S3GenResult(
                            index=work_item.index,
                            audio_tensor=audio_tensor.cpu(),
                            chunk_meta=work_item.chunk_meta,
                            error=None
                        )
                else:
                    # Fallback to default stream
                    with torch.inference_mode():
                        # Check if this is turbo S3Gen
                        if hasattr(full_model, 's3gen') and hasattr(full_model.s3gen, 'meanflow'):
                            # Turbo S3Gen path (2 diffusion steps)
                            # Debug logging (once for first chunk from first worker)
                            if worker_id == 0 and work_item.index == 0:
                                logger.info(f"[Worker {worker_id}] Turbo mode detected (meanflow=True)")
                                logger.info(f"[Worker {worker_id}] _cached_s3gen_ref: {full_model._cached_s3gen_ref is not None}")
                                logger.info(f"[Worker {worker_id}] work_item.s3gen_ref: {work_item.s3gen_ref is not None}")

                            # Use cached ref_dict from model wrapper instead of work_item
                            ref_dict = full_model._cached_s3gen_ref or work_item.s3gen_ref

                            if worker_id == 0 and work_item.index == 0:
                                logger.info(f"[Worker {worker_id}] final ref_dict: {ref_dict is not None}")

                            if ref_dict is None:
                                raise RuntimeError(f"No voice conditioning available for turbo S3Gen (chunk {work_item.index})")

                            wav, _ = full_model.s3gen.inference(
                                speech_tokens=work_item.speech_tokens.squeeze(0),
                                ref_dict=ref_dict,
                                n_cfm_timesteps=2,
                            )
                            # wav is already a Tensor, not numpy array
                            audio_tensor = wav.unsqueeze(0).to(full_model.device)
                        else:
                            # Standard ChatterboxTTS path
                            audio_tensor = full_model.generate_from_tokens(
                                speech_tokens=work_item.speech_tokens
                            )
                        result = S3GenResult(
                            index=work_item.index,
                            audio_tensor=audio_tensor.cpu(),
                            chunk_meta=work_item.chunk_meta,
                            error=None
                        )
                processed_count += 1
            except Exception as e:
                logger.error(f"[Audio Worker {worker_id}] Error processing chunk {work_item.index}: {e}")
                result = S3GenResult(
                    index=work_item.index,
                    audio_tensor=None,
                    chunk_meta=work_item.chunk_meta,
                    error=e
                )

            # Free memory after each chunk
            torch.cuda.empty_cache()
            result_queue.put(result)
            work_queue.task_done()

        except queue.Empty:
            continue

# Global variables for model/device cache
_MODEL_CACHE: Dict[str, Any] = {}
_DEVICE_CACHE: Optional[torch.device] = None
_MAX_WORKERS = 4  # Default pool size

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def shutdown_cached_model():
    """
    Shutdown and unload cached Turbo model to free GPU resources.
    Call this after audio conversion is complete.
    """
    global _MODEL_CACHE
    if 'turbo_s3gen' in _MODEL_CACHE:
        model = _MODEL_CACHE['turbo_s3gen']
        if hasattr(model, 'shutdown'):
            model.shutdown()
        del _MODEL_CACHE['turbo_s3gen']
        logger.info("✅ Turbo model cache cleared")


def _get_device():
    """Load the model dynamically based on TTS_BACKEND and VLLM_MODEL_VARIANT.
    Args:
    config (dict): Configuration dictionary containing backend and model variant information.
    Returns:
    model: The loaded model instance.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        if torch.cuda.is_available():
            _DEVICE_CACHE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            _DEVICE_CACHE = torch.device("mps")
        else:
            _DEVICE_CACHE = torch.device("cpu")
    return _DEVICE_CACHE


def _load_model(config):
    """Dynamically load the model based on TTS_BACKEND and VLLM_MODEL_VARIANT"""
    global _MODEL_CACHE

    # Check if turbo-hybrid mode is requested
    backend = getattr(config, 'TTS_BACKEND', 'standard').lower()

    if backend == 'turbo-hybrid':
        # Load ONLY Turbo S3Gen (tokens come from VLLM Phase 1)
        cache_key = 'turbo_s3gen'

        if cache_key not in _MODEL_CACHE:
            try:
                from src.chatterbox_turbo.models.s3gen import S3Gen as TurboS3Gen
                from safetensors.torch import load_file

                device = _get_device()
                logger.info(f"Loading Turbo S3Gen (meanflow) for token-to-audio conversion on {device}...")

                # Determine turbo checkpoint directory
                turbo_ckpt = getattr(config, 'TURBO_CKPT_DIR', None)
                if turbo_ckpt is None:
                    # Try to auto-detect from HF cache
                    import os
                    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
                    candidates = list(hf_cache.glob("models--ResembleAI--chatterbox-turbo/snapshots/*"))
                    if candidates:
                        turbo_ckpt = candidates[0]
                    else:
                        logger.error("Turbo model not found. Set TURBO_CKPT_DIR or download from HF.")
                        return None

                turbo_ckpt = Path(turbo_ckpt)
                if not turbo_ckpt.exists():
                    logger.error(f"Turbo checkpoint directory not found: {turbo_ckpt}")
                    return None

                # Load Turbo S3Gen with meanflow
                s3gen = TurboS3Gen(meanflow=True)
                weights_path = turbo_ckpt / "s3gen_meanflow.safetensors"
                if not weights_path.exists():
                    logger.error(f"Turbo S3Gen weights not found: {weights_path}")
                    return None

                weights = load_file(weights_path)
                s3gen.load_state_dict(weights, strict=True)
                s3gen.to(device).eval()

                # Load voice encoder for conditioning
                from src.chatterbox_turbo.models.voice_encoder import VoiceEncoder
                ve = VoiceEncoder()
                ve_weights = load_file(turbo_ckpt / "ve.safetensors")
                ve.load_state_dict(ve_weights)
                ve.to(device).eval()

                # Wrap in a simple container to provide common interface
                class TurboS3GenWrapper:
                    """A class that wraps TurboS3Gen for generating speech and managing audio conditions.
                    Initialization sets up the S3Gen instance, vocal effects engine, and device.
                    Provides method to get audio conditioning from an audio file using torchaudio.
                    """
                    def __init__(self, s3gen, ve, device):
                        """Initializes an object with S3Gen, voice encoder, and device.
                        Args:
                        s3gen (S3Gen): The S3Gen instance.
                        ve (VoiceEncoder): The voice encoder instance.
                        device (str or torch.device): The device to run on, e.g., 'cuda' or 'cpu'.
                        Returns:
                        None
                        """
                        self.s3gen = s3gen
                        self.ve = ve
                        self.device = device
                        self.sr = 24000  # Turbo S3Gen sample rate
                        self._cached_s3gen_ref = None  # Cache ref_dict to avoid serialization issues

                    def get_audio_conditionals(self, voice_path):
                        """Get voice conditioning from audio file"""
                        try:
                            import torchaudio
                            from pathlib import Path

                            # Ensure voice_path is a string
                            voice_path_str = str(voice_path)
                            logger.info(f"[Turbo] Loading voice from: {voice_path_str}")

                            # Check file exists
                            if not Path(voice_path_str).exists():
                                raise FileNotFoundError(f"Voice file not found: {voice_path_str}")

                            # Load audio
                            audio, sr = torchaudio.load(voice_path_str)
                            logger.info(f"[Turbo] Loaded audio: shape={audio.shape}, sr={sr}")

                            # Resample if needed
                            if sr != 16000:
                                logger.info(f"[Turbo] Resampling from {sr} to 16000 Hz")
                                audio = torchaudio.functional.resample(audio, sr, 16000)

                            # Convert stereo to mono if needed
                            if audio.shape[0] > 1:
                                logger.info(f"[Turbo] Converting stereo to mono")
                                audio = audio.mean(dim=0, keepdim=True)

                            # Use embed_ref() to create properly structured ref_dict
                            with torch.inference_mode():
                                # embed_ref() expects 16kHz audio and handles all preprocessing:
                                # - Tokenizes with S3Tokenizer
                                # - Converts to 24kHz mel-spectrograms
                                # - Generates speaker embedding (x-vector)
                                s3gen_ref = self.s3gen.embed_ref(
                                    ref_wav=audio.squeeze(0),  # Already 16kHz from earlier processing
                                    ref_sr=16000,
                                    device=self.device
                                )
                                logger.info(f"[Turbo] Generated ref_dict with keys: {list(s3gen_ref.keys())}")

                            # Cache for parallel workers
                            self._cached_s3gen_ref = s3gen_ref
                            logger.info(f"[Turbo] ✅ Cached ref_dict successfully")

                            # Extract speaker embedding for return value
                            speaker_emb = s3gen_ref.get("embedding")

                            return s3gen_ref, speaker_emb

                        except Exception as e:
                            logger.error(f"[Turbo] ❌ get_audio_conditionals() failed: {e}", exc_info=True)
                            raise  # Re-raise to let caller handle it

                    def generate_from_tokens(self, speech_tokens):
                        """Generate audio from tokens using turbo inference"""
                        raise NotImplementedError("Use inference() method for turbo S3Gen")
                    
                    def shutdown(self):
                        """Unload Turbo S3Gen model and free GPU resources"""
                        del self.s3gen
                        del self.ve
                        import torch
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        print("🔌 Turbo S3Gen model unloaded and GPU resources cleared")

                wrapper = TurboS3GenWrapper(s3gen, ve, device)
                _MODEL_CACHE[cache_key] = wrapper
                logger.info(f"✅ Loaded Turbo S3Gen (meanflow=True, 2 diffusion steps)")

            except Exception as e:
                logger.error(f"Error loading Turbo S3Gen model: {e}", exc_info=True)
                return None

        return _MODEL_CACHE[cache_key]

    else:
        # Existing standard ChatterboxTTS loading path
        variant = config.VLLM_MODEL_VARIANT.lower()
        if variant not in _MODEL_CACHE:
            try:
                # Explicitly import standard ChatterboxTTS to avoid vLLM shadowing
                import importlib.util
                spec = importlib.util.spec_from_file_location("chatterbox.tts", str(project_root / "src" / "chatterbox" / "tts.py"))
                if spec is None or spec.loader is None:
                    raise ImportError("Could not load standard ChatterboxTTS module")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                ChatterboxTTS = module.ChatterboxTTS

                device = _get_device()
                logger.info(f"Loading ChatterboxTTS ({variant}) for token-to-audio conversion on {device}...")

                if variant == "multilingual":
                    ckpt_dir = config.VLLM_MULTILINGUAL_CKPT_DIR or Path("t3-model-multilingual")
                else:
                    variant = "english"
                    ckpt_dir = config.VLLM_ENGLISH_CKPT_DIR or Path("t3-model")

                if not ckpt_dir.exists():
                    raise FileNotFoundError(f"Model checkpoint directory not found: {ckpt_dir}")

                model = ChatterboxTTS.from_local(
                    ckpt_dir=str(ckpt_dir),
                    device=str(device),
                )
                # Configure supported parameters
                model.set_diffusion_steps(config.S3GEN_DIFFUSION_STEPS)
                model.set_fp16(config.VLLM_S3GEN_USE_FP16)
                _MODEL_CACHE[variant] = model
                logger.info("ChatterboxTTS model loaded for token-to-audio.")
            except ImportError:
                logger.error("VLLM backend not installed. Cannot perform token-to-audio conversion.")
                return None
            except Exception as e:
                logger.error(f"Error loading ChatterboxTTS model: {e}")
                return None
        return _MODEL_CACHE.get(variant)


def cleanup_model_cache():
    """
    Explicitly cleanup all cached models and free VRAM.
    Called after Phase 2 (token-to-audio conversion) to free GPU memory.
    """
    global _MODEL_CACHE

    logger.info("🧹 Cleaning up model cache...")

    for key, model in list(_MODEL_CACHE.items()):
        try:
            if hasattr(model, 'shutdown'):
                # Call shutdown() if model has it (TurboS3GenWrapper)
                logger.info(f"  Shutting down model: {key}")
                model.shutdown()
            elif hasattr(model, 's3gen'):
                # Fallback cleanup for turbo models
                logger.info(f"  Cleaning up turbo model: {key}")
                if hasattr(model, 's3gen'):
                    del model.s3gen
                if hasattr(model, 've'):
                    del model.ve
            elif hasattr(model, 'cpu'):
                # Standard model cleanup - move to CPU first
                logger.info(f"  Moving model to CPU and clearing: {key}")
                model.cpu()

            del model
        except Exception as e:
            logger.warning(f"Failed to cleanup model {key}: {e}")

    # Clear the cache dict
    _MODEL_CACHE.clear()
    logger.info("  Model cache cleared")

    # Force garbage collection
    import gc
    gc.collect()
    gc.collect()  # Call twice to ensure cleanup

    # Clear CUDA cache
    try:
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        logger.warning(f"CUDA cleanup warning: {e}")

    logger.info("✅ Model cache cleaned - VRAM freed")


def _tensor_to_audio_segment(audio_tensor, sample_rate: int) -> AudioSegment:
    """Converts a PyTorch tensor to a pydub AudioSegment."""
    # Ensure tensor is detached and on CPU
    wav_np = audio_tensor.detach().cpu().numpy()

    # Ensure it's a 1D array (mono)
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)

    # Scale the waveform data to pydub's expected range (16-bit PCM, roughly +/- 32767)
    # The VLLM model often outputs float32 in range [-1.0, 1.0]
    wav_int16 = (wav_np * 32767).astype(np.int16)

    # Create AudioSegment
    return AudioSegment(
        wav_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=wav_int16.dtype.itemsize,
        channels=1
    )


class TokenToAudioConverter:
    """
    Handles the conversion of pre-generated T3 tokens into audio chunks
    using the standard ChatterboxTTS model or a VLLM S3Gen process.
    """

    def __init__(self, json_path: Path, audio_output_dir: Path, voice_path: Optional[Path], config: Any, progress_callback=None):
        """Initializes a new instance of the class.
        Args:
        json_path (Path): The path to the JSON file containing chunks tokens.
        audio_output_dir (Path): The directory where audio chunks will be saved.
        voice_path (Optional[Path]): The path to the voice model or file, if applicable.
        config (Any): Configuration settings for the instance.
        progress_callback (Callable[[int, str], None], optional): Callback function for progress updates.
        """
        self.json_path = json_path
        self.audio_chunks_dir = audio_output_dir
        self.voice_path = voice_path
        self.config = config
        self.progress_callback = progress_callback
        self.model = None

    def _emit_progress(self, percentage: int, message: str):
        """Emits progress using a callback function.
        Args:
        - percentage (int): The current progress percentage.
        - message (str): A message describing the current progress.
        Returns: None
        Loads the chunks tokens JSON file.
        Args: None
        Returns:
        - dict: The loaded JSON data.
        Ensures the output directory exists.
        Args: None
        Returns: None
        """
        if self.progress_callback:
            self.progress_callback(percentage, message)
        logger.info(f"Progress: {percentage}% - {message}")

    def load_json(self):
        """Loads the chunks tokens JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Tokens JSON file not found at {self.json_path}")
        with open(self.json_path, 'r') as f:
            return json.load(f)

    def setup_output_dir(self):
        """Ensures the output directory exists."""
        self.audio_chunks_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Loads the appropriate S3Gen model."""
        self.model = _load_model(self.config)
        return self.model is not None

    def _process_one_chunk(self, chunk: Dict[str, Any]):
        """Processes a single token chunk and saves the audio."""
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")
        tokens = chunk.get("tokens")
        s3gen_ref = chunk.get("s3gen_ref")  # Cond embedding (optional)

        if tokens is None or self.model is None or chunk_id is None:
            logger.warning(f"Skipping chunk {chunk_id}: Missing tokens or model not loaded.")
            return False

        audio_path = self.audio_chunks_dir / f"{chunk_id}.wav"
        if audio_path.exists():
            logger.info(f"Skipping chunk {chunk_id}: Audio already exists.")
            return True

        # Pad tokens to expected length (60) if shorter
        if len(tokens) < 60:
            tokens = tokens + [0] * (60 - len(tokens))
        elif len(tokens) > 60:
            tokens = tokens[:60]  # Truncate if longer
        # Handle list of lists (multiple token sequences per chunk)
        if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
            # Flatten all token sequences for this chunk
            flattened_tokens = []
            for seq in tokens:
                flattened_tokens.extend(seq)
            tokens = flattened_tokens
        # Use full token sequence (no truncation)
        tokens_tensor = torch.tensor(tokens, device=_get_device()).unsqueeze(0)

        # Load conditional reference if present (e.g., from vllm_wav_generator)
        cond_emb = None
        if s3gen_ref:
            try:
                # Assuming s3gen_ref is a dict compatible with loading conds
                cond_emb = torch.tensor(s3gen_ref["cond_emb"], device=_get_device()).unsqueeze(0)
            except Exception:
                logger.error(f"Failed to load s3gen_ref for chunk {chunk_id}")
                pass

        try:
            # Generate audio from tokens using S3Gen decoder
            audio_tensor = self.model.generate_from_tokens(speech_tokens)

            # Apply post-processing (trimming + boundary silence)
            audio_segment = _tensor_to_audio_segment(audio_tensor, self.model.sr)

            # Get boundary_type from chunk data
            boundary_type = chunk.get("boundary_type", "none")

            # Apply trimming and boundary silence
            processed_audio = process_audio_with_trimming_and_silence(
                audio_segment=audio_segment,
                boundary_type=boundary_type,
                enable_trimming=ENABLE_AUDIO_TRIMMING
            )

            # Export processed audio to WAV
            processed_audio.export(str(audio_path), format="wav")

            # FIX: Add post-export silence (missing from prior implementation) to restore any truncated trailing silence
            add_chunk_end_silence(audio_path) # <<< FUNCTIONAL FIX APPLIED HERE

            return True

        except Exception as e:
            logger.error(f"Failed to convert tokens to audio for chunk {chunk_id}: {e}", exc_info=True)
            return False

    def process_tokens_in_batches(self, chunks: List[Dict[str, Any]]) -> int:
        """Process token chunks with parallel S3Gen workers for maximum speed."""
        total_chunks = len(chunks)
        processed_count = 0

        # Filter chunks that have tokens
        valid_chunks = [c for c in chunks if c.get("speech_tokens") or c.get("tokens")]

        if not valid_chunks:
            logger.warning("No chunks with tokens found")
            return 0

        # Determine optimal worker count based on VRAM
        num_workers = determine_worker_count(PARALLEL_S3GEN_WORKERS) if ENABLE_PARALLEL_S3GEN else 1

        if num_workers > 1:
            logger.info(f"[SPEED] Using {num_workers} parallel S3Gen workers for maximum performance")
        else:
            logger.info("[SPEED] Using sequential processing (VRAM limited)")

        # Get S3Gen reference for voice conditioning
        try:
            logger.info(f"Loading voice conditioning from: {self.voice_path}")
            s3gen_ref, _ = self.model.get_audio_conditionals(str(self.voice_path))
            diffusion_steps = getattr(self.config, 'S3GEN_DIFFUSION_STEPS', 25)
            logger.info(f"✅ Voice conditioning loaded successfully")

            # For turbo models, verify cache was set
            if hasattr(self.model, '_cached_s3gen_ref'):
                if self.model._cached_s3gen_ref is not None:
                    logger.info(f"✅ Turbo S3Gen ref_dict cached successfully")
                else:
                    logger.error(f"❌ Turbo S3Gen cache is still None after get_audio_conditionals()!")

        except Exception as e:
            logger.error(f"❌ Failed to get S3Gen reference: {e}", exc_info=True)

            # For turbo models, this is fatal - cannot proceed without ref_dict
            if hasattr(self.model, 's3gen') and hasattr(self.model.s3gen, 'meanflow'):
                raise RuntimeError(
                    f"Turbo-hybrid mode requires voice conditioning. "
                    f"Failed to load from {self.voice_path}: {e}"
                )

            # For standard models, try fallback
            logger.warning(f"Trying fallback conditioning for standard model")
            s3gen_ref = self.model.conds.gen if hasattr(self.model, 'conds') else None
            diffusion_steps = 25

        if num_workers > 1 and len(valid_chunks) > 1:
            # PARALLEL PROCESSING with CUDA streams
            work_queue = queue.Queue(maxsize=50)
            result_queue = queue.Queue()
            stop_event = Event()

            # Create CUDA streams for parallel GPU execution
            cuda_streams = [torch.cuda.Stream() for _ in range(num_workers)]
            logger.info(f"[SPEED] Created {num_workers} CUDA streams for parallel processing")

            # Start worker threads
            workers = []
            for worker_id in range(num_workers):
                worker = Thread(
                    target=s3gen_worker,
                    args=(worker_id, self.model, work_queue, result_queue, stop_event, cuda_streams[worker_id]),
                    daemon=False
                )
                worker.start()
                workers.append(worker)

            # Queue all work items
            for idx, chunk in enumerate(valid_chunks):
                tokens = chunk.get("speech_tokens") or chunk.get("tokens")
                print(f"DEBUG: chunk {idx}, tokens type: {type(tokens)}, len: {len(tokens) if hasattr(tokens, '__len__') else 'no len'}, first 5: {tokens[:5] if hasattr(tokens, '__getitem__') and len(tokens) > 0 else 'empty'}")
                # Handle list of lists (multiple token sequences per chunk)
                if isinstance(tokens, list) and len(tokens) > 0 and isinstance(tokens[0], list):
                    # Flatten all token sequences for this chunk
                    flattened_tokens = []
                    for seq in tokens:
                        flattened_tokens.extend(seq)
                    tokens = flattened_tokens
                # Use full token sequence (no truncation)
                speech_tokens = torch.tensor(tokens, device=_get_device()).unsqueeze(0)

                work_item = S3GenWorkItem(
                    index=idx,
                    speech_tokens=speech_tokens,
                    chunk_meta=chunk,
                    s3gen_ref=s3gen_ref,
                    diffusion_steps=diffusion_steps
                )
                work_queue.put(work_item)

            # Send poison pills
            for _ in range(num_workers):
                work_queue.put(None)

            # Collect results and process audio
            results_dict = {}
            for _ in range(len(valid_chunks)):
                result = result_queue.get()
                if result.error:
                    logger.error(f"Failed to process chunk {result.chunk_meta.get('chunk_id')}: {result.error}")
                    continue
                results_dict[result.index] = result

            # Wait for workers
            for worker in workers:
                worker.join(timeout=10.0)

            # Process results in order
            for idx in range(len(valid_chunks)):
                if idx not in results_dict:
                    continue

                result = results_dict[idx]
                chunk = result.chunk_meta
                chunk_id = chunk.get("chunk_id")

                try:
                    # Convert tensor to audio and save
                    audio_segment = _tensor_to_audio_segment(result.audio_tensor.squeeze(0), self.model.sr)

                    # Apply post-processing
                    boundary_type = chunk.get("boundary_type", "none")
                    processed_audio = process_audio_with_trimming_and_silence(
                        audio_segment=audio_segment,
                        boundary_type=boundary_type,
                        enable_trimming=getattr(self.config, 'ENABLE_AUDIO_TRIMMING', True)
                    )

                    # Save to file
                    audio_path = self.audio_chunks_dir / f"{chunk_id}.wav"
                    processed_audio.export(str(audio_path), format="wav")
                    add_chunk_end_silence(audio_path)

                    processed_count += 1

                    # Progress update
                    if processed_count % 5 == 0 or processed_count == len(valid_chunks):
                        pct = 10 + int((processed_count / total_chunks) * 85)  # Range 10-95%
                        self._emit_progress(pct, f"Generated {processed_count}/{total_chunks} chunks")

                except Exception as e:
                    logger.error(f"Failed to save audio for chunk {chunk_id}: {e}")

        else:
            # SEQUENTIAL FALLBACK
            logger.info("[SPEED] Using sequential processing")
            for idx, chunk in enumerate(tqdm(valid_chunks, desc="T2A Conversion")):
                if self._process_one_chunk(chunk):
                    processed_count += 1

                if processed_count % 5 == 0 or idx == len(valid_chunks) - 1:
                    pct = 10 + int((processed_count / total_chunks) * 85)
                    self._emit_progress(pct, f"Generated {processed_count}/{total_chunks} chunks")

        return processed_count

    def run(self) -> Tuple[bool, int, int]:
        """Main execution flow for token-to-audio conversion with parallel S3Gen processing."""
        self._emit_progress(5, "Loading tokens and model...")

        try:
            chunks = self.load_json()
            self.setup_output_dir()
            if not self.load_model():
                return False, 0, 0
        except Exception as e:
            logger.error(f"Setup failed: {e}", exc_info=True)
            return False, 0, 0

        total_chunks = len(chunks)

        # Ensure chunks have unique IDs; if not, add one based on index
        for idx, chunk in enumerate(chunks):
            if chunk.get("chunk_id") is None:
                chunk["chunk_id"] = f"chunk_{idx+1:05}"

        # Use parallel processing for maximum speed
        generated_count = self.process_tokens_in_batches(chunks)

        skipped_count = total_chunks - generated_count

        self._emit_progress(100, f"Conversion complete. Generated: {generated_count}, Skipped: {skipped_count}")

        # Cleanup model after conversion completes
        if self.model and hasattr(self.model, 'shutdown'):
            try:
                logger.info("🧹 Cleaning up S3Gen model in TokenToAudioConverter...")
                self.model.shutdown()
            except Exception as e:
                logger.warning(f"Model cleanup failed in TokenToAudioConverter: {e}")

        return True, generated_count, skipped_count


def generate_audio_from_tokens(
    json_path: Path,
    audio_output_dir: Path,
    voice_path: Optional[Path],
    config: Any,
    progress_callback=None,
) -> Tuple[bool, int, int]:
    """Top-level function to run the token-to-audio conversion."""
    try:
        converter = TokenToAudioConverter(
            json_path, audio_output_dir, voice_path, config, progress_callback
        )
        return converter.run()
    except Exception as e:
        logger.error(f"Token-to-audio process failed: {e}", exc_info=True)
        return False, 0, 0
