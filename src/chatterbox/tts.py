from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
# import perth  # DISABLED: watermarking removed
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


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
        """Moves the model and its components to a specified device.
        Args:
        device (str or torch.device): The target device for the model and its components.
        Returns:
        self: The modified instance with all tensors moved to the specified device.
        Saves the current state of the model to a file.
        Args:
        fpath (Path): Path where the model state should be saved.
        Returns:
        None
        """
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """Saves the current state of the object to a file.
        Args:
        fpath (Path): The path where the state will be saved.
        Returns:
        None
        """
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        """Loads a ChatterboxTTS model from a file.
        Args:
        fpath (str): The path to the model file.
        map_location (str or torch.device, optional): Where to load the model. Default is 'cpu'.
        Returns:
        ChatterboxTTS: An instance of the loaded model.
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    """A class for handling Text-to-Speech (TTS) operations using conditional models and voice encoders.
    Manages initialization and properties for TTS processing.
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        """Initializes a new instance of a voice synthesis system.
        Args:
        t3 (T3): The T3 model for text processing.
        s3gen (S3Gen): The S3 generation module for audio synthesis.
        ve (VoiceEncoder): The voice encoding component.
        tokenizer (EnTokenizer): The tokenizer for converting text to tokens.
        device (str): The hardware device ('cpu' or 'cuda') to run the model on.
        conds (Conditionals, optional): Additional conditional parameters.
        Returns:
        None
        """
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # self.watermarker = perth.PerthImplicitWatermarker()  # DISABLED: watermarking removed

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        """Initialize a ChatterboxTTS instance from a local checkpoint directory.
        Args:
        ckpt_dir (str): Path to the checkpoint directory.
        device (str): Device to load the model onto ('cpu', 'mps', or CUDA).
        Returns:
        ChatterboxTTS: The initialized ChatterboxTTS instance.
        """
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        """This function loads a pre-trained model from specified local files and returns an instance of ChatterboxTTS.
        Args:
        device (str): The device on which to load the model ("mps" or "cpu").
        Returns:
        ChatterboxTTS: An instance of the loaded ChatterboxTTS model.
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        """Loads a reference WAV file, resamples it to 16k, extracts a segment, and prepares conditional embeddings for speech synthesis. Args: wav_fpath (str): Path to the WAV file. exaggeration (float): Exaggeration factor. Returns: Prepared conditional embeddings and prompt tokens."""
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        """Generates audio based on the provided text and optional audio prompt.
        Args:
        text (str): The text to be converted to audio.
        audio_prompt_path (str, optional): Path to an audio prompt file. If provided, conditions will be prepared using this audio.
        exaggeration (float, optional): Controls the emotional intensity of the generated audio. Default is 0.5.
        cfg_weight (float, optional): Configuration weight for the generation process. Default is 0.5.
        temperature (float, optional): Temperature for controlling randomness in the generation process. Default is 0.8.
        Returns:
        None
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            # watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)  # DISABLED: watermarking removed
        return torch.from_numpy(wav).unsqueeze(0)

    def set_diffusion_steps(self, n_timesteps: int):
        """Set the number of diffusion timesteps for S3Gen inference."""
        if hasattr(self.s3gen, 'flow'):
            self.s3gen.flow.n_timesteps = n_timesteps

    def set_fp16(self, enabled: bool):
        """Enable or disable FP16 (half precision) for S3Gen inference."""
        if hasattr(self.s3gen, 'flow'):
            self.s3gen.flow.fp16 = enabled

    def shutdown(self):
        """
        Unload ChatterboxTTS model and free GPU resources
        """
        # Explicitly delete model references
        del self.t3
        del self.s3gen
        del self.ve
        if hasattr(self, 'conds'):
            del self.conds
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        # Clear CUDA memory
        import torch
        torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        print("🔌 ChatterboxTTS model unloaded and GPU resources cleared")

    def generate_from_tokens(self, speech_tokens):
        """Generate audio from pre-tokenized speech tokens."""
        assert self.conds is not None, "Please `prepare_conditionals` first"

        # Ensure tokens are on the correct device and shape
        if isinstance(speech_tokens, list):
            speech_tokens = torch.tensor(speech_tokens, device=self.device)
        elif torch.is_tensor(speech_tokens):
            speech_tokens = speech_tokens.to(self.device)
        else:
            raise ValueError(f"Unsupported speech_tokens type: {type(speech_tokens)}")

        # Ensure proper shape [B, T]
        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        elif len(speech_tokens.shape) > 2:
            raise ValueError(f"speech_tokens must be 1D or 2D tensor, got shape {speech_tokens.shape}")

        print(f"DEBUG: generate_from_tokens - input tokens shape: {speech_tokens.shape}, first 10: {speech_tokens[0][:10].cpu().tolist()}")

        # Filter valid tokens (same as in generate method) - COMMENTED OUT TO PRESERVE PAUSE TOKENS
        # speech_tokens = speech_tokens[speech_tokens < 6561]

        print(f"DEBUG: generate_from_tokens - after filtering < 6561, shape: {speech_tokens.shape}, first 10: {speech_tokens[0][:10].cpu().tolist()}")
        print(f"DEBUG: generate_from_tokens - tokens >= 6561 found: {torch.sum(speech_tokens >= 6561).item()}")

        with torch.inference_mode():
            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            # watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)  # DISABLED: watermarking removed
        return torch.from_numpy(wav).unsqueeze(0)

    def get_audio_conditionals(self, audio_path: str):
        """Load voice conditioning from audio file path."""
        try:
            self.prepare_conditionals(audio_path)
            return self.conds.gen, None
        except Exception as e:
            print(f"Warning: Failed to load conditioning from {audio_path}: {e}")
            # Return default conditioning if available
            if self.conds is not None:
                return self.conds.gen, None
            else:
                raise RuntimeError(f"No conditioning available after failure to load from {audio_path}")