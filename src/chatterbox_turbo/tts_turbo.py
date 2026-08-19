import os
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import numpy as np
import pyloudnorm as ln

from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.s3gen.const import S3GEN_SIL
from .text_utils import chunk_text
import logging
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modules.audio_processor import (
    process_audio_with_trimming_and_silence,
    add_contextual_silence,
    ENABLE_AUDIO_TRIMMING,
)
from modules.pause_utils import parse_pause_tags, insert_pauses_into_audio_tensor
logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


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
        ("…", ", "),
        (":", ","),
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
        """Moves model and its parameters to a specified device.
        Args:
        device: The target device (e.g., 'cpu', 'cuda').
        Returns:
        The modified model instance.
        Saves the model's state dictionary to a file.
        Args:
        fpath: Path to save the model file.
        """
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        """Saves the current state of an object to a file.
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
        """Loads a ChatterboxTurboTTS model from a file.
        Args:
        fpath (str): The path to the file containing the model.
        map_location (str or torch.device, optional): Where to load the tensors. Defaults to "cpu".
        Returns:
        ChatterboxTurboTTS: A new instance of ChatterboxTurboTTS loaded from the specified file.
        """
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTurboTTS:
    """ChatterboxTurboTTS: A class for handling text-to-speech synthesis using advanced models and encoders.
    Encodings:
    - ENC_COND_LEN: Length of encoding conditions for text processing.
    - DEC_COND_LEN: Length of decoding conditions for audio synthesis.
    Initialization parameters include models, encoders, tokenizer, device, and optional conditions.
    """
    ENC_COND_LEN = 15 * S3_SR
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
        """Initialize the TTS system.
        Args:
        t3 (T3): The T3 model instance.
        s3gen (S3Gen): The S3 generation model instance.
        ve (VoiceEncoder): The voice encoder instance.
        tokenizer (EnTokenizer): The text tokenizer instance.
        device (str): The device to run the models on, e.g., 'cpu' or 'cuda'.
        conds (Conditionals, optional): Optional conditional inputs.
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

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        """Creates a new instance of ChatterboxTurboTTS from a local checkpoint directory and device.
        Args:
        ckpt_dir (str or Path): The directory containing the model checkpoint.
        device (str): The device to load the model onto ("cpu", "mps", etc.).
        Returns:
        ChatterboxTurboTTS: A new instance of the model.
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

        # Turbo specific hp
        hp = T3Config(text_tokens_dict_size=50276)
        hp.llama_config_name = "GPT2_medium"
        hp.speech_tokens_dict_size = 6563
        hp.input_pos_emb = None
        hp.speech_cond_prompt_len = 375
        hp.use_perceiver_resampler = False
        hp.emotion_adv = False

        t3 = T3(hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        del t3.tfmr.wte
        t3.to(device).eval()

        s3gen = S3Gen(meanflow=True)
        weights = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(
            weights, strict=True
        )
        s3gen.to(device).eval()

        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != 50276:
            print(f"WARNING: Tokenizer len {len(tokenizer)} != 50276")

        conds = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        """Initializes a `ChatterboxTurboTTS` instance using pre-trained weights.
        Args:
        device (str): The hardware device to load the model onto, e.g., "mps", "cpu".
        Returns:
        ChatterboxTurboTTS: A new instance of `ChatterboxTurboTTS`.
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        # PRIORITY: Use cached files first (no download, no overwrite risk)
        cache_dir = Path.home() / ".cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots"
        if cache_dir.exists():
            # Find the most recent snapshot directory
            snapshots = list(cache_dir.glob("*"))
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
                if (latest_snapshot / "ve.safetensors").exists():
                    print(f"✅ Using cached turbo model from {latest_snapshot}")
                    return cls.from_local(str(latest_snapshot), device)
        
        # FALLBACK: Download only if cache missing
        print("⚠️ No cached turbo model found, attempting download...")
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Turbo model requires HuggingFace authentication.\n"
                "Set environment variable: export HF_TOKEN=your_hf_token\n"
                "Or login with: huggingface-cli login"
            )
        
        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=hf_token,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        """Loads a WAV file and normalizes its loudness to -27 LUFS if required.
        Args:
        wav_fpath (str): Path to the WAV file.
        exaggeration (float, optional): Exaggeration factor for conditional effects. Default is 0.5.
        norm_loudness (bool, optional): Whether to normalize the loudness of the audio. Default is True.
        Returns:
        np.ndarray: Normalized or unaltered audio data as NumPy array.
        """
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isfinite(gain_linear) and gain_linear > 0.0:
                wav = wav * gain_linear
        except Exception as e:
            print(f"Warning: Error in norm_loudness, skipping: {e}")

        return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        """Loads and normalizes a reference WAV file, resamples it to 16kHz, and prepares it for use as a condition in a neural voice cloning model.
        Args:
        wav_fpath (str): Path to the input WAV file.
        exaggeration (float, optional): Factor to adjust pitch. Default is 0.5.
        norm_loudness (bool, optional): Whether to normalize loudness. Default is True.
        Returns:
        dict: A dictionary containing the prepared reference audio and its embedding.
        """
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        assert len(s3gen_ref_wav) / _sr > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, _sr)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        ref_16k_wav = ref_16k_wav.astype(np.float32)  # Cast to float32 to prevent dtype mismatch

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
        exaggeration=0.0,  # IGNORED for Turbo (always uses exaggeration=0.0)
        cfg_weight=3.0,    # IGNORED for Turbo (Turbo doesn't use CFG)
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        min_p=0.0,         # IGNORED for Turbo  
        repetition_penalty=1.2,
        chunk_text_enabled=True,  # Enable internal chunking if text is long
        return_tokens=False,  # Return tokens for diagnostic comparison
        boundary_type=None,  # Boundary type for post-processing silence
    ):
        """Generates a response to the given text using specified parameters.
        Args:
        - text (str): The input text.
        - exaggeration (float): IGNORED for Turbo (always uses exaggeration=0.0).
        - cfg_weight (float): IGNORED for Turbo (Turbo doesn't use CFG).
        - temperature (float): Controls randomness of output.
        - top_k (int): Limits the next word selection to the top K most likely words.
        - top_p (float): Nucleus sampling probability threshold.
        - min_p (float): IGNORED for Turbo.
        - repetition_penalty (float): Penalizes repeated tokens in the generated text.
        - chunk_text_enabled (bool): Enables internal chunking if text is long.
        - return_tokens (bool): Returns tokens for diagnostic comparison.
        - boundary_type (str): Boundary type for post-processing silence.
        Returns:
        - str: The generated response.
        """

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Check for pause tags in text (inline pauses)
        if text and '[pause:' in text:
            logger.info("🎵 Detected pause tags in text - parsing and inserting pauses")
            
            # Parse pause tags to get segments and pause durations
            segments, pause_durations = parse_pause_tags(text)
            
            if len(segments) > 1 and pause_durations:
                logger.info(f"🎵 Processing {len(segments)} text segments with {len(pause_durations)} pauses")
                
                # Generate audio for each text segment
                segment_audios = []
                all_segment_tokens = []
                
                for i, segment_text in enumerate(segments):
                    if segment_text.strip():
                        logger.debug(f"Generating audio for segment {i+1}/{len(segments)}: '{segment_text[:50]}...'")
                        
                        if return_tokens:
                            result = self._generate_single(
                                segment_text.strip(),
                                repetition_penalty=repetition_penalty,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                return_tokens=True,
                            )
                            audio_segment, tokens = result
                            all_segment_tokens.append(tokens)
                        else:
                            audio_segment = self._generate_single(
                                segment_text.strip(),
                                repetition_penalty=repetition_penalty,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                            )
                        
                        # Keep the batch dimension (1, samples) for insert_pauses_into_audio_tensor
                        # _generate_single returns shape (1, samples), which is what we need
                        segment_audios.append(audio_segment)
                
                # Insert pauses between audio segments
                if len(segment_audios) > 1 and pause_durations:
                    final_audio = insert_pauses_into_audio_tensor(segment_audios, pause_durations, self.sr)
                    # insert_pauses_into_audio_tensor returns shape (1, total_samples), which is correct
                    logger.info(f"✅ Inserted {len(pause_durations)} pauses into audio")
                else:
                    # Just concatenate if no pauses or single segment
                    if len(segment_audios) == 1:
                        final_audio = segment_audios[0]
                    else:
                        final_audio = torch.cat(segment_audios, dim=-1)
                
                # Apply boundary post-processing if specified
                if boundary_type and boundary_type != "none":
                    final_audio = self._apply_postprocessing(final_audio, boundary_type)
                
                if return_tokens:
                    return final_audio, all_segment_tokens
                else:
                    return final_audio
        
        # Norm text (for non-pause-tag processing)
        text = punc_norm(text)
        
        # Chunk text if needed
        if chunk_text_enabled and len(text) > max_chunk_chars:
            text_chunks = chunk_text(text, max_chars=max_chunk_chars)
            print(f"Text split into {len(text_chunks)} chunks for processing")
            
            # Generate audio for each chunk and concatenate
            all_wavs = []
            all_tokens = []
            for i, chunk in enumerate(text_chunks):
                print(f"Processing chunk {i+1}/{len(text_chunks)}: {chunk[:50]}...")
                if return_tokens:
                    result = self._generate_single(
                        chunk,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        return_tokens=True,
                    )
                    wav, tokens = result
                    all_tokens.append(tokens)
                else:
                    wav = self._generate_single(
                        chunk,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
                all_wavs.append(wav.squeeze(0))
            
            # Concatenate all audio chunks
            final_audio = torch.cat(all_wavs, dim=-1).unsqueeze(0)
            
            # Apply post-processing if boundary_type is specified
            if boundary_type and boundary_type != "none":
                final_audio = self._apply_postprocessing(final_audio, boundary_type)
            
            if return_tokens:
                return final_audio, all_tokens
            else:
                return final_audio
        else:
            result = self._generate_single(
                text,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_tokens=return_tokens,
            )
            
            # Unpack result if returning tokens
            if return_tokens:
                audio_output, captured_tokens = result
            else:
                audio_output = result
            
            # Apply post-processing if boundary_type is specified
            if boundary_type and boundary_type != "none":
                audio_output = self._apply_postprocessing(audio_output, boundary_type)
            
            if return_tokens:
                return audio_output, captured_tokens
            else:
                return audio_output
    
    def _generate_single(
        self,
        text,
        repetition_penalty=1.2,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        return_tokens=False,
    ):
        """Generate audio for a single text chunk."""
        # Tokenize text
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        speech_tokens = speech_tokens.to(self.device)
        silence = torch.tensor([S3GEN_SIL, S3GEN_SIL, S3GEN_SIL]).long().to(self.device)
        speech_tokens = torch.cat([speech_tokens, silence])
        
        # Capture tokens if requested (for diagnostic comparison with vLLM)
        if return_tokens:
            captured_tokens = speech_tokens.cpu().tolist()

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
            n_cfm_timesteps=2,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        audio_output = torch.from_numpy(wav).unsqueeze(0)
        
        if return_tokens:
            return audio_output, captured_tokens
        else:
            return audio_output
    
    def _apply_postprocessing(self, audio_tensor, boundary_type):
        """
        Apply post-processing (trimming and silence) to audio tensor.
        Uses the same process as standard backend.
        """
        from pydub import AudioSegment
        import io
        
        if boundary_type is None or boundary_type == "none":
            # No post-processing needed
            return audio_tensor
        
        logger.info(f"🔧 Applying post-processing for boundary_type={boundary_type}")
        
        try:
            # Convert tensor to numpy array
            audio_np = audio_tensor.squeeze(0).cpu().numpy()
            original_duration = len(audio_np) / self.sr
            
            # Convert numpy array to bytes (WAV format)
            import soundfile as sf
            wav_bytes = io.BytesIO()
            sf.write(wav_bytes, audio_np, self.sr, format='WAV')
            wav_bytes.seek(0)
            
            # Create AudioSegment
            audio_segment = AudioSegment.from_wav(wav_bytes)
            
            # Apply post-processing
            processed_audio = process_audio_with_trimming_and_silence(
                audio_segment=audio_segment,
                boundary_type=boundary_type,
                enable_trimming=ENABLE_AUDIO_TRIMMING
            )
            
            new_duration = len(processed_audio) / 1000.0  # pydub gives milliseconds
            added_silence = new_duration - original_duration
            
            logger.info(f"✅ Post-processing complete: {original_duration:.2f}s → {new_duration:.2f}s (added {added_silence:.0f}ms silence)")
            
            # Convert back to tensor
            import numpy as np
            processed_np = np.array(processed_audio.get_array_of_samples(), dtype=np.float32)
            # Normalize to [-1, 1] range
            processed_np = processed_np / (2**15)  # 16-bit audio
            
            return torch.from_numpy(processed_np).unsqueeze(0)
            
        except Exception as e:
            logger.warning(f"⚠️ Post-processing failed for boundary_type={boundary_type}: {e}")
            # Return original tensor if post-processing fails
            return audio_tensor
    
    def shutdown(self):
        """
        Unload Turbo TTS model and free GPU resources
        """
        # Explicitly delete model references
        del self.t3
        del self.s3gen
        del self.ve
        del self.conds
        del self.tokenizer
        
        # Clear CUDA memory
        import torch
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("🔌 Turbo TTS model unloaded and GPU resources cleared")
