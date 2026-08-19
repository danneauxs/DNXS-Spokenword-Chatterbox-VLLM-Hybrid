from pathlib import Path

import librosa
import torch
# import perth  # DISABLED: watermarking removed
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen


REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    """Class for handling voice conversion tasks using a reference dictionary."""
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict=None,
    ):
        """Initializes a new instance of the class with S3Gen, device, and optional reference dictionary.
        Args:
        s3gen (S3Gen): The S3Gen object.
        device (str): The device to use for tensors.
        ref_dict (dict, optional): A dictionary containing references. Defaults to None.
        Returns:
        None
        """
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        # self.watermarker = perth.PerthImplicitWatermarker()  # DISABLED: watermarking removed
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        """Loads a ChatterboxVC model from a local directory.
        Args:
        ckpt_dir (str): Path to the checkpoint directory.
        device (str): Device to load the model on ('cpu', 'mps', or cuda).
        Returns:
        ChatterboxVC: Loaded model instance.
        """
        ckpt_dir = Path(ckpt_dir)
        
        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None
            
        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states['gen']

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxVC':
        """Initializes the model from a pretrained checkpoint on a specified device.
        Args:
        - device (str): The hardware device to load the model onto ('mps', 'cuda', or 'cpu').
        Returns:
        - ChatterboxVC: An instance of the ChatterboxVC class loaded with the pretrained model.
        """
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"
            
        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        """Sets the target voice for the model using a WAV file.
        Args:
        target_voice_path (str): Path to the WAV file containing the reference audio.
        Returns:
        None
        """
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        """Generates audio using a specified voice or previously prepared conditionals.
        Args:
        audio (str): Path to input audio file.
        target_voice_path (str, optional): Path to target voice model. If not provided, requires `prepare_conditionals` to be called first.
        Returns:
        None
        """
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        else:
            assert self.ref_dict is not None, "Please `prepare_conditionals` first or specify `target_voice_path`"

        with torch.inference_mode():
            audio_16, _ = librosa.load(audio, sr=S3_SR)
            audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None, ]

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            # watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)  # DISABLED: watermarking removed
        return torch.from_numpy(wav).unsqueeze(0)