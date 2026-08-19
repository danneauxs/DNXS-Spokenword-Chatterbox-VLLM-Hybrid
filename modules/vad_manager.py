"""
Silero VAD Manager Module
=========================

Centralized Voice Activity Detection using Silero VAD.
Provides efficient, cached VAD model loading and reusable functions
for speech detection, endpoint trimming, and quality validation.

Key Features:
- Singleton VAD model loading (no repeated downloads)
- Speech timestamp detection with configurable thresholds
- Speech endpoint detection for audio trimming
- Gap detection for TTS failure identification
- Quality scoring based on speech continuity

Performance:
- <1ms latency per 30ms audio chunk on CPU
- 1-2 MB model size (lightweight)
- Supports 8kHz and 16kHz sample rates
"""

import torch
import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pydub import AudioSegment

# Suppress PyTorch Hub verbosity
os.environ.setdefault("TORCH_HUB_VERBOSE", "0")
os.environ.setdefault("PYTHONWARNINGS", "ignore")


class SileroVADManager:
    """Singleton manager for Silero VAD model"""
    
    _instance = None
    _model = None
    _utils = None
    _initialized = False
    
    def __new__(cls):
        """Creates a singleton instance of the VAD manager.
        Args:
        cls (type): The class type.
        Returns:
        VADManager: The singleton instance of the VAD manager.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize VAD manager (singleton pattern)"""
        if not self._initialized:
            self._load_model()
            SileroVADManager._initialized = True
    
    def _load_model(self):
        """Load Silero VAD model and utilities"""
        if SileroVADManager._model is not None:
            return  # Already loaded
        
        # Temporarily suppress logging
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                verbose=False,
                trust_repo=True
            )
            
            SileroVADManager._model = model
            SileroVADManager._utils = utils
            
            logging.getLogger().setLevel(old_level)
            logging.info("✅ Silero VAD model loaded successfully")
            
        except Exception as e:
            logging.getLogger().setLevel(old_level)
            logging.error(f"❌ Failed to load Silero VAD model: {e}")
            raise
    
    @property
    def model(self):
        """Get speech timestamps from audio data.
        Args:
        audio (bytes): Audio data as bytes.
        sample_rate (int): Sample rate of the audio in Hz.
        Returns:
        List[Tuple[float, float]]: List of tuples representing the start and end times of detected speech segments.
        """
        """Get VAD model"""
        if SileroVADManager._model is None:
            self._load_model()
        return SileroVADManager._model
    
    @property
    def utils(self):
        """Returns speech timestamps for the given audio.
        Args:
        audio_path_or_tensor (str or Tensor): Path to the audio file or preloaded audio tensor.
        sampling_rate (int, optional): Audio sampling rate. Defaults to 16000.
        threshold (float, optional): Silence detection threshold. Defaults to 0.5.
        min_speech_duration_ms (int, optional): Minimum speech duration in milliseconds. Defaults to 250.
        min_silence_duration_ms (int, optional): Minimum silence duration in milliseconds. Defaults to 100.
        Returns:
        List[Dict[str, int]]: A list of dictionaries containing start and end times of detected speech segments.
        """
        """Get VAD utilities"""
        if SileroVADManager._utils is None:
            self._load_model()
        return SileroVADManager._utils
    
    def get_speech_timestamps(
        self,
        audio_path_or_tensor,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100
    ) -> List[Dict[str, int]]:
        """
        Get speech timestamps from audio.
        
        Args:
            audio_path_or_tensor: Path to audio file or torch tensor
            sampling_rate: Audio sample rate (8000 or 16000)
            threshold: Speech probability threshold (0.0-1.0)
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence duration between segments
        
        Returns:
            List of dicts with 'start' and 'end' keys (sample indices)
        """
        get_speech_timestamps, _, read_audio, _, _ = self.utils
        
        # Load audio if path provided
        if isinstance(audio_path_or_tensor, (str, Path)):
            wav = read_audio(str(audio_path_or_tensor), sampling_rate=sampling_rate)
        else:
            wav = audio_path_or_tensor
        
        # Get timestamps
        speech_segments = get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=sampling_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )
        
        return speech_segments
    
    def find_speech_endpoint(
        self,
        audio_path_or_tensor,
        sampling_rate=16000
    ) -> Optional[int]:
        """
        Find the end of the last speech segment in milliseconds.
        
        Args:
            audio_path_or_tensor: Path to audio file or torch tensor
            sampling_rate: Audio sample rate
        
        Returns:
            End timestamp in milliseconds, or None if no speech detected
        """
        speech_segments = self.get_speech_timestamps(
            audio_path_or_tensor,
            sampling_rate=sampling_rate
        )
        
        if not speech_segments:
            return None
        
        # Return last segment end in milliseconds
        last_end_sample = speech_segments[-1]['end']
        return int(last_end_sample * 1000 / sampling_rate)
    
    def detect_speech_gaps(
        self,
        audio_path_or_tensor,
        sampling_rate=16000,
        max_natural_gap_ms=1000
    ) -> List[Tuple[int, int]]:
        """
        Detect abnormally long gaps between speech segments.
        
        Args:
            audio_path_or_tensor: Path to audio file or torch tensor
            sampling_rate: Audio sample rate
            max_natural_gap_ms: Maximum expected natural pause duration
        
        Returns:
            List of (gap_start_ms, gap_end_ms) tuples for suspicious gaps
        """
        speech_segments = self.get_speech_timestamps(
            audio_path_or_tensor,
            sampling_rate=sampling_rate
        )
        
        if len(speech_segments) < 2:
            return []  # Need at least 2 segments to have gaps
        
        suspicious_gaps = []
        
        for i in range(len(speech_segments) - 1):
            gap_start_sample = speech_segments[i]['end']
            gap_end_sample = speech_segments[i + 1]['start']
            
            gap_duration_ms = (gap_end_sample - gap_start_sample) * 1000 / sampling_rate
            
            if gap_duration_ms > max_natural_gap_ms:
                gap_start_ms = int(gap_start_sample * 1000 / sampling_rate)
                gap_end_ms = int(gap_end_sample * 1000 / sampling_rate)
                suspicious_gaps.append((gap_start_ms, gap_end_ms))
        
        return suspicious_gaps
    
    def calculate_speech_quality_score(
        self,
        audio_path_or_tensor,
        sampling_rate=16000,
        total_duration_ms=None
    ) -> float:
        """
        Calculate quality score based on speech continuity.
        
        Args:
            audio_path_or_tensor: Path to audio file or torch tensor
            sampling_rate: Audio sample rate
            total_duration_ms: Total audio duration (auto-detect if None)
        
        Returns:
            Quality score 0.0-1.0 (higher is better)
        """
        speech_segments = self.get_speech_timestamps(
            audio_path_or_tensor,
            sampling_rate=sampling_rate
        )
        
        if not speech_segments:
            return 0.0  # No speech detected
        
        # Calculate total speech duration
        total_speech_samples = sum(
            seg['end'] - seg['start'] for seg in speech_segments
        )
        total_speech_ms = total_speech_samples * 1000 / sampling_rate
        
        # Get total audio duration
        if total_duration_ms is None:
            if isinstance(audio_path_or_tensor, (str, Path)):
                audio = AudioSegment.from_file(str(audio_path_or_tensor))
                total_duration_ms = len(audio)
            else:
                total_duration_ms = len(audio_path_or_tensor) * 1000 / sampling_rate
        
        # Speech coverage ratio
        speech_ratio = min(total_speech_ms / total_duration_ms, 1.0)
        
        # Continuity score (penalize fragmentation)
        avg_segment_duration_ms = total_speech_ms / len(speech_segments)
        continuity_score = min(avg_segment_duration_ms / 1000, 1.0)  # Normalize to 1s
        
        # Combined score (weighted)
        quality_score = (speech_ratio * 0.6) + (continuity_score * 0.4)
        
        return quality_score
    
    def detect_trailing_artifact(
        self,
        audio_path_or_tensor,
        sampling_rate=16000,
        artifact_threshold_ms=200,
        min_artifact_energy=0.01
    ) -> bool:
        """
        Detect if audio has trailing non-speech artifacts.
        
        Args:
            audio_path_or_tensor: Path to audio file or torch tensor
            sampling_rate: Audio sample rate
            artifact_threshold_ms: Minimum trailing duration to consider
            min_artifact_energy: Minimum RMS energy for artifact detection
        
        Returns:
            True if trailing artifact detected, False otherwise
        """
        speech_segments = self.get_speech_timestamps(
            audio_path_or_tensor,
            sampling_rate=sampling_rate
        )
        
        if not speech_segments:
            return False  # No speech, can't determine artifact
        
        # Get last speech end time
        last_speech_end_ms = speech_segments[-1]['end'] * 1000 / sampling_rate
        
        # Load audio to check total duration and trailing energy
        if isinstance(audio_path_or_tensor, (str, Path)):
            audio = AudioSegment.from_file(str(audio_path_or_tensor))
        else:
            # Convert tensor to AudioSegment for analysis
            import tempfile
            import soundfile as sf
            
            temp_path = tempfile.mktemp(suffix='.wav')
            sf.write(temp_path, audio_path_or_tensor.numpy(), sampling_rate)
            audio = AudioSegment.from_file(temp_path)
            os.unlink(temp_path)
        
        total_duration_ms = len(audio)
        trailing_duration_ms = total_duration_ms - last_speech_end_ms
        
        # Check if trailing audio is significant
        if trailing_duration_ms < artifact_threshold_ms:
            return False
        
        # Analyze trailing segment energy
        trailing_segment = audio[int(last_speech_end_ms):]
        trailing_rms = trailing_segment.rms / trailing_segment.max_possible_amplitude
        
        # Artifact if trailing has significant energy
        return trailing_rms > min_artifact_energy


# Singleton instance
_vad_manager = None


def get_vad_manager() -> SileroVADManager:
    """Get or create singleton VAD manager instance"""
    global _vad_manager
    if _vad_manager is None:
        _vad_manager = SileroVADManager()
    return _vad_manager
