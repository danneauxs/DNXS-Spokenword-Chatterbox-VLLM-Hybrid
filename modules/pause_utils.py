"""
Pause Tag Processing Utilities for vLLM Pipeline
=================================================

Handles parsing of [pause] tags and silence generation for TTS synthesis.
Integrates with Chatterbox vLLM pipeline.
"""

import re
from typing import List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False


def parse_pause_tags(text: str) -> Tuple[List[str], List[float]]:
    """
    Parse text with [pause:Xms] tags and extract segments with pause durations.

    Args:
        text: Input text containing [pause:Xms] tags

    Returns:
        tuple: (segments, pause_durations)
            segments: List of text segments between pauses
            pause_durations: List of pause durations in seconds between segments

    Example:
        "Hello [pause:150ms] world" → (["Hello", "world"], [0.15])
    """
    # Pattern to match [pause:Xms] - no capturing groups for split
    split_pattern = r'\[pause:\d+ms\]'
    pause_pattern = r'\[pause:(\d+)ms\]'

    # Split text at pause tags, keeping the tags
    parts = re.split(f'({split_pattern})', text)

    segments = []
    pause_durations = []

    current_segment = ""
    i = 0
    while i < len(parts):
        part = parts[i]

        if re.match(pause_pattern, part):
            # This is a pause tag - extract duration
            if current_segment.strip():
                segments.append(current_segment.strip())

            match = re.match(r'\[pause:(\d+)ms\]', part)
            if match:
                duration_ms = int(match.group(1))
                duration_sec = duration_ms / 1000.0
            else:
                duration_sec = 0.0  # fallback
            pause_durations.append(duration_sec)
            current_segment = ""
        else:
            # This is text content
            current_segment += part

        i += 1

    # Add the final segment if it exists
    if current_segment.strip():
        segments.append(current_segment.strip())

    # Defensive: If we have trailing pauses (more pauses than segments-1), trim them
    # This handles old/bad data that ended with pause tags
    expected_pause_count = len(segments) - 1
    if len(pause_durations) > expected_pause_count:
        # Trim trailing pauses
        pause_durations = pause_durations[:expected_pause_count]

    return segments, pause_durations


def convert_inline_markers_to_pause_tags(text: str) -> str:
    """
    Convert shorthand inline pause markers ("~1", "~2") into [pause:Xms] tags.

    Lets users type "~1"/"~2" directly in source text as a manual pause shorthand;
    gated by config.ENABLE_INLINE_PAUSES, durations from config.INLINE_PAUSE_1_MS /
    INLINE_PAUSE_2_MS. Ported from the equivalent marker syntax in the sibling Turbo
    project (src/chatterbox/tts.py), adapted to this project's [pause:Xms] (integer
    milliseconds) tag format rather than that project's [pause:Xs] seconds format.

    Args:
        text: Input text potentially containing ~1/~2 markers.

    Returns:
        Text with ~1/~2 replaced by [pause:Xms] tags; unchanged if disabled.
    """
    from config.config import ENABLE_INLINE_PAUSES, INLINE_PAUSE_1_MS, INLINE_PAUSE_2_MS

    if not ENABLE_INLINE_PAUSES:
        return text

    inline_map = {
        "~1": max(0, int(INLINE_PAUSE_1_MS)),
        "~2": max(0, int(INLINE_PAUSE_2_MS)),
    }

    def _repl(m):
        """Replaces a matched ~1/~2 marker with its configured [pause:Xms] tag."""
        ms = inline_map.get(m.group(0))
        return f"[pause:{ms}ms]" if ms is not None else m.group(0)

    return re.sub(r"~[12]", _repl, text)


def create_silence_tensor(duration_sec: float, sample_rate: int = 24000, device=None):
    """
    Create a tensor of silence for the specified duration.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Audio sample rate
        device: Device to create tensor on (cuda/cpu). If None, uses CPU.

    Returns:
        torch.Tensor: Silence audio tensor with shape (1, num_samples)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for create_silence_tensor")

    num_samples = int(duration_sec * sample_rate)
    if device is not None:
        silence = torch.zeros(1, num_samples, device=device)
    else:
        silence = torch.zeros(1, num_samples)
    return silence


def create_silence_numpy(duration_sec: float, sample_rate: int = 24000):
    """
    Create a numpy array of silence for the specified duration.

    Args:
        duration_sec: Duration in seconds
        sample_rate: Audio sample rate

    Returns:
        np.ndarray: Silence audio array
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for create_silence_numpy")

    num_samples = int(duration_sec * sample_rate)
    silence = np.zeros(num_samples, dtype=np.float32)
    return silence


def insert_pauses_into_audio_tensor(audio_segments: List,
                                   pause_durations: List[float],
                                   sample_rate: int = 24000):
    """
    Concatenate audio segments with silence pauses between them.

    Args:
        audio_segments: List of audio tensors for each text segment
        pause_durations: List of pause durations in seconds between segments
        sample_rate: Audio sample rate

    Returns:
        torch.Tensor: Combined audio with pauses inserted
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is required for insert_pauses_into_audio_tensor")

    if not audio_segments:
        return torch.empty(0, 0)

    if len(audio_segments) - 1 != len(pause_durations):
        raise ValueError(f"Number of pause durations ({len(pause_durations)}) must be one less than number of audio segments ({len(audio_segments)})")

    # Detect device from first audio segment
    device = audio_segments[0].device if hasattr(audio_segments[0], 'device') else None

    result_segments = [audio_segments[0]]

    for i, audio in enumerate(audio_segments[1:], 1):
        # Add silence pause on same device as audio
        pause_sec = pause_durations[i-1]
        silence = create_silence_tensor(pause_sec, sample_rate, device=device)
        result_segments.extend([silence, audio])

    return torch.cat(result_segments, dim=1)


def insert_pauses_into_audio_numpy(audio_segments: List,
                                  pause_durations: List[float],
                                  sample_rate: int = 24000):
    """
    Concatenate numpy audio segments with silence pauses between them.

    Args:
        audio_segments: List of audio arrays for each text segment
        pause_durations: List of pause durations in seconds between segments
        sample_rate: Audio sample rate

    Returns:
        np.ndarray: Combined audio with pauses inserted
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required for insert_pauses_into_audio_numpy")

    if not audio_segments:
        return np.array([], dtype=np.float32)

    if len(audio_segments) - 1 != len(pause_durations):
        raise ValueError(f"Number of pause durations ({len(pause_durations)}) must be one less than number of audio segments ({len(audio_segments)})")

    result_segments = [audio_segments[0]]

    for i, audio in enumerate(audio_segments[1:], 1):
        # Add silence pause
        pause_sec = pause_durations[i-1]
        silence = create_silence_numpy(pause_sec, sample_rate)
        result_segments.extend([silence, audio])

    return np.concatenate(result_segments)


def validate_pause_text(text: str) -> bool:
    """
    Validate that pause tags in text are properly formatted.

    Args:
        text: Text containing pause tags

    Returns:
        bool: True if all tags are valid
    """
    # Check for properly formatted tags
    tag_pattern = r'\[pause:\d+ms\]'
    invalid_tags = re.findall(r'\[pause:[^\]]*\]', text)

    for tag in invalid_tags:
        if not re.match(tag_pattern, tag):
            return False

    return True


def insert_pause_tokens(token_lists: List[List[int]], pause_durations: List[float]) -> List[List[int]]:
    """
    Insert pause tokens between token sequences.

    Args:
        token_lists: List of token sequences for each text segment
        pause_durations: List of pause durations in seconds between segments

    Returns:
        List[List[int]]: Token sequences with pauses inserted

    Note: This is for future use with token-level pause insertion.
    Currently, pauses are handled at the audio level.
    """
    if not token_lists:
        return []

    if len(token_lists) - 1 != len(pause_durations):
        raise ValueError(f"Number of pause durations ({len(pause_durations)}) must be one less than number of token lists ({len(token_lists)})")

    # For now, just concatenate the tokens
    # Future implementation could insert special pause tokens
    result = [token_lists[0]]

    for i, tokens in enumerate(token_lists[1:], 1):
        # Insert pause tokens here if needed
        # For now, just append
        result.append(tokens)

    return result