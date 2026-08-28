"""
Pause Tag Processing Utilities for vLLM Pipeline
=================================================

Handles parsing of [pause] tags and silence generation for TTS synthesis.
Integrates with Chatterbox vLLM pipeline.
"""

import re
from typing import List, Tuple

from modules.punctuation_pauses import split_t3_sentences

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


def is_speech_text_segment(text: str) -> bool:
    """Return True when a pause-split piece has spoken alphanumeric content.

    Quote-only or punctuation-only fragments as standalone T3 prompts cause
    context-free hallucinations (same failure Pocket documented).

    Args:
        text: One text segment after pause-tag splitting.

    Returns:
        True if T3 should generate this piece.
    """
    return any(ch.isalnum() for ch in str(text or ""))


def _is_short_orphan(text: str, min_words: int = 3) -> bool:
    """Return True for a mid-clause leftover too small to be its own T3 prompt.

    Complete micro-sentences (Yes. / No!) are kept. Bare tails without sentence
    punctuation are merged back so T3 is not asked to start on "he said".

    Args:
        text: Speech segment.
        min_words: Word-count floor; at or above this, keep as its own prompt.

    Returns:
        True when this fragment should be glued onto the previous segment.
    """
    stripped = str(text or "").strip()
    if not stripped or not is_speech_text_segment(stripped):
        return True
    words = stripped.split()
    if len(words) >= min_words:
        return False
    return stripped[-1] not in ".!?"


def parse_pause_tags(text: str) -> Tuple[List[str], List[float]]:
    """Parse [pause:Xms] tags into T3 text segments and silence durations.

    Zero-ms tags do not split. Mute/punctuation-only pieces are dropped.
    Short orphans without sentence-end punctuation merge into the previous
    segment so T3 never sees a two-word clause as a new utterance. Each
    remaining piece is then split on real sentence ends so vLLM can stop
    after one sentence without dropping the next. A pause-only chunk returns
    no speech segments and one silence duration so the decoder can render it
    as silence instead of speech.

    Args:
        text: Input text containing [pause:Xms] tags.

    Returns:
        (segments, pause_durations) with one fewer pause than segments.
    """
    split_pattern = r"\[pause:\d+ms\]"
    pause_pattern = r"\[pause:(\d+)ms\]"
    parts = re.split(f"({split_pattern})", text or "")

    raw_segments: List[str] = []
    raw_pauses: List[float] = []
    current = ""
    pending_pause: float = 0.0

    def _flush_current() -> None:
        """Commit buffered speech and attach any pending pause to it."""
        nonlocal current, pending_pause
        piece = current.strip()
        current = ""
        if not piece:
            return
        raw_segments.append(piece)
        if pending_pause > 0:
            raw_pauses.append(pending_pause)
            pending_pause = 0.0

    for part in parts:
        match = re.match(pause_pattern, part)
        if match:
            duration_sec = int(match.group(1)) / 1000.0
            if duration_sec <= 0:
                # 0 ms: keep accumulating into the same T3 prompt.
                continue
            _flush_current()
            pending_pause += duration_sec
            continue
        current += part
    _flush_current()
    if pending_pause > 0 and raw_segments:
        raw_pauses.append(pending_pause)
        pending_pause = 0.0

    segments: List[str] = []
    pauses: List[float] = []
    for i, seg in enumerate(raw_segments):
        pause_before = raw_pauses[i - 1] if i > 0 and i - 1 < len(raw_pauses) else None
        if not is_speech_text_segment(seg):
            continue
        if segments and _is_short_orphan(seg):
            segments[-1] = f"{segments[-1]} {seg}".strip()
            continue
        if segments and pause_before is not None:
            pauses.append(pause_before)
        segments.append(seg)

    if not segments:
        if pending_pause > 0:
            return [], [pending_pause]
        if raw_pauses:
            return [], raw_pauses
        fallback = re.sub(split_pattern, " ", text or "").strip()
        return ([fallback] if fallback else [text or ""]), []

    return _expand_segments_to_sentences(segments, pauses)


def _expand_segments_to_sentences(
    segments: List[str], pauses: List[float]
) -> Tuple[List[str], List[float]]:
    """Turn each pause-tag segment into one T3 prompt per sentence.

    Extra sentences join with 0 ms so S3Gen still concatenates them inside the
    same chunk. Original pause-tag gaps stay between the pieces they split.

    Args:
        segments: Speech pieces after pause-tag parsing.
        pauses: Seconds of silence between those pieces.

    Returns:
        (sentence_prompts, pauses) with one fewer pause than prompts.
    """
    new_segments: List[str] = []
    new_pauses: List[float] = []
    inter_segment_pauses = pauses[: max(0, len(segments) - 1)]
    trailing_pauses = pauses[len(segments) - 1 :] if len(pauses) >= len(segments) else []
    for index, segment in enumerate(segments):
        parts = split_t3_sentences(segment) or [segment]
        for part_index, part in enumerate(parts):
            if new_segments:
                if part_index == 0 and index > 0 and index - 1 < len(inter_segment_pauses):
                    new_pauses.append(inter_segment_pauses[index - 1])
                else:
                    new_pauses.append(0.0)
            new_segments.append(part)
    if not new_segments:
        return segments, pauses
    if trailing_pauses:
        new_pauses.extend(trailing_pauses)
    return new_segments, new_pauses


def convert_inline_markers_to_pause_tags(text: str) -> str:
    """
    Convert numeric inline pause markers such as "~500" into [pause:Xms] tags.

    Every numeric marker converts directly from milliseconds, independent of GUI
    checkbox state.

    Args:
        text: Input text potentially containing numeric ~N markers.

    Returns:
        Text with supported markers replaced by [pause:Xms] tags.
    """
    def _repl(m):
        """Replace one numeric marker with its millisecond pause tag."""
        ms = int(m.group(1))
        if ms <= 0:
            return ""
        return f"[pause:{ms}ms]"

    return re.sub(r"~(\d+)", _repl, text)


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
