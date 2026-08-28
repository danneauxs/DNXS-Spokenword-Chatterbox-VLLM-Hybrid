"""Fast PCM concat, peak measurement, and ffmpeg concat-list helpers.

Chunk WAVs from this pipeline are uncompressed 24 kHz PCM. Stitching them by
copying frames avoids a full-book ffmpeg round-trip. FFmpeg concat-copy is the
fallback when headers differ.
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_PCM_COPY_FRAMES = 65536
_AAC_CODEC_CACHE: Dict[str, str] = {}
_EXPORT_WORKER_CAP = 8


def wav_duration_seconds(path: Path) -> float:
    """Return duration of a PCM WAV from its header, without decoding audio.

    Args:
        path: Path to a WAV file.

    Returns:
        Duration in seconds. Uses a 1 Hz floor if the header rate is zero.
    """
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate() or 1
        return float(frames) / float(rate)


def write_concat_list(wav_paths: Sequence[str], list_path: Path) -> Path:
    """Write an ffmpeg concat demuxer list using quoted absolute paths.

    Args:
        wav_paths: Ordered source audio paths.
        list_path: Destination .txt path.

    Returns:
        The written list_path.
    """
    list_path = Path(list_path)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for wav in wav_paths:
        safe = str(Path(wav).resolve()).replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return list_path


def _wav_pcm_format(handle: wave.Wave_read) -> Tuple[int, int, int, str]:
    """Return channels, sample width, frame rate, and compression type."""
    return (
        handle.getnchannels(),
        handle.getsampwidth(),
        handle.getframerate(),
        handle.getcomptype(),
    )


def stitch_pcm_wavs(wav_paths: Sequence[str], output_wav: Path) -> Path:
    """Concatenate identical PCM WAV files by copying frames after one header.

    Args:
        wav_paths: Ordered source WAV paths. All must share channels, width,
            rate, and uncompressed PCM.
        output_wav: Destination WAV path.

    Returns:
        The written output_wav path.

    Raises:
        ValueError: Empty input, compressed WAV, or a format mismatch.
    """
    if not wav_paths:
        raise ValueError("No WAV files to stitch")
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    expected: Optional[Tuple[int, int, int, str]] = None
    with wave.open(str(output_wav), "wb") as dest:
        for wav in wav_paths:
            with wave.open(str(wav), "rb") as source:
                fmt = _wav_pcm_format(source)
                if expected is None:
                    if fmt[3] not in {"NONE", "not compressed"}:
                        raise ValueError(f"Compressed WAV cannot be stitched: {wav}")
                    expected = fmt
                    dest.setnchannels(fmt[0])
                    dest.setsampwidth(fmt[1])
                    dest.setframerate(fmt[2])
                elif fmt != expected:
                    raise ValueError(
                        f"WAV format mismatch for {wav}: {fmt} != {expected}"
                    )
                while True:
                    payload = source.readframes(_PCM_COPY_FRAMES)
                    if not payload:
                        break
                    dest.writeframes(payload)
    return output_wav


def concat_wavs(
    wav_paths: Sequence[str], output_wav: Path, ffmpeg_path: str = "ffmpeg"
) -> Path:
    """Concatenate WAV files, preferring a native PCM stitch over ffmpeg.

    Args:
        wav_paths: Ordered source WAV paths.
        output_wav: Destination WAV path.
        ffmpeg_path: FFmpeg executable used only when stitch fails.

    Returns:
        The written output_wav path.
    """
    output_wav = Path(output_wav)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        return stitch_pcm_wavs(wav_paths, output_wav)
    except Exception as exc:
        logger.info("Native PCM stitch unavailable (%s); using FFmpeg concat", exc)
    list_path = output_wav.with_suffix(".concat.txt")
    write_concat_list(wav_paths, list_path)
    try:
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_wav),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass
    return output_wav


def measure_pcm_peak(wav_paths: Sequence[str]) -> float:
    """Return the maximum absolute PCM sample across files as a 0..1 fraction.

    Args:
        wav_paths: Ordered WAV paths to scan in 65536-frame blocks.

    Returns:
        Peak magnitude in [0, 1]. Silent or empty input returns 0.0.
    """
    peak = 0
    max_abs = 1
    try:
        import numpy as np
    except ImportError:
        np = None
    for wav in wav_paths:
        with wave.open(str(wav), "rb") as source:
            width = source.getsampwidth()
            max_abs = (1 << (8 * width - 1)) - 1
            while True:
                payload = source.readframes(_PCM_COPY_FRAMES)
                if not payload:
                    break
                if np is not None and width == 2:
                    samples = np.frombuffer(payload, dtype="<i2")
                    if samples.size:
                        peak = max(peak, int(np.max(np.abs(samples))))
                    continue
                import array

                typecode = {1: "b", 2: "h", 4: "i"}.get(width)
                if typecode is None:
                    continue
                values = array.array(typecode)
                values.frombytes(payload)
                if values:
                    peak = max(peak, max(abs(int(sample)) for sample in values))
    if peak <= 0 or max_abs <= 0:
        return 0.0
    return min(1.0, float(peak) / float(max_abs))


def peak_volume_filter(peak: float, target_db: float) -> Optional[str]:
    """Return an FFmpeg volume= filter that places measured peak at target_db.

    Args:
        peak: Measured 0..1 peak from measure_pcm_peak.
        target_db: Desired peak level in dBFS.

    Returns:
        An FFmpeg audio-filter string, or None when no gain is needed.
    """
    if peak <= 0.0:
        return None
    current_db = 20.0 * math.log10(peak)
    gain_db = float(target_db) - current_db
    if abs(gain_db) < 0.05:
        return None
    gain_db = max(-24.0, min(24.0, gain_db))
    return f"volume={gain_db:.3f}dB"


def export_worker_count(job_count: int) -> int:
    """Return a conservative parallel-encode cap so disk does not thrash.

    Args:
        job_count: Number of chapter encode jobs.

    Returns:
        Worker count between 1 and min(job_count, CPU, 8).
    """
    cpu = os.cpu_count() or 2
    return max(1, min(job_count, cpu, _EXPORT_WORKER_CAP))


def aac_codec(ffmpeg_path: str) -> str:
    """Return libfdk_aac when this FFmpeg build has it, else native aac.

    Args:
        ffmpeg_path: FFmpeg executable to probe.

    Returns:
        Encoder name string.
    """
    cached = _AAC_CODEC_CACHE.get(ffmpeg_path)
    if cached:
        return cached
    codec = "aac"
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and "libfdk_aac" in (result.stdout or ""):
            codec = "libfdk_aac"
    except Exception as exc:
        logger.debug("FFmpeg encoder probe failed: %s", exc)
    _AAC_CODEC_CACHE[ffmpeg_path] = codec
    return codec


def aac_encode_args(
    ffmpeg_path: str, sample_rate: int, bitrate: str = "128k"
) -> List[str]:
    """Build AAC encode arguments, skipping a no-op resample when rate is 0.

    Args:
        ffmpeg_path: FFmpeg executable used for the encoder probe.
        sample_rate: Requested output rate. Pass 0 to leave the source rate.
        bitrate: CBR AAC bitrate.

    Returns:
        FFmpeg argument list starting at the codec flags.
    """
    codec = aac_codec(ffmpeg_path)
    args: List[str] = ["-c:a", codec, "-b:a", bitrate]
    if sample_rate > 0:
        args = ["-ar", str(sample_rate), *args]
    if codec == "aac":
        args.extend(["-aac_coder", "fast"])
    return args


def run_ffmpeg_checked(cmd: Sequence[str]) -> None:
    """Run an FFmpeg command and raise with stderr on failure.

    Args:
        cmd: Full FFmpeg argument list including the executable.
    """
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed ({result.returncode}): {result.stderr[-4000:]}"
        )
