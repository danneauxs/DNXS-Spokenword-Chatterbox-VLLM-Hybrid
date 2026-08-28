"""NeMo Parakeet Stage 1 transcriber with a faster-whisper-like surface.

Loads nvidia/parakeet-tdt-0.6b-v3 lazily. Missing NeMo is a clear runtime error
so faster-whisper / whisper.cpp still work without that stack installed.
"""

from __future__ import annotations

import importlib.util
import logging
import wave
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

PARAKEET_TDT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"


@dataclass
class _Seg:
    """Minimal segment with .text for the ASR daemon comparator path."""

    text: str


def parakeet_is_available() -> bool:
    """Return True when NeMo ASR can be imported in this interpreter."""
    return (
        importlib.util.find_spec("nemo") is not None
        and importlib.util.find_spec("nemo.collections.asr") is not None
    )


def _extract_text(output: Any) -> str:
    """Pull transcript text from NeMo's version-dependent return shapes.

    Args:
        output: String or NeMo Hypothesis-like object.

    Returns:
        Stripped transcript, or empty string.
    """
    if isinstance(output, str):
        return output.strip()
    text = getattr(output, "text", None)
    if text is None and isinstance(output, dict):
        text = output.get("text")
    return str(text or "").strip()


def _audio_duration_seconds(audio_path: str) -> float | None:
    """Read WAV duration from its header without decoding samples."""
    try:
        with wave.open(audio_path, "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate else None
    except (OSError, wave.Error, ZeroDivisionError):
        return None


class ParakeetAsrModel:
    """One resident Parakeet TDT model for sequential chunk transcribes."""

    def __init__(self, model_name: str = PARAKEET_TDT_MODEL, device: str = "cuda"):
        """Load Parakeet TDT once.

        Args:
            model_name: HuggingFace / NGC id. Default is Pocket's TDT 0.6b.
            device: cuda or cpu.

        Raises:
            RuntimeError: NeMo is not installed.
        """
        if not parakeet_is_available():
            raise RuntimeError(
                "NeMo ASR is not installed. Install optional nemo-toolkit[asr] "
                "in this venv to use Parakeet Stage 1."
            )
        import nemo.collections.asr as nemo_asr

        name = (model_name or "").strip()
        if not name or name.lower() in {"parakeet", "parakeet-tdt", "parakeet-tdt-0.6b-v3"}:
            name = PARAKEET_TDT_MODEL
        self.model_name = name
        self.device = "cuda" if str(device).lower() in {"cuda", "gpu"} else "cpu"
        logger.info("Loading Parakeet %s on %s", self.model_name, self.device)
        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        if self.device == "cuda":
            self._model = self._model.cuda()
        self._model.eval()
        logger.info("Parakeet ready (%s)", self.device)

    def transcribe(
        self,
        audio_path: Any,
        language: str = "en",
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,
        vad_parameters: Optional[dict] = None,
        **kwargs: Any,
    ) -> Tuple[List[_Seg], dict]:
        """Transcribe one WAV path. Extra kwargs match the Whisper daemon call.

        Args:
            audio_path: Filesystem path to the chunk WAV.
            language: Ignored; Parakeet TDT here is English-centric.
            condition_on_previous_text: Ignored.
            vad_filter: Ignored.
            vad_parameters: Ignored.

        Returns:
            (segments, info) with one segment exposing ``.text``.
        """
        _ = language, condition_on_previous_text, vad_filter, vad_parameters, kwargs
        outputs = self._model.transcribe([str(audio_path)], timestamps=False)
        if not outputs:
            return [], {"language": "en", "engine": "parakeet"}
        text = _extract_text(outputs[0])
        if not text:
            return [], {"language": "en", "engine": "parakeet"}
        return [_Seg(text=text)], {"language": "en", "engine": "parakeet"}

    def transcribe_many(self, audio_paths: List[Any], batch_size: int = 16) -> List[str]:
        """Transcribe multiple WAV paths with this one loaded Parakeet model.

        NeMo accepts a path list and handles its own minibatching.  Older NeMo
        builds do not accept ``batch_size`` here, so retain a compatible retry
        without that optional argument rather than falling back to one model per
        clip.

        Args:
            audio_paths: WAV paths in the result order required by the caller.
            batch_size: Requested NeMo transcription minibatch size.

        Returns:
            One stripped transcript per supplied audio path.
        """
        paths = [str(path) for path in audio_paths]
        if not paths:
            return []

        # Pocket's proven path uses NeMo's default Lhotse loader and batch 4.
        # Keep long clips out of normal batches because their activations can
        # overflow CUDA even when every ordinary clip is safe.
        requested_batch_size = max(1, int(batch_size or 1))
        short_paths: List[str] = []
        long_paths: List[str] = []
        for path in paths:
            duration = _audio_duration_seconds(path)
            if duration is not None and duration > 25.0:
                long_paths.append(path)
            else:
                short_paths.append(path)

        grouped_paths = [(short_paths, min(requested_batch_size, 4))]
        grouped_paths.extend(([path], 1) for path in long_paths)
        transcripts_by_path: dict[str, str] = {}
        for group_paths, group_batch_size in grouped_paths:
            if not group_paths:
                continue
            logger.info(
                "Parakeet transcribing %d clip(s) with batch_size=%d%s",
                len(group_paths),
                group_batch_size,
                " (long-clip isolation)" if len(group_paths) == 1 and group_batch_size == 1 else "",
            )
            try:
                outputs = self._model.transcribe(
                    group_paths,
                    timestamps=False,
                    batch_size=group_batch_size,
                )
            except TypeError:
                # Older NeMo versions reject batch_size; retain Pocket's
                # compatible full-list call rather than creating per-clip models.
                outputs = self._model.transcribe(group_paths, timestamps=False)
            if len(outputs) != len(group_paths):
                raise RuntimeError(
                    f"Parakeet returned {len(outputs)} transcripts for {len(group_paths)} paths"
                )
            transcripts_by_path.update(
                (path, _extract_text(output)) for path, output in zip(group_paths, outputs)
            )
        return [transcripts_by_path.get(path, "") for path in paths]
