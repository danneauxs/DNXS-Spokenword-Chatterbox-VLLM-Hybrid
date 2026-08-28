"""whisper.cpp backend for ASR QC via pywhispercpp.

Resamples 24 kHz TTS WAVs to 16 kHz (required by whisper.cpp) and exposes a
faster-whisper-compatible transcribe() surface for validate_single_chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Map GUI/config model names → pywhispercpp ggml model ids (prefer English).
_CPP_MODEL_MAP = {
    "tiny": "tiny.en",
    "tiny.en": "tiny.en",
    "base": "base.en",
    "base.en": "base.en",
    "small": "small.en",
    "small.en": "small.en",
    "medium": "medium.en",
    "medium.en": "medium.en",
    "large-v3": "large-v3",
    "large": "large-v3",
    # Distil names: fall back to closest ggml English size if no distil ggml.
    "distil-small.en": "small.en",
    "distil-medium.en": "medium.en",
    "distil-large-v3": "large-v3",
}

# Minimum expected ggml file size (bytes). Incomplete downloads cause
# "not all tensors loaded" / segfaults — reject before inference.
_CPP_MIN_FILE_BYTES = {
    "tiny.en": 70 * 1024 * 1024,
    "base.en": 130 * 1024 * 1024,
    "small.en": 450 * 1024 * 1024,
    "medium.en": 1400 * 1024 * 1024,
    "large-v3": 2800 * 1024 * 1024,
}


@dataclass
class _Seg:
    """Minimal segment with .text for validate_single_chunk."""

    text: str


class WhisperCppAsrModel:
    """Loaded whisper.cpp model kept warm for many short-chunk transcribes."""

    def __init__(
        self,
        model_name: str = "base",
        n_threads: int = 2,
        use_gpu: bool = False,
    ):
        """Load ggml model via pywhispercpp.

        Args:
            model_name: GUI/config name (base, tiny, …) or ggml id (base.en).
            n_threads: CPU threads per model instance (keep low if many workers).
            use_gpu: If True and build has GPU, use it; else CPU.
        """
        from pywhispercpp.model import Model

        from model_download_progress import ensure_ggml_model

        ggml_name = _CPP_MODEL_MAP.get(model_name.strip(), model_name.strip())
        self.model_name = model_name
        self.ggml_name = ggml_name
        self.n_threads = max(1, int(n_threads))
        self.use_gpu = bool(use_gpu)
        logger.info(
            "Loading whisper.cpp model %s (ggml=%s) threads=%s gpu=%s "
            "(weights already ensured with progress if this is first use)",
            model_name,
            ggml_name,
            self.n_threads,
            self.use_gpu,
        )
        # Safe cache hit if caller already downloaded with progress
        model_path = ensure_ggml_model(ggml_name, progress_cb=None)
        logger.warning(
            "Initializing whisper.cpp context on %s for %s — can take 10–60s "
            "for medium/large; not stuck…",
            "GPU" if self.use_gpu else "CPU",
            ggml_name,
        )
        self._model = Model(
            model_path,  # local path avoids silent re-download inside Model
            n_threads=self.n_threads,
            print_realtime=False,
            print_progress=False,
            redirect_whispercpp_logs_to=False,
            context_params={"use_gpu": self.use_gpu},
            language="en",
            no_context=True,
            single_segment=False,
        )
        logger.info("whisper.cpp model %s ready (gpu=%s)", ggml_name, self.use_gpu)

    def transcribe(
        self,
        audio_path: Any,
        language: str = "en",
        condition_on_previous_text: bool = False,
        vad_filter: bool = False,
        vad_parameters: Optional[dict] = None,
        **kwargs: Any,
    ) -> Tuple[List[_Seg], dict]:
        """Transcribe a WAV path or mono float32 array (16 kHz preferred).

        Matches faster-whisper's surface: path **or** preloaded numpy array
        (standalone pipeline loads once and passes arrays).

        Args:
            audio_path: Filesystem path, or ``np.ndarray`` mono samples.
            language: Language code pin.
            condition_on_previous_text: Ignored (cpp uses no_context).
            vad_filter: Ignored (TTS path disables VAD upstream).
            vad_parameters: Ignored.
            **kwargs: Extra args ignored for API compat.

        Returns:
            (segments, info) with segment objects exposing ``.text`` (and
            ``.start``/``.end`` when available).
        """
        _ = condition_on_previous_text, vad_filter, vad_parameters, kwargs
        if isinstance(audio_path, np.ndarray):
            # Already decoded by load_audio_mono_16k / caller
            audio = np.ascontiguousarray(audio_path, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1).astype(np.float32)
        else:
            audio, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
            if audio is None or len(audio) == 0:
                return [], {"language": language or "en", "engine": "whisper_cpp"}
            audio = np.ascontiguousarray(audio, dtype=np.float32)

        if audio is None or len(audio) == 0:
            return [], {"language": language or "en", "engine": "whisper_cpp"}

        segments = self._model.transcribe(
            audio,
            language=(language or "en"),
            no_context=True,
        )
        out: List[_Seg] = []
        if segments:
            for seg in segments:
                text = getattr(seg, "text", None)
                if text is None:
                    text = str(seg)
                text = (text or "").strip()
                if text:
                    # Preserve times when pywhispercpp provides them (pack mapping)
                    start = float(getattr(seg, "t0", getattr(seg, "start", 0.0)) or 0.0)
                    end = float(getattr(seg, "t1", getattr(seg, "end", 0.0)) or 0.0)
                    # Some builds return centiseconds for t0/t1
                    if start > 1000 or end > 1000:
                        start, end = start / 100.0, end / 100.0
                    s = _Seg(text=text)
                    s.start = start  # type: ignore[attr-defined]
                    s.end = end  # type: ignore[attr-defined]
                    out.append(s)
        return out, {"language": language or "en", "engine": "whisper_cpp"}


def whisper_cpp_cuda_bundle_available() -> bool:
    """Return whether installed pywhispercpp ships a GGML CUDA backend library.

    A ``context_params={"use_gpu": True}`` request does not compile CUDA into a
    CPU-only wheel.  Inspecting the packaged GGML libraries lets the daemon
    report its resolved device truthfully before loading an expensive model.

    Returns:
        True only when the installed pywhispercpp distribution includes a
        platform CUDA GGML shared library.
    """
    try:
        import pywhispercpp

        package_root = Path(pywhispercpp.__file__).resolve().parent.parent
        library_dir = package_root / "pywhispercpp.libs"
        patterns = ("libggml-cuda*.so*", "libggml-cuda*.dylib", "ggml-cuda*.dll")
        return any(
            path.is_file()
            for pattern in patterns
            for path in library_dir.glob(pattern)
        )
    except (ImportError, OSError, TypeError):
        return False


def load_whisper_cpp_model(
    model_name: str = "base",
    force_device: Optional[str] = None,
    n_threads: int = 2,
    progress_cb=None,
) -> Tuple[Optional[WhisperCppAsrModel], Optional[str]]:
    """Load whisper.cpp backend; returns (model, device_label).

    Args:
        model_name: Config/GUI model name.
        force_device: 'cpu' forces CPU; 'cuda'/'gpu' allows GPU if available.
        n_threads: Threads per instance.
        progress_cb: Optional download/load progress callback.
    """
    try:
        from model_download_progress import ensure_ggml_model, format_load_banner

        forced = (force_device or "").strip().lower()
        requested_gpu = forced in ("cuda", "gpu")
        cuda_bundle = whisper_cpp_cuda_bundle_available()
        # A CPU-only wheel accepts context_params but still runs CPU kernels.
        # Resolve it here so logs, daemon JSON, and reports do not claim GPU work.
        use_gpu = requested_gpu and cuda_bundle
        if requested_gpu and not cuda_bundle:
            message = (
                "whisper.cpp CUDA requested, but installed pywhispercpp has no "
                "libggml-cuda backend; resolving this worker to CPU. "
                "Run ASR/install_pywhispercpp_cuda.sh in a CUDA-capable shell "
                "to enable GPU inference."
            )
            logger.warning(message)
            print(f"⚠️ {message}", flush=True)
        ggml = _CPP_MODEL_MAP.get(model_name.strip(), model_name.strip())
        print(
            format_load_banner(
                "whisper_cpp", model_name, "cuda" if use_gpu else "cpu", 1
            ),
            flush=True,
        )
        # Explicit progress-aware download before construct
        ensure_ggml_model(ggml, progress_cb=progress_cb)
        model = WhisperCppAsrModel(
            model_name=model_name,
            n_threads=n_threads,
            use_gpu=use_gpu,
        )
        device = "cuda" if use_gpu else "cpu"
        print(
            f"✅ whisper.cpp loaded {model.ggml_name} on {device.upper()} "
            f"threads={n_threads}",
            flush=True,
        )
        return model, device
    except Exception as exc:
        print(f"❌ whisper.cpp load failed: {exc}", flush=True)
        logger.exception("whisper.cpp load failed")
        return None, None
