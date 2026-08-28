#!/usr/bin/env python3
"""Model download progress helpers for ASR (ggml + faster-whisper).

Keeps the UI from looking frozen during multi‑GB downloads by logging
progress to the logging module and an optional callback.

Callback signature::

    def progress_cb(event: dict) -> None:
        # event keys: phase, message, percent (0-100 or None),
        #             bytes_done, bytes_total, model
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("asr_model_download")

ProgressCallback = Optional[Callable[[dict], None]]

# ggml expected minimum sizes (bytes) for incomplete-file detection
_GGML_MIN = {
    "tiny.en": 70 * 1024 * 1024,
    "base.en": 130 * 1024 * 1024,
    "small.en": 450 * 1024 * 1024,
    "medium.en": 1400 * 1024 * 1024,
    "large-v3": 2800 * 1024 * 1024,
}


def _emit(cb: ProgressCallback, **kwargs) -> None:
    """Send a progress event to logging and optional callback."""
    msg = kwargs.get("message", "")
    pct = kwargs.get("percent")
    level = kwargs.get("level", "info")
    line = msg if pct is None else f"{msg} ({pct:.0f}%)"
    if level == "warning":
        logger.warning(line)
    else:
        logger.info(line)
    if cb:
        try:
            cb(kwargs)
        except Exception:
            pass


def ensure_ggml_model(
    ggml_name: str,
    progress_cb: ProgressCallback = None,
    chunk_size: int = 1024 * 256,
) -> str:
    """Ensure ggml model is fully downloaded; return absolute path.

    Args:
        ggml_name: e.g. base.en, medium.en
        progress_cb: Optional UI progress callback.
        chunk_size: Download chunk size in bytes.

    Returns:
        Absolute path to ggml-*.bin

    Raises:
        RuntimeError: On incomplete file that cannot be repaired, or download fail.
    """
    import requests
    from platformdirs import user_data_dir
    from pywhispercpp.constants import AVAILABLE_MODELS, MODELS_BASE_URL, MODELS_PREFIX_URL

    if ggml_name not in AVAILABLE_MODELS:
        # allow raw names still
        pass

    models_dir = Path(user_data_dir("pywhispercpp")) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    file_path = models_dir / f"ggml-{ggml_name}.bin"
    min_bytes = _GGML_MIN.get(ggml_name, 0)

    if file_path.is_file():
        size = file_path.stat().st_size
        if min_bytes and size < min_bytes:
            _emit(
                progress_cb,
                phase="download",
                level="warning",
                model=ggml_name,
                message=(
                    f"Incomplete ggml {file_path.name}: {size/1e6:.0f} MB "
                    f"(need ≥{min_bytes/1e6:.0f} MB) — deleting and re-downloading"
                ),
                percent=None,
                bytes_done=size,
                bytes_total=min_bytes,
            )
            file_path.unlink(missing_ok=True)
        else:
            _emit(
                progress_cb,
                phase="cache",
                model=ggml_name,
                message=f"ggml model already on disk: {file_path.name} ({size/1e6:.0f} MB)",
                percent=100,
                bytes_done=size,
                bytes_total=size,
            )
            return str(file_path.resolve())

    url = f"{MODELS_BASE_URL}/{MODELS_PREFIX_URL}-{ggml_name}.bin"
    _emit(
        progress_cb,
        phase="download_start",
        level="warning",
        model=ggml_name,
        message=(
            f"Downloading whisper.cpp model '{ggml_name}' — this can take several minutes "
            f"for medium/large. Program is NOT stuck. URL: {url}"
        ),
        percent=0,
        bytes_done=0,
        bytes_total=0,
    )

    tmp_path = file_path.with_suffix(".bin.partial")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        t0 = time.time()
        last_emit = 0.0
        with open(tmp_path, "wb") as fh:
            for data in resp.iter_content(chunk_size=chunk_size):
                if not data:
                    continue
                fh.write(data)
                done += len(data)
                now = time.time()
                # throttle UI updates ~4/sec
                if now - last_emit >= 0.25 or (total and done >= total):
                    last_emit = now
                    pct = (100.0 * done / total) if total else None
                    elapsed = max(0.001, now - t0)
                    speed = done / elapsed
                    eta = ((total - done) / speed) if total and speed > 0 else 0
                    _emit(
                        progress_cb,
                        phase="download",
                        model=ggml_name,
                        message=(
                            f"Downloading {ggml_name}: {done/1e6:.1f}/"
                            f"{(total/1e6) if total else '?'} MB "
                            f"@ {speed/1e6:.1f} MB/s ETA {eta:.0f}s"
                        ),
                        percent=pct,
                        bytes_done=done,
                        bytes_total=total,
                    )
        tmp_path.replace(file_path)
        size = file_path.stat().st_size
        if min_bytes and size < min_bytes:
            file_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download of {ggml_name} incomplete ({size/1e6:.0f} MB < {min_bytes/1e6:.0f} MB)"
            )
        _emit(
            progress_cb,
            phase="download_done",
            model=ggml_name,
            message=f"Download complete: {file_path.name} ({size/1e6:.0f} MB)",
            percent=100,
            bytes_done=size,
            bytes_total=size,
        )
        return str(file_path.resolve())
    except Exception:
        if tmp_path.is_file():
            tmp_path.unlink(missing_ok=True)
        raise


def ensure_faster_whisper_model(
    model_name: str,
    progress_cb: ProgressCallback = None,
) -> None:
    """Trigger faster-whisper / HF snapshot download with progress logs.

    WhisperModel() will use the cache after this. Safe to call even if cached.

    Args:
        model_name: faster-whisper size id (base, distil-small.en, …).
        progress_cb: Optional UI progress callback.
    """
    try:
        from faster_whisper.utils import _MODELS, download_model
    except Exception as exc:
        _emit(
            progress_cb,
            phase="warning",
            level="warning",
            model=model_name,
            message=f"Could not preflight faster-whisper download: {exc}",
            percent=None,
        )
        return

    repo = _MODELS.get(model_name, model_name)
    _emit(
        progress_cb,
        phase="download_start",
        level="warning",
        model=model_name,
        message=(
            f"Ensuring faster-whisper model '{model_name}' (HF: {repo}). "
            "If missing, download may take several minutes — not stuck."
        ),
        percent=0,
    )

    # Enable HF tqdm → logging via custom tqdm subclass
    try:
        from tqdm.auto import tqdm as std_tqdm

        class _LogTqdm(std_tqdm):
            """tqdm that mirrors HF download progress into our callback."""

            def __init__(self, *args, **kwargs):
                """Initializes tqdm with logging enabled, sets up last update timestamp."""
                kwargs["disable"] = False
                super().__init__(*args, **kwargs)
                self._last = 0.0

            def update(self, n=1):
                """Updates progress bar with rate limit, emits detailed download status."""
                r = super().update(n)
                now = time.time()
                if now - self._last < 0.5 and self.n < (self.total or 0):
                    return r
                self._last = now
                total = self.total or 0
                pct = (100.0 * self.n / total) if total else None
                _emit(
                    progress_cb,
                    phase="download",
                    model=model_name,
                    message=f"HF download {model_name}: {self.n}/{total or '?'} {self.unit}",
                    percent=pct,
                    bytes_done=self.n,
                    bytes_total=total,
                )
                return r

        # temporarily patch download_model kwargs via monkeypatch of snapshot
        import faster_whisper.utils as fw_utils
        import huggingface_hub

        orig = huggingface_hub.snapshot_download

        def _snap(*args, **kwargs):
            """Overrides snapshot_download to use _LogTqdm and emits download progress."""
            kwargs["tqdm_class"] = _LogTqdm
            # drop disabled_tqdm if present
            return orig(*args, **kwargs)

        huggingface_hub.snapshot_download = _snap
        try:
            path = download_model(model_name, local_files_only=False)
            _emit(
                progress_cb,
                phase="download_done",
                model=model_name,
                message=f"faster-whisper model ready: {path}",
                percent=100,
            )
        finally:
            huggingface_hub.snapshot_download = orig
    except Exception as exc:
        # local_files_only path or network — still try load later
        _emit(
            progress_cb,
            phase="warning",
            level="warning",
            model=model_name,
            message=f"Pre-download note for {model_name}: {exc}",
            percent=None,
        )


def format_load_banner(engine: str, model: str, device: str, workers: int) -> str:
    """Human banner before model load (printed to logs/UI)."""
    return (
        f"⏳ Loading ASR engine={engine} model={model} device={device} workers={workers}. "
        f"First-time downloads can take minutes — UI will show progress; not stuck."
    )
