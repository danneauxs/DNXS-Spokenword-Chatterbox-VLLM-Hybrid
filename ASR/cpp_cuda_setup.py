#!/usr/bin/env python3
"""Detect / build CUDA-enabled pywhispercpp for whisper.cpp GPU ASR.

Yes on the GUI dialog must **run the build**, not continue without CUDA libs.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[str], None]]

ROOT = Path(__file__).resolve().parents[1]
ASR_DIR = Path(__file__).resolve().parent
INSTALL_SCRIPT = ASR_DIR / "install_pywhispercpp_cuda.sh"


def cpp_cuda_build_available(python_exe: Optional[str] = None) -> bool:
    """Return True if this interpreter's pywhispercpp has libggml-cuda.

    Args:
        python_exe: Interpreter to probe (default: current).

    Returns:
        Whether CUDA ggml backend libraries are present.
    """
    py = python_exe or sys.executable
    code = (
        "import pathlib\n"
        "try:\n"
        " import pywhispercpp\n"
        " site = pathlib.Path(pywhispercpp.__file__).resolve().parent.parent\n"
        " print('1' if list(site.glob('**/libggml-cuda*')) else '0')\n"
        "except Exception:\n"
        " print('0')\n"
    )
    try:
        out = subprocess.check_output(
            [py, "-c", code],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).strip()
        return out.endswith("1")
    except Exception:
        return False


def nvcc_available() -> bool:
    """True if nvcc is on PATH (needed to build CUDA pywhispercpp)."""
    return shutil.which("nvcc") is not None


def build_cpp_cuda(
    progress_cb: ProgressCb = None,
    python_exe: Optional[str] = None,
) -> Tuple[bool, str]:
    """Run the CUDA source build into the active/main venv.

    Args:
        progress_cb: Optional callable receiving log lines for UI.
        python_exe: Target interpreter (unused by shell script; script picks venv).

    Returns:
        (success, message)
    """
    def emit(line: str) -> None:
        """Emits a line of text to logger and progress callback."""
        logger.info("%s", line)
        if progress_cb:
            try:
                progress_cb(line)
            except Exception:
                pass

    if cpp_cuda_build_available(python_exe):
        return True, "CUDA pywhispercpp already built"

    if not nvcc_available():
        return (
            False,
            "nvcc (CUDA toolkit) not found. Install CUDA toolkit, then retry, "
            "or use backend faster_whisper with device cuda.",
        )

    if not INSTALL_SCRIPT.is_file():
        return False, f"Build script missing: {INSTALL_SCRIPT}"

    if not os.access(INSTALL_SCRIPT, os.X_OK):
        try:
            INSTALL_SCRIPT.chmod(INSTALL_SCRIPT.stat().st_mode | 0o111)
        except Exception:
            pass

    emit("Starting CUDA pywhispercpp source build (5–15+ minutes)…")
    emit(f"Script: {INSTALL_SCRIPT}")
    try:
        proc = subprocess.Popen(
            ["bash", str(INSTALL_SCRIPT)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            emit(line.rstrip())
        rc = proc.wait(timeout=3600)
        if rc != 0:
            return False, f"Build failed with exit code {rc}"
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        return False, "Build timed out after 1 hour"
    except Exception as exc:
        return False, f"Build failed: {exc}"

    if not cpp_cuda_build_available(python_exe or sys.executable):
        return (
            False,
            "Build finished but libggml-cuda still not found. "
            "Check build log; use faster_whisper+cuda instead.",
        )
    return True, "CUDA pywhispercpp build succeeded"


def build_cpp_cuda_async(
    on_line: ProgressCb = None,
    on_done: Optional[Callable[[bool, str], None]] = None,
    python_exe: Optional[str] = None,
) -> threading.Thread:
    """Start build_cpp_cuda in a daemon thread.

    Args:
        on_line: Progress line callback (any thread).
        on_done: Called with (ok, message) when finished.
        python_exe: Target interpreter for probe after build.

    Returns:
        The started Thread.
    """
    def _run() -> None:
        """Starts a thread to run CUDA build process."""
        ok, msg = build_cpp_cuda(progress_cb=on_line, python_exe=python_exe)
        if on_done:
            try:
                on_done(ok, msg)
            except Exception:
                pass

    t = threading.Thread(target=_run, name="cpp-cuda-build", daemon=True)
    t.start()
    return t
