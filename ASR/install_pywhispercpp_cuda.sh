#!/usr/bin/env bash
# Build/install pywhispercpp with CUDA into the MAIN project venv (single env).
# Requires: nvcc, cmake, g++, CUDA toolkit, NVIDIA driver, patchelf.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Prefer main venv; fall back to ASR/venv only for legacy installs
if [[ -x "$ROOT_DIR/venv/bin/pip" ]]; then
  VENV_PIP="$ROOT_DIR/venv/bin/pip"
  VENV_PY="$ROOT_DIR/venv/bin/python"
  SITE="$ROOT_DIR/venv/lib/python3.12/site-packages"
elif [[ -x "$SCRIPT_DIR/venv/bin/pip" ]]; then
  echo "WARNING: using legacy ASR/venv — prefer main project venv/"
  VENV_PIP="$SCRIPT_DIR/venv/bin/pip"
  VENV_PY="$SCRIPT_DIR/venv/bin/python"
  SITE="$SCRIPT_DIR/venv/lib/python3.12/site-packages"
else
  echo "No venv found. Create main venv first:"
  echo "  cd $ROOT_DIR && python3 -m venv venv && source venv/bin/activate"
  echo "  pip install -r requirements.txt"
  exit 1
fi
LIBS="$SITE/pywhispercpp.libs"

if ! command -v nvcc >/dev/null; then
  echo "nvcc not found — install CUDA toolkit first"
  exit 1
fi
if ! command -v patchelf >/dev/null; then
  echo "patchelf not found — install with: sudo apt install patchelf"
  exit 1
fi

echo "=== Target venv: $VENV_PY ==="
echo "=== Uninstall old pywhispercpp ==="
"$VENV_PIP" uninstall -y pywhispercpp || true

echo "=== Build from source with GGML_CUDA=1 ==="
export GGML_CUDA=1
export WHISPER_CUDA=1
export CMAKE_ARGS="-DGGML_CUDA=ON"
GGML_CUDA=1 CMAKE_ARGS="-DGGML_CUDA=ON" \
  "$VENV_PIP" install --no-cache-dir --force-reinstall --no-binary=pywhispercpp \
  "git+https://github.com/absadiki/pywhispercpp.git"

echo "=== Fix delocated libcuda (must use system driver libcuda) ==="
if [[ -d "$LIBS" ]]; then
  CUDA_HASHED=$(ls "$LIBS"/libcuda-*.so* 2>/dev/null | head -1 || true)
  if [[ -n "${CUDA_HASHED:-}" ]]; then
    HASH_NAME=$(basename "$CUDA_HASHED")
    echo "Replace NEEDED $HASH_NAME → libcuda.so.1"
    for f in "$LIBS"/lib*.so*; do
      if patchelf --print-needed "$f" 2>/dev/null | grep -qx "$HASH_NAME"; then
        patchelf --replace-needed "$HASH_NAME" libcuda.so.1 "$f"
        echo "  patched $(basename "$f")"
      fi
    done
    rm -f "$LIBS"/libcuda-*
    echo "Removed bundled libcuda"
  else
    echo "No bundled libcuda (ok)"
  fi
fi

"$VENV_PIP" install -q 'numpy>=1.26,<2.5' || true

echo "=== Verify CUDA load ==="
cd "$SCRIPT_DIR"
"$VENV_PY" - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
from whisper_cpp_backend import load_whisper_cpp_model

m, d = load_whisper_cpp_model("base", force_device="cuda", n_threads=2)
if m is None or d != "cuda":
    raise SystemExit(f"FAIL: expected cuda model, got model={m is not None} device={d}")
print("OK: whisper.cpp on GPU")
import pathlib
site = pathlib.Path(__import__("pywhispercpp").__file__).resolve().parent.parent
print("libggml-cuda:", list(site.glob("**/libggml-cuda*"))[:3])
PY

echo "=== Done. Use main venv python for ASR (whisper_cpp + cuda). ==="
