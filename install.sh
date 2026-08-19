#!/usr/bin/env bash
# Install all runtime dependencies for portable Pipeline 4.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3.10 or newer is required."
    exit 1
fi

PYTHON_VERSION="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "Python 3.10 or newer is required. Found $PYTHON_VERSION."
    exit 1
fi

if [[ ! -d "venv" ]]; then
    echo "Creating virtual environment in $SCRIPT_DIR/venv..."
    python3 -m venv venv
fi

if [[ ! -f "venv/bin/activate" ]]; then
    echo "Virtual environment is incomplete: venv/bin/activate is missing." >&2
    echo "Remove the incomplete venv directory and run this installer again." >&2
    exit 1
fi

# Activate for the rest of this installer so every python and pip command uses
# the newly created environment rather than the system interpreter.
source "$SCRIPT_DIR/venv/bin/activate"
hash -r

echo "Using virtual environment: $VIRTUAL_ENV"
python -m pip install --upgrade pip setuptools wheel

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU detected. Pipeline 4 requires an NVIDIA CUDA GPU."
    exit 1
fi

CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)"
if [[ -z "$CUDA_VERSION" ]] || ! python -c "raise SystemExit(0 if float('$CUDA_VERSION') >= 12.6 else 1)"; then
    echo "NVIDIA driver supporting CUDA 12.6 or newer required for Pipeline 4."
    exit 1
fi

echo "Installing CUDA PyTorch and Pipeline 4 dependencies..."
python -m pip install torch==2.7.1 torchaudio==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

python - <<'PY'
import torch
import vllm
from PyQt5 import QtCore

if not torch.cuda.is_available():
    raise SystemExit("PyTorch installed, but CUDA is unavailable.")

print(f"CUDA ready: {torch.cuda.get_device_name(0)}")
print(f"vLLM ready: {vllm.__version__}")
print(f"PyQt ready: {QtCore.QT_VERSION_STR}")
PY

echo
echo "Install complete. Run ./0launch_gui.sh."
echo "First launch downloads Pipeline 4 model files into ./models/."
