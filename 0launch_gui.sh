#!/usr/bin/env bash
# Launch the Pipeline 4 GUI from this portable Distribution directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "venv/bin/activate" || ! -x "venv/bin/python" ]]; then
    echo "Pipeline 4 is not installed. Run ./install.sh first."
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"
hash -r
echo "Using virtual environment: $VIRTUAL_ENV"

export CHATTERBOX_TTS_BACKEND="turbo-hybrid"
export CHATTERBOX_T3_SOURCE="${CHATTERBOX_T3_SOURCE:-multilingual-v3}"
export CHATTERBOX_T3_LANGUAGE="${CHATTERBOX_T3_LANGUAGE:-en}"
export CHATTERBOX_CKPT_DIR="$SCRIPT_DIR/models/chatterbox"
export CHATTERBOX_MTL_CKPT_DIR="$SCRIPT_DIR/models/chatterbox-mtl-v3"
export CHATTERBOX_MTL_V2_CKPT_DIR="$SCRIPT_DIR/models/chatterbox-mtl-v2"
export TURBO_CKPT_DIR="$SCRIPT_DIR/models/chatterbox-turbo"
export VLLM_ENGLISH_CKPT_DIR="$SCRIPT_DIR/models/vllm-t3"
export VLLM_MULTILINGUAL_CKPT_DIR="$SCRIPT_DIR/models/vllm-t3-mtl-v3"
export VLLM_MULTILINGUAL_V2_CKPT_DIR="$SCRIPT_DIR/models/vllm-t3-mtl-v2"
export VLLM_TURBO_CKPT_DIR="$SCRIPT_DIR/models/vllm-t3-turbo"
export PYTHONPATH="$SCRIPT_DIR/chatterbox-vllm/src:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_ENABLE_SDPA="1"
export VLLM_HOST_IP="127.0.0.1"

python -m modules.model_bootstrap
exec python chatterbox_gui.py "$@"
