#!/usr/bin/env bash
# Launch Chatterbox standalone ASR validator with the selected project Python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -n "${CHATTERBOX_PYTHON:-}" ]]; then
    PYTHON_BIN="$CHATTERBOX_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
elif [[ -x "/home/danno/.pyenv/versions/my-project-3.10/bin/python" ]]; then
    PYTHON_BIN="/home/danno/.pyenv/versions/my-project-3.10/bin/python"
else
    PYTHON_BIN="python3"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/asr_gui.py" "$@"
