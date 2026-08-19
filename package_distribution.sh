#!/usr/bin/env bash
# Create one self-contained compressed Pipeline 4 distribution archive.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_NAME="$(basename "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="$PARENT_DIR/${PACKAGE_NAME}_${TIMESTAMP}.tar.gz"
DRY_RUN=false

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Create compressed archive containing runnable Distribution files, models, and venv.

Options:
  -o, --output PATH  Archive path (default: $OUTPUT_PATH)
  -n, --dry-run      Validate archive input without writing archive
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output)
            [[ $# -ge 2 ]] || { echo "Missing output path." >&2; exit 2; }
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

required_paths=(
    "0launch_gui.sh"
    "install.sh"
    "chatterbox_gui.py"
    "config"
    "modules"
    "src"
    "wrapper"
)

for required_path in "${required_paths[@]}"; do
    if [[ ! -e "$SCRIPT_DIR/$required_path" ]]; then
        echo "Required package path missing: $required_path" >&2
        exit 1
    fi
done

if [[ "$OUTPUT_PATH" != /* ]]; then
    OUTPUT_PATH="$PWD/$OUTPUT_PATH"
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

tar_excludes=(
    "--exclude=$PACKAGE_NAME/venv"
    "--exclude=$PACKAGE_NAME/models"
    "--exclude=$PACKAGE_NAME/Audiobook"
    "--exclude=$PACKAGE_NAME/logs"
    "--exclude=$PACKAGE_NAME/DocDNA"
    "--exclude=$PACKAGE_NAME/.git"
    "--exclude=$PACKAGE_NAME/.codex"
    "--exclude=$PACKAGE_NAME/.opencode"
    "--exclude=$PACKAGE_NAME/.mcp.json"
    "--exclude=$PACKAGE_NAME/opencode.json"
    "--exclude=$PACKAGE_NAME/AGENTS.md"
    "--exclude=$PACKAGE_NAME/CLAUDE.md"
    "--exclude=$PACKAGE_NAME/term.log"
    "--exclude=$PACKAGE_NAME/__pycache__"
    "--exclude=$PACKAGE_NAME/*/__pycache__"
    "--exclude=$PACKAGE_NAME/*/*/__pycache__"
    "--exclude=$PACKAGE_NAME/*/*/*/__pycache__"
    "--exclude=$PACKAGE_NAME/*.bak"
    "--exclude=$PACKAGE_NAME/*.backup"
    "--exclude=$PACKAGE_NAME/*/*.bak"
    "--exclude=$PACKAGE_NAME/*/*.backup"
    "--exclude=$PACKAGE_NAME/*/*/*.bak"
    "--exclude=$PACKAGE_NAME/*/*/*.backup"
    "--exclude=$PACKAGE_NAME/*~"
    "--exclude=$PACKAGE_NAME/*/*~"
    "--exclude=$PACKAGE_NAME/*/*/*~"
)

tar_options=(
    "--owner=0"
    "--group=0"
    "--numeric-owner"
    "--no-xattrs"
    "--no-acls"
)

if "$DRY_RUN"; then
    echo "Validating package contents; no archive will be written."
    tar -cf /dev/null "${tar_options[@]}" "${tar_excludes[@]}" \
        -C "$PARENT_DIR" "$PACKAGE_NAME"
    echo "Package validation passed."
    exit 0
fi

echo "Creating: $OUTPUT_PATH"
tar -czf "$OUTPUT_PATH" "${tar_options[@]}" "${tar_excludes[@]}" \
    -C "$PARENT_DIR" "$PACKAGE_NAME"

archive_size="$(du -h "$OUTPUT_PATH" | cut -f1)"
echo "Archive created: $OUTPUT_PATH ($archive_size)"
