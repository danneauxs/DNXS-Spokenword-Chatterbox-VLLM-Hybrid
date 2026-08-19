#!/usr/bin/env python3
"""Generate existing chunk metadata through Pipeline 4 for the GUI."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import TEXT_INPUT_ROOT
from modules.file_manager import ensure_voice_sample_compatibility, list_voice_samples
from modules.tts_engine import get_best_available_device, process_book_folder


def generate_audiobook_from_json(json_path, voice_name, temp_setting=None):
    """Generate an existing chunks JSON through vLLM T3 and Turbo S3Gen.

    Args:
        json_path: Path to the existing ``chunks_info.json`` file.
        voice_name: Voice sample stem selected in the GUI.
        temp_setting: Retained for GUI API compatibility; JSON values win.

    Returns:
        Tuple of success flag, status message, and final audiobook path.
    """
    try:
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        with json_file.open("r", encoding="utf-8") as json_stream:
            json_records = json.load(json_stream)
        if not isinstance(json_records, list):
            raise ValueError("JSON file must contain a list of chunk records")
        if not all(isinstance(item, dict) for item in json_records):
            raise ValueError("JSON file records must be JSON objects")

        chunks = [
            item
            for item in json_records
            if not item.get("_metadata", False)
        ]
        if not chunks or any(not isinstance(item.get("text"), str) for item in chunks):
            raise ValueError("JSON file must contain text chunk records")

        if "Audiobook" not in json_file.parts:
            raise ValueError("JSON path must be inside an Audiobook/<book>/ directory")
        audiobook_index = json_file.parts.index("Audiobook")
        if audiobook_index + 1 >= len(json_file.parts):
            raise ValueError("Cannot determine book name from JSON path")
        book_name = json_file.parts[audiobook_index + 1]
        book_dir = TEXT_INPUT_ROOT / book_name
        if not book_dir.is_dir():
            raise FileNotFoundError(
                f"Matching source book directory not found: {book_dir}"
            )

        voice_path = next(
            (voice for voice in list_voice_samples() if voice.stem == voice_name),
            None,
        )
        if voice_path is None:
            available = [voice.stem for voice in list_voice_samples()]
            return False, f"Voice '{voice_name}' not found. Available: {available}", None
        voice_path = ensure_voice_sample_compatibility(voice_path)

        device = get_best_available_device()
        first_params = next(
            (item.get("tts_params", {}) for item in chunks if item.get("tts_params")),
            {},
        )
        from config import config as runtime_config

        runtime_config.TTS_BACKEND = "turbo-hybrid"
        print(f"Pipeline 4 JSON generation: {len(chunks)} chunks")
        print(f"Using existing JSON: {json_file}")
        print("Backend forced to turbo-hybrid for exact Pipeline 4 testing")
        if temp_setting is not None:
            print("Using per-chunk JSON parameters; GUI temperature is ignored")

        final_m4b_path, _, _ = process_book_folder(
            book_dir=book_dir,
            voice_path=voice_path,
            tts_params=first_params,
            device=device,
            enable_asr=False,
            existing_json_path=json_file,
        )
        if final_m4b_path and Path(final_m4b_path).exists():
            return (
                True,
                "Pipeline 4 JSON generation completed successfully",
                str(final_m4b_path),
            )
        return False, "Pipeline 4 completed without producing final audiobook", None
    except Exception as exc:
        message = f"Pipeline 4 JSON generation error: {exc}"
        print(f"❌ {message}")
        return False, message, None


def get_book_name_from_json_path(json_path):
    """Extract the audiobook book name from a JSON path."""
    json_file = Path(json_path)
    if "Audiobook" in json_file.parts:
        audiobook_index = json_file.parts.index("Audiobook")
        if audiobook_index + 1 < len(json_file.parts):
            return json_file.parts[audiobook_index + 1]
    if json_file.stem.endswith("_chunks"):
        return json_file.stem.replace("_chunks", "")
    return json_file.stem


if __name__ == "__main__":
    print("GUI JSON Generator - use from GUI or import as module")
