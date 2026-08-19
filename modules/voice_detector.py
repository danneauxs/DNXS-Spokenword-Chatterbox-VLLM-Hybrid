"""
Voice Detection Module
Handles voice detection from multiple sources: JSON metadata, log files, filenames
"""

import json
import os
import re
import sys
from pathlib import Path
from config.config import AUDIOBOOK_ROOT
from modules.file_manager import list_voice_samples


def get_likely_voices_for_book(book_name, chunks_json_path=None):
    """
    Get voice candidates from the book TTS directory and configured samples.

    A matching ``voice_used`` value in JSON is placed first when its generated
    TTS-ready file exists.

    Returns: list of (voice_name, voice_path, detection_method) tuples
    """
    print(f"🔍 Finding likely voices for book: {book_name}")
    print(f"🔎 voice_detector file: {__file__}")
    print(f"🔎 cwd: {os.getcwd()}")
    print(f"🔎 sys.path[0:10]: {sys.path[:10]}")
    print(f"🔎 AUDIOBOOK_ROOT: {AUDIOBOOK_ROOT}")
    likely_voices = []

    # JSON name is authoritative for the generated TTS copy.
    tts_dir = None
    if chunks_json_path:
        tts_dir = get_tts_dir_from_json(chunks_json_path)
        print(f"🔎 Repair JSON path: {chunks_json_path}")
        print(f"🔎 Derived TTS dir: {tts_dir}")
        print(f"🔎 Derived TTS dir exists: {tts_dir.exists() if tts_dir else None}")
        print(f"🔎 Derived TTS dir is_dir: {tts_dir.is_dir() if tts_dir else None}")
        voice_from_json = get_voice_from_json(chunks_json_path)
        print(f"🔎 voice_used from JSON: {voice_from_json}")
        if voice_from_json:
            print(f"🔎 Looking for JSON voice: {voice_from_json}")
            voice_path = find_voice_in_tts_dir(tts_dir, voice_from_json)
            if voice_path:
                likely_voices.append((voice_from_json, voice_path, "json_metadata"))
                print(f"✅ Voice found in TTS directory from JSON: {voice_from_json}")
            else:
                print(
                    f"⚠️ No direct TTS match for {voice_from_json} in {tts_dir}"
                )

    for voice_name, voice_path in get_voices_from_tts_dir(tts_dir, book_name):
        if not any(v[1] == voice_path for v in likely_voices):
            likely_voices.append((voice_name, voice_path, "tts_directory"))

    for voice_path in list_voice_samples():
        voice_name = voice_path.stem.removesuffix("_ttsready")
        if not any(v[1] == voice_path for v in likely_voices):
            likely_voices.append((voice_name, voice_path, "voice_samples"))
            print(f"🔎 Added voice sample fallback: {voice_name} | {voice_path}")

    if not likely_voices:
        print(f"⚠️ No likely voices detected for {book_name}")
    else:
        print(f"📋 Found {len(likely_voices)} likely voice candidates")

    return likely_voices


def detect_voice_for_book(book_name, chunks_json_path=None):
    """
    Detect the most likely voice for a book (returns first candidate)
    For backwards compatibility with existing code
    """
    likely_voices = get_likely_voices_for_book(book_name, chunks_json_path)
    if likely_voices:
        return likely_voices[0]  # Return the first (most likely) candidate
    return None, None, "not_found"


def get_voice_from_json(json_path):
    """Extract stored voice name from standard JSON metadata."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        print(f"🔎 JSON read path: {json_path}")
        print(f"🔎 JSON first 200 chars: {content[:200]!r}")

        if '"voice_used":' in content:
            data = json.loads(content)
            print(f"🔎 JSON type: {type(data).__name__}")
            if isinstance(data, dict) and "voice_used" in data:
                print(f"🔎 JSON top-level voice_used: {data['voice_used']}")
                return data["voice_used"]
            if isinstance(data, list) and data and "voice_used" in data[0]:
                print(f"🔎 JSON first-item voice_used: {data[0]['voice_used']}")
                return data[0]["voice_used"]

        # Check for voice as comment in JSON (fallback option)
        voice_comment_match = re.search(
            r"//\s*voice:\s*([^\n]+)", content, re.IGNORECASE
        )
        if voice_comment_match:
            return voice_comment_match.group(1).strip()

    except Exception as e:
        print(f"⚠️ Error reading JSON for voice info: {e}")

    return None


def get_voice_from_log(book_name):
    """Extract voice information from run.log file"""
    audiobook_root = Path(AUDIOBOOK_ROOT)
    log_file = audiobook_root / book_name / "run.log"

    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Voice: ") or line.startswith("Voice used: "):
                        voice_name = line.split(": ", 1)[1].strip()
                        return voice_name
        except Exception as e:
            print(f"⚠️ Error reading run log: {e}")

    return None


def get_voices_from_filenames(book_name):
    """Extract voice names from existing audiobook filename patterns (may return multiple)"""
    audiobook_root = Path(AUDIOBOOK_ROOT)
    book_dir = audiobook_root / book_name

    if not book_dir.exists():
        return []

    found_voices = []

    # Look for WAV files with voice pattern: BookName [VoiceName].wav
    for wav_file in book_dir.glob("*.wav"):
        match = re.search(r"\[([^\]]+)\]\.wav$", wav_file.name)
        if match:
            voice_name = match.group(1)
            if voice_name not in found_voices:
                found_voices.append(voice_name)

    # Look for M4B files with voice pattern: BookName[VoiceName].m4b
    for m4b_file in book_dir.glob("*.m4b"):
        match = re.search(r"\[([^\]]+)\]\.m4b$", m4b_file.name)
        if match:
            voice_name = match.group(1)
            if voice_name not in found_voices:
                found_voices.append(voice_name)

    return found_voices


def get_voice_from_filename(book_name):
    """Extract voice name from existing audiobook filename patterns (backwards compatibility)"""
    voices = get_voices_from_filenames(book_name)
    return voices[0] if voices else None


def find_voice_file_by_name(voice_name):
    """Find voice file by name in Voice_Samples directory"""
    voice_files = list_voice_samples()

    # Exact match first
    for voice_file in voice_files:
        if voice_file.stem == voice_name:
            return voice_file

    # Partial match (case insensitive)
    voice_name_lower = voice_name.lower()
    for voice_file in voice_files:
        if voice_name_lower in voice_file.stem.lower():
            return voice_file

    return None


def find_voice_in_tts_dir(tts_dir, voice_name):
    """Find exact generated TTS voice copy for a stored voice name."""
    for candidate_name, candidate_path in get_voices_from_tts_dir(tts_dir):
        print(
            f"🔎 Compare JSON voice '{voice_name}' to TTS candidate '{candidate_name}' ({candidate_path.name})"
        )
        if candidate_name == voice_name:
            print(f"✅ Match found: {candidate_path}")
            return candidate_path
    print(f"⚠️ No exact TTS match for voice '{voice_name}' in {tts_dir}")
    return None


def get_tts_dir_from_json(json_path):
    """Resolve book TTS directory from selected chunks JSON path."""
    json_path = Path(json_path)
    return json_path.parent.parent


def get_voices_from_tts_dir(tts_dir, book_name=None):
    """List voice WAV files directly inside book TTS storage."""
    if tts_dir is None:
        if not book_name:
            return []
        tts_dir = Path(AUDIOBOOK_ROOT) / book_name / "TTS"
    else:
        tts_dir = Path(tts_dir)

    if not tts_dir.is_dir():
        print(f"⚠️ TTS dir not found: {tts_dir}")
        return []

    voices = []
    wav_files = sorted(tts_dir.glob("*.wav"), key=lambda path: path.stem.lower())
    print(f"🔎 Scanning TTS WAV files in: {tts_dir}")
    for wav_file in wav_files:
        print(f"🔎 TTS file candidate: {wav_file.name}")
    for voice_path in wav_files:
        voice_name = voice_path.stem.removesuffix("_ttsready")
        print(
            f"🔎 Derived candidate name: '{voice_name}' from stem '{voice_path.stem}'"
        )
        voices.append((voice_name, voice_path))
    print(f"🔎 Total TTS WAV candidates: {len(voices)}")
    return voices


def add_voice_to_json(json_path, voice_name, method="metadata"):
    """
    Add voice information to JSON file

    method options:
    - "metadata": Add as top-level metadata
    - "comment": Add as comment that doesn't affect parsing
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        if method == "metadata":
            # Add voice as metadata to JSON structure
            data = json.loads(content)

            if isinstance(data, list):
                # For list format, add metadata as first element or update existing
                if (
                    data
                    and isinstance(data[0], dict)
                    and not any(key.startswith("text") for key in data[0].keys())
                ):
                    # First element is already metadata
                    data[0]["voice_used"] = voice_name
                else:
                    # Insert metadata as first element
                    metadata = {"voice_used": voice_name, "_metadata": True}
                    data.insert(0, metadata)
            elif isinstance(data, dict):
                # For dict format, add to top level
                data["voice_used"] = voice_name

            # Save updated JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif method == "comment":
            # Add voice as comment at the top of file
            voice_comment = f"// voice: {voice_name}\n"

            if not content.startswith("// voice:"):
                content = voice_comment + content
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(content)

        print(
            f"✅ Added voice '{voice_name}' to {json_path.name} using {method} method"
        )
        return True

    except Exception as e:
        print(f"❌ Error adding voice to JSON: {e}")
        return False


def remove_voice_comment_from_json(json_path):
    """Remove voice comment from JSON file for clean processing"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Remove voice comment lines
        lines = content.split("\n")
        filtered_lines = [
            line for line in lines if not line.strip().startswith("// voice:")
        ]

        if len(filtered_lines) != len(lines):
            # Comments were removed, save cleaned version
            cleaned_content = "\n".join(filtered_lines)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(cleaned_content)
            return True

    except Exception as e:
        print(f"⚠️ Error cleaning JSON comments: {e}")

    return False
