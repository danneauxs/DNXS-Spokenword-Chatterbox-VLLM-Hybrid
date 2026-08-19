"""
Combine Only Tool
Standalone tool for combining existing audio chunks into final audiobook
"""

import re
import time
from datetime import timedelta
from pathlib import Path

from config.config import *
from modules.file_manager import (
    get_audio_files_in_directory,
    combine_audio_chunks,
    convert_to_m4b,
    add_metadata_to_m4b,
    find_book_files,
)
from modules.audio_processor import get_wav_duration


def combine_audio_for_book(book_path_str, voice_name=None):
    """Combine audio chunks for a specific book (GUI-friendly version)"""

    book_path = Path(book_path_str)

    print(f"\n{CYAN}🔗 Combining Audio Chunks for: {book_path.name}{RESET}")
    print("=" * 60)

    # Setup paths
    tts_dir = book_path / "TTS"
    audio_chunks_dir = tts_dir / "audio_chunks"

    if not audio_chunks_dir.exists():
        print(f"{RED}❌ No audio_chunks folder found in {book_path}{RESET}")
        print("💡 Make sure this book has been processed with TTS generation first.")
        return False

    # Find audio chunks
    chunk_paths = get_audio_files_in_directory(audio_chunks_dir)

    if not chunk_paths:
        print(f"{RED}❌ No chunk_*.wav files found in {audio_chunks_dir}{RESET}")
        print("💡 Expected files like: chunk_00001.wav, chunk_00002.wav, etc.")
        return False

    print(f"\n📦 Found {GREEN}{len(chunk_paths)}{RESET} audio chunks")

    # Verify chunk sequence
    missing_chunks = verify_chunk_sequence(chunk_paths)
    if missing_chunks:
        print(f"\n⚠️ {YELLOW}Warning: Missing chunks detected:{RESET}")
        for chunk_num in missing_chunks[:10]:  # Show first 10 missing
            print(f"   Missing: chunk_{chunk_num:05}.wav")
        if len(missing_chunks) > 10:
            print(f"   ... and {len(missing_chunks) - 10} more")
        print(
            f"{YELLOW}🔄 Continuing with available chunks for GUI operation...{RESET}"
        )

    # Display chunk info
    total_duration = sum(get_wav_duration(chunk_path) for chunk_path in chunk_paths)
    duration_str = str(timedelta(seconds=int(total_duration)))

    print("\n📊 Chunk Analysis:")
    print(f"   Total Chunks: {GREEN}{len(chunk_paths)}{RESET}")
    print(f"   Total Duration: {GREEN}{duration_str}{RESET}")
    print(f"   Average Chunk: {GREEN}{total_duration/len(chunk_paths):.1f}s{RESET}")

    # Perform the actual combine operation
    return _perform_combine_operation(
        book_path, chunk_paths, total_duration, voice_name
    )


def _perform_combine_operation(book_path, chunk_paths, total_duration, voice_name=None):
    """Perform the actual audio combining operation"""
    from datetime import timedelta

    basename = book_path.name

    # Determine file naming based on voice
    if voice_name:
        file_suffix = f" [{voice_name}]"
    else:
        file_suffix = "_combined"

    # Start timing
    start_time = time.time()

    # Create concat file and combine
    print("\n🔗 Combining audio chunks...")
    combined_wav_path = book_path / f"{basename}{file_suffix}.wav"

    try:
        combine_audio_chunks(chunk_paths, combined_wav_path)
        print(f"✅ Combined WAV created: {combined_wav_path.name}")
    except Exception as e:
        print(f"{RED}❌ Failed to combine chunks: {e}{RESET}")
        return False

    # Find metadata files
    text_book_dir = TEXT_INPUT_ROOT / basename
    book_files = find_book_files(text_book_dir)
    text_files, cover_file, nfo_file = (
        book_files["text"],
        book_files["cover"],
        book_files["nfo"],
    )

    if not cover_file:
        print(f"⚠️ {YELLOW}No cover image found in {text_book_dir}{RESET}")
    else:
        print(f"📸 Using cover: {cover_file.name}")

    if not nfo_file:
        print(f"⚠️ {YELLOW}No book.nfo metadata found in {text_book_dir}{RESET}")
    else:
        print(f"📝 Using metadata: {nfo_file.name}")

    # M4B conversion
    print("\n📱 Converting to M4B audiobook...")
    temp_m4b_path = book_path / "temp_output.m4b"
    final_m4b_path = book_path / f"{basename}{file_suffix}.m4b"

    try:
        convert_to_m4b(combined_wav_path, temp_m4b_path)
        add_metadata_to_m4b(temp_m4b_path, final_m4b_path, cover_file, nfo_file)
        print(f"✅ M4B audiobook created: {final_m4b_path.name}")
    except FileNotFoundError:
        print(f"{RED}❌ FFmpeg not available - cannot create M4B audiobook{RESET}")
        print(f"{YELLOW}📁 WAV file is available at: {combined_wav_path}{RESET}")
        print(f"{YELLOW}💡 To create M4B files, install FFmpeg and try again{RESET}")
        return False
    except Exception as e:
        print(f"{RED}❌ Failed to create M4B: {e}{RESET}")
        return False

    # Calculate final timing
    elapsed_total = time.time() - start_time
    elapsed_td = timedelta(seconds=int(elapsed_total))

    # Verify final file
    if final_m4b_path.exists():
        final_size = final_m4b_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📦 Final file size: {GREEN}{final_size:.1f} MB{RESET}")

        # Calculate efficiency
        realtime_factor = total_duration / elapsed_total if elapsed_total > 0 else 0
        duration_str = str(timedelta(seconds=int(total_duration)))

        print(f"\n🎉 {GREEN}Combine completed successfully!{RESET}")
        print("📊 Final Statistics:")
        print(f"   Audio Duration: {GREEN}{duration_str}{RESET}")
        print(f"   Processing Time: {GREEN}{elapsed_td}{RESET}")
        print(f"   Realtime Factor: {GREEN}{realtime_factor:.2f}x{RESET}")
        print(f"   Output Location: {GREEN}{final_m4b_path}{RESET}")

        # Clean up temp files
        try:
            if temp_m4b_path.exists():
                temp_m4b_path.unlink()
                print(f"🧹 Cleaned up temporary file: {temp_m4b_path.name}")
        except Exception as e:
            print(f"⚠️ Could not clean up temp file: {e}")

        return True
    else:
        print(f"{RED}❌ Final M4B file was not created successfully{RESET}")
        return False


def run_combine_only_mode():
    """Combine existing chunks into audiobook (CLI version)"""
    print(f"\n{CYAN}🔗 Combine-Only Mode: Assembling Existing Audio Chunks{RESET}")
    print("=" * 60)

    # Show available audiobooks
    books = sorted([d for d in AUDIOBOOK_ROOT.iterdir() if d.is_dir()])
    if not books:
        print(f"{RED}❌ No folders found in Audiobook/ directory.{RESET}")
        print("💡 Make sure you have processed books with audio chunks to combine.")
        return None

    print(f"{CYAN}Available audiobooks to combine:{RESET}")
    for i, book in enumerate(books):
        # Check if it has audio chunks
        audio_chunks_dir = book / "TTS" / "audio_chunks"
        if audio_chunks_dir.exists():
            chunk_count = len(list(audio_chunks_dir.glob("chunk_*.wav")))
            status = f"({chunk_count} chunks)" if chunk_count > 0 else "(no chunks)"
            print(f"  [{i}] {book.name} {status}")
        else:
            print(f"  [{i}] {book.name} (no TTS folder)")

    # Book selection
    while True:
        try:
            idx = int(input(f"\n{YELLOW}Select audiobook index: {RESET}"))
            if 0 <= idx < len(books):
                break
            else:
                print(
                    f"{RED}Invalid selection. Please enter a number between 0 and {len(books)-1}.{RESET}"
                )
        except (ValueError, KeyboardInterrupt):
            print(f"{RED}Invalid selection. Please try again.{RESET}")
        except EOFError:
            print(f"\n{RED}❌ Input error - unable to read selection.{RESET}")
            return None
        except Exception as e:
            print(f"{RED}❌ Unexpected error: {e}{RESET}")
            return None

    selected_book = books[idx]
    basename = selected_book.name

    print(f"\n🎯 Selected: {BOLD}{basename}{RESET}")

    # Setup paths
    tts_dir = selected_book / "TTS"
    audio_chunks_dir = tts_dir / "audio_chunks"

    if not audio_chunks_dir.exists():
        print(f"{RED}❌ No audio_chunks folder found in {selected_book}{RESET}")
        print("💡 Make sure this book has been processed with TTS generation first.")
        return None

    # Find audio chunks
    chunk_paths = get_audio_files_in_directory(audio_chunks_dir)

    if not chunk_paths:
        print(f"{RED}❌ No chunk_*.wav files found in {audio_chunks_dir}{RESET}")
        print("💡 Expected files like: chunk_00001.wav, chunk_00002.wav, etc.")
        return None

    print(f"\n📦 Found {GREEN}{len(chunk_paths)}{RESET} audio chunks")

    # Verify chunk sequence
    missing_chunks = verify_chunk_sequence(chunk_paths)
    if missing_chunks:
        print(f"\n⚠️ {YELLOW}Warning: Missing chunks detected:{RESET}")
        for chunk_num in missing_chunks[:10]:  # Show first 10 missing
            print(f"   Missing: chunk_{chunk_num:05}.wav")
        if len(missing_chunks) > 10:
            print(f"   ... and {len(missing_chunks) - 10} more")

        try:
            continue_anyway = (
                input(f"\n{YELLOW}Continue with incomplete chunks? [y/N]: {RESET}")
                .strip()
                .lower()
            )
            if continue_anyway != "y":
                print("🛑 Combine operation cancelled.")
                return None
        except (EOFError, KeyboardInterrupt):
            print(f"\n{RED}🛑 Combine operation cancelled.{RESET}")
            return None

    # Display chunk info
    total_duration = sum(get_wav_duration(chunk_path) for chunk_path in chunk_paths)
    duration_str = str(timedelta(seconds=int(total_duration)))

    print("\n📊 Chunk Analysis:")
    print(f"   Total Chunks: {GREEN}{len(chunk_paths)}{RESET}")
    print(f"   Total Duration: {GREEN}{duration_str}{RESET}")
    print(f"   Average Chunk: {GREEN}{total_duration/len(chunk_paths):.1f}s{RESET}")

    # Use the shared combine operation (CLI doesn't pass voice name)
    success = _perform_combine_operation(selected_book, chunk_paths, total_duration)

    if success:
        return selected_book / f"{basename}_combined.m4b"
    else:
        return None


def verify_chunk_sequence(chunk_paths):
    """Verify chunk sequence and return missing chunk numbers"""
    chunk_numbers = []

    for chunk_path in chunk_paths:
        match = re.match(r"chunk_(\d+)\.wav", chunk_path.name)
        if match:
            chunk_numbers.append(int(match.group(1)))

    if not chunk_numbers:
        return []

    chunk_numbers.sort()
    expected_range = range(1, max(chunk_numbers) + 1)
    missing = [num for num in expected_range if num not in chunk_numbers]

    return missing


def list_available_books_for_combine():
    """List books available for combine operation"""
    books_info = []

    if not AUDIOBOOK_ROOT.exists():
        return books_info

    for book_dir in AUDIOBOOK_ROOT.iterdir():
        if not book_dir.is_dir():
            continue

        audio_chunks_dir = book_dir / "TTS" / "audio_chunks"
        if not audio_chunks_dir.exists():
            continue

        chunk_paths = get_audio_files_in_directory(audio_chunks_dir)
        if not chunk_paths:
            continue

        # Calculate total duration
        try:
            total_duration = sum(
                get_wav_duration(chunk_path) for chunk_path in chunk_paths
            )
            duration_str = str(timedelta(seconds=int(total_duration)))
        except:
            duration_str = "Unknown"

        books_info.append(
            {
                "name": book_dir.name,
                "path": book_dir,
                "chunk_count": len(chunk_paths),
                "duration": duration_str,
            }
        )

    return books_info


def quick_combine(book_name):
    """Quick combine operation for specific book (CLI usage)"""
    book_path = AUDIOBOOK_ROOT / book_name

    if not book_path.exists():
        print(f"{RED}❌ Book '{book_name}' not found in Audiobook directory{RESET}")
        return None

    audio_chunks_dir = book_path / "TTS" / "audio_chunks"
    chunk_paths = get_audio_files_in_directory(audio_chunks_dir)

    if not chunk_paths:
        print(f"{RED}❌ No audio chunks found for '{book_name}'{RESET}")
        return None

    print(f"🔗 Quick combining {len(chunk_paths)} chunks for '{book_name}'...")

    # Use same logic as main function but without interactive prompts
    combined_wav_path = book_path / f"{book_name}_quick_combined.wav"
    final_m4b_path = book_path / f"{book_name}_quick_combined.m4b"

    combine_audio_chunks(chunk_paths, combined_wav_path)

    temp_m4b_path = book_path / "temp_quick.m4b"

    try:
        convert_to_m4b(combined_wav_path, temp_m4b_path)
        # Simple M4B without metadata for quick operation
        temp_m4b_path.rename(final_m4b_path)
    except FileNotFoundError:
        print(f"{RED}❌ FFmpeg not available - cannot create M4B audiobook{RESET}")
        print(f"{YELLOW}📁 WAV file is available at: {combined_wav_path}{RESET}")
        return None

    print(f"✅ Quick combine complete: {final_m4b_path}")
    return final_m4b_path


# apply_playback_speed_to_m4b function removed - now uses unified convert_to_m4b from modules.file_manager

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # CLI usage: python combine_only.py "Book Name"
        book_name = sys.argv[1]
        quick_combine(book_name)
    else:
        # Interactive mode
        run_combine_only_mode()
