import subprocess
import os


def play_chunk_audio(path):
    """Plays an audio file using ffplay.
    Args:
    - path (str): The path to the audio file.
    Returns:
    None
    """
    if not os.path.exists(path):
        print(f"❌ Audio file not found: {path}")
        return
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Error playing audio: {e}")
