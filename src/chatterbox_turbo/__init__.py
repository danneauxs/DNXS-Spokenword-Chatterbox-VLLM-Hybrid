"""Chatterbox package init (kept minimal to avoid heavy side effects on import).

Import submodules explicitly, e.g.:
  from src.chatterbox.tts_turbo import ChatterboxTurboTTS
  from src.chatterbox.vc import ChatterboxVC
"""

from .tts_turbo import ChatterboxTurboTTS
from .vc import ChatterboxVC

__all__ = ['ChatterboxTurboTTS']
