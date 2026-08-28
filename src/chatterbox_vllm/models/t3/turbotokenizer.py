"""vLLM tokenizer wrapper for Turbo T3's GPT2 tokenizer files."""

from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizer


def _turbo_tokenizer_dir() -> str:
    """Return the directory that holds Turbo GPT2 tokenizer JSON/merges."""
    env = os.environ.get("TURBO_CKPT_DIR")
    if env:
        return env
    return str(Path(__file__).resolve().parents[3] / "models" / "chatterbox-turbo")


class TurboTokenizer:
    """GPT2 tokenizer plus vLLM max_token_id covering speech-token offset ids.

    Prefill markers sit at 50276-50278 and decode speech ids at 51000+.
    HuggingFace GPT2TokenizerFast has no max_token_id; vLLM requires it.
    """

    def __init__(self, inner):
        """Wrap a GPT2 tokenizer and advertise the extended vLLM token id range."""
        object.__setattr__(self, "_inner", inner)
        # Speech offset 51000 + vocab 6563, plus padding.
        object.__setattr__(self, "max_token_id", 51000 + 6563 + 64)

    def __getattr__(self, name):
        """Delegate unknown attributes to the HuggingFace GPT2 tokenizer."""
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        """Tokenize like HuggingFace GPT2TokenizerFast."""
        return self._inner(*args, **kwargs)

    def __len__(self):
        """Return GPT2 vocab size so vLLM detokenizer checks succeed."""
        return len(self._inner)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Instantiate from TURBO_CKPT_DIR GPT2 tokenizer files."""
        return cls(AutoTokenizer.from_pretrained(_turbo_tokenizer_dir()))
