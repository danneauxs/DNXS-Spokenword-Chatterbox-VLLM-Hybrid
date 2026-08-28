#!/usr/bin/env python3
"""Check pause tagging does not split T3 mid-clause or on abbreviations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.pause_utils import parse_pause_tags
from modules.pause_utils import convert_inline_markers_to_pause_tags
from modules.punctuation_pauses import add_pause_tags_to_text


def main() -> int:
    """Run pause-split cases and return 0 on success."""
    tagged, _ = add_pause_tags_to_text(
        "Hello, there, she said. Next sentence starts.", "period"
    )
    assert "." not in tagged, tagged
    assert "she said [pause:" in tagged and "] Next" in tagged, tagged
    segs, pauses = parse_pause_tags(tagged)
    # Comma default 0: period split only, not five comma cuts.
    assert len(segs) <= 2, (tagged, segs)

    tagged2, _ = add_pause_tags_to_text("Dr. Smith went home. Next arrived.", "period")
    segs2, _ = parse_pause_tags(tagged2)
    joined = " ".join(segs2)
    assert "Dr." in tagged2 or "Dr. Smith" in joined or "Smith" in joined, tagged2
    assert not any(s.strip() in {"Dr", "Dr."} for s in segs2), segs2

    segs3, pauses3 = parse_pause_tags("Hello [pause:0ms] world")
    assert len(segs3) == 1, segs3
    assert "Hello" in segs3[0] and "world" in segs3[0]

    segs4, _ = parse_pause_tags("A long enough first sentence here [pause:300ms] he said")
    assert len(segs4) == 1, segs4

    segs5, p5 = parse_pause_tags(
        "The cat sat on the mat.[pause:300ms] The dog ran home."
    )
    assert len(segs5) == 2, segs5
    assert len(p5) == 1 and p5[0] == 0.3

    segs6, p6 = parse_pause_tags("[pause:5000ms]")
    assert segs6 == [], segs6
    assert len(p6) == 1 and p6[0] == 5.0, p6

    segs7, p7 = parse_pause_tags("One. [pause:500ms] [pause:1000ms] Two.")
    assert segs7 == ["One.", "Two."], segs7
    assert len(p7) == 1 and p7[0] == 1.5, p7

    tagged_existing, _ = add_pause_tags_to_text(
        "Already paused.[pause:400ms] Next sentence.", "period"
    )
    assert tagged_existing == (
        "Already paused [pause:400ms] Next sentence [pause:300ms] "
    ), tagged_existing

    assert convert_inline_markers_to_pause_tags("Before ~1 ~500 ~1000") == (
        "Before [pause:1ms] [pause:500ms] [pause:1000ms]"
    )

    segs8, _ = parse_pause_tags('He said [pause:150ms] "')
    assert len(segs8) == 1, segs8

    print("ok: pause splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
