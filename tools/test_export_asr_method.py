#!/usr/bin/env python3
"""Smoke-check PCM stitch, spoken-compare, chapter ids, and export plan."""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ASR.spoken_compare import compare_spoken, normalize_spoken
from modules.asr_stages import (
    STAGE2_BACKENDS,
    backend_missing_reason,
    daemon_worker_count,
    normalize_backend,
    recommended_stage_two_model,
)
from modules.pause_utils import parse_pause_tags
from modules.punctuation_pauses import min_speech_tokens_for_text, split_t3_sentences
from modules.audio_export import concat_wavs, measure_pcm_peak, stitch_pcm_wavs
from modules.chapter_export import build_chapter_export_plan
from modules.chapter_headers import assign_chapter_ids, match_chapter_header


def _write_silence(path: Path, frames: int = 2400, rate: int = 24000) -> None:
    """Write a tiny 16-bit mono PCM WAV of zeros."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


def main() -> int:
    """Run method checks and return 0 on success."""
    assert match_chapter_header("Chapter One begins here") is not None
    assert match_chapter_header("this chapter is prose") is None
    chunks = [
        {"index": 0, "text": "Front matter sentence.", "boundary_type": "period"},
        {"index": 1, "text": "Chapter 1 The Start", "boundary_type": "chapter_start"},
        {"index": 2, "text": "More story.", "boundary_type": "paragraph_end"},
    ]
    stamped = assign_chapter_ids(chunks)
    assert stamped[0]["chapter_id"] == 0
    assert stamped[1]["chapter_id"] == 1
    assert stamped[2]["chapter_id"] == 1

    compared = compare_spoken("I have 2 apples.", "I have two apples.")
    assert compared["passed"] or compared["score"] > 0.7, compared
    apples_norm = normalize_spoken("I have 2 apples.")
    assert "2" not in apples_norm or "two" in apples_norm, apples_norm

    # Pocket comparator: spelling / contraction / phonetic equivalents pass.
    wasent = compare_spoken(
        "And I thought this one wasent.",
        "And I thought this one wasn't.",
        threshold=0.60,
    )
    assert wasent["passed"], wasent
    hullo = compare_spoken(
        "He frowned. Hullo! What's he up to now, d'you think?",
        "He frowned, hello, what's he up to now, do you think?",
        threshold=0.60,
    )
    assert hullo["passed"], hullo
    sickroom = compare_spoken(
        "Where is the sickroom, Watson?",
        "Where is the sick room, Watson?",
        threshold=0.60,
    )
    assert sickroom["passed"], sickroom
    objekt = compare_spoken(
        "The objekt being, of course, to discredit young Lord Whitechurch.",
        "The object being, of course, to discredit young Lord Whitechurch.",
        threshold=0.60,
    )
    assert objekt["passed"], objekt
    extra = compare_spoken(
        "Hello there.",
        "Hello there extra Holmes.",
        threshold=0.60,
    )
    assert not extra["passed"], extra

    carstairs_ok = compare_spoken(
        "They were together, according to Carstairs. According to Carstairs. Again, we must leave them on our list.",
        "They were together, according to car-stairs, according to car-stairs. Again, we must leave them on our list.",
        threshold=0.60,
    )
    assert carstairs_ok["passed"], carstairs_ok
    assert carstairs_ok.get("failure_type") != "unexpected_repetition", carstairs_ok

    carstairs_extra = compare_spoken(
        "They were together, according to Carstairs. Again, we must leave them on our list.",
        "They were together, according to Carstairs. According to Carstairs. Again, we must leave them on our list.",
        threshold=0.60,
    )
    assert not carstairs_extra["passed"], carstairs_extra

    stutter = compare_spoken("Hello.", "Hello hello hello.", threshold=0.60)
    assert not stutter["passed"], stutter

    chunk95 = (
        "It was expected that one would possess the judgement of Solomon. "
        "Rather neat, that, I thought. I see."
    )
    sents = split_t3_sentences(chunk95)
    assert len(sents) == 3, sents
    assert sents[0].endswith("Solomon.")
    assert sents[-1] == "I see."
    dr = split_t3_sentences("Dr. Smith went home. Then he slept.")
    assert len(dr) == 2, dr
    assert min_speech_tokens_for_text("I see.", 1000) >= 2
    segs, pauses = parse_pause_tags(chunk95)
    assert len(segs) == 3, segs
    assert pauses == [0.0, 0.0], pauses

    assert recommended_stage_two_model("base") == "medium"
    assert recommended_stage_two_model("large-v3-turbo") == "disabled"
    assert normalize_backend("whisper.cpp") == "whisper_cpp"
    assert normalize_backend("parakeet") == "parakeet"
    assert "parakeet" not in STAGE2_BACKENDS
    assert daemon_worker_count("parakeet", 4, "cuda") == 1
    assert daemon_worker_count("faster_whisper", 4, "cuda", "medium") == 1
    assert daemon_worker_count("faster_whisper", 4, "cuda", "base") == 4
    assert daemon_worker_count("faster_whisper", 4, "cpu", "medium") == 4
    assert daemon_worker_count("whisper_cpp", 8, "cuda") in {2, 8}
    for engine, module in (
        ("faster_whisper", "faster_whisper"),
        ("whisper_cpp", "pywhispercpp"),
    ):
        reason = backend_missing_reason(engine)
        try:
            __import__(module)
            assert reason is None, reason
        except ImportError:
            assert reason is not None

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        wavs = []
        for i in range(3):
            dest = audio_dir / f"chunk_{i:05d}.wav"
            _write_silence(dest)
            wavs.append(str(dest))
        out = tmp_path / "book.wav"
        stitch_pcm_wavs(wavs, out)
        assert out.exists() and out.stat().st_size > 44
        concat_wavs(wavs, tmp_path / "book2.wav")
        peak = measure_pcm_peak(wavs)
        assert peak == 0.0
        timeline, chapters, mode, headings = build_chapter_export_plan(
            stamped,
            audio_dir,
            chapterize=True,
            max_chapter_minutes=0,
            chapter_mode="headings_only",
        )
        assert headings is True
        assert len(chapters) == 2
        assert mode == "headings_only"
        print("ok: stitch, spoken-compare, chapter plan")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
