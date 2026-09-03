"""Regression tests for existing-folder ASR safety behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.run_asr_folder import run_folder


class ExistingFolderAsrSafetyTests(unittest.TestCase):
    """Ensure an unavailable ASR backend cannot erase confirmed listener work."""

    def test_all_unscored_stage_one_preserves_confirmed_failure_report(self):
        """Stop before Stage 2 and retain confirmed failures when Stage 1 cannot load."""
        tasks = [{
            "chunk_id": "7",
            "wav_path": "/fixture/chunk_00007.wav",
            "expected_text": "Expected speech.",
            "threshold": 0.65,
        }]
        rows = [{
            "chunk_id": "7",
            "expected_text": "Expected speech.",
            "asr_text": "",
            "score": 0.0,
            "passed": False,
            "error": "NeMo ASR is not installed.",
        }]
        metadata = {"backend": "parakeet", "model": "fixture", "chunks": 1}
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir) / "TTS"
            tts_dir.mkdir()
            confirmed = tts_dir / "asr_confirmed_failures.json"
            original = [{"chunk_id": 99, "expected_text": "Keep this."}]
            confirmed.write_text(json.dumps(original), encoding="utf-8")

            with (
                patch("tools.run_asr_folder._load_tasks", return_value=tasks),
                patch("tools.run_asr_folder.load_known_asr_passes", return_value={}),
                patch("tools.run_asr_folder.load_accepted_asr_fuzzies", return_value={}),
                patch("tools.run_asr_folder._run_stage", return_value=(rows, metadata)) as run_stage,
            ):
                summary = run_folder(
                    tts_dir,
                    "parakeet",
                    "base",
                    "whisper_cpp",
                    "medium",
                    "cuda",
                    0.65,
                )

            self.assertFalse(summary["success"])
            self.assertEqual(summary["stage1_unscored"], 1)
            self.assertIn("NeMo ASR is not installed", summary["stage1_error"])
            self.assertEqual(run_stage.call_count, 1)
            self.assertEqual(json.loads(confirmed.read_text(encoding="utf-8")), original)
            self.assertFalse((tts_dir / "asr_stage2.json").exists())


if __name__ == "__main__":
    unittest.main()
