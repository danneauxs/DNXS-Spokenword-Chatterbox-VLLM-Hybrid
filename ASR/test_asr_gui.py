"""Focused regression tests for standalone Chatterbox ASR tool helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ASR.asr_gui import (
    backend_preflight_error,
    build_validation_command,
    load_manual_review_rows,
    project_python,
    rescore_report,
    resolve_tts_dir,
    write_manual_repair_report,
    write_diagnostic_report,
)


class StandaloneAsrGuiTests(unittest.TestCase):
    """Verify report replay stays non-destructive and command wiring is production-safe."""

    def test_rescore_preserves_source_and_writes_new_report(self):
        """Replay a list report without overwriting source rows or source file bytes."""
        source_rows = [{
            "chunk_id": 7,
            "expected_text": "The wrist-watch stopped.",
            "asr_text": "The wristwatch stopped.",
            "backend": "whisper_cpp",
        }]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "asr_confirmed_failures.json"
            source.write_text(json.dumps(source_rows), encoding="utf-8")
            original_bytes = source.read_bytes()
            summary = rescore_report(source)
            output = root / "asr_confirmed_failures_rescored.json"
            rows = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(summary["passed"], 1)
            self.assertTrue(rows[0]["passed"])
            self.assertEqual(rows[0]["backend"], "whisper_cpp")
            self.assertEqual(rows[0]["rescore_status"], "rescored")

    def test_rescore_preserves_wrapped_report_shape_and_unscored_rows(self):
        """Keep object reports wrapped and retain operational rows as unscored."""
        payload = {
            "records": [
                {"chunk_id": 2, "expected_text": "Ready.", "asr_text": "Ready."},
                {"chunk_id": 3, "expected_text": "Broken.", "asr_text": "", "error": "decode failed"},
            ],
            "run_name": "fixture",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "report.json"
            output = Path(temp_dir) / "rescored.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            summary = rescore_report(source, output)
            replay = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(summary["scored"], 1)
            self.assertEqual(summary["unscored"], 1)
            self.assertEqual(replay["run_name"], "fixture")
            self.assertTrue(replay["records"][0]["passed"])
            self.assertEqual(replay["records"][1]["rescore_status"], "unscored")

    def test_diagnostic_snapshot_preserves_raw_lines_and_alignment_evidence(self):
        """Write one review-ready report containing source, transcript, and failure facts."""
        stage_row = {
            "chunk_id": "4",
            "expected_text": "The wrist-watch stopped.",
            "asr_text": "The wristwatch stopped.",
            "passed": True,
            "score": 1.0,
            "ref_normalized": "the wrist watch stopped",
            "hyp_normalized": "the wristwatch stopped",
            "alignment_operations": [{"op": "boundary_resegmentation"}],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir) / "TTS"
            tts_dir.mkdir()
            (tts_dir / "audio_chunks").mkdir()
            (tts_dir / "asr_stage1.json").write_text(json.dumps([stage_row]), encoding="utf-8")
            summary = write_diagnostic_report(tts_dir)
            report = json.loads(Path(summary["output_report"]).read_text(encoding="utf-8"))

            self.assertEqual(report["report_type"], "chatterbox_asr_diagnostic")
            self.assertEqual(summary["records"], 1)
            record = report["records"][0]
            self.assertEqual(record["original_text_raw"], stage_row["expected_text"])
            self.assertEqual(record["asr_text_raw"], stage_row["asr_text"])
            self.assertEqual(record["comparison"]["alignment_operations"], stage_row["alignment_operations"])

    def test_path_resolution_and_command_use_chatterbox_runner(self):
        """Resolve supported folders and invoke only the Chatterbox ASR-only CLI."""
        with tempfile.TemporaryDirectory() as temp_dir:
            book = Path(temp_dir) / "Book"
            tts = book / "TTS"
            (tts / "audio_chunks").mkdir(parents=True)
            self.assertEqual(resolve_tts_dir(book), tts)
            self.assertEqual(resolve_tts_dir(tts / "audio_chunks"), tts)
            command = build_validation_command(
                tts, "faster_whisper", "base", "whisper_cpp", "medium", "cpu", 0.65
            )
            self.assertIn("tools/run_asr_folder.py", command[1])
            self.assertEqual(command[0], project_python())
            self.assertEqual(command[-2:], ["--threshold", "0.65"])

    def test_parakeet_preflight_uses_project_venv_not_launcher_python(self):
        """Permit Parakeet when project venv has NeMo despite a system-Python GUI launch."""
        self.assertIsNone(backend_preflight_error("parakeet", project_python()))
        self.assertIsNone(backend_preflight_error("faster_whisper", project_python()))

    def test_manual_review_uses_offset_text_and_writes_repair_rows(self):
        """Pair zero-based WAVs with one-based text and emit Repair-compatible rows."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir) / "TTS"
            audio_dir = tts_dir / "audio_chunks"
            text_dir = tts_dir / "text_chunks"
            audio_dir.mkdir(parents=True)
            text_dir.mkdir()
            for index, text in enumerate(("First source line.", "Second source line.")):
                (audio_dir / f"chunk_{index:05d}.wav").write_bytes(b"RIFF")
                (text_dir / f"chunk_{index + 1:05d}.txt").write_text(text, encoding="utf-8")

            rows = load_manual_review_rows(tts_dir)
            summary = write_manual_repair_report(tts_dir, rows, {1})
            report = json.loads(Path(summary["output_report"]).read_text(encoding="utf-8"))

            self.assertEqual([row["expected_text"] for row in rows], ["First source line.", "Second source line."])
            self.assertEqual(summary["selected"], 1)
            self.assertEqual(report[0]["chunk_id"], 1)
            self.assertEqual(report[0]["expected_text"], "Second source line.")
            self.assertFalse(report[0]["passed"])


if __name__ == "__main__":
    unittest.main()
