"""Focused tests for persistent GUI batch queue storage and report formatting."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from modules.batch_queue import (
    append_batch_report,
    load_queue,
    parse_asr_summary,
    parse_timestamped_run_log,
    save_queue,
)


class BatchQueueTests(unittest.TestCase):
    """Verify queue persistence and append-only report fields without ML imports."""

    def test_queue_round_trip_is_ordered(self):
        """Persisted queue restores ordered JSON-safe snapshots unchanged."""
        jobs = [{"id": "first"}, {"id": "second"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            save_queue(root, jobs)
            self.assertEqual(load_queue(root), jobs)

    def test_report_uses_metrics_and_increments_run_number(self):
        """Reports preserve paths, ASR counts, and monotonic run labels."""
        job = {
            "id": "job-1",
            "book_dir": "/books/Emerge",
            "text_file": "/books/Emerge/input.txt",
            "voice_path": "/voices/reader.wav",
            "tts_params": {
                "t3_source": "multilingual-v3",
                "s3gen_decoder": "turbo",
                "exaggeration": 0.55,
                "temperature": 0.85,
                "cfg_weight": 0.30,
                "min_p": 0.05,
                "top_p": 0.95,
                "repetition_penalty": 1.2,
            },
            "quality_params": {
                "asr_stage1_backend": "parakeet",
                "asr_stage1_model": "multilingual",
                "asr_stage2_backend": "whisper_cpp",
                "asr_stage2_model": "medium",
            },
        }
        metrics = {
            "phase_1_time": "0:01:00",
            "phase_2_time": "0:02:00",
            "asr_stage1": "0:00:10",
            "asr_stage2": "0:00:20",
            "elapsed_time": "0:03:00",
            "realtime_raw": "2.00x",
            "total_elapsed": "0:04:00",
            "audio_duration": "0:06:00",
            "realtime_total": "1.50x",
            "stage1_fails": "3",
            "stage2_fails": "1",
            "still_fails": "0",
        }
        now = datetime(2026, 8, 27, 12, 0, 0)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            append_batch_report(root, job, now, now, True, metrics)
            append_batch_report(root, job, now, now, False, error="boom")
            report = (root / "batch_run_report.txt").read_text(encoding="utf-8")
        self.assertIn("Run #1", report)
        self.assertIn("Run #2", report)
        self.assertIn("Text Input Path: /books/Emerge/input.txt", report)
        self.assertIn("Voice Path: /voices/reader.wav", report)
        self.assertIn("T3: Multi V3", report)
        self.assertIn("Min-P: 0.05", report)
        self.assertIn("Top-P: 0.95", report)
        self.assertIn("Repetition Penalty: 1.2", report)
        self.assertIn("Stage 2 fails: 1", report)
        self.assertIn("Error: boom", report)

    def test_parsers_read_engine_log_and_summary(self):
        """Timestamped engine fields map to report metric keys without ambiguity."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "run_0827-1200.log"
            log_path.write_text(
                "Phase 1 Time: 0:01:00\nASR Stage 1: 0:00:10\nFails: 3\n"
                "ASR Stage 2: 0:00:20\nFails: 1\nRealtime RAW: 2.00x\n",
                encoding="utf-8",
            )
            summary_path = root / "asr_run_summary.json"
            summary_path.write_text(json.dumps({"regen_still_failed": 2}), encoding="utf-8")
            metrics = parse_timestamped_run_log(log_path)
            self.assertEqual(metrics["phase_1_time"], "0:01:00")
            self.assertEqual(metrics["stage1_fails"], "3")
            self.assertEqual(metrics["stage2_fails"], "1")
            self.assertEqual(parse_asr_summary(summary_path), {"still_fails": "2"})


if __name__ == "__main__":
    unittest.main()
