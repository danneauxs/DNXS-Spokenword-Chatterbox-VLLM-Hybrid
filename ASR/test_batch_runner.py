"""Focused tests for backend-aware short-lived ASR batch dispatch."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ASR.batch_runner import (
    ASRBatchConfig,
    _result_from_transcript,
    run_asr_batch,
    run_asr_batch_isolated,
)


class _Segment:
    """Minimal timed segment accepted by the batch runner's transcript adapter."""

    def __init__(self, text: str, start: float = 0.0, end: float = 1.0):
        """Store transcript text and a timestamp interval for packed mapping tests."""
        self.text = text
        self.start = start
        self.end = end


class _WhisperModel:
    """Small deterministic Whisper-compatible fake for batch runner tests."""

    def transcribe(self, _audio, **_kwargs):
        """Return one correct transcript without reading a real model."""
        return [_Segment("hello")], {"language": "en"}


class _ParakeetModel:
    """Parakeet-compatible fake exposing the production multi-file method."""

    def transcribe_many(self, audio_paths, batch_size):
        """Return one correct transcript for every requested path and batch size."""
        assert batch_size > 0
        return ["hello" for _ in audio_paths]


def _task(chunk_id: str, wav_path: Path) -> dict:
    """Build one normal runner task with intentionally simple comparison text."""
    return {
        "chunk_id": chunk_id,
        "wav_path": str(wav_path),
        "expected_text": "hello",
        "threshold": 0.65,
    }


class BatchRunnerTests(unittest.TestCase):
    """Exercise supported engine/device dispatch without heavyweight ASR models."""

    def test_transcript_rows_preserve_boundary_policy_evidence(self):
        """Forward comparator version and boundary operations into report-ready rows."""
        row = _result_from_transcript(
            {
                "chunk_id": "5924",
                "expected_text": "Never mind that.",
                "threshold": 0.99,
            },
            "Nevermind that.",
            "parakeet",
            "cuda",
        )

        self.assertTrue(row["passed"])
        self.assertEqual(row["comparison_policy_version"], "boundary-resegmentation-v2")
        self.assertTrue(any(
            item["op"] == "boundary_resegmentation"
            and item.get("boundary_method") == "exact_surface"
            for item in row["alignment_operations"]
        ))
        self.assertTrue(any(
            item.get("kind") == "boundary_resegmentation"
            for item in row["accepted_phrase_equivalences"]
        ))

    def test_cpu_engine_paths_preserve_selected_backend_and_score(self):
        """CPU Faster-Whisper, CPP, and Parakeet return scored rows for all tasks."""
        with tempfile.TemporaryDirectory() as tmp:
            tasks = [_task("0", Path(tmp) / "a.wav"), _task("1", Path(tmp) / "b.wav")]

            def fake_load(config, n_threads=2):
                """Return the matching fake engine and a truthful CPU device label."""
                _ = n_threads
                return (_ParakeetModel() if config.backend == "parakeet" else _WhisperModel()), "cpu"

            with patch("ASR.batch_runner._load_one_model", side_effect=fake_load):
                for backend in ("faster_whisper", "whisper_cpp", "parakeet"):
                    run = run_asr_batch(
                        tasks,
                        ASRBatchConfig(backend, "base", "cpu", workers=2),
                    )
                    self.assertEqual(set(run.results), {"0", "1"})
                    self.assertTrue(all(row["passed"] for row in run.results.values()))
                    self.assertTrue(all(row["backend"] == backend for row in run.results.values()))
                    self.assertEqual(run.actual_devices, ["cpu"])

    def test_cuda_faster_whisper_fallback_stays_at_one_cpu_model(self):
        """A CUDA load fallback cannot recreate four Medium CPU model contexts."""
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            tasks = [_task("0", Path(tmp) / "a.wav")]

            def fake_load(config, n_threads=2):
                """Simulate CUDA failure resolving to one CPU model on both attempts."""
                calls.append((config.requested_device, n_threads))
                return _WhisperModel(), "cpu"

            with patch("ASR.batch_runner._load_one_model", side_effect=fake_load):
                run = run_asr_batch(
                    tasks,
                    ASRBatchConfig("faster_whisper", "medium", "cuda", workers=4),
                )
            self.assertEqual(run.requested_device, "cuda")
            self.assertEqual(run.actual_devices, ["cpu"])
            self.assertEqual(run.effective_workers, 1)
            self.assertEqual(len(calls), 2)  # Initial CUDA attempt plus one CPU fallback.

    def test_cuda_faster_whisper_uses_packed_one_model_path(self):
        """GPU Faster-Whisper uses one model while two clips are packed and scored."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task("0", root / "a.wav"), _task("1", root / "b.wav")]
            for task in tasks:
                samples = (np.zeros(16_000, dtype=np.int16)).tobytes()
                with wave.open(task["wav_path"], "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(16_000)
                    wav.writeframes(samples)

            with patch("ASR.batch_runner._load_one_model", return_value=(_WhisperModel(), "cuda")) as load:
                run = run_asr_batch(
                    tasks,
                    ASRBatchConfig("faster_whisper", "medium", "cuda", workers=4, pack_size=2),
                )
            self.assertEqual(load.call_count, 1)
            self.assertEqual(run.actual_devices, ["cuda"])
            self.assertEqual(run.effective_workers, 1)
            self.assertTrue(all(row["passed"] for row in run.results.values()))

    def test_cuda_whisper_cpp_forces_one_medium_context(self):
        """CUDA CPP ignores CPU worker count to prevent native Medium-model OOM."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = [_task("0", root / "a.wav"), _task("1", root / "b.wav")]
            with patch("ASR.batch_runner._load_one_model", return_value=(_WhisperModel(), "cuda")) as load:
                run = run_asr_batch(
                    tasks,
                    ASRBatchConfig("whisper_cpp", "medium", "cuda", workers=4),
                )
            self.assertEqual(load.call_count, 1)
            self.assertEqual(run.actual_devices, ["cuda"])
            self.assertEqual(run.effective_workers, 1)
            self.assertTrue(all(row["passed"] for row in run.results.values()))

    def test_isolated_runner_returns_errors_and_removes_its_job_directory(self):
        """A child-process ASR error returns unscored rows without job-file residue."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = run_asr_batch_isolated(
                [_task("0", root / "a.wav")],
                ASRBatchConfig("unsupported_backend", "base", "cpu"),
                root,
                stage_label="test",
            )
            self.assertIn("0", run.results)
            self.assertTrue(run.results["0"]["error"])
            self.assertEqual(list(root.glob(".asr_batch_*")), [])
            self.assertTrue((root / "asr_batch.log").is_file())


if __name__ == "__main__":
    unittest.main()
