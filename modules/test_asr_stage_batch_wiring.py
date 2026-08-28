"""Regression test for every supported Stage-1 × Stage-2 batch selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ASR.batch_runner import ASRBatchRun
from modules import tts_engine


def _row(chunk_id: str, score: float, backend: str) -> dict:
    """Build a fully scored runner result suitable for Phase-3 report assembly."""
    passed = score >= 0.65
    return {
        "chunk_id": chunk_id,
        "score": score,
        "passed": passed,
        "classification": "PASS" if passed else "FAIL",
        "asr_text": "hello",
        "expected_text": "hello",
        "backend": backend,
        "device": "cuda",
        "error": None,
    }


class StageBatchWiringTests(unittest.TestCase):
    """Confirm all allowed engine pairs retain their selected runner config."""

    def test_all_stage_backend_pairs_dispatch_without_daemon(self):
        """Run the nine supported Stage-1/Stage-2 combinations with fake ASR batches."""
        stage1_backends = ("faster_whisper", "whisper_cpp", "parakeet")
        stage2_backends = ("disabled", "faster_whisper", "whisper_cpp")
        for stage1_backend in stage1_backends:
            for stage2_backend in stage2_backends:
                for requested_device in ("cpu", "cuda"):
                    with (
                        self.subTest(
                            stage1=stage1_backend,
                            stage2=stage2_backend,
                            device=requested_device,
                        ),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        calls = []

                        def fake_run(tasks, config, *_args, **_kwargs):
                            """Return a Stage-1 candidate and a Stage-2 pass when requested."""
                            task_list = list(tasks)
                            calls.append(config)
                            score = 0.20 if len(calls) == 1 else 0.90
                            rows = {
                                str(task["chunk_id"]): _row(
                                    str(task["chunk_id"]), score, config.backend
                                )
                                for task in task_list
                            }
                            return ASRBatchRun(
                                rows,
                                config.backend,
                                config.model_size,
                                config.requested_device,
                                ["cuda"],
                                1,
                                0.01,
                                0.02,
                            )

                        root = Path(tmp)
                        with (
                            patch.object(tts_engine, "ASR_STAGE1_BACKEND", stage1_backend),
                            patch.object(tts_engine, "ASR_STAGE1_MODEL", "base"),
                            patch.object(
                                tts_engine,
                                "ASR_STAGE2_BACKEND",
                                stage2_backend if stage2_backend != "disabled" else "faster_whisper",
                            ),
                            patch.object(
                                tts_engine,
                                "ASR_STAGE2_MODEL",
                                "disabled" if stage2_backend == "disabled" else "medium",
                            ),
                            patch.object(tts_engine, "ASR_WORKERS", 4),
                            patch.object(tts_engine, "ENABLE_REGENERATION_LOOP", False),
                            patch.object(tts_engine, "MAX_REGENERATION_ATTEMPTS", 0),
                            patch("ASR.batch_runner.run_asr_batch_isolated", side_effect=fake_run),
                        ):
                            tts_engine._run_phase3_regen(
                                chunk_meta={
                                    "0": {
                                        "text": "hello",
                                        "tts_params": {},
                                        "segments": ["hello"],
                                        "pauses": [],
                                        "boundary_type": "none",
                                    }
                                },
                                tokens_dict={"0": []},
                                local_scores={},
                                tts_dir=root,
                                audio_chunks_dir=root / "audio_chunks",
                                cond_emb=None,
                                ckpt_dir=root,
                                device="cuda",
                                variant="standard",
                                decoder_type="standard",
                                voice_path=None,
                                turbo_ckpt_dir=None,
                                asr_enabled=True,
                                asr_device=requested_device,
                                asr_threshold=0.65,
                                true_start_time=0.0,
                            )

                        self.assertEqual(calls[0].backend, stage1_backend)
                        self.assertEqual(calls[0].requested_device, requested_device)
                        if stage1_backend == "parakeet":
                            self.assertEqual(calls[0].model_size, "parakeet-tdt-0.6b-v3")
                        if stage2_backend == "disabled":
                            self.assertEqual(len(calls), 1)
                        else:
                            self.assertEqual(len(calls), 2)
                            self.assertEqual(calls[1].backend, stage2_backend)
                            self.assertEqual(calls[1].model_size, "medium")


if __name__ == "__main__":
    unittest.main()
