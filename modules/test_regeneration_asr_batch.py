"""Focused test that regeneration scores all retry candidates in one ASR batch."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ASR.batch_runner import ASRBatchConfig, ASRBatchRun
from modules.regeneration_engine import regenerate_failed_chunks


class _FakeProcessor:
    """Minimal retry-token processor replacement that has no model side effects."""

    def __init__(self, **_kwargs):
        """Accept the production constructor surface without loading vLLM."""

    def shutdown(self):
        """Mirror production cleanup after fake retry-token generation."""


class _FakeAudio:
    """Small exportable audio replacement for candidate-file lifecycle coverage."""

    def export(self, path: str, format: str):
        """Write a non-empty candidate placeholder accepted by archive/copy code."""
        _ = format
        Path(path).write_bytes(b"fake wav")


class _FakeDecoder:
    """Decode retry tokens into distinct exportable candidate placeholders."""

    def __init__(self, **_kwargs):
        """Accept the production decoder constructor surface without S3Gen."""

    def decode_and_assemble_chunk(self, *_args, **_kwargs):
        """Return one audio placeholder and the normal unused debug payload."""
        return _FakeAudio(), {}

    def shutdown(self):
        """Mirror production decoder release after retry candidates are emitted."""


class RegenerationBatchTests(unittest.TestCase):
    """Verify retries reach one selected-engine score call, never one call per WAV."""

    def test_retry_candidates_are_scored_in_one_selected_backend_batch(self):
        """Two generated retries become one whisper.cpp batch with two tasks."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "audio_chunks"
            output_dir.mkdir()
            (output_dir / "chunk_00000.wav").write_bytes(b"original")
            captured = []

            def fake_batch(tasks, config, *_args, **_kwargs):
                """Record the exact retry task set and make attempt two the winner."""
                task_list = list(tasks)
                captured.append((task_list, config))
                rows = {
                    task["chunk_id"]: {
                        "chunk_id": task["chunk_id"],
                        "score": 0.9 if task["chunk_id"].endswith("r2") else 0.99,
                        "passed": True,
                        "asr_text": "hello",
                        "expected_text": "hello",
                        "backend": config.backend,
                        "device": "cuda",
                        "error": None if task["chunk_id"].endswith("r2") else "synthetic ASR error",
                    }
                    for task in task_list
                }
                return ASRBatchRun(rows, config.backend, config.model_size, "cuda", ["cuda"], 2, 0.1, 0.2)

            with (
                patch("modules.vllm_batch_processor.VllmBatchProcessor", _FakeProcessor),
                patch("modules.vllm_decoder.VllmDecoder", _FakeDecoder),
                patch(
                    "modules.regeneration_engine._generate_regeneration_tokens_batched",
                    return_value={"0": [(1, {}, [[]]), (2, {}, [[]])]},
                ),
                patch("ASR.batch_runner.run_asr_batch_isolated", side_effect=fake_batch),
            ):
                outcome = regenerate_failed_chunks(
                    failed=[
                        {
                            "chunk_id": "0",
                            "text": "hello",
                            "tts_params": {},
                            "segments": ["hello"],
                            "pauses": [],
                            "boundary_type": "none",
                            "original_score": 0.1,
                        }
                    ],
                    cond_emb=None,
                    ckpt_dir=Path(tmp),
                    device="cuda",
                    variant="standard",
                    decoder_type="standard",
                    voice_path=None,
                    turbo_ckpt_dir=None,
                    audio_output_dir=output_dir,
                    asr_client=None,
                    asr_threshold=0.65,
                    asr_batch_config=ASRBatchConfig("whisper_cpp", "medium", "cuda", workers=4),
                    local_scoring_enabled=False,
                    max_attempts=2,
                    report_dir=Path(tmp),
                )

            self.assertEqual(len(captured), 1)
            self.assertEqual([task["chunk_id"] for task in captured[0][0]], ["0_r1", "0_r2"])
            self.assertEqual(captured[0][1].backend, "whisper_cpp")
            self.assertEqual(outcome["regenerated"], 1)
            self.assertEqual(outcome["report"][0]["score"], 0.0)
            self.assertEqual(outcome["report"][1]["score"], 0.9)
            self.assertTrue((output_dir / "Failed" / "chunk_00000_attempt1.wav").exists())
            self.assertTrue((output_dir / "Failed" / "chunk_00000_attempt2.wav").exists())


if __name__ == "__main__":
    unittest.main()
