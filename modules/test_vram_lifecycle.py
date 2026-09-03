"""Regression coverage for T3 and S3Gen release boundaries."""

from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.vllm_batch_processor import VllmBatchProcessor
from modules.vllm_decoder import VllmDecoder
from modules.regeneration_engine import regenerate_failed_chunks
from src.chatterbox_vllm.tts import ChatterboxTTS


class _FakeModel:
    """Record shutdown calls without loading a GPU model."""

    def __init__(self):
        """Initialize a zero-side-effect model replacement."""
        self.shutdown_calls = 0

    def shutdown(self):
        """Record one release request from the owner under test."""
        self.shutdown_calls += 1


class VramLifecycleTests(unittest.TestCase):
    """Verify cleanup remains safe on normal and failing phase transitions."""

    def test_batch_processor_shutdown_is_idempotent(self):
        """Release a Phase-1 model once even when two cleanup paths meet."""
        processor = object.__new__(VllmBatchProcessor)
        model = _FakeModel()
        processor.model = model

        with patch("modules.vllm_batch_processor.torch.cuda.is_available", return_value=False):
            processor.shutdown()
            processor.shutdown()

        self.assertEqual(model.shutdown_calls, 1)
        self.assertIsNone(processor.model)

    def test_decoder_shutdown_detaches_model_after_model_error(self):
        """Detach S3Gen ownership even when backend shutdown raises an error."""
        decoder = object.__new__(VllmDecoder)

        class _FailingModel:
            """Raise from shutdown to exercise the decoder's finally cleanup."""

            def shutdown(self):
                """Raise the deliberate backend-release failure."""
                raise RuntimeError("synthetic decoder failure")

        decoder._model = _FailingModel()
        decoder._s3gen_ref = object()

        with patch("modules.vllm_decoder.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "synthetic decoder failure"):
                decoder.shutdown()

        self.assertIsNone(decoder._model)
        self.assertIsNone(decoder._s3gen_ref)

    def test_t3_shutdown_detaches_phase_one_tensors_after_engine_error(self):
        """Break engine ownership and tensor references before re-raising an error."""
        model = object.__new__(ChatterboxTTS)

        class _FailingEngineCore:
            """Represent a vLLM core that errors while releasing itself."""

            def __init__(self):
                """Expose the executor link held by EngineCore."""
                self.model_executor = object()

            def shutdown(self):
                """Raise the deliberate EngineCore release failure."""
                raise RuntimeError("synthetic engine failure")

        engine_core = _FailingEngineCore()
        llm_engine = types.SimpleNamespace(
            engine_core=engine_core,
            model_executor=object(),
        )
        model.t3 = types.SimpleNamespace(llm_engine=llm_engine)
        model.t3_config = object()
        model.t3_cond_enc = object()
        model.t3_speech_emb = object()
        model.t3_speech_pos_emb = object()
        model.s3gen = object()
        model.ve = object()
        model.default_conds = object()

        with patch("src.chatterbox_vllm.tts.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "synthetic engine failure"):
                model.shutdown()

        self.assertIsNone(engine_core.model_executor)
        self.assertIsNone(llm_engine.engine_core)
        self.assertIsNone(llm_engine.model_executor)
        for attribute in (
            "t3",
            "t3_config",
            "t3_cond_enc",
            "t3_speech_emb",
            "t3_speech_pos_emb",
            "s3gen",
            "ve",
            "default_conds",
        ):
            self.assertFalse(hasattr(model, attribute), attribute)

    def test_regeneration_releases_t3_and_s3gen_after_decode_error(self):
        """Release both retry-phase models before a decode error reaches the GUI."""
        created_processors = []
        created_decoders = []

        class _Processor:
            """Record retry T3 cleanup without constructing a vLLM engine."""

            def __init__(self, **_kwargs):
                """Register the fake retry T3 processor instance."""
                self.shutdown_calls = 0
                created_processors.append(self)

            def shutdown(self):
                """Record the required retry T3 release boundary."""
                self.shutdown_calls += 1

        class _Decoder:
            """Fail decoding after recording the required S3Gen release call."""

            def __init__(self, **_kwargs):
                """Register the fake retry decoder instance."""
                self.shutdown_calls = 0
                created_decoders.append(self)

            def decode_and_assemble_chunk(self, *_args, **_kwargs):
                """Raise a synthetic decode error before candidate WAV creation."""
                raise RuntimeError("synthetic retry decode failure")

            def shutdown(self):
                """Record the required retry S3Gen release boundary."""
                self.shutdown_calls += 1

        with (
            patch("modules.vllm_batch_processor.VllmBatchProcessor", _Processor),
            patch("modules.vllm_decoder.VllmDecoder", _Decoder),
            patch(
                "modules.regeneration_engine._generate_regeneration_tokens_batched",
                return_value={"0": [(1, {}, [[]])]},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic retry decode failure"):
                regenerate_failed_chunks(
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
                    ckpt_dir="unused",
                    device="cuda",
                    variant="english",
                    decoder_type="turbo",
                    voice_path=None,
                    turbo_ckpt_dir=None,
                    audio_output_dir=Path("unused"),
                    asr_client=None,
                    asr_threshold=0.65,
                    max_attempts=1,
                )

        self.assertEqual(created_processors[0].shutdown_calls, 1)
        self.assertEqual(created_decoders[0].shutdown_calls, 1)


if __name__ == "__main__":
    unittest.main()
