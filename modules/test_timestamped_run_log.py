"""Focused tests for book-local timestamped TTS run logs."""

from __future__ import annotations

import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


def _install_tts_engine_import_stubs():
    """Install minimal stub modules so tts_engine can import without ML deps."""
    module_stubs = {
        "torch": {},
        "torchaudio": {},
        "modules.vllm_batch_processor": {"VllmBatchProcessor": object},
        "modules.vllm_decoder": {"VllmDecoder": object},
        "modules.text_processor": {
            "smart_punctuate": lambda text: text,
            "sentence_chunk_text": lambda text, **kwargs: [text],
            "detect_content_boundaries": lambda text: [],
        },
        "modules.punctuation_pauses": {"add_pause_tags_to_text": lambda text: text},
        "modules.pause_utils": {
            "convert_inline_markers_to_pause_tags": lambda text: text,
        },
        "modules.audio_processor": {
            "pause_for_chunk_review": lambda *args, **kwargs: None,
            "get_chunk_audio_duration": lambda *args, **kwargs: 0.0,
            "has_mid_energy_drop": lambda *args, **kwargs: False,
        },
        "modules.audio_export": {
            "wav_duration_seconds": lambda *args, **kwargs: 20.0,
        },
        "modules.terminal_logger": {"start_terminal_logging": lambda *args, **kwargs: None},
        "modules.file_manager": {
            "setup_book_directories": lambda *args, **kwargs: None,
            "find_book_files": lambda *args, **kwargs: None,
            "ensure_voice_sample_compatibility": lambda *args, **kwargs: None,
            "combine_audio_chunks": lambda *args, **kwargs: None,
            "get_audio_files_in_directory": lambda *args, **kwargs: [],
            "convert_to_m4b": lambda *args, **kwargs: None,
            "add_metadata_to_m4b": lambda *args, **kwargs: None,
            "wipe_chunk_outputs": lambda *args, **kwargs: None,
        },
        "vaderSentiment.vaderSentiment": {
            "SentimentIntensityAnalyzer": object,
        },
        "wrapper.chunk_loader": {"save_chunks": lambda *args, **kwargs: None},
    }

    package_names = (
        "src",
        "src.chatterbox",
        "src.chatterbox.models",
        "vaderSentiment",
        "wrapper",
    )
    for package_name in package_names:
        if package_name not in sys.modules:
            package_stub = types.ModuleType(package_name)
            package_stub.__path__ = []
            sys.modules[package_name] = package_stub

    tokenizers_stub = types.ModuleType("src.chatterbox.models.tokenizers")

    class EnTokenizer:
        """Placeholder tokenizer used only so tts_engine import succeeds."""

    tokenizers_stub.EnTokenizer = EnTokenizer
    module_stubs["src.chatterbox.models.tokenizers"] = {
        "EnTokenizer": EnTokenizer
    }

    for module_name, attrs in module_stubs.items():
        if module_name in sys.modules:
            continue
        module_stub = types.ModuleType(module_name)
        for attr_name, attr_value in attrs.items():
            setattr(module_stub, attr_name, attr_value)
        if "." in module_name:
            module_stub.__package__ = module_name.rpartition(".")[0]
        sys.modules[module_name] = module_stub


_install_tts_engine_import_stubs()

from modules.tts_engine import _finalize_book_output, _write_timestamped_tts_run_log


class TimestampedRunLogTests(unittest.TestCase):
    """Verify complete settings and measurements are preserved per completed run."""

    def test_finalize_defines_total_realtime_before_writing_legacy_log(self):
        """Exercise finalization so total realtime exists on the log-writing path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_dir = root / "audio_chunks"
            audio_dir.mkdir()
            chunk_path = audio_dir / "chunk_00001.wav"
            chunk_path.touch()

            with (
                patch("modules.tts_engine.get_audio_files_in_directory", return_value=[chunk_path]),
                patch("modules.tts_engine.pause_for_chunk_review"),
                patch("modules.tts_engine.log_run"),
                patch("modules.tts_engine.time.time", return_value=100.0),
                patch("modules.tts_engine.WRITE_M4B", False, create=True),
                patch("modules.tts_engine.WRITE_MP3", False, create=True),
                patch("modules.tts_engine.WRITE_WAV", False, create=True),
            ):
                _, _, run_log_lines = _finalize_book_output(
                    audio_chunks_dir=audio_dir,
                    output_root=root,
                    voice_path=root / "voice.wav",
                    book_dir=root / "Book",
                    cover_file=None,
                    nfo_file=None,
                    run_log_lines=[],
                    start_time=95.0,
                    tts_params={
                        "exaggeration": 0.5,
                        "cfg_weight": 0.4,
                        "temperature": 0.85,
                    },
                    total_start_time=90.0,
                    generation_elapsed=5.0,
                )
                _, _, legacy_run_log_lines = _finalize_book_output(
                    audio_chunks_dir=audio_dir,
                    output_root=root,
                    voice_path=root / "voice.wav",
                    book_dir=root / "Book",
                    cover_file=None,
                    nfo_file=None,
                    run_log_lines=[],
                    start_time=95.0,
                    tts_params={
                        "exaggeration": 0.5,
                        "cfg_weight": 0.4,
                        "temperature": 0.85,
                    },
                    generation_elapsed=5.0,
                )

        self.assertIn("Realtime RAW: 4.00x", run_log_lines)
        self.assertIn("Realtime Total: 2.00x", run_log_lines)
        self.assertIn("Realtime Total: 4.00x", legacy_run_log_lines)

    def test_writes_requested_fields_from_runtime_values(self):
        """Write every requested setting and measured Stage 1/2 result to TTS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = _write_timestamped_tts_run_log(
                Path(temp_dir),
                {
                    "text_file": "/books/Sherlock/source.txt",
                    "voice_sample": "Watson.wav",
                    "vader_enabled": True,
                    "asr_enabled": True,
                    "stage1_backend": "parakeet",
                    "stage1_model": "parakeet-tdt-0.6b-v3",
                    "stage2_backend": "whisper_cpp",
                    "stage2_model": "medium",
                    "t3_encoder": "multilingual-v3",
                    "tts_params": {
                        "exaggeration": 0.5,
                        "temperature": 0.85,
                        "min_p": 0.05,
                        "top_p": 1.0,
                        "repetition_penalty": 1.2,
                        "cfg_weight": 0.4,
                    },
                },
                {
                    "elapsed_seconds": 120,
                    "phase1_elapsed": 35,
                    "phase2_elapsed": 85,
                    "total_elapsed_seconds": 150,
                    "audio_seconds": 600,
                    "chunk_count": 268,
                    "write_m4b": True,
                    "write_mp3": False,
                    "write_wav": True,
                    "chapterize": True,
                    "chapter_mode": "headings_only",
                    "asr_summary": {
                        "stage1_backend": "parakeet",
                        "stage1_model": "parakeet-tdt-0.6b-v3",
                        "stage1_elapsed_s": 12,
                        "stage1_failed": 30,
                        "stage2_backend": "whisper_cpp",
                        "stage2_model": "medium",
                        "stage2_ran": True,
                        "stage2_elapsed_s": 48,
                        "stage2_failed": 13,
                    },
                },
                timestamp="0821-1234",
            )
            contents = log_path.read_text(encoding="utf-8")

        self.assertEqual(log_path.name, "run_0821-1234.log")
        for expected in (
            "Text File: /books/Sherlock/source.txt",
            "Voice Sample: Watson.wav",
            "VADER: True",
            "ASR: True",
            "Stage 1: parakeet:parakeet-tdt-0.6b-v3",
            "Stage 2: whisper_cpp:medium",
            "Exaggeration: 0.5",
            "Temperature: 0.85",
            "Min-p: 0.05",
            "Top-P: 1.0",
            "Rep. Penalty: 1.2",
            "CFG: 0.4",
            "T3 Encoder: multilingual-v3",
            "Output Type: M4B, WAV",
            "Chapterise: True",
            "Type: headings_only",
            "Elapsed Time: 0:02:00",
            "Phase 1 Time: 0:00:35",
            "Phase 2 Time: 0:01:25",
            "ASR Stage 1: 0:00:12",
            "Fails: 30",
            "ASR Stage 2: 0:00:48",
            "Fails: 13",
            "Total Elapsed: 0:02:30",
            "Realtime RAW: 5.00x",
            "Realtime Total: 4.00x",
            "Audio Duration: 0:10:00",
            "Chunks: 268",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, contents)

    def test_same_minute_run_uses_a_new_file_instead_of_overwriting(self):
        """Keep separate logs when two completed runs share an MMDD-HHMM timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tts_dir = Path(temp_dir)
            metadata = {
                "asr_enabled": False,
                "tts_params": {},
                "run_timestamp": "0821-1234",
            }
            stats = {"elapsed_seconds": 1, "audio_seconds": 1}
            first = _write_timestamped_tts_run_log(tts_dir, metadata, stats)
            second = _write_timestamped_tts_run_log(tts_dir, metadata, stats)

        self.assertEqual(first.name, "run_0821-1234.log")
        self.assertEqual(second.name, "run_0821-1234_02.log")


if __name__ == "__main__":
    unittest.main()
