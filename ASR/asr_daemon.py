#!/usr/bin/env python3
"""
ASR Worker Daemon - Runs in ASR venv with persistent Whisper models.
Communicates with main process via file-based task/result queue.

This daemon loads Whisper model once per worker and reuses it for all validations,
avoiding the 2-4 second model loading overhead per chunk that subprocess approach had.
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

# Globals for persistent model (loaded once per worker)
_model = None
_device = None

def _load_whisper_model(model_size="base", device="cpu"):
    """
    Load Whisper model, on GPU if requested (falling back to CPU on any failure).

    CPU is the safe default -- it never contends with TTS models for VRAM. GPU is
    opt-in (Tab 2's "Run ASR on GPU" checkbox): faster per-transcription, and safe
    on cards with headroom, but this worker process shares the same GPU as the main
    process's S3Gen decoder while Phase 2 is still running, so a card under heavier
    load than 8GB-with-headroom could still see contention.

    Args:
        model_size: Whisper model name (tiny/base/small/medium/large*), selected
            by the GUI's SAFE/MODERATE/INSANE tier (see recommend_asr_models).
        device: "cpu" or "cuda".
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        logging.error(f"Failed to import faster-whisper: {e}")
        raise

    if device == "cuda":
        try:
            logging.info(f"🖥️  Loading Whisper {model_size} on GPU")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            logging.info(f"✅ Whisper {model_size} loaded on GPU")
            return model, "cuda"
        except Exception as e:
            logging.warning(f"GPU load failed ({e}), falling back to CPU")

    logging.info(f"🖥️  Loading Whisper {model_size} on CPU")
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logging.info(f"✅ Whisper {model_size} loaded on CPU")
        return model, "cpu"
    except Exception as e:
        logging.error(f"Failed to load Whisper model on CPU: {e}")
        raise


def _initialize_worker(model_size="base", device="cpu"):
    """Load Whisper model once per worker process."""
    global _model, _device
    if _model is not None:
        return

    try:
        _model, _device = _load_whisper_model(model_size, device)
        logging.info(f"Worker {os.getpid()} loaded ASR model ({model_size}) on {_device}")
    except Exception as e:
        logging.error(f"Worker {os.getpid()} failed to load ASR model: {e}")
        raise


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _validate_chunk(task_data: dict) -> dict:
    """Validate a single chunk (runs in worker with persistent model)."""
    global _model

    if _model is None:
        _initialize_worker()

    chunk_id = task_data['chunk_id']
    wav_path = Path(task_data['wav_path'])
    expected_text = task_data['expected_text']
    threshold = task_data['threshold']

    # Wait for file (handle async save)
    max_wait = 30
    start = time.time()
    while not wav_path.exists() and (time.time() - start) < max_wait:
        time.sleep(0.1)

    if not wav_path.exists():
        return {
            'chunk_id': chunk_id,
            'passed': False,
            'score': 0.0,
            'error': f'File not found: {wav_path}',
            'asr_text': ''
        }

    try:
        # Transcribe with persistent model
        segments, _ = _model.transcribe(
            str(wav_path),
            language="en",  # Force English - audio is always English TTS
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}
        )
        asr_text = " ".join([seg.text for seg in segments]).strip()

        # Compute similarity
        expected_norm = _normalize_text(expected_text)
        asr_norm = _normalize_text(asr_text)
        score = SequenceMatcher(None, expected_norm, asr_norm).ratio()

        return {
            'chunk_id': chunk_id,
            'passed': score >= threshold,
            'score': score,
            'asr_text': asr_text,
            'expected_text': expected_text,
            'error': None
        }
    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'asr_text': ''
        }


def run_daemon(
    queue_dir: Path, results_dir: Path, num_workers: int = 4,
    model_size: str = "base", device: str = "cpu",
):
    """Main daemon loop: watch queue, process tasks, write results."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"ASR daemon starting with {num_workers} workers (model: {model_size}, device: {device})")

    queue_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Write PID file
    pid_file = queue_dir.parent / "asr_daemon.pid"
    pid_file.write_text(f"{os.getpid()}\n{time.time()}")

    executor = ProcessPoolExecutor(
        max_workers=num_workers, initializer=_initialize_worker, initargs=(model_size, device)
    )
    pending_futures = {}  # future -> chunk_id

    try:
        while True:
            # Update heartbeat
            pid_file.write_text(f"{os.getpid()}\n{time.time()}")

            # Submit new tasks
            for task_file in sorted(queue_dir.glob("*.task.json")):
                try:
                    task_data = json.loads(task_file.read_text())
                    chunk_id = task_data['chunk_id']

                    # Submit to worker pool
                    future = executor.submit(_validate_chunk, task_data)
                    pending_futures[future] = chunk_id

                    # Remove task file
                    task_file.unlink()
                    logging.info(f"Queued {chunk_id}")
                except Exception as e:
                    logging.error(f"Failed to process task {task_file}: {e}")

            # Collect completed results
            done_futures = [f for f in pending_futures if f.done()]
            for future in done_futures:
                chunk_id = pending_futures.pop(future)
                try:
                    result = future.result()
                    result_file = results_dir / f"{chunk_id}.result.json"
                    result_file.write_text(json.dumps(result, indent=2))
                    logging.info(f"Completed {chunk_id} (score: {result['score']:.3f})")
                except Exception as e:
                    logging.error(f"Error processing {chunk_id}: {e}")

            # Check for shutdown signal
            shutdown_file = queue_dir.parent / "asr_daemon.shutdown"
            if shutdown_file.exists():
                logging.info("Shutdown signal received")
                shutdown_file.unlink()
                break

            time.sleep(0.1)  # Polling interval

    finally:
        executor.shutdown(wait=True)
        pid_file.unlink(missing_ok=True)
        logging.info("ASR daemon stopped")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model-size", type=str, default="base")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_daemon(args.queue_dir, args.results_dir, args.workers, args.model_size, args.device)
