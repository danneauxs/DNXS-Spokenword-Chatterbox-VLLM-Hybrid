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
from concurrent.futures.process import BrokenProcessPool

_ASR_DIR = Path(__file__).resolve().parent
if str(_ASR_DIR) not in sys.path:
    sys.path.insert(0, str(_ASR_DIR))
from spoken_compare import compare_spoken, strip_chatterbox_pause_tags

# Globals for persistent model (loaded once per worker)
_model = None
_device = None
_backend = "faster_whisper"


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


def _require_backend_installed(backend: str) -> None:
    """Exit-quality check so a missing optional engine cannot start a dead pool.

    Args:
        backend: Canonical or alias backend name.

    Raises:
        RuntimeError: Required package is not importable in this venv.
    """
    kind = str(backend or "faster_whisper").strip().lower().replace("-", "_")
    if kind in {"parakeet", "parakeet_tdt", "nemo", "nemo_parakeet"}:
        from parakeet_backend import parakeet_is_available

        if not parakeet_is_available():
            raise RuntimeError(
                "NeMo ASR is not installed. Install optional nemo-toolkit[asr] "
                "in this venv to use Parakeet Stage 1, or pick faster-whisper."
            )
        return
    if kind in {"whisper_cpp", "whispercpp", "cpp"}:
        try:
            import pywhispercpp  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "pywhispercpp is not installed. pip install pywhispercpp "
                "(GPU: ASR/install_pywhispercpp_cuda.sh) or pick faster-whisper."
            ) from exc


def _load_asr_engine(model_size="base", device="cpu", backend="faster_whisper"):
    """Load the selected Stage transcriber into this worker.

    Args:
        model_size: Whisper size or Parakeet id.
        device: cpu or cuda.
        backend: faster_whisper, whisper_cpp, or parakeet.

    Returns:
        (model, device_label)
    """
    kind = str(backend or "faster_whisper").strip().lower().replace("-", "_")
    if kind in {"whisper_cpp", "whispercpp", "cpp"}:
        from whisper_cpp_backend import load_whisper_cpp_model

        model, loaded = load_whisper_cpp_model(
            model_size, force_device=device, n_threads=2
        )
        if model is None and str(device).lower() in {"cuda", "gpu"}:
            logging.warning("whisper.cpp GPU load failed; retrying on CPU")
            model, loaded = load_whisper_cpp_model(
                model_size, force_device="cpu", n_threads=2
            )
        if model is None:
            raise RuntimeError("whisper.cpp failed to load")
        return model, loaded or device
    if kind in {"parakeet", "parakeet_tdt", "nemo", "nemo_parakeet"}:
        from parakeet_backend import ParakeetAsrModel

        model = ParakeetAsrModel(model_name=model_size, device=device)
        return model, model.device
    return _load_whisper_model(model_size, device)


def _initialize_worker(model_size="base", device="cpu", backend="faster_whisper"):
    """Load the chosen ASR engine once per worker process."""
    global _model, _device, _backend
    if _model is not None:
        return

    try:
        _backend = backend
        _model, _device = _load_asr_engine(model_size, device, backend)
        logging.info(
            "Worker %s loaded %s (%s) on %s",
            os.getpid(),
            backend,
            model_size,
            _device,
        )
    except Exception as e:
        logging.error(f"Worker {os.getpid()} failed to load ASR model: {e}")
        raise


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    return re.sub(r'[^\w\s]', '', text.lower()).strip()


def _validate_chunk(task_data: dict) -> dict:
    """Validate one chunk and return the worker's resolved backend/device.

    The returned device is set after model load, not copied from the GUI request,
    so a CPU fallback is visible to the parent process and persisted reports.
    """
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
            'asr_text': '',
            'backend': _backend,
            'device': _device,
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

        compared = compare_spoken(
            strip_chatterbox_pause_tags(expected_text),
            asr_text,
            threshold=threshold,
        )
        return {
            'chunk_id': chunk_id,
            'passed': compared['passed'],
            'score': compared['score'],
            'classification': compared['classification'],
            'failure_type': compared.get('failure_type', ''),
            'asr_text': asr_text,
            'expected_text': expected_text,
            'ref_normalized': compared.get('ref_normalized', ''),
            'hyp_normalized': compared.get('hyp_normalized', ''),
            'extra_content_words': compared.get('extra_content_words', 0),
            'missing_words': compared.get('missing_words', []),
            'extra_words': compared.get('extra_words', []),
            'coverage_score': compared.get('coverage_score'),
            'phonetic_score': compared.get('phonetic_score'),
            'accepted_equivalences': compared.get('accepted_equivalences', []),
            'explanation': compared.get('explanation', ''),
            'backend': _backend,
            'device': _device,
            'error': None
        }
    except Exception as e:
        return {
            'chunk_id': chunk_id,
            'passed': False,
            'score': 0.0,
            'error': str(e),
            'asr_text': '',
            'backend': _backend,
            'device': _device,
        }


def run_daemon(
    queue_dir: Path, results_dir: Path, num_workers: int = 4,
    model_size: str = "base", device: str = "cpu", backend: str = "faster_whisper",
):
    """Watch the task queue, run persistent ASR workers, and write result JSON.

    A requested CUDA device can resolve to CPU when pywhispercpp lacks its
    GGML CUDA bundle.  Match parent worker sizing so that fallback case keeps
    configured CPU parallelism instead of retaining CUDA's conservative cap.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    kind = str(backend or "faster_whisper").strip().lower().replace("-", "_")
    workers = max(1, int(num_workers or 1))
    if kind in {"parakeet", "parakeet_tdt", "nemo", "nemo_parakeet"}:
        workers = 1
    elif kind in {"whisper_cpp", "whispercpp", "cpp"} and device == "cuda":
        try:
            from whisper_cpp_backend import whisper_cpp_cuda_bundle_available
            import torch

            if whisper_cpp_cuda_bundle_available() and torch.cuda.is_available():
                workers = min(workers, 2)
        except (ImportError, OSError, RuntimeError):
            # Keep CPU parallelism when CUDA capability probing itself is unavailable.
            pass
    num_workers = workers
    logging.info(
        "ASR daemon starting with %s workers (backend: %s, model: %s, device: %s)",
        num_workers,
        backend,
        model_size,
        device,
    )
    _require_backend_installed(backend)

    queue_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Write PID file
    pid_file = queue_dir.parent / "asr_daemon.pid"
    pid_file.write_text(f"{os.getpid()}\n{time.time()}")

    executor = ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_initialize_worker,
        initargs=(model_size, device, backend),
    )
    pending_futures = {}  # future -> chunk_id
    pool_dead = False

    try:
        while True:
            # Update heartbeat
            pid_file.write_text(f"{os.getpid()}\n{time.time()}")

            if pool_dead:
                logging.error("ASR worker pool is dead; exiting so the GUI does not wait")
                break

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
                except BrokenProcessPool as e:
                    pool_dead = True
                    logging.error("ASR worker pool died: %s", e)
                    break
                except Exception as e:
                    err = str(e)
                    if "process pool is not usable" in err:
                        pool_dead = True
                        logging.error("ASR worker pool died: %s", e)
                        break
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
    parser.add_argument(
        "--backend",
        type=str,
        default="faster_whisper",
        help="faster_whisper, whisper_cpp, or parakeet",
    )
    args = parser.parse_args()

    run_daemon(
        args.queue_dir,
        args.results_dir,
        args.workers,
        args.model_size,
        args.device,
        args.backend,
    )
