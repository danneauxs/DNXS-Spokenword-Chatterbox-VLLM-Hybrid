"""Backend-aware, short-lived batch ASR runner for completed TTS WAV files.

The runner deliberately owns every ASR model only for one completed Stage 1,
Stage 2, or regeneration batch.  It has no daemon, queue directory, or child
process pool, so completing a batch cannot leave a model resident for a later
book.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

try:
    from .spoken_compare import compare_spoken, strip_chatterbox_pause_tags
except ImportError:  # Supports direct execution from the ASR directory.
    from spoken_compare import compare_spoken, strip_chatterbox_pause_tags


logger = logging.getLogger(__name__)
_TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ASRBatchConfig:
    """Describe one selected ASR engine invocation without changing UI choices.

    Args:
        backend: ``faster_whisper``, ``whisper_cpp``, or Stage-1 ``parakeet``.
        model_size: Selected Whisper model name or Parakeet model id.
        requested_device: ``cpu`` or ``cuda`` from the GUI's shared ASR toggle.
        workers: Requested CPU workers or model contexts where that is safe.
            CUDA whisper.cpp always uses one context because native CUDA OOM
            aborts its process instead of raising a recoverable Python error.
        pack_size: Number of clips per Faster-Whisper GPU decode pack.
        pack_silence_s: Silence placed between packed clips.
        allow_cpu_fallback: Whether one explicit CPU fallback is permitted when
            a Faster-Whisper CUDA model cannot load.
    """

    backend: str
    model_size: str
    requested_device: str
    workers: int = 4
    pack_size: int = 8
    pack_silence_s: float = 0.75
    allow_cpu_fallback: bool = True


@dataclass
class ASRBatchRun:
    """Return completed ASR rows plus runtime facts for one short-lived batch."""

    results: Dict[str, Dict[str, Any]]
    backend: str
    model_size: str
    requested_device: str
    actual_devices: List[str]
    effective_workers: int
    load_seconds: float
    elapsed_seconds: float


def _canonical_backend(value: str | None) -> str:
    """Normalize UI/backend aliases without importing GUI modules."""
    token = str(value or "faster_whisper").strip().lower().replace("-", "_")
    aliases = {
        "whispercpp": "whisper_cpp",
        "cpp": "whisper_cpp",
        "pywhispercpp": "whisper_cpp",
        "parakeet_tdt": "parakeet",
        "nemo": "parakeet",
        "nemo_parakeet": "parakeet",
    }
    return aliases.get(token, token)


def _canonical_device(value: str | None) -> str:
    """Normalize a requested device to the runner's CPU/CUDA vocabulary."""
    return "cuda" if str(value or "cpu").strip().lower() in {"cuda", "gpu"} else "cpu"


def _task_id(task: Dict[str, Any]) -> str:
    """Return one stable string id from a batch task, raising on malformed input."""
    if "chunk_id" not in task:
        raise ValueError("ASR batch task is missing chunk_id")
    return str(task["chunk_id"])


def _error_result(task: Dict[str, Any], backend: str, device: str, error: Exception | str) -> Dict[str, Any]:
    """Build an unscored operational-error row that cannot authorize regeneration."""
    return {
        "chunk_id": _task_id(task),
        "passed": False,
        "score": 0.0,
        "asr_text": "",
        "expected_text": str(task.get("expected_text") or ""),
        "backend": backend,
        "device": device,
        "error": str(error),
    }


def _result_from_transcript(
    task: Dict[str, Any], transcript: str, backend: str, device: str
) -> Dict[str, Any]:
    """Compare one transcript and preserve all evidence required by ASR reports."""
    expected_text = str(task.get("expected_text") or "")
    threshold = float(task.get("threshold") or 0.0)
    compared = compare_spoken(
        strip_chatterbox_pause_tags(expected_text), transcript, threshold=threshold
    )
    return {
        "chunk_id": _task_id(task),
        **compared,
        "asr_text": transcript,
        "expected_text": expected_text,
        "backend": backend,
        "device": device,
        "error": None,
    }


def _load_one_model(config: ASRBatchConfig, n_threads: int = 2) -> tuple[Any, str]:
    """Load exactly one selected engine and return its resolved runtime device.

    Faster-Whisper's CUDA fallback intentionally creates only one CPU model.
    The old daemon created four Medium CPU fallbacks after four GPU OOMs.
    """
    backend = _canonical_backend(config.backend)
    device = _canonical_device(config.requested_device)
    if backend == "whisper_cpp":
        try:
            from .whisper_cpp_backend import load_whisper_cpp_model
        except ImportError:
            from whisper_cpp_backend import load_whisper_cpp_model

        model, actual = load_whisper_cpp_model(
            config.model_size, force_device=device, n_threads=max(1, int(n_threads))
        )
        if model is None:
            raise RuntimeError("whisper.cpp model load failed")
        return model, str(actual or device)
    if backend == "parakeet":
        try:
            from .parakeet_backend import ParakeetAsrModel
        except ImportError:
            from parakeet_backend import ParakeetAsrModel

        model = ParakeetAsrModel(model_name=config.model_size, device=device)
        return model, model.device
    if backend != "faster_whisper":
        raise ValueError(f"Unsupported ASR backend: {config.backend}")

    from faster_whisper import WhisperModel

    if device == "cuda":
        try:
            logger.info("Loading Faster-Whisper %s on CUDA", config.model_size)
            return WhisperModel(config.model_size, device="cuda", compute_type="float16"), "cuda"
        except Exception as exc:
            if not config.allow_cpu_fallback:
                raise RuntimeError(f"Faster-Whisper CUDA load failed: {exc}") from exc
            logger.warning(
                "Faster-Whisper CUDA load failed (%s); using one explicit CPU fallback", exc
            )
            print(
                "⚠️ Faster-Whisper CUDA model load failed; using one CPU fallback "
                "instead of spawning multiple CPU models.",
                flush=True,
            )
    logger.info("Loading Faster-Whisper %s on CPU", config.model_size)
    return WhisperModel(config.model_size, device="cpu", compute_type="int8"), "cpu"


def _release_models(models: Iterable[Any]) -> None:
    """Release every batch-owned model and return CUDA cache to later TTS work."""
    for model in models:
        try:
            close = getattr(model, "close", None)
            if callable(close):
                close()
        except Exception:
            logger.debug("ASR model close failed", exc_info=True)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    gc.collect()


def _transcribe_path(model: Any, wav_path: Path, backend: str) -> str:
    """Transcribe one WAV path using a selected non-packed model context."""
    segments, _info = model.transcribe(
        str(wav_path),
        language="en",
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(str(getattr(segment, "text", "") or "").strip() for segment in segments).strip()


def _run_model_pool(
    tasks: List[Dict[str, Any]], config: ASRBatchConfig, progress_callback: Optional[Callable]
) -> ASRBatchRun:
    """Run CPP GPU/CPU or Faster-Whisper CPU work with owned model contexts.

    CUDA whisper.cpp is intentionally one context and one transcription at a
    time. Its native allocator terminates the child on a second model OOM, so
    runtime capacity probing cannot safely admit multiple Medium models.
    """
    started = time.monotonic()
    backend = _canonical_backend(config.backend)
    requested_device = _canonical_device(config.requested_device)
    wanted = max(1, int(config.workers or 1))
    if backend == "whisper_cpp" and requested_device == "cuda":
        # pywhispercpp turns CUDA model-allocation failure into SIGSEGV; never
        # probe capacity by loading a second model in the isolated child.
        wanted = 1
    threads_per = max(1, (os.cpu_count() or 4) // wanted) if requested_device == "cpu" else 2
    models: List[Any] = []
    actual_devices: List[str] = []
    load_started = time.monotonic()
    try:
        for _index in range(wanted):
            model, actual = _load_one_model(config, n_threads=threads_per)
            actual = _canonical_device(actual)
            if models and actual != actual_devices[0]:
                # Do not mix GPU and CPU contexts after a partial CUDA admission.
                _release_models([model])
                break
            models.append(model)
            actual_devices.append(actual)
            # A Faster-Whisper CUDA failure is an explicit one-model CPU fallback.
            if backend == "faster_whisper" and requested_device == "cuda" and actual == "cpu":
                break
    except Exception as exc:
        logger.exception("ASR model-pool load failed")
        if not models:
            results = {
                _task_id(task): _error_result(task, backend, requested_device, exc)
                for task in tasks
            }
            return ASRBatchRun(
                results, backend, config.model_size, requested_device, [], 0,
                time.monotonic() - load_started, time.monotonic() - started,
            )

    load_seconds = time.monotonic() - load_started
    if not models:
        results = {
            _task_id(task): _error_result(task, backend, requested_device, "no ASR model admitted")
            for task in tasks
        }
        return ASRBatchRun(
            results, backend, config.model_size, requested_device, [], 0,
            load_seconds, time.monotonic() - started,
        )

    free_models: queue.Queue[Any] = queue.Queue()
    for model in models:
        free_models.put(model)
    actual = actual_devices[0]
    results: Dict[str, Dict[str, Any]] = {}
    completed = 0
    result_lock = threading.Lock()

    def _work(task: Dict[str, Any]) -> Dict[str, Any]:
        """Borrow one model context for a single transcription and comparison."""
        model = free_models.get()
        try:
            transcript = _transcribe_path(model, Path(task["wav_path"]), backend)
            return _result_from_transcript(task, transcript, backend, actual)
        except Exception as exc:
            return _error_result(task, backend, actual, exc)
        finally:
            free_models.put(model)

    try:
        with ThreadPoolExecutor(max_workers=len(models), thread_name_prefix="asr-model") as pool:
            futures = [pool.submit(_work, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                with result_lock:
                    results[str(row["chunk_id"])] = row
                    completed += 1
                    done = completed
                if progress_callback is not None:
                    progress_callback(done, len(tasks), row)
    finally:
        _release_models(models)
    return ASRBatchRun(
        results, backend, config.model_size, requested_device, sorted(set(actual_devices)), len(models),
        load_seconds, time.monotonic() - started,
    )


def _load_audio_mono_16k(wav_path: Path) -> np.ndarray:
    """Load one WAV as contiguous mono float32 at 16 kHz for packed Whisper."""
    import librosa
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError("audio file has 0 frames")
    if int(sample_rate) != _TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=_TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _transcribe_segments(model: Any, audio: np.ndarray) -> List[Dict[str, Any]]:
    """Decode timed Faster-Whisper segments from one packed 16 kHz audio array."""
    segments, _info = model.transcribe(
        audio,
        language="en",
        condition_on_previous_text=False,
        vad_filter=False,
        word_timestamps=False,
        without_timestamps=False,
    )
    return [
        {
            "start": float(getattr(segment, "start", 0.0) or 0.0),
            "end": float(getattr(segment, "end", 0.0) or 0.0),
            "text": str(getattr(segment, "text", "") or "").strip(),
        }
        for segment in segments
        if str(getattr(segment, "text", "") or "").strip()
    ]


def _pack_audio(items: List[Dict[str, Any]], silence_s: float) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    """Concatenate loaded WAVs with separators and retain regions for transcript mapping."""
    silence = np.zeros(max(0, int(round(silence_s * _TARGET_SAMPLE_RATE))), dtype=np.float32)
    parts: List[np.ndarray] = []
    regions: List[Dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(items):
        if index and silence.size:
            parts.append(silence)
            cursor += float(silence.size) / _TARGET_SAMPLE_RATE
        audio = item["audio"]
        start = cursor
        cursor += float(audio.size) / _TARGET_SAMPLE_RATE
        parts.append(audio)
        regions.append({"task": item["task"], "audio": audio, "start": start, "end": cursor})
    return np.concatenate(parts), regions


def _map_segments_to_regions(segments: List[Dict[str, Any]], regions: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map packed transcript segments to each clip by segment midpoint or overlap."""
    mapped: Dict[str, List[str]] = {_task_id(region["task"]): [] for region in regions}
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"] or start)
        midpoint = (start + end) * 0.5
        selected = None
        for region in regions:
            if region["start"] - 0.001 <= midpoint <= region["end"] + 0.001:
                selected = region
                break
        if selected is None and regions:
            selected = max(
                regions,
                key=lambda region: min(end, region["end"]) - max(start, region["start"]),
            )
        if selected is not None and segment["text"]:
            mapped[_task_id(selected["task"])].append(segment["text"])
    return {task_id: " ".join(words).strip() for task_id, words in mapped.items()}


def _run_faster_whisper_gpu(
    tasks: List[Dict[str, Any]], config: ASRBatchConfig, progress_callback: Optional[Callable]
) -> ASRBatchRun:
    """Run Pocket-style one-model packed GPU Faster-Whisper transcription."""
    started = time.monotonic()
    backend = "faster_whisper"
    load_started = time.monotonic()
    try:
        model, actual = _load_one_model(replace(config, workers=1), n_threads=2)
    except Exception as exc:
        results = {
            _task_id(task): _error_result(task, backend, "cuda", exc) for task in tasks
        }
        return ASRBatchRun(results, backend, config.model_size, "cuda", [], 0, time.monotonic() - load_started, time.monotonic() - started)
    load_seconds = time.monotonic() - load_started
    if _canonical_device(actual) != "cuda":
        _release_models([model])
        # CUDA failed once. Preserve an explicit one-model CPU fallback instead of
        # reattempting the GPU and creating the old four-Medium CPU pile-up.
        fallback = _run_model_pool(
            tasks, replace(config, requested_device="cpu", workers=1), progress_callback
        )
        fallback.requested_device = "cuda"
        fallback.load_seconds += load_seconds
        return fallback

    workers = max(1, int(config.workers or 1))
    loader_count = max(2, workers // 2)
    scorer_count = max(2, workers - loader_count)
    load_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, loader_count * 2, config.pack_size * 2))
    score_queue: queue.Queue[Any] = queue.Queue(maxsize=max(8, scorer_count * 2))
    stop = threading.Event()
    results: Dict[str, Dict[str, Any]] = {}
    results_lock = threading.Lock()
    done = 0

    def _emit(task: Dict[str, Any], transcript: str = "", error: Optional[str] = None) -> None:
        """Queue one transcript or operational error for parallel comparison."""
        score_queue.put((task, transcript, error))

    def _flush(pack: List[Dict[str, Any]]) -> None:
        """Decode one pack, then solo-retry only empty or failing mapped clips."""
        if not pack:
            return
        try:
            if len(pack) == 1:
                item = pack[0]
                _emit(item["task"], " ".join(segment["text"] for segment in _transcribe_segments(model, item["audio"])))
                return
            packed, regions = _pack_audio(pack, float(config.pack_silence_s))
            hypotheses = _map_segments_to_regions(_transcribe_segments(model, packed), regions)
            for region in regions:
                task = region["task"]
                transcript = hypotheses.get(_task_id(task), "")
                try:
                    preliminary = _result_from_transcript(task, transcript, backend, "cuda")
                    need_solo = not transcript or not preliminary.get("passed")
                except Exception:
                    preliminary = {"score": 0.0}
                    need_solo = True
                if need_solo:
                    solo = " ".join(segment["text"] for segment in _transcribe_segments(model, region["audio"]))
                    if not transcript or _result_from_transcript(task, solo, backend, "cuda").get("score", 0.0) >= preliminary.get("score", 0.0):
                        transcript = solo
                _emit(task, transcript)
        except Exception as exc:
            logger.exception("Packed Faster-Whisper decode failed; returning ASR errors for this pack")
            for item in pack:
                _emit(item["task"], error=f"ASR transcription error: {exc}")

    def _loader(task_slice: List[Dict[str, Any]]) -> None:
        """Read assigned WAVs once and send ready PCM arrays to the GPU thread."""
        for task in task_slice:
            if stop.is_set():
                break
            try:
                audio = _load_audio_mono_16k(Path(task["wav_path"]))
                load_queue.put({"task": task, "audio": audio, "error": None})
            except Exception as exc:
                load_queue.put({"task": task, "audio": None, "error": str(exc)})
        load_queue.put(None)

    def _gpu_loop() -> None:
        """Own the sole GPU model and convert loaded PCM into scored work items."""
        finished = 0
        pack: List[Dict[str, Any]] = []
        try:
            while finished < loader_count:
                item = load_queue.get()
                if item is None:
                    finished += 1
                    continue
                if item["error"]:
                    _emit(item["task"], error=item["error"])
                    continue
                pack.append(item)
                if len(pack) >= max(1, int(config.pack_size)):
                    _flush(pack)
                    pack = []
            _flush(pack)
        except Exception as exc:
            logger.exception("Faster-Whisper GPU loop failed")
            stop.set()
            for task in tasks:
                _emit(task, error=f"ASR GPU loop failed: {exc}")
        finally:
            for _ in range(scorer_count):
                score_queue.put(None)

    def _scorer() -> None:
        """Compare transcripts in CPU threads while the GPU decodes later packs."""
        nonlocal done
        while True:
            item = score_queue.get()
            if item is None:
                return
            task, transcript, error = item
            row = _error_result(task, backend, "cuda", error) if error else _result_from_transcript(task, transcript, backend, "cuda")
            with results_lock:
                results[_task_id(task)] = row
                done += 1
                current = done
            if progress_callback is not None:
                progress_callback(current, len(tasks), row)

    slices: List[List[Dict[str, Any]]] = [[] for _ in range(loader_count)]
    for index, task in enumerate(tasks):
        slices[index % loader_count].append(task)
    loaders = [threading.Thread(target=_loader, args=(part,), daemon=True) for part in slices]
    scorers = [threading.Thread(target=_scorer, daemon=True) for _ in range(scorer_count)]
    gpu_thread = threading.Thread(target=_gpu_loop, daemon=True)
    for thread in loaders + scorers + [gpu_thread]:
        thread.start()
    for thread in loaders + [gpu_thread] + scorers:
        thread.join()
    _release_models([model])
    for task in tasks:
        results.setdefault(
            _task_id(task), _error_result(task, backend, "cuda", "ASR runner returned no result")
        )
    return ASRBatchRun(
        results, backend, config.model_size, "cuda", ["cuda"], 1, load_seconds,
        time.monotonic() - started,
    )


def _run_parakeet(
    tasks: List[Dict[str, Any]], config: ASRBatchConfig, progress_callback: Optional[Callable]
) -> ASRBatchRun:
    """Run one Parakeet model's multi-file transcription API for Stage 1 only."""
    started = time.monotonic()
    load_started = time.monotonic()
    try:
        model, actual = _load_one_model(replace(config, workers=1), n_threads=1)
    except Exception as exc:
        results = {_task_id(task): _error_result(task, "parakeet", config.requested_device, exc) for task in tasks}
        return ASRBatchRun(results, "parakeet", config.model_size, config.requested_device, [], 0, time.monotonic() - load_started, time.monotonic() - started)
    load_seconds = time.monotonic() - load_started
    results: Dict[str, Dict[str, Any]] = {}
    try:
        # Pocket's stable Parakeet path uses NeMo's default batch four; the
        # backend isolates clips longer than 25 seconds before transcription.
        initial_batch_size = 4 if str(config.requested_device).lower() in {"cuda", "gpu"} else 16
        logger.info("Parakeet inference batch_size=%d", initial_batch_size)
        transcripts = model.transcribe_many(
            [Path(task["wav_path"]) for task in tasks],
            batch_size=initial_batch_size,
        )
        if len(transcripts) != len(tasks):
            raise RuntimeError(f"Parakeet returned {len(transcripts)} transcripts for {len(tasks)} tasks")
        for index, (task, transcript) in enumerate(zip(tasks, transcripts), start=1):
            row = _result_from_transcript(task, transcript, "parakeet", actual)
            results[_task_id(task)] = row
            if progress_callback is not None:
                progress_callback(index, len(tasks), row)
    except Exception as exc:
        logger.exception("Parakeet batch transcription failed")
        results = {_task_id(task): _error_result(task, "parakeet", actual, exc) for task in tasks}
    finally:
        _release_models([model])
    return ASRBatchRun(results, "parakeet", config.model_size, config.requested_device, [actual], 1, load_seconds, time.monotonic() - started)


def run_asr_batch(
    tasks: Iterable[Dict[str, Any]],
    config: ASRBatchConfig,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> ASRBatchRun:
    """Validate a completed batch using the exact selected backend/model/device.

    Faster-Whisper CUDA takes Pocket's one-model packed route.  Parakeet takes
    its one-model multi-file route.  CPP GPU/CPU and Faster-Whisper CPU use
    safely owned in-process model contexts.  Every route releases its models
    before returning.
    """
    task_list = list(tasks)
    backend = _canonical_backend(config.backend)
    normalized = replace(
        config,
        backend=backend,
        requested_device=_canonical_device(config.requested_device),
        workers=max(1, int(config.workers or 1)),
        pack_size=max(1, int(config.pack_size or 1)),
        pack_silence_s=max(0.0, float(config.pack_silence_s or 0.0)),
    )
    if not task_list:
        return ASRBatchRun({}, backend, normalized.model_size, normalized.requested_device, [], 0, 0.0, 0.0)
    if backend == "parakeet":
        return _run_parakeet(task_list, normalized, progress_callback)
    if backend == "faster_whisper" and normalized.requested_device == "cuda":
        return _run_faster_whisper_gpu(task_list, normalized, progress_callback)
    return _run_model_pool(task_list, normalized, progress_callback)


def _failed_run(
    tasks: Iterable[Dict[str, Any]], config: ASRBatchConfig, error: Exception | str
) -> ASRBatchRun:
    """Return unscored rows when an isolated batch process cannot complete."""
    backend = _canonical_backend(config.backend)
    device = _canonical_device(config.requested_device)
    return ASRBatchRun(
        {
            _task_id(task): _error_result(task, backend, device, error)
            for task in tasks
        },
        backend,
        config.model_size,
        device,
        [],
        0,
        0.0,
        0.0,
    )


def _run_to_json(run: ASRBatchRun) -> Dict[str, Any]:
    """Convert a completed batch result into the small JSON IPC payload."""
    return {
        "results": run.results,
        "backend": run.backend,
        "model_size": run.model_size,
        "requested_device": run.requested_device,
        "actual_devices": run.actual_devices,
        "effective_workers": run.effective_workers,
        "load_seconds": run.load_seconds,
        "elapsed_seconds": run.elapsed_seconds,
    }


def _run_from_json(payload: Dict[str, Any]) -> ASRBatchRun:
    """Rebuild a batch result returned by the short-lived child process."""
    return ASRBatchRun(
        results={str(key): value for key, value in dict(payload.get("results") or {}).items()},
        backend=str(payload.get("backend") or "faster_whisper"),
        model_size=str(payload.get("model_size") or "base"),
        requested_device=_canonical_device(payload.get("requested_device")),
        actual_devices=[str(device) for device in payload.get("actual_devices") or []],
        effective_workers=max(0, int(payload.get("effective_workers") or 0)),
        load_seconds=float(payload.get("load_seconds") or 0.0),
        elapsed_seconds=float(payload.get("elapsed_seconds") or 0.0),
    )


def run_asr_batch_isolated(
    tasks: Iterable[Dict[str, Any]],
    config: ASRBatchConfig,
    work_dir: Path | str,
    stage_label: str,
    timeout_seconds: float = 1_800.0,
) -> ASRBatchRun:
    """Run one batch in a child process so completed models cannot leak into a book.

    This is Pocket's isolation property without its persistent daemon: the
    parent writes only a job JSON, one process loads models and scores every
    task, writes results, and then exits. No worker descendants are created by
    this runner, so process exit is also model-memory release.

    Args:
        tasks: Completed WAV validation tasks for one Stage 1, Stage 2, or retry batch.
        config: Exact selected backend/model/device configuration.
        work_dir: Book-local TTS directory for transient job files and batch log.
        stage_label: Human-readable stage token used in job/log diagnostics.
        timeout_seconds: Hard upper bound for the complete batch child process.

    Returns:
        Completed ASRBatchRun, or unscored error rows if child setup/execution fails.
    """
    task_list = list(tasks)
    if not task_list:
        return run_asr_batch(task_list, config)
    safe_label = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(stage_label or "batch")
    )
    batch_dir = Path(work_dir) / f".asr_batch_{safe_label}_{time.time_ns()}"
    input_path = batch_dir / "input.json"
    output_path = batch_dir / "output.json"
    log_path = Path(work_dir) / "asr_batch.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--batch-input",
        str(input_path),
        "--batch-output",
        str(output_path),
    ]
    started = time.monotonic()
    try:
        batch_dir.mkdir(parents=True, exist_ok=False)
        input_path.write_text(
            json.dumps({"config": asdict(config), "tasks": task_list}, ensure_ascii=False),
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
            log_file.write(
                f"\n=== ASR batch start stage={safe_label} backend={config.backend} "
                f"model={config.model_size} requested_device={config.requested_device} "
                f"tasks={len(task_list)} ===\n"
            )
            completed = subprocess.run(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
                check=False,
            )
            log_file.write(
                f"=== ASR batch exit stage={safe_label} code={completed.returncode} "
                f"elapsed={time.monotonic() - started:.2f}s ===\n"
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"ASR batch child exited {completed.returncode}; inspect {log_path}"
            )
        if not output_path.is_file():
            raise RuntimeError(f"ASR batch child wrote no result file; inspect {log_path}")
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        run = _run_from_json(payload)
        for task in task_list:
            run.results.setdefault(
                _task_id(task),
                _error_result(task, run.backend, run.requested_device, "ASR batch returned no result"),
            )
        return run
    except Exception as exc:
        logger.exception("Isolated ASR batch failed (%s)", safe_label)
        return _failed_run(task_list, config, exc)
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)


def _main() -> int:
    """Execute one JSON-described ASR batch inside the isolation child process."""
    parser = argparse.ArgumentParser(description="Short-lived ASR batch runner")
    parser.add_argument("--batch-input", required=True, help="JSON job payload path")
    parser.add_argument("--batch-output", required=True, help="JSON result payload path")
    args = parser.parse_args()
    input_path = Path(args.batch_input)
    output_path = Path(args.batch_output)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        config = ASRBatchConfig(**dict(payload.get("config") or {}))
        run = run_asr_batch(list(payload.get("tasks") or []), config)
        output_path.write_text(
            json.dumps(_run_to_json(run), ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return 0
    except Exception:
        logger.exception("ASR batch child failed before producing results")
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
