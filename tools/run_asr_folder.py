"""Run the existing two-stage ASR checks against completed TTS audio."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ASR.batch_runner import ASRBatchConfig, run_asr_batch_isolated
from modules.asr_stages import (
    PARAKEET_MODEL,
    accepted_asr_fuzzy_reason,
    is_parakeet_backend,
    is_stage_two_disabled,
    known_asr_pass_reason,
    load_accepted_asr_fuzzies,
    load_known_asr_passes,
    normalize_backend,
    normalize_model_name,
)

logger = logging.getLogger("run_asr_folder")


def _resolve_tts_dir(folder: Path) -> Path:
    """Resolve a supplied book or TTS path to its actual TTS directory."""
    candidate = folder.expanduser().resolve()
    if candidate.name.lower() == "tts":
        return candidate
    nested = candidate / "TTS"
    if nested.is_dir():
        return nested
    raise ValueError(f"No TTS directory found at {candidate} or {nested}")


def _chunk_id(path: Path) -> str:
    """Extract the numeric chunk id from a canonical chunk WAV filename."""
    stem = path.stem
    if not stem.startswith("chunk_"):
        raise ValueError(f"Unexpected chunk filename: {path.name}")
    return str(int(stem.removeprefix("chunk_")))


def _collect_text_files(tts_dir: Path) -> Dict[int, Path]:
    """Index chunk text files from the supported TTS folder locations."""
    indexed: Dict[int, Path] = {}
    for directory in (tts_dir / "audio_chunks", tts_dir, tts_dir / "text_chunks"):
        for path in directory.glob("chunk_*.txt"):
            try:
                indexed[int(path.stem.removeprefix("chunk_"))] = path
            except ValueError:
                continue
    return indexed


def _choose_text_offset(audio_ids: List[int], text_ids: Iterable[int]) -> int:
    """Choose zero-based, one-based, or legacy text numbering by match count."""
    available = set(text_ids)
    choices = {offset: sum((audio_id + offset) in available for audio_id in audio_ids) for offset in (0, 1, -1)}
    return max(choices, key=choices.get)


def _load_tasks(tts_dir: Path, threshold: float) -> List[Dict[str, Any]]:
    """Build ASR tasks from existing chunk WAVs and matching source text."""
    audio_dir = tts_dir / "audio_chunks"
    if not audio_dir.is_dir():
        raise ValueError(f"Missing audio_chunks directory: {audio_dir}")
    wavs = []
    for wav_path in audio_dir.glob("chunk_*.wav"):
        try:
            _chunk_id(wav_path)
        except ValueError:
            # Repair/Oregen artifacts share the prefix but are not source chunks.
            logger.debug("Ignoring noncanonical chunk WAV: %s", wav_path.name)
            continue
        wavs.append(wav_path)
    wavs.sort(key=lambda path: int(_chunk_id(path)))
    if not wavs:
        raise ValueError(f"No chunk_*.wav files found in {audio_dir}")
    text_files = _collect_text_files(tts_dir)
    audio_ids = [int(_chunk_id(path)) for path in wavs]
    text_offset = _choose_text_offset(audio_ids, text_files)
    logger.info("Resolved text chunk numbering offset: %+d", text_offset)
    tasks: List[Dict[str, Any]] = []
    missing_text: List[str] = []
    for wav_path in wavs:
        chunk_id = _chunk_id(wav_path)
        text_path = text_files.get(int(chunk_id) + text_offset)
        expected_text = text_path.read_text(encoding="utf-8").strip() if text_path else ""
        if not expected_text:
            missing_text.append(chunk_id)
        tasks.append(
            {
                "chunk_id": chunk_id,
                "wav_path": str(wav_path),
                "expected_text": expected_text,
                "threshold": threshold,
            }
        )
    if missing_text:
        logger.warning("Missing expected text for %d chunk(s); those rows cannot score", len(missing_text))
    return tasks


def _write_json(path: Path, payload: Any) -> None:
    """Write UTF-8 JSON with stable formatting for later repair review."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, title: str, rows: Iterable[Dict[str, Any]], threshold: float) -> None:
    """Write a compact human-readable report alongside machine-readable JSON."""
    row_list = list(rows)
    lines = [title, "=" * len(title), f"Threshold: {threshold:.2f}", f"Rows: {len(row_list)}", ""]
    for row in row_list:
        status = "PASS" if row.get("passed") else ("FAIL" if _is_scored(row) else "ASR_ERROR")
        lines.extend(
            [
                f"chunk_{int(row.get('chunk_id', 0)):05d}: {status} score={float(row.get('score', 0.0) or 0.0):.3f}",
                f"  original: {str(row.get('expected_text') or row.get('text') or '')[:240]}",
                f"  heard:    {str(row.get('asr_text') or '')[:240]}",
                f"  error:    {row.get('error') or ''}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _is_scored(row: Dict[str, Any]) -> bool:
    """Return true only for a completed ASR comparison, not an operational error."""
    return bool(row) and "score" in row and not row.get("error")


def _unscored_error_summary(rows: Iterable[Dict[str, Any]]) -> str:
    """Return one compact representative error when an ASR stage cannot score.

    The batch runner intentionally returns an operational-error row for every
    task rather than calling an engine setup failure an audio failure.  This
    summary makes that distinction visible to standalone callers before they
    decide whether to preserve a prior confirmed-failure report.
    """
    errors = {
        str(row.get("error") or "ASR returned no scored result").strip()
        for row in rows
        if not _is_scored(row)
    }
    if not errors:
        return ""
    first = sorted(errors)[0]
    suffix = f" (+{len(errors) - 1} distinct error(s))" if len(errors) > 1 else ""
    return first + suffix


def _confirmed_failures(
    rows: Iterable[Dict[str, Any]],
    known_passes: Dict[Tuple[int, str], str],
    accepted_fuzzies: Dict[Tuple[int, str], str],
) -> List[Dict[str, Any]]:
    """Return scored failures after applying listener-approved ASR decisions.

    An override never masks a broken ASR run: unscored rows remain operational
    errors because they cannot safely be classified from prior listening. Manual
    passes become passed rows; accepted fuzzies remain failed ASR observations
    but are excluded from further verification and regeneration.
    """
    failures: List[Dict[str, Any]] = []
    for row in rows:
        if not _is_scored(row) or row.get("passed"):
            continue
        reason = known_asr_pass_reason(
            known_passes, row.get("chunk_id"), row.get("expected_text") or row.get("text") or ""
        )
        if reason:
            row["passed"] = True
            row["classification"] = "MANUAL_PASS"
            row["manual_pass"] = True
            row["manual_pass_reason"] = reason
            continue
        fuzzy_reason = accepted_asr_fuzzy_reason(
            accepted_fuzzies,
            row.get("chunk_id"),
            row.get("expected_text") or row.get("text") or "",
        )
        if fuzzy_reason:
            row["classification"] = "ACCEPTED_FUZZY"
            row["accepted_fuzzy"] = True
            row["accepted_fuzzy_reason"] = fuzzy_reason
            row["regeneration_exempt"] = True
            continue
        failures.append(row)
    return failures


def _run_stage(
    tasks: List[Dict[str, Any]],
    tts_dir: Path,
    backend: str,
    model: str,
    device: str,
    label: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run one isolated ASR stage and return rows plus runtime metadata."""
    if not tasks:
        return [], {"backend": backend, "model": model, "device": device, "chunks": 0, "elapsed_seconds": 0.0}
    started = time.monotonic()
    config = ASRBatchConfig(
        backend=backend,
        model_size=model,
        requested_device=device,
        workers=1 if backend == "parakeet" else 4,
    )
    run = run_asr_batch_isolated(tasks, config, tts_dir, stage_label=label)
    rows = [run.results.get(str(task["chunk_id"]), {}) for task in tasks]
    elapsed = time.monotonic() - started
    for row, task in zip(rows, tasks):
        row.setdefault("chunk_id", task["chunk_id"])
        row.setdefault("expected_text", task.get("expected_text", ""))
    return rows, {
        "backend": run.backend,
        "model": run.model_size,
        "requested_device": run.requested_device,
        "actual_devices": run.actual_devices,
        "effective_workers": run.effective_workers,
        "chunks": len(tasks),
        "elapsed_seconds": elapsed,
    }


def run_folder(
    folder: Path,
    stage1_backend: str,
    stage1_model: str,
    stage2_backend: str,
    stage2_model: str,
    device: str,
    threshold: float,
) -> Dict[str, Any]:
    """Run Stage 1 and Stage 2 against existing TTS chunks without TTS work."""
    tts_dir = _resolve_tts_dir(folder)
    tasks = _load_tasks(tts_dir, threshold)
    known_passes = load_known_asr_passes(tts_dir)
    accepted_fuzzies = load_accepted_asr_fuzzies(tts_dir)
    stage1_backend = normalize_backend(stage1_backend)
    stage1_model = PARAKEET_MODEL if is_parakeet_backend(stage1_backend) else normalize_model_name(stage1_model)
    stage2_backend = normalize_backend(stage2_backend)
    stage2_model = normalize_model_name(stage2_model)

    print(f"[ASR-only] TTS folder: {tts_dir}")
    if known_passes:
        print(f"[ASR-only] Loaded {len(known_passes)} text-validated manual no-fail override(s)")
    if accepted_fuzzies:
        print(f"[ASR-only] Loaded {len(accepted_fuzzies)} listener-approved fuzzy exemption(s)")
    print(f"[ASR-only] Stage 1: {stage1_backend}/{stage1_model} on {device} ({len(tasks)} chunks)")
    stage1_rows, stage1_meta = _run_stage(tasks, tts_dir, stage1_backend, stage1_model, device, "stage1")
    stage1_failures = _confirmed_failures(stage1_rows, known_passes, accepted_fuzzies)
    stage1_unscored = len(stage1_rows) - sum(_is_scored(row) for row in stage1_rows)
    _write_json(tts_dir / "asr_stage1.json", stage1_rows)
    _write_json(tts_dir / "asr_stage1_failures.json", stage1_failures)
    _write_report(tts_dir / "asr_stage1_report.txt", "ASR Stage 1 inspection", stage1_rows, threshold)

    if stage1_unscored == len(stage1_rows):
        error_summary = _unscored_error_summary(stage1_rows)
        summary = {
            "tts_dir": str(tts_dir),
            "threshold": threshold,
            "success": False,
            "stage1": stage1_meta,
            "stage1_failures": 0,
            "stage1_unscored": stage1_unscored,
            "stage1_error": error_summary,
            "stage2": {
                "backend": stage2_backend,
                "model": stage2_model,
                "chunks": 0,
                "skipped": True,
                "reason": "Stage 1 produced no scored rows",
            },
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        _write_json(tts_dir / "asr_run_summary.json", summary)
        print(
            "[ASR-only] ERROR: Stage 1 produced no scored rows. "
            "Stage 2 was not run and the prior asr_confirmed_failures.json was preserved."
        )
        print(f"[ASR-only] Stage 1 error: {error_summary}")
        return summary

    stage2_rows: List[Dict[str, Any]] = []
    stage2_failures: List[Dict[str, Any]] = []
    stage2_meta: Dict[str, Any] = {"backend": stage2_backend, "model": stage2_model, "chunks": 0, "skipped": True}
    if stage1_failures and not is_stage_two_disabled(stage2_model):
        stage2_tasks = [
            {
                "chunk_id": row["chunk_id"],
                "wav_path": str(tts_dir / "audio_chunks" / f"chunk_{int(row['chunk_id']):05d}.wav"),
                "expected_text": row.get("expected_text", ""),
                "threshold": threshold,
            }
            for row in stage1_failures
        ]
        print(f"[ASR-only] Stage 2: {stage2_backend}/{stage2_model} on {device} ({len(stage2_tasks)} candidates)")
        stage2_rows, stage2_meta = _run_stage(stage2_tasks, tts_dir, stage2_backend, stage2_model, device, "stage2")
        stage2_failures = _confirmed_failures(stage2_rows, known_passes, accepted_fuzzies)
    else:
        print("[ASR-only] Stage 2 skipped: no Stage-1 failures or Stage 2 disabled")
    _write_json(tts_dir / "asr_stage2.json", stage2_rows)
    _write_json(tts_dir / "asr_confirmed_failures.json", stage2_failures or (stage1_failures if not stage2_rows else []))
    _write_report(tts_dir / "asr_stage2_report.txt", "ASR Stage 2 inspection", stage2_rows, threshold)
    _write_report(
        tts_dir / "asr_confirmed_report.txt",
        "ASR confirmed failures",
        stage2_failures or (stage1_failures if not stage2_rows else []),
        threshold,
    )
    summary = {
        "tts_dir": str(tts_dir),
        "threshold": threshold,
        "success": True,
        "stage1": stage1_meta,
        "stage1_failures": len(stage1_failures),
        "stage1_unscored": stage1_unscored,
        "stage2": stage2_meta,
        "stage2_failures": len(stage2_failures),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(tts_dir / "asr_run_summary.json", summary)
    print(
        f"[ASR-only] Complete: Stage 1 failures={len(stage1_failures)}, "
        f"unscored={stage1_unscored}, Stage 2 confirmed={len(stage2_failures)}"
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ASR-only folder validation."""
    parser = argparse.ArgumentParser(description="Run ASR Stage 1/2 on an existing TTS folder.")
    parser.add_argument("folder", type=Path, help="TTS folder or book folder containing TTS/")
    parser.add_argument("--stage1-backend", default="faster_whisper", choices=("faster_whisper", "whisper_cpp", "parakeet"))
    parser.add_argument("--stage1-model", default="base")
    parser.add_argument("--stage2-backend", default="faster_whisper", choices=("faster_whisper", "whisper_cpp"))
    parser.add_argument("--stage2-model", default="medium")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--threshold", type=float, default=0.65)
    return parser


def main(argv: List[str] | None = None) -> int:
    """Parse arguments, run both requested ASR stages, and return a shell status."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args(argv)
    try:
        summary = run_folder(
            args.folder,
            args.stage1_backend,
            args.stage1_model,
            args.stage2_backend,
            args.stage2_model,
            args.device,
            args.threshold,
        )
        if not summary.get("success", True):
            return 2
    except Exception as exc:
        logger.exception("ASR-only run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
