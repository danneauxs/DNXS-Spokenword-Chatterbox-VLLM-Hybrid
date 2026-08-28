"""Regenerate confirmed ASR failures with the original unified Chatterbox model.

This is a standalone listening-test tool.  It intentionally uses the same
single-model English T3/S3Gen stack as Repair-tab resynthesis, not Pipeline 4's
separate vLLM T3 and Turbo S3Gen stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    # whisper_cpp_backend retains this legacy top-level import.
    sys.path.insert(0, str(project_root / "ASR"))

from ASR.batch_runner import (
    ASRBatchConfig,
    _load_one_model,
    _release_models,
    _result_from_transcript,
    _transcribe_path,
)
from config import config as runtime_config
from config.config import (
    REGEN_EXAGGERATION_ADJUSTMENT,
    REGEN_TEMPERATURE_ADJUSTMENT,
    TTS_PARAM_MIN_EXAGGERATION,
    TTS_PARAM_MIN_TEMPERATURE,
)


logger = logging.getLogger("run_regen_review")
_UNIFIED_CHECKPOINT_FILES = ("ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json")


def _resolve_tts_dir(folder: Path) -> Path:
    """Resolve either a book directory or its TTS directory to the TTS folder."""
    candidate = folder.expanduser().resolve()
    if candidate.name.lower() == "tts":
        return candidate
    nested = candidate / "TTS"
    if nested.is_dir():
        return nested
    raise ValueError(f"No TTS directory found at {candidate} or {nested}")


def _load_failures(tts_dir: Path, report: str) -> List[Dict[str, Any]]:
    """Load unique, text-bearing failure rows from one TTS-local JSON report."""
    report_path = Path(report)
    if not report_path.is_absolute():
        report_path = tts_dir / report_path
    if not report_path.is_file():
        raise FileNotFoundError(f"ASR failure report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ASR failure report must contain a JSON list: {report_path}")
    rows: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    for row in payload:
        if not isinstance(row, dict) or row.get("chunk_id") is None:
            continue
        chunk_id = int(row["chunk_id"])
        if chunk_id in seen_ids:
            continue
        text = str(row.get("text") or row.get("expected_text") or "").strip()
        if not text:
            logger.warning("Skipping chunk_%05d because its report row has no text", chunk_id)
            continue
        copied = dict(row)
        copied["chunk_id"] = chunk_id
        copied["text"] = text
        rows.append(copied)
        seen_ids.add(chunk_id)
    if not rows:
        raise ValueError(f"No usable failed chunks found in {report_path}")
    return rows


def _params_for_repair_attempt(base_params: Dict[str, Any], attempt_number: int) -> Dict[str, Any]:
    """Return Repair-equivalent first-attempt parameters and later retry variants.

    Repair-tab resynthesis sends a chunk's saved values unchanged.  Oregen's
    first attempt must do the same; later attempts intentionally lower only
    temperature and exaggeration from that original baseline.
    """
    if attempt_number == 1:
        return dict(base_params)
    adjusted = dict(base_params)
    adjusted["temperature"] = max(
        TTS_PARAM_MIN_TEMPERATURE,
        float(base_params.get("temperature", 0.8))
        - REGEN_TEMPERATURE_ADJUSTMENT * (attempt_number - 1),
    )
    adjusted["exaggeration"] = max(
        TTS_PARAM_MIN_EXAGGERATION,
        float(base_params.get("exaggeration", 0.5))
        - REGEN_EXAGGERATION_ADJUSTMENT * (attempt_number - 1),
    )
    return adjusted


def _validate_unified_checkpoint() -> Path:
    """Confirm all original unified-model assets exist before any Oregen file is written."""
    checkpoint_dir = Path(runtime_config.CHATTERBOX_CKPT_DIR).expanduser().resolve()
    missing = [name for name in _UNIFIED_CHECKPOINT_FILES if not (checkpoint_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Original unified Chatterbox checkpoint is incomplete at {checkpoint_dir}; "
            f"missing {', '.join(missing)}"
        )
    return checkpoint_dir


def _write_text_sidecar(output_dir: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    """Write exact expected text beside Oregen WAVs for listening review."""
    text_path = output_dir / "Oregen_text_chunks.txt"
    lines: List[str] = []
    for row in rows:
        lines.extend((f"chunk_{int(row['chunk_id']):05d}", str(row["text"]), ""))
    text_path.write_text("\n".join(lines), encoding="utf-8")
    return text_path


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as H:MM:SS.mmm for terminal and UI display."""
    total_seconds = max(0.0, float(seconds))
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    if hours:
        return f"{hours}:{minutes:02d}:{secs:06.3f}"
    return f"{minutes}:{secs:06.3f}"


def _original_chunk_wav_path(tts_dir: Path, chunk_id: int) -> Path:
    """Return source chunk WAV path for one reviewed chunk."""
    return Path(tts_dir) / "audio_chunks" / f"chunk_{chunk_id:05d}.wav"


def _check_output_targets(output_dir: Path, rows: Iterable[Dict[str, Any]], attempts: int, overwrite: bool) -> None:
    """Refuse to silently replace prior Oregen attempt WAVs."""
    collisions = [
        output_dir / f"chunk_{int(row['chunk_id']):05d}_attempt{attempt}.wav"
        for row in rows
        for attempt in range(1, attempts + 1)
        if (output_dir / f"chunk_{int(row['chunk_id']):05d}_attempt{attempt}.wav").exists()
    ]
    if collisions and not overwrite:
        sample = ", ".join(path.name for path in collisions[:4])
        raise FileExistsError(
            f"Oregen already contains {len(collisions)} target WAV(s), including {sample}. "
            "Use --overwrite only when replacement is intentional."
        )


def _load_live_cpp_scorer(model_size: str, requested_device: str):
    """Load one whisper.cpp checker and return its resolved runtime device.

    Args:
        model_size: whisper.cpp model name to load.
        requested_device: Requested runtime device for the ASR checker.

    Returns:
        Tuple of loaded model and resolved device label.

    Raises:
        RuntimeError: Requested CUDA ASR could not stay on GPU.
    """
    config = ASRBatchConfig(
        backend="whisper_cpp",
        model_size=model_size,
        requested_device=requested_device,
        workers=1,
    )
    model, actual_device = _load_one_model(config, n_threads=2)
    if str(requested_device).lower() == "cuda" and actual_device != "cuda":
        _release_models([model])
        raise RuntimeError(
            f"Live Oregen scorer requested CUDA but resolved to {actual_device}; "
            "GPU ASR did not load."
        )
    return model, actual_device


def _score_attempt(
    model: Any, wav_path: Path, row: Dict[str, Any], threshold: float, asr_device: str
) -> Dict[str, Any]:
    """Transcribe and compare one generated WAV with the live whisper.cpp ASR model."""
    task = {
        "chunk_id": str(row["chunk_id"]),
        "wav_path": str(wav_path),
        "expected_text": row["text"],
        "threshold": threshold,
    }
    try:
        transcript = _transcribe_path(model, wav_path, "whisper_cpp")
        return _result_from_transcript(task, transcript, "whisper_cpp", asr_device)
    except Exception as exc:
        return {
            "chunk_id": str(row["chunk_id"]),
            "passed": False,
            "score": 0.0,
            "asr_text": "",
            "expected_text": row["text"],
            "backend": "whisper_cpp",
            "device": asr_device,
            "error": str(exc),
        }


def _hydrate_repair_rows(tts_dir: Path, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace report metadata with source chunks used by the Repair tab.

    Failure reports choose which chunk IDs to test.  Repair itself obtains text,
    parameters, and boundary type from ``text_chunks/chunks_info.json``.  This
    makes Oregen use those same authoritative chunk records.
    """
    chunks_path = tts_dir / "text_chunks" / "chunks_info.json"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Repair source chunks file not found: {chunks_path}")
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Repair source chunks must contain a JSON list: {chunks_path}")
    by_index = {
        int(chunk["index"]): dict(chunk)
        for chunk in payload
        if isinstance(chunk, dict) and chunk.get("index") is not None
    }
    hydrated: List[Dict[str, Any]] = []
    missing: List[int] = []
    for row in rows:
        chunk_id = int(row["chunk_id"])
        source = by_index.get(chunk_id)
        if source is None:
            missing.append(chunk_id)
            continue
        source["chunk_id"] = chunk_id
        source["source_report_score"] = row.get(
            "score", row.get("original_score", row.get("best_score"))
        )
        hydrated.append(source)
    if missing:
        preview = ", ".join(f"{chunk_id:05d}" for chunk_id in missing[:8])
        raise ValueError(
            f"{len(missing)} failure-report ID(s) are absent from Repair chunks_info.json: {preview}"
        )
    return hydrated


def run_regeneration_review(
    folder: Path,
    voice_path: Path,
    report: str = "asr_confirmed_failures.json",
    max_attempts: int = 3,
    threshold: float = 0.65,
    device: str = "cuda",
    asr_device: str = "cpu",
    asr_model: str = "medium",
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Generate and immediately score retries through the Repair-tab pipeline.

    Each retry calls ``wrapper.chunk_synthesizer.synthesize_chunk`` with an
    explicit Oregen target.  That is the exact generator used by Repair,
    including its optimized model, voice preparation, and audio finishing.
    A chunk stops at its first spoken-comparison pass.
    """
    if device != "cuda":
        raise ValueError("This listening-test tool currently requires --device cuda")
    if max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    tts_dir = _resolve_tts_dir(folder)
    voice_path = voice_path.expanduser().resolve()
    if not voice_path.is_file():
        raise FileNotFoundError(f"Voice sample not found: {voice_path}")
    failure_rows = _load_failures(tts_dir, report)
    rows = _hydrate_repair_rows(tts_dir, failure_rows)
    checkpoint_dir = _validate_unified_checkpoint()
    output_dir = tts_dir / "audio_chunks" / "Oregen"
    output_dir.mkdir(parents=True, exist_ok=True)
    _check_output_targets(output_dir, rows, max_attempts, overwrite)
    text_path = _write_text_sidecar(output_dir, rows)
    report_name = Path(report).name

    print(f"[Oregen] TTS folder: {tts_dir}")
    print(f"[Oregen] Report: {report_name} | rows: {len(rows)} | output: {output_dir}")
    print(f"[Oregen] Generator: Repair-tab unified ChatterboxTTS pipeline from {checkpoint_dir}")
    print(f"[Oregen] Live ASR: whisper_cpp/{asr_model} on requested {asr_device} (one warm model)")

    asr_scorer = None
    asr_scorer_device = asr_device
    manifest_rows: List[Dict[str, Any]] = []
    started = time.monotonic()
    try:
        from wrapper.chunk_synthesizer import synthesize_chunk

        for position, row in enumerate(rows, start=1):
            chunk_id = int(row["chunk_id"])
            attempts: List[Dict[str, Any]] = []
            accepted_attempt: Optional[int] = None
            print(f"[Oregen] {position}/{len(rows)} chunk_{chunk_id:05d}: testing up to {max_attempts} attempt(s)")
            for attempt_number in range(1, max_attempts + 1):
                params = _params_for_repair_attempt(dict(row.get("tts_params") or {}), attempt_number)
                wav_path = output_dir / f"chunk_{chunk_id:05d}_attempt{attempt_number}.wav"
                repair_chunk = dict(row)
                repair_chunk["tts_params"] = params
                written_path = synthesize_chunk(
                    repair_chunk,
                    chunk_id,
                    tts_dir.parent.name,
                    tts_dir / "audio_chunks",
                    chunks_json_path=tts_dir / "text_chunks" / "chunks_info.json",
                    override_voice_name=voice_path.stem,
                    override_voice_path=voice_path,
                    output_path=wav_path,
                )
                if not written_path:
                    raise RuntimeError(
                        f"Repair-equivalent synthesis failed for chunk_{chunk_id:05d} attempt {attempt_number}"
                    )
                if asr_scorer is None:
                    # Match Repair's first generation before placing optional ASR
                    # beside the unified T3/S3Gen model in GPU memory.
                    asr_scorer, asr_scorer_device = _load_live_cpp_scorer(
                        asr_model, asr_device
                    )
                score_row = _score_attempt(
                    asr_scorer,
                    wav_path,
                    row,
                    threshold,
                    asr_scorer_device,
                )
                score_row.update(
                    {
                        "attempt": attempt_number,
                        "wav_path": str(wav_path),
                        "params": params,
                        "unsupported_original_model_params": {
                            key: params[key]
                            for key in ("min_p", "top_p", "repetition_penalty")
                            if key in params
                        },
                    }
                )
                attempts.append(score_row)
                print(
                    f"[Oregen] chunk_{chunk_id:05d} attempt {attempt_number}: "
                    f"score={float(score_row.get('score', 0.0) or 0.0):.3f} "
                    f"passed={bool(score_row.get('passed'))}"
                )
                if bool(score_row.get("passed")) and not score_row.get("error"):
                    accepted_attempt = attempt_number
                    break
            original_wav_path = _original_chunk_wav_path(tts_dir, chunk_id)
            accepted_wav_path = (
                output_dir / f"chunk_{chunk_id:05d}_attempt{accepted_attempt}.wav"
                if accepted_attempt is not None
                else None
            )
            manifest_rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": row["text"],
                    "threshold": threshold,
                    "source_report_score": row.get("source_report_score"),
                    "accepted_attempt": accepted_attempt,
                    "original_wav_path": str(original_wav_path),
                    "accepted_wav_path": str(accepted_wav_path) if accepted_wav_path else None,
                    "attempts": attempts,
                }
            )
    finally:
        if asr_scorer is not None:
            _release_models([asr_scorer])

    elapsed_seconds = round(time.monotonic() - started, 3)
    manifest = {
        "tts_dir": str(tts_dir),
        "report": str(report),
        "report_name": report_name,
        "voice_path": str(voice_path),
        "output_dir": str(output_dir),
        "text_sidecar": str(text_path),
        "book_folder": str(tts_dir.parent),
        "chunk_count": len(rows),
        "threshold": threshold,
        "max_attempts": max_attempts,
        "tts_device": device,
        "generator": "original_unified_chatterbox",
        "generator_checkpoint": str(checkpoint_dir),
        "asr_backend": "whisper_cpp",
        "asr_device": asr_scorer_device,
        "asr_model": asr_model,
        "elapsed_seconds": elapsed_seconds,
        "rows": manifest_rows,
    }
    manifest_path = output_dir / "Oregen_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Oregen] Complete. Review manifest: {manifest_path}")
    print(f"[Oregen] Runtime: {_format_elapsed(elapsed_seconds)}")
    return manifest


def open_review_window(manifest: Dict[str, Any]) -> None:
    """Open a Tkinter player showing original and accepted audio for each chunk."""
    import tkinter as tk
    from tkinter import ttk

    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("pygame is required for Oregen playback") from exc
    entries = list(manifest["rows"])
    if not entries:
        raise RuntimeError("No Oregen rows were generated, so there is nothing to review")
    root = tk.Tk()
    root.title("Oregen ASR Listening Review")
    root.minsize(1020, 520)
    current = tk.IntVar(value=0)
    header = ttk.Label(root, padding=(12, 12, 12, 4), font=("TkDefaultFont", 11, "bold"))
    header.pack(anchor="w")
    meta = ttk.Label(root, padding=(12, 0, 12, 8))
    meta.pack(anchor="w")
    text = tk.Text(root, wrap="word", height=14, padx=12, pady=10)
    text.pack(fill="both", expand=True, padx=12, pady=4)
    controls = ttk.Frame(root, padding=12)
    controls.pack(fill="x")
    status = ttk.Label(root, padding=(12, 0, 12, 10))
    status.pack(anchor="w")

    def _selected_row() -> Dict[str, Any]:
        """Return currently selected manifest row."""
        return entries[current.get()]

    def _selected_paths() -> tuple[Path, Optional[Path]]:
        """Return original and accepted WAV paths for current row."""
        row = _selected_row()
        original = Path(row["original_wav_path"])
        accepted = Path(row["accepted_wav_path"]) if row.get("accepted_wav_path") else None
        return original, accepted

    def render() -> None:
        """Render selected chunk metadata and immutable source text."""
        row = _selected_row()
        accepted_attempt = row.get("accepted_attempt")
        header.configure(
            text=(
                f"{current.get() + 1}/{len(entries)}  chunk_{int(row['chunk_id']):05d}  "
                f"accepted attempt={accepted_attempt if accepted_attempt is not None else 'none'}"
            )
        )
        meta.configure(
            text=(
                f"Book: {manifest.get('book_folder', manifest['tts_dir'])}    "
                f"Report: {manifest.get('report_name', Path(manifest['report']).name)}    "
                f"Chunks: {manifest.get('chunk_count', len(entries))}    "
                f"Run time: {_format_elapsed(float(manifest.get('elapsed_seconds', 0.0) or 0.0))}"
            )
        )
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.insert("1.0", row["text"])
        text.configure(state="disabled")
        status.configure(
            text=(
                f"Original: {row['original_wav_path']}    "
                f"Accepted: {row.get('accepted_wav_path') or 'none'}"
            )
        )
        if accepted_button is not None:
            accepted_button.configure(state="normal" if row.get("accepted_wav_path") else "disabled")

    def play_original() -> None:
        """Play source chunk WAV through pygame's mixer."""
        original, _accepted = _selected_paths()
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(original))
        pygame.mixer.music.play()

    def play_accepted() -> None:
        """Play accepted regeneration WAV through pygame's mixer."""
        _original, accepted = _selected_paths()
        if accepted is None:
            raise RuntimeError("Current chunk has no accepted regeneration WAV")
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(accepted))
        pygame.mixer.music.play()

    def stop() -> None:
        """Stop active playback without changing selected review entry."""
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def move(delta: int) -> None:
        """Move selection with wraparound and stop any active audio first."""
        stop()
        current.set((current.get() + delta) % len(entries))
        render()

    ttk.Button(controls, text="Previous", command=lambda: move(-1)).pack(side="left")
    ttk.Button(controls, text="Play Original", command=play_original).pack(side="left", padx=6)
    accepted_button = ttk.Button(controls, text="Play Accepted", command=play_accepted)
    accepted_button.pack(side="left", padx=6)
    ttk.Button(controls, text="Stop", command=stop).pack(side="left")
    ttk.Button(controls, text="Next", command=lambda: move(1)).pack(side="left", padx=6)
    ttk.Label(controls, text=f"Text file: {manifest['text_sidecar']}").pack(side="right")
    render()
    root.mainloop()


def _build_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for unified-model Oregen listening tests."""
    parser = argparse.ArgumentParser(description="Regenerate ASR failures with original ChatterboxTTS and open an Oregen review player.")
    parser.add_argument("folder", type=Path, help="TTS folder or book folder containing TTS/")
    parser.add_argument("--voice", required=True, type=Path, help="Voice WAV used by original book generation")
    parser.add_argument("--report", default="asr_confirmed_failures.json", help="TTS-local failure JSON, default: asr_confirmed_failures.json")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--asr-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--asr-model", default="medium")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacement of same-named Oregen WAVs")
    parser.add_argument("--no-window", action="store_true", help="Generate and score files without opening Tkinter")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run unified regeneration and then optionally open the Tkinter review window."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _build_parser().parse_args(argv)
    try:
        manifest = run_regeneration_review(
            folder=args.folder,
            voice_path=args.voice,
            report=args.report,
            max_attempts=args.max_attempts,
            threshold=args.threshold,
            device=args.device,
            asr_device=args.asr_device,
            asr_model=args.asr_model,
            overwrite=args.overwrite,
        )
        if not args.no_window:
            open_review_window(manifest)
    except Exception as exc:
        logger.exception("Oregen review failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
