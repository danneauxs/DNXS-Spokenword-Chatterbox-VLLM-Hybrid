#!/usr/bin/env python3
"""Standalone Chatterbox ASR validator and saved-report rescore utility.

The audio-validation action delegates to ``tools/run_asr_folder.py`` so it
uses Chatterbox's production short-lived batch runner and report contracts.
The report-rescore action instead replays saved transcripts through the active
``spoken_compare`` policy.  It never touches WAVs or overwrites the selected
source report.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import os
import queue
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

_ASR_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _ASR_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from ASR.spoken_compare import compare_spoken


_CONFIG_PATH = _ASR_DIR / "asr_gui_config.json"
_REPORT_LIST_KEYS = ("records", "results", "failures")
_DIAGNOSTIC_SOURCE_REPORTS = (
    ("stage1", "asr_stage1.json"),
    ("stage2", "asr_stage2.json"),
    ("confirmed", "asr_confirmed_failures.json"),
)
_BACKENDS = ("faster_whisper", "whisper_cpp", "parakeet")
_STAGE_TWO_BACKENDS = ("faster_whisper", "whisper_cpp")
_MODELS = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
_MANUAL_REPAIR_REPORT_FILENAME = "asr_manual_repair_selection.json"


def resolve_tts_dir(selected: Path) -> Path:
    """Resolve a book, TTS, or audio-chunks path to its Chatterbox TTS folder."""
    candidate = selected.expanduser().resolve()
    if candidate.name == "audio_chunks":
        candidate = candidate.parent
    if candidate.name == "TTS" and (candidate / "audio_chunks").is_dir():
        return candidate
    nested = candidate / "TTS"
    if (nested / "audio_chunks").is_dir():
        return nested
    raise ValueError(
        "Select a TTS folder containing audio_chunks, its audio_chunks folder, "
        "or a book folder containing TTS/audio_chunks."
    )


def _read_json(path: Path) -> Any:
    """Read one UTF-8 JSON file with an actionable error message."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON: {exc}") from exc


def _record_container(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Return report rows and their wrapper key, rejecting ambiguous payloads."""
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload, None
    if isinstance(payload, dict):
        for key in _REPORT_LIST_KEYS:
            rows = payload.get(key)
            if isinstance(rows, list) and all(isinstance(item, dict) for item in rows):
                return rows, key
    raise ValueError("Report must be a JSON list or an object containing records/results/failures.")


def _row_reference(row: Dict[str, Any]) -> str:
    """Extract source text from supported Chatterbox and legacy ASR row fields."""
    for key in ("expected_text", "text", "reference_text", "ref_text_raw"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _row_transcript(row: Dict[str, Any]) -> str:
    """Extract saved ASR transcript from supported report row fields."""
    for key in ("asr_text", "transcribed_text", "hyp_text_raw", "transcript"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    return ""


def default_rescore_path(report_path: Path) -> Path:
    """Return a non-destructive sibling filename for a comparator replay."""
    return report_path.with_name(f"{report_path.stem}_rescored.json")


def load_manual_review_rows(folder: Path) -> List[Dict[str, Any]]:
    """Load canonical WAV/text review rows using production offset resolution.

    Audio chunks are zero-based while this project's text chunks are normally
    one-based.  Reusing the standalone runner's task builder keeps manual
    review attached to the same source text that production validation uses.

    Args:
        folder: Book, TTS, or audio_chunks directory selected by the user.

    Returns:
        Ordered rows containing chunk id, WAV path, and expected text.
    """
    from tools.run_asr_folder import _load_tasks

    tts_dir = resolve_tts_dir(folder)
    tasks = _load_tasks(tts_dir, threshold=0.65)
    missing = [task["chunk_id"] for task in tasks if not str(task.get("expected_text") or "").strip()]
    if missing:
        raise ValueError(
            f"{len(missing)} audio chunks have no matching text chunk; first: {', '.join(missing[:8])}"
        )
    return [
        {
            "chunk_id": int(task["chunk_id"]),
            "wav_path": str(task["wav_path"]),
            "expected_text": str(task["expected_text"]),
        }
        for task in tasks
    ]


def write_manual_repair_report(
    folder: Path,
    rows: Iterable[Dict[str, Any]],
    selected_chunk_ids: Iterable[int],
) -> Dict[str, Any]:
    """Write selected listener failures in the Repair tab's failure-row format.

    The output intentionally contains no ASR claim.  Each row records the
    listener-selected chunk id and expected source text, which is all Repair
    needs to locate the chunk and regenerate it.

    Args:
        folder: Book, TTS, or audio_chunks directory for report placement.
        rows: Manual-review rows returned by :func:`load_manual_review_rows`.
        selected_chunk_ids: Audio chunk ids checked by the listener.

    Returns:
        Output path and number of selected chunks.
    """
    tts_dir = resolve_tts_dir(folder)
    selected = {int(chunk_id) for chunk_id in selected_chunk_ids}
    records = []
    for row in rows:
        chunk_id = int(row["chunk_id"])
        if chunk_id not in selected:
            continue
        text = str(row.get("expected_text") or "")
        records.append(
            {
                "chunk_id": chunk_id,
                "expected_text": text,
                "text": text,
                "asr_text": "",
                "passed": False,
                "classification": "MANUAL_REPAIR_SELECTION",
                "failure_type": "listener_selected_repair",
                "score": 0.0,
                "prose_score": 0.0,
                "id_score": 1.0,
                "coverage_score": 0.0,
                "phonetic_score": 0.0,
                "explanation": "Selected during manual audio review.",
                "backend": "manual_review",
                "device": "",
                "error": None,
                "manual_pass": False,
                "accepted_fuzzy": False,
                "regeneration_exempt": False,
            }
        )
    destination = tts_dir / _MANUAL_REPAIR_REPORT_FILENAME
    destination.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_report": str(destination), "selected": len(records)}


def project_python() -> str:
    """Return Chatterbox's configured venv Python instead of GUI's launcher Python.

    A desktop shortcut or terminal can start this GUI with system Python, but
    validation must use the same project venv as ``0launch_gui.sh``.  An
    explicit ``CHATTERBOX_PYTHON`` override remains available for a deliberate
    alternate runtime.
    """
    configured = os.environ.get("CHATTERBOX_PYTHON")
    if configured and Path(configured).expanduser().is_file():
        return str(Path(configured).expanduser().resolve())
    for candidate in (_ROOT_DIR / "venv" / "bin" / "python", _ROOT_DIR / ".venv" / "bin" / "python"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return sys.executable


def rescore_report(report_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Replay saved ASR transcripts through the active Chatterbox comparator.

    Operational-error rows remain unscored.  Every comparable row receives
    fresh comparator evidence while preserving its original backend, audio, and
    source fields.  The source report is read only; output is always a sibling
    or explicitly supplied different path.
    """
    source = report_path.expanduser().resolve()
    destination = (output_path or default_rescore_path(source)).expanduser().resolve()
    if destination == source:
        raise ValueError("Rescore output must differ from the source report.")
    payload = _read_json(source)
    rows, wrapper_key = _record_container(payload)
    rescored_rows: List[Dict[str, Any]] = []
    scored = 0
    passed = 0
    unscored = 0
    for original in rows:
        row = deepcopy(original)
        reference = _row_reference(row)
        transcript = _row_transcript(row)
        if row.get("error") or not reference or not transcript:
            row["rescore_status"] = "unscored"
            unscored += 1
            rescored_rows.append(row)
            continue
        threshold = float(row.get("threshold") or 0.65)
        compared = compare_spoken(reference, transcript, threshold=threshold)
        row.update(compared)
        row["rescore_status"] = "rescored"
        row["rescore_threshold"] = threshold
        scored += 1
        passed += int(bool(compared.get("passed")))
        rescored_rows.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if wrapper_key is None:
        output_payload: Any = rescored_rows
    else:
        output_payload = deepcopy(payload)
        output_payload[wrapper_key] = rescored_rows
    destination.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_report": str(source),
        "output_report": str(destination),
        "rows": len(rows),
        "scored": scored,
        "passed": passed,
        "failed": scored - passed,
        "unscored": unscored,
        "rescored_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path = destination.with_name(f"{destination.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _diagnostic_record(stage: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract raw comparison inputs and failure evidence from one ASR row."""
    original = _row_reference(row)
    transcript = _row_transcript(row)
    comparison_keys = (
        "passed", "classification", "failure_type", "score", "prose_score",
        "id_score", "coverage_score", "phonetic_score", "explanation",
        "missing_words", "extra_words", "missing_tokens", "extra_tokens",
        "substitutions", "critical_mismatches", "repetition_details",
        "ref_normalized", "hyp_normalized", "alignment_operations",
        "accepted_equivalences", "accepted_phrase_equivalences",
        "identifier_comparisons", "error", "backend", "device",
    )
    return {
        "stage": stage,
        "chunk_id": row.get("chunk_id"),
        "original_text_raw": original,
        "asr_text_raw": transcript,
        "comparison": {key: row.get(key) for key in comparison_keys if key in row},
    }


def write_diagnostic_report(folder: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Save one portable diagnostic snapshot from current Chatterbox ASR reports.

    The snapshot does not rescore, regenerate audio, or alter stage reports.
    It simply gathers each available stage's raw source/ASR lines and the
    comparator evidence needed to explain a failure during later review.
    """
    tts_dir = resolve_tts_dir(folder)
    destination = output_path or (tts_dir / "asr_diagnostic_report.json")
    destination = destination.expanduser().resolve()
    records: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    for stage, filename in _DIAGNOSTIC_SOURCE_REPORTS:
        source = tts_dir / filename
        if not source.is_file():
            continue
        payload = _read_json(source)
        rows, _wrapper = _record_container(payload)
        sources.append({"stage": stage, "path": str(source), "rows": len(rows)})
        records.extend(_diagnostic_record(stage, row) for row in rows)
    if not sources:
        raise ValueError("No asr_stage1.json, asr_stage2.json, or asr_confirmed_failures.json found.")
    report = {
        "report_type": "chatterbox_asr_diagnostic",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tts_dir": str(tts_dir),
        "sources": sources,
        "records": records,
    }
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_report": str(destination),
        "sources": len(sources),
        "records": len(records),
    }


def build_validation_command(
    folder: Path,
    stage1_backend: str,
    stage1_model: str,
    stage2_backend: str,
    stage2_model: str,
    device: str,
    threshold: float,
) -> List[str]:
    """Build the production ASR-only command without launching a shell."""
    return [
        project_python(),
        str(_ROOT_DIR / "tools" / "run_asr_folder.py"),
        str(folder),
        "--stage1-backend",
        stage1_backend,
        "--stage1-model",
        stage1_model,
        "--stage2-backend",
        stage2_backend,
        "--stage2-model",
        stage2_model,
        "--device",
        device,
        "--threshold",
        f"{threshold:.2f}",
    ]


def backend_preflight_error(backend: str, python_executable: str) -> Optional[str]:
    """Return a missing-runtime explanation before an ASR batch can clobber reports.

    Parakeet is optional in this checkout.  Its import must succeed in the
    project Python selected for the batch child, not merely the interpreter
    that happened to launch the Tk window.
    """
    if backend != "parakeet":
        return None
    try:
        completed = subprocess.run(
            [python_executable, "-c", "import nemo.collections.asr"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception as exc:
        return (
            f"Could not preflight Parakeet with {python_executable}: "
            f"{type(exc).__name__}: {exc}"
        )
    if completed.returncode:
        detail = (completed.stderr or "NeMo import failed").strip().splitlines()[-1]
        return (
            f"Parakeet cannot run with {python_executable}: {detail}. "
            "Select faster_whisper/whisper_cpp or set CHATTERBOX_PYTHON to "
            "the Chatterbox environment containing nemo-toolkit[asr]."
        )
    return None


def load_gui_config() -> Dict[str, Any]:
    """Load persisted standalone-GUI selections, returning defaults on error."""
    try:
        payload = _read_json(_CONFIG_PATH)
        return payload if isinstance(payload, dict) else {}
    except ValueError:
        return {}


def save_gui_config(config: Dict[str, Any]) -> None:
    """Persist last-used standalone-GUI selections without failing a run."""
    try:
        _CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


class ManualRepairReviewWindow:
    """Show every WAV/text pair in a scrollable listener-review table."""

    def __init__(self, parent: tk.Misc, tts_dir: Path, rows: List[Dict[str, Any]]) -> None:
        """Create an all-chunk listener-review window for one TTS folder."""
        self.tts_dir = tts_dir
        self.rows = rows
        self.selected_chunk_ids: set[int] = set()
        self.row_heights = [self._row_height(str(row["expected_text"])) for row in rows]
        self.row_starts: List[int] = []
        next_start = 0
        for height in self.row_heights:
            self.row_starts.append(next_start)
            next_start += height
        self.total_table_height = next_start
        self.window = tk.Toplevel(parent)
        self.window.title("Manual Audio Review for Repair")
        self.window.geometry("1100x760")
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()

    def _build_ui(self) -> None:
        """Build scrollable per-chunk playback, marking, and report controls."""
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text=f"{len(self.rows)} chunks — play and mark only real failures").pack(anchor="w")

        header = ttk.Frame(outer, padding=(4, 2))
        header.pack(fill="x")
        ttk.Label(header, text="Audio", width=8).grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Repair", width=13).grid(row=0, column=1, sticky="w")
        ttk.Label(header, text="Source text").grid(row=0, column=2, sticky="w")

        table = ttk.Frame(outer)
        table.pack(fill="both", expand=True, pady=(8, 6))
        self.canvas = tk.Canvas(table, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(table, orient="vertical", command=self._scroll_canvas)
        self.canvas.configure(yscrollcommand=self._update_scrollbar, scrollregion=(0, 0, 1, self.total_table_height))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", self._resize_canvas_rows)
        self.canvas.bind("<Button-1>", self._click_row)
        self.canvas.bind("<MouseWheel>", self._mouse_wheel)
        self.canvas.bind("<Button-4>", self._mouse_wheel)
        self.canvas.bind("<Button-5>", self._mouse_wheel)
        self.window.after_idle(self._draw_visible_rows)

        controls = ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Button(controls, text="Stop playback", command=self._stop_playback).pack(side="left")
        ttk.Button(
            controls,
            text="Write Repair Report",
            command=self._write_report,
        ).pack(side="right")
        ttk.Label(outer, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", pady=(8, 0))

    @staticmethod
    def _row_height(text: str) -> int:
        """Estimate a wrapped text row height without creating an off-screen widget."""
        line_count = sum(max(1, (len(line) + 104) // 105) for line in text.splitlines() or [""])
        return max(42, 12 + (line_count * 18))

    def _update_scrollbar(self, first: str, last: str) -> None:
        """Update scrollbar position and redraw only rows visible in the canvas."""
        self.scrollbar.set(first, last)
        self.window.after_idle(self._draw_visible_rows)

    def _scroll_canvas(self, *args: str) -> None:
        """Apply scrollbar movement then repaint the small visible row window."""
        self.canvas.yview(*args)
        self._draw_visible_rows()

    def _resize_canvas_rows(self, _event: tk.Event) -> None:
        """Repaint virtual rows after the canvas viewport width or height changes."""
        self._draw_visible_rows()

    def _draw_visible_rows(self) -> None:
        """Render only viewport rows, avoiding thousands of Tk button pixmaps."""
        if not self.rows or self.canvas.winfo_width() <= 1:
            return
        top = int(self.canvas.canvasy(0))
        bottom = top + self.canvas.winfo_height()
        first = max(0, bisect_right(self.row_starts, top) - 1)
        last = min(len(self.rows), bisect_right(self.row_starts, bottom) + 1)
        width = max(1, self.canvas.winfo_width())
        self.canvas.delete("review-row")
        for index in range(first, last):
            row = self.rows[index]
            chunk_id = int(row["chunk_id"])
            y = self.row_starts[index]
            height = self.row_heights[index]
            fill = "#edf4ff" if index % 2 == 0 else "#ffffff"
            self.canvas.create_rectangle(0, y, width, y + height, fill=fill, outline="#d0d0d0", tags="review-row")
            self.canvas.create_rectangle(6, y + 9, 64, y + 33, fill="#e6e6e6", outline="#777777", tags="review-row")
            self.canvas.create_text(35, y + 21, text="Play", tags="review-row")
            checked = chunk_id in self.selected_chunk_ids
            self.canvas.create_rectangle(78, y + 11, 94, y + 27, fill="#ffffff", outline="#555555", tags="review-row")
            if checked:
                self.canvas.create_text(86, y + 19, text="X", tags="review-row")
            self.canvas.create_text(103, y + 19, text=f"{chunk_id:05d}", anchor="w", tags="review-row")
            self.canvas.create_text(
                160,
                y + 7,
                text=str(row["expected_text"]),
                anchor="nw",
                justify="left",
                width=max(150, width - 172),
                tags="review-row",
            )

    def _click_row(self, event: tk.Event) -> None:
        """Play or mark the virtual table row whose canvas control was clicked."""
        y = int(self.canvas.canvasy(event.y))
        index = bisect_right(self.row_starts, y) - 1
        if index < 0 or index >= len(self.rows) or y >= self.row_starts[index] + self.row_heights[index]:
            return
        x = int(self.canvas.canvasx(event.x))
        row = self.rows[index]
        if 6 <= x <= 64:
            self._play_row(row)
        elif 78 <= x <= 145:
            self._toggle_mark(int(row["chunk_id"]))

    def _mouse_wheel(self, event: tk.Event) -> str:
        """Scroll virtual rows with Linux wheel buttons or standard wheel deltas."""
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            self.canvas.yview_scroll(-3, "units")
        else:
            self.canvas.yview_scroll(3, "units")
        self._draw_visible_rows()
        return "break"

    def _toggle_mark(self, chunk_id: int) -> None:
        """Toggle one virtual table checkbox and refresh its visible marker."""
        if chunk_id in self.selected_chunk_ids:
            self.selected_chunk_ids.discard(chunk_id)
        else:
            self.selected_chunk_ids.add(chunk_id)
        self.status_var.set(f"Marked for Repair: {len(self.selected_chunk_ids)}")
        self._draw_visible_rows()

    def _play_row(self, row: Dict[str, Any]) -> None:
        """Play one table row's WAV with pygame from the project runtime."""
        wav_path = Path(str(row["wav_path"]))
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.music.load(str(wav_path))
            pygame.mixer.music.play()
            self.status_var.set(f"Playing {wav_path.name}")
        except Exception as exc:
            self.status_var.set(f"Playback failed: {exc}")
            messagebox.showerror("Playback failed", f"Could not play {wav_path.name}:\n{exc}", parent=self.window)

    def _stop_playback(self) -> None:
        """Stop active pygame playback without affecting the listener selection."""
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
        except Exception:
            pass
        self.status_var.set("Playback stopped")

    def _write_report(self) -> None:
        """Write current marked chunks to the Repair-loadable selection report."""
        try:
            summary = write_manual_repair_report(self.tts_dir, self.rows, self.selected_chunk_ids)
            self.status_var.set(
                f"Saved {summary['selected']} marked chunk(s): {Path(summary['output_report']).name}"
            )
            messagebox.showinfo(
                "Repair report saved",
                f"Saved {summary['selected']} chunks to\n{summary['output_report']}",
                parent=self.window,
            )
        except Exception as exc:
            self.status_var.set(f"Report save failed: {exc}")
            messagebox.showerror("Report save failed", str(exc), parent=self.window)


class AsrGuiApp:
    """Provide a small Tk front end for production validation and report replay."""

    def __init__(self, root: tk.Tk) -> None:
        """Create controls, restore selections, and start GUI log polling."""
        self.root = root
        self.root.title("Chatterbox ASR Validator")
        self.root.geometry("900x650")
        saved = load_gui_config()
        self.events: queue.Queue[Tuple[str, str]] = queue.Queue()
        self.running = False
        self.manual_review_windows: List[ManualRepairReviewWindow] = []
        self.book_var = tk.StringVar(value=str(saved.get("folder", "")))
        self.report_var = tk.StringVar(value=str(saved.get("report", "")))
        self.stage1_backend_var = tk.StringVar(value=str(saved.get("stage1_backend", "faster_whisper")))
        self.stage1_model_var = tk.StringVar(value=str(saved.get("stage1_model", "base")))
        self.stage2_backend_var = tk.StringVar(value=str(saved.get("stage2_backend", "faster_whisper")))
        self.stage2_model_var = tk.StringVar(value=str(saved.get("stage2_model", "medium")))
        self.device_var = tk.StringVar(value=str(saved.get("device", "cuda")))
        self.threshold_var = tk.DoubleVar(value=float(saved.get("threshold", 0.65)))
        self.status_var = tk.StringVar(value="Ready")
        self._build_ui()
        self.root.after(125, self._poll_events)

    def _build_ui(self) -> None:
        """Build selection controls, actions, status area, and scrolling log."""
        folder = ttk.LabelFrame(self.root, text="Book / TTS folder", padding=8)
        folder.pack(fill="x", padx=10, pady=(10, 4))
        ttk.Entry(folder, textvariable=self.book_var).pack(side="left", fill="x", expand=True)
        ttk.Button(folder, text="Browse…", command=self._browse_folder).pack(side="left", padx=(6, 0))

        settings = ttk.LabelFrame(self.root, text="Production ASR validation", padding=8)
        settings.pack(fill="x", padx=10, pady=4)
        self._combo(settings, "Stage 1 backend", self.stage1_backend_var, _BACKENDS, 0, 0)
        self._combo(settings, "Stage 1 model", self.stage1_model_var, _MODELS, 0, 2)
        self._combo(settings, "Stage 2 backend", self.stage2_backend_var, _STAGE_TWO_BACKENDS, 1, 0)
        self._combo(settings, "Stage 2 model", self.stage2_model_var, ("Disabled", *_MODELS), 1, 2)
        self._combo(settings, "Device", self.device_var, ("cuda", "cpu"), 2, 0)
        ttk.Label(settings, text="Threshold").grid(row=2, column=2, sticky="w", padx=(18, 4), pady=3)
        ttk.Spinbox(settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=8).grid(row=2, column=3, sticky="w", pady=3)

        actions = ttk.Frame(self.root, padding=4)
        actions.pack(fill="x", padx=10)
        self.validate_button = ttk.Button(actions, text="Run Audio Validation", command=self._start_validation)
        self.validate_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.diagnostic_button = ttk.Button(
            actions,
            text="Save Diagnostic Report",
            command=self._start_diagnostic_report,
        )
        self.diagnostic_button.pack(side="left")
        self.review_button = ttk.Button(
            actions,
            text="Manual Review for Repair",
            command=self._open_manual_review,
        )
        self.review_button.pack(side="left", padx=(6, 0))

        report = ttk.LabelFrame(self.root, text="Test comparator changes against saved report", padding=8)
        report.pack(fill="x", padx=10, pady=4)
        ttk.Entry(report, textvariable=self.report_var).pack(side="left", fill="x", expand=True)
        ttk.Button(report, text="Browse…", command=self._browse_report).pack(side="left", padx=(6, 0))
        self.rescore_button = ttk.Button(report, text="Rescore Saved Report", command=self._start_rescore)
        self.rescore_button.pack(side="left", padx=(6, 0))

        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(fill="x", padx=10, pady=4)
        log_frame = ttk.LabelFrame(self.root, text="Run log", padding=6)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log = tk.Text(log_frame, wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _combo(
        self,
        parent: ttk.LabelFrame,
        label: str,
        variable: tk.StringVar,
        values: Iterable[str],
        row: int,
        column: int,
    ) -> None:
        """Place one labeled readonly selector in the validation settings grid."""
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0 if column == 0 else 18, 4), pady=3)
        ttk.Combobox(parent, textvariable=variable, values=tuple(values), state="readonly", width=20).grid(row=row, column=column + 1, sticky="w", pady=3)

    def _browse_folder(self) -> None:
        """Choose a Chatterbox book or TTS folder for audio validation."""
        selected = filedialog.askdirectory(title="Select Chatterbox book or TTS folder")
        if selected:
            self.book_var.set(selected)

    def _browse_report(self) -> None:
        """Choose a saved JSON report to replay through current comparator code."""
        selected = filedialog.askopenfilename(
            title="Select ASR JSON report",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self.report_var.set(selected)

    def _open_manual_review(self) -> None:
        """Open one sequential WAV/text listener review for the selected TTS folder."""
        try:
            tts_dir = resolve_tts_dir(Path(self.book_var.get()))
            rows = load_manual_review_rows(tts_dir)
        except (OSError, TypeError, ValueError) as exc:
            messagebox.showerror("Manual review unavailable", str(exc), parent=self.root)
            return
        review = ManualRepairReviewWindow(self.root, tts_dir, rows)
        self.manual_review_windows.append(review)
        self.status_var.set(f"Manual review loaded: {len(rows)} chunks")

    def _save_selections(self) -> None:
        """Persist current controls before a background operation starts."""
        save_gui_config({
            "folder": self.book_var.get(),
            "report": self.report_var.get(),
            "stage1_backend": self.stage1_backend_var.get(),
            "stage1_model": self.stage1_model_var.get(),
            "stage2_backend": self.stage2_backend_var.get(),
            "stage2_model": self.stage2_model_var.get(),
            "device": self.device_var.get(),
            "threshold": self.threshold_var.get(),
        })

    def _set_running(self, running: bool, status: str) -> None:
        """Toggle actions while a worker runs and publish its current status."""
        self.running = running
        state = "disabled" if running else "normal"
        self.validate_button.configure(state=state)
        self.rescore_button.configure(state=state)
        self.diagnostic_button.configure(state=state)
        self.review_button.configure(state=state)
        self.status_var.set(status)

    def _start_validation(self) -> None:
        """Validate settings and launch production audio validation in a thread."""
        try:
            tts_dir = resolve_tts_dir(Path(self.book_var.get()))
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid folder", str(exc), parent=self.root)
            return
        command = build_validation_command(
            tts_dir,
            self.stage1_backend_var.get(),
            self.stage1_model_var.get(),
            self.stage2_backend_var.get(),
            self.stage2_model_var.get(),
            self.device_var.get(),
            float(self.threshold_var.get()),
        )
        preflight_error = backend_preflight_error(self.stage1_backend_var.get(), command[0])
        if preflight_error:
            self._append_log(preflight_error)
            messagebox.showerror("ASR backend unavailable", preflight_error, parent=self.root)
            return
        self._save_selections()
        self._set_running(True, "Running production ASR validation…")
        threading.Thread(target=self._run_validation, args=(command,), daemon=True).start()

    def _run_validation(self, command: List[str]) -> None:
        """Run the production CLI and stream combined output back to the Tk thread."""
        self.events.put(("log", "$ " + " ".join(command)))
        try:
            process = subprocess.Popen(
                command,
                cwd=_ROOT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("log", line.rstrip()))
            status = "Validation complete" if process.wait() == 0 else "Validation failed; inspect log"
        except OSError as exc:
            status = f"Validation launch failed: {exc}"
        self.events.put(("done", status))

    def _start_rescore(self) -> None:
        """Validate selected report and replay it in a worker without audio work."""
        report = Path(self.report_var.get()).expanduser()
        if not report.is_file():
            messagebox.showerror("Invalid report", "Select an existing ASR JSON report.", parent=self.root)
            return
        self._save_selections()
        self._set_running(True, "Rescoring saved transcripts…")
        threading.Thread(target=self._run_rescore, args=(report,), daemon=True).start()

    def _run_rescore(self, report: Path) -> None:
        """Replay one report and send its summary or error back to the Tk thread."""
        try:
            summary = rescore_report(report)
            self.events.put(("log", json.dumps(summary, indent=2)))
            status = f"Rescore complete: {summary['passed']}/{summary['scored']} passed"
        except Exception as exc:
            status = f"Rescore failed: {exc}"
        self.events.put(("done", status))

    def _start_diagnostic_report(self) -> None:
        """Save current stage evidence without rerunning ASR or changing audio."""
        try:
            tts_dir = resolve_tts_dir(Path(self.book_var.get()))
        except (TypeError, ValueError) as exc:
            messagebox.showerror("Invalid folder", str(exc), parent=self.root)
            return
        self._save_selections()
        self._set_running(True, "Saving ASR diagnostic report…")
        threading.Thread(target=self._run_diagnostic_report, args=(tts_dir,), daemon=True).start()

    def _run_diagnostic_report(self, tts_dir: Path) -> None:
        """Write diagnostic snapshot and return its path to the Tk event queue."""
        try:
            summary = write_diagnostic_report(tts_dir)
            self.events.put(("log", json.dumps(summary, indent=2)))
            status = f"Diagnostic report saved: {Path(summary['output_report']).name}"
        except Exception as exc:
            status = f"Diagnostic report failed: {exc}"
        self.events.put(("done", status))

    def _poll_events(self) -> None:
        """Render worker events on the Tk thread and schedule the next poll."""
        try:
            while True:
                kind, message = self.events.get_nowait()
                if kind == "log":
                    self._append_log(message)
                elif kind == "done":
                    self._set_running(False, message)
        except queue.Empty:
            pass
        self.root.after(125, self._poll_events)

    def _append_log(self, message: str) -> None:
        """Append one status line while keeping the read-only log widget safe."""
        self.log.configure(state="normal")
        self.log.insert("end", message + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parsing for headless saved-report replays and GUI startup."""
    parser = argparse.ArgumentParser(description="Standalone Chatterbox ASR validation and report-rescore tool.")
    parser.add_argument("--rescore-report", type=Path, help="Replay one saved JSON report using current spoken_compare.py")
    parser.add_argument("--output", type=Path, help="Output JSON path for --rescore-report; defaults beside source")
    parser.add_argument("--no-gui", action="store_true", help="Require --rescore-report and do not start Tk")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run headless rescore mode when requested, otherwise open the Tk tool."""
    args = _build_parser().parse_args(argv)
    if args.rescore_report:
        try:
            print(json.dumps(rescore_report(args.rescore_report, args.output), indent=2))
            return 0
        except Exception as exc:
            print(f"ASR rescore failed: {exc}", file=sys.stderr)
            return 1
    if args.no_gui:
        print("--no-gui requires --rescore-report", file=sys.stderr)
        return 2
    root = tk.Tk()
    AsrGuiApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
