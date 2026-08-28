"""Persistent GUI conversion queue and append-only batch-run reporting."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


QUEUE_FILENAME = "batch_queue.json"
REPORT_FILENAME = "batch_run_report.txt"


def load_queue(audiobook_root: Path) -> list[dict[str, Any]]:
    """Load queued conversion snapshots without silently accepting bad data."""
    path = Path(audiobook_root) / QUEUE_FILENAME
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(job, dict) for job in data):
        raise ValueError(f"Invalid batch queue format: {path}")
    return data


def save_queue(audiobook_root: Path, jobs: list[dict[str, Any]]) -> Path:
    """Atomically persist ordered queue snapshots under Audiobook root."""
    root = Path(audiobook_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / QUEUE_FILENAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(jobs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def parse_timestamped_run_log(path: Path | None) -> dict[str, str]:
    """Extract completed-run measurements from the engine's timestamped log."""
    if path is None or not path.exists():
        return {}

    fields: dict[str, str] = {}
    active_asr_stage = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        key, value = (part.strip() for part in raw_line.split(":", 1))
        if key == "ASR Stage 1":
            active_asr_stage = "stage1"
            fields["asr_stage1"] = value
        elif key == "ASR Stage 2":
            active_asr_stage = "stage2"
            fields["asr_stage2"] = value
        elif key == "Fails" and active_asr_stage:
            fields[f"{active_asr_stage}_fails"] = value
        else:
            fields[key.lower().replace(" ", "_")] = value
    return fields


def parse_asr_summary(path: Path) -> dict[str, str]:
    """Extract final regeneration failure count from the current ASR summary."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    still_fails = data.get("regen_still_failed")
    return {"still_fails": str(still_fails)} if still_fails is not None else {}


def append_batch_report(
    audiobook_root: Path,
    job: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    success: bool,
    metrics: dict[str, str] | None = None,
    error: str = "",
) -> Path:
    """Append one completed or failed queue job to the permanent text report."""
    root = Path(audiobook_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / REPORT_FILENAME
    run_number = _next_run_number(path)
    metrics = metrics or {}
    tts = dict(job.get("tts_params") or {})
    quality = dict(job.get("quality_params") or {})
    text_path = Path(str(job.get("text_file", "")))
    voice_path = Path(str(job.get("voice_path", "")))
    stage1 = f"{quality.get('asr_stage1_backend', 'faster_whisper')} {quality.get('asr_stage1_model', 'base')}"
    stage2_model = quality.get("asr_stage2_model", "medium")
    stage2 = "Disabled" if str(stage2_model).lower() == "disabled" else f"{quality.get('asr_stage2_backend', 'faster_whisper')} {stage2_model}"
    lines = [
        f"Run #{run_number}",
        f"Batch Job: {job.get('id', '')}",
        f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Finished: {finished_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Book Folder: {job.get('book_dir', '')}",
        f"Text Input Path: {text_path}",
        f"Text Input Filename: {text_path.name}",
        f"Voice Path: {voice_path}",
        f"Voice Filename: {voice_path.name}",
        "",
        f"T3: {_t3_label(tts.get('t3_source'))}",
        f"S3: {_s3_label(tts.get('s3gen_decoder'))}",
        f"Exaggeration: {tts.get('exaggeration', '')}",
        f"Temperature: {tts.get('temperature', '')}",
        f"CFG: {tts.get('cfg_weight', '')}",
        f"Min-P: {tts.get('min_p', '')}",
        f"Top-P: {tts.get('top_p', '')}",
        f"Repetition Penalty: {tts.get('repetition_penalty', '')}",
        f"Stage 1: {stage1}",
        f"Stage 2: {stage2}",
        "",
        f"Phase 1: {_value(metrics, 'phase_1_time')}",
        f"Phase 2: {_value(metrics, 'phase_2_time')}",
        f"ASR Stage 1: {_value(metrics, 'asr_stage1')}",
        f"ASR Stage 2: {_value(metrics, 'asr_stage2')}",
        f"Elapsed Time: {_value(metrics, 'elapsed_time')}",
        f"Realtime RAW: {_value(metrics, 'realtime_raw')}",
        "",
        f"Total Elapsed: {_value(metrics, 'total_elapsed')}",
        f"Audio Duration: {_value(metrics, 'audio_duration')}",
        f"Realtime Total: {_value(metrics, 'realtime_total')}",
        "",
        f"Stage 1 fails: {_value(metrics, 'stage1_fails')}",
        f"Stage 2 fails: {_value(metrics, 'stage2_fails')}",
        f"Still fails: {_value(metrics, 'still_fails')}",
        f"Status: {'Completed' if success else 'Failed'}",
    ]
    if error:
        lines.append(f"Error: {error}")
    with path.open("a", encoding="utf-8") as report:
        report.write("\n".join(lines) + "\n\n________________________________________\n\n")
    return path


def _next_run_number(report_path: Path) -> int:
    """Return next monotonic report run number without changing prior entries."""
    if not report_path.exists():
        return 1
    matches = re.findall(r"^Run #(\d+)$", report_path.read_text(encoding="utf-8"), re.MULTILINE)
    return max((int(value) for value in matches), default=0) + 1


def _value(metrics: dict[str, str], key: str) -> str:
    """Return recorded metric or stable placeholder when a job failed early."""
    return str(metrics.get(key) or "N/A")


def _t3_label(source: Any) -> str:
    """Convert stored T3 identifier into the concise report label."""
    return {
        "multilingual-v2": "Multi V2",
        "multilingual-v3": "Multi V3",
        "english": "English",
    }.get(str(source), str(source or "N/A"))


def _s3_label(decoder: Any) -> str:
    """Convert stored S3 identifier into the concise report label."""
    return {"turbo": "Turbo", "standard": "Standard"}.get(
        str(decoder), str(decoder or "N/A")
    )
