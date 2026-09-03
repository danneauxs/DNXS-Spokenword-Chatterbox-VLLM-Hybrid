"""Stage 1 / Stage 2 Whisper model ladder for two-stage ASR verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STAGE_TWO_DISABLED = "disabled"
KNOWN_ASR_PASSES_FILENAME = "asr_known_passes.json"
ACCEPTED_ASR_FUZZIES_FILENAME = "asr_accepted_fuzzies.json"

BACKEND_FASTER_WHISPER = "faster_whisper"
BACKEND_WHISPER_CPP = "whisper_cpp"
BACKEND_PARAKEET = "parakeet"
PARAKEET_MODEL = "parakeet-tdt-0.6b-v3"
STAGE1_BACKENDS = (BACKEND_FASTER_WHISPER, BACKEND_WHISPER_CPP, BACKEND_PARAKEET)
STAGE2_BACKENDS = (BACKEND_FASTER_WHISPER, BACKEND_WHISPER_CPP)

_BACKEND_ALIASES = {
    "faster_whisper": BACKEND_FASTER_WHISPER,
    "faster-whisper": BACKEND_FASTER_WHISPER,
    "whisper": BACKEND_FASTER_WHISPER,
    "whisper_cpp": BACKEND_WHISPER_CPP,
    "whisper-cpp": BACKEND_WHISPER_CPP,
    "cpp": BACKEND_WHISPER_CPP,
    "parakeet": BACKEND_PARAKEET,
    "parakeet_tdt": BACKEND_PARAKEET,
    "nemo": BACKEND_PARAKEET,
    "nemo_parakeet": BACKEND_PARAKEET,
}

_MODEL_RANKS = {
    "tiny": 0,
    "base": 1,
    "small": 2,
    "distil-small.en": 2,
    "medium": 3,
    "distil-medium.en": 3,
    "large": 4,
    "large-v2": 4,
    "large-v3": 4,
    "large-v3-turbo": 5,
    "distil-large-v3": 5,
    "parakeet-tdt-0.6b-v3": 1,
    "parakeet": 1,
}

_NEXT_HIGHER = {
    "tiny": "base",
    "base": "small",
    "small": "medium",
    "medium": "large-v3",
    "large": "large-v3-turbo",
    "large-v2": "large-v3-turbo",
    "large-v3": "large-v3-turbo",
    "distil-small.en": "distil-medium.en",
    "distil-medium.en": "medium",
}


def normalize_backend(name: str | None) -> str:
    """Return a canonical ASR backend id.

    Args:
        name: GUI or config backend string.

    Returns:
        faster_whisper, whisper_cpp, or parakeet. Unknown values become faster_whisper.
    """
    token = str(name or "").strip().lower().replace(" ", "_").replace(".", "_")
    return _BACKEND_ALIASES.get(token, BACKEND_FASTER_WHISPER)


def is_parakeet_backend(name: str | None) -> bool:
    """Return True when this backend is NVIDIA Parakeet TDT."""
    return normalize_backend(name) == BACKEND_PARAKEET


def backend_missing_reason(name: str | None) -> Optional[str]:
    """Return a human error if this ASR engine is not importable, else None.

    Args:
        name: GUI or config backend id.

    Returns:
        None when the engine can load; otherwise an install hint.
    """
    kind = normalize_backend(name)
    if kind == BACKEND_FASTER_WHISPER:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return "faster-whisper is not installed in this venv."
        return None
    if kind == BACKEND_WHISPER_CPP:
        try:
            import pywhispercpp  # noqa: F401
        except ImportError:
            return (
                "whisper.cpp needs pywhispercpp in this venv. "
                "CPU: pip install pywhispercpp. "
                "GPU: bash ASR/install_pywhispercpp_cuda.sh"
            )
        return None
    if kind == BACKEND_PARAKEET:
        asr_dir = Path(__file__).resolve().parent.parent / "ASR"
        import sys

        if str(asr_dir) not in sys.path:
            sys.path.insert(0, str(asr_dir))
        from parakeet_backend import parakeet_is_available

        if not parakeet_is_available():
            return "Parakeet needs nemo-toolkit[asr] in this venv."
        return None
    return None


def backend_is_available(name: str | None) -> bool:
    """Return True when this ASR engine can be imported in the current venv."""
    return backend_missing_reason(name) is None


def _cpp_cuda_runtime_ready() -> bool:
    """Return whether whisper.cpp can actually start CUDA workers in this venv.

    The UI can request CUDA while a CPU-only pywhispercpp wheel is installed.
    In that case worker sizing must remain CPU-parallel instead of applying the
    more conservative multi-model CUDA cap before the daemon has loaded.

    Returns:
        True only when the GGML CUDA bundle and CUDA runtime are both available.
    """
    try:
        asr_dir = Path(__file__).resolve().parent.parent / "ASR"
        import sys

        if str(asr_dir) not in sys.path:
            sys.path.insert(0, str(asr_dir))
        from whisper_cpp_backend import whisper_cpp_cuda_bundle_available
        import torch

        return whisper_cpp_cuda_bundle_available() and torch.cuda.is_available()
    except (ImportError, OSError, RuntimeError):
        return False


def daemon_worker_count(
    backend: str | None,
    requested: int,
    device: str,
    model_size: str | None = None,
) -> int:
    """Clamp ASR worker processes for the selected engine.

    Parakeet is one resident GPU model. A usable whisper.cpp CUDA deployment
    keeps at most two copies so VRAM is not filled with ggml clones. Medium and
    larger faster-whisper CUDA models use one process-model: four concurrent
    model loads have been observed to OOM and silently fall back to CPU. A CUDA
    request that will resolve to CPU remains CPU-parallel.

    Args:
        backend: Canonical or alias backend name.
        requested: GUI/config worker request.
        device: cpu or cuda.
        model_size: Selected model name, used for faster-whisper CUDA sizing.

    Returns:
        Worker count >= 1.
    """
    wanted = max(1, int(requested or 1))
    kind = normalize_backend(backend)
    if kind == BACKEND_PARAKEET:
        return 1
    if (
        kind == BACKEND_FASTER_WHISPER
        and str(device).lower() in {"cuda", "gpu"}
        and model_rank(normalize_model_name(model_size or "base")) >= _MODEL_RANKS["medium"]
    ):
        return 1
    if kind == BACKEND_WHISPER_CPP and str(device).lower() in {"cuda", "gpu"}:
        return min(wanted, 2) if _cpp_cuda_runtime_ready() else wanted
    return wanted


def normalize_model_name(model_name: str) -> str:
    """Return a normalized ASR model token for comparisons and reporting."""
    return (model_name or "").strip().lower().replace("_", "-")


def is_stage_two_disabled(model_name: str | None) -> bool:
    """Return True when Stage 2 is the Disabled sentinel or an empty value."""
    token = normalize_model_name(model_name or "")
    return token in {"", "disabled", "none", "off"}


def model_rank(model_name: str) -> int:
    """Rank ASR model size on the Stage 1/2 ladder."""
    normalized = normalize_model_name(model_name)
    if is_stage_two_disabled(normalized):
        return -1
    return _MODEL_RANKS.get(
        normalized, 3 if "medium" in normalized or "large" in normalized else 1
    )


def next_higher(model_name: str) -> Optional[str]:
    """Return the next-larger model on the locked ladder, or None at the top.

    Args:
        model_name: Stage 1 or Stage 2 model token.

    Returns:
        The next model name, or None when no larger verifier exists.
    """
    key = normalize_model_name(model_name)
    if key in {"large-v3-turbo", "distil-large-v3"}:
        return None
    return _NEXT_HIGHER.get(key)


def model_is_higher(candidate: str, baseline: str) -> bool:
    """Return True when candidate is strictly larger than baseline."""
    if is_stage_two_disabled(candidate) or is_stage_two_disabled(baseline):
        return False
    return model_rank(candidate) > model_rank(baseline)


def recommended_stage_two_model(stage_one_model: str) -> str:
    """Return the default Stage 2 Whisper model for a Stage 1 selection.

    Tiny/base/small jump to medium. Stage 1 at medium or above takes one
    ladder step, or Disabled at the top.

    Args:
        stage_one_model: Selected Stage 1 Whisper model.

    Returns:
        A Whisper model name, or ``disabled`` when no larger Stage 2 exists.
    """
    key = normalize_model_name(stage_one_model)
    if key in {"large-v3-turbo", "distil-large-v3"}:
        return STAGE_TWO_DISABLED
    if key == "distil-small.en":
        return "distil-medium.en"
    if model_rank(key) < _MODEL_RANKS["medium"]:
        return "medium"
    nxt = next_higher(key)
    if nxt is None:
        return STAGE_TWO_DISABLED
    if not model_is_higher(nxt, key):
        return "large-v3"
    return nxt


def _known_pass_text_hash(text: str) -> str:
    """Return a whitespace-stable fingerprint for an approved source chunk."""
    normalized = " ".join(str(text or "").casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_known_asr_passes(tts_dir: Path) -> Dict[Tuple[int, str], str]:
    """Load text-validated manual ASR pass overrides from one book's TTS folder.

    Overrides are scoped to both a chunk id and a normalized source-text hash.
    A changed chunk at the same id therefore cannot inherit an old approval.

    Args:
        tts_dir: Book TTS directory that may contain asr_known_passes.json.

    Returns:
        Mapping of (chunk id, source-text hash) to a human-readable reason.
    """
    path = Path(tts_dir) / KNOWN_ASR_PASSES_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("passes", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return {}
    known: Dict[Tuple[int, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            chunk_id = int(entry["chunk_id"])
        except (KeyError, TypeError, ValueError):
            continue
        text_hash = str(entry.get("text_sha256") or "")
        if len(text_hash) != 64:
            continue
        known[(chunk_id, text_hash)] = str(entry.get("reason") or "manual no-fail review")
    return known


def known_asr_pass_reason(
    known_passes: Dict[Tuple[int, str], str], chunk_id: int | str, text: str
) -> Optional[str]:
    """Return an approved-pass reason only when chunk id and source text both match."""
    try:
        key = (int(chunk_id), _known_pass_text_hash(text))
    except (TypeError, ValueError):
        return None
    return known_passes.get(key)


def load_accepted_asr_fuzzies(tts_dir: Path) -> Dict[Tuple[int, str], str]:
    """Load text-validated fuzzy approvals that suppress regeneration only.

    Each approval is scoped to a chunk id and normalized source-text hash. Unlike
    ``asr_known_passes.json``, these entries do not change an ASR failure into a
    pass; they only prevent a listener-approved, context-repaired near-homophone
    from advancing to Stage 2 or regeneration.

    Args:
        tts_dir: Book TTS directory that may contain asr_accepted_fuzzies.json.

    Returns:
        Mapping of (chunk id, source-text hash) to a human-readable reason.
    """
    path = Path(tts_dir) / ACCEPTED_ASR_FUZZIES_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    entries = payload.get("accepted_fuzzies", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        return {}
    accepted: Dict[Tuple[int, str], str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            chunk_id = int(entry["chunk_id"])
        except (KeyError, TypeError, ValueError):
            continue
        text_hash = str(entry.get("text_sha256") or "")
        if len(text_hash) != 64:
            continue
        accepted[(chunk_id, text_hash)] = str(
            entry.get("reason") or "listener-approved context-repaired fuzzy"
        )
    return accepted


def accepted_asr_fuzzy_reason(
    accepted_fuzzies: Dict[Tuple[int, str], str], chunk_id: int | str, text: str
) -> Optional[str]:
    """Return a fuzzy-approval reason only when chunk id and source text match."""
    try:
        key = (int(chunk_id), _known_pass_text_hash(text))
    except (TypeError, ValueError):
        return None
    return accepted_fuzzies.get(key)


# Repair-tab load order keeps canonical one-row-per-chunk reports ahead of
# diagnostic/full-stage files, while still exposing every available source.
ASR_FAILURE_REPORTS = (
    ("asr_failed_regenerations.json", "Failed regenerations"),
    ("asr_investigation_failures.json", "Investigation failures"),
    ("asr_conservative_fuzzy.json", "Conservative fuzzy review"),
    ("asr_false_positives.json", "ASR false positives"),
    ("asr_near_matches.json", "ASR near matches"),
    ("asr_significant_mismatches.json", "ASR significant mismatches"),
    ("asr_manual_repair_selection.json", "Manual audio review selection"),
    ("asr_confirmed_failures.json", "Stage 2 confirmed failures"),
    ("asr_remaining_failures.json", "Still failed after regen"),
    ("asr_stage2.json", "Stage 2 report (failed rows)"),
    ("asr_stage1_failures.json", "Stage 1 failures"),
)


def discover_asr_failure_reports(tts_dir: Path) -> List[Tuple[Path, str]]:
    """Return existing ASR failure JSON files under a book's TTS folder.

    Args:
        tts_dir: Book TTS directory (parent of audio_chunks/).

    Returns:
        List of (path, label) in preferred load order.
    """
    tts_dir = Path(tts_dir)
    found: List[Tuple[Path, str]] = []
    for filename, label in ASR_FAILURE_REPORTS:
        path = tts_dir / filename
        if path.exists() and path.stat().st_size > 2:
            found.append((path, label))
    return found


def _chunk_id_from_row(row: Dict[str, Any]) -> Optional[int]:
    """Return a 0-based chunk index from a failure-report row."""
    for key in ("chunk_id", "index", "chunk_index"):
        if key in row and row[key] is not None:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                continue
    return None


def parse_asr_failure_rows(path: Path) -> List[Dict[str, Any]]:
    """Load normalized failed-chunk rows from any supported ASR report.

    Stage 2 full reports include passing rows; those are dropped. Canonical
    regeneration and investigation reports already contain failed chunks, and
    duplicate chunk ids are collapsed so attempt-level data cannot duplicate
    Repair Tool entries.

    Args:
        path: JSON file written by Phase 3.

    Returns:
        List of dicts with at least chunk_id, plus score/text when present.

    Raises:
        ValueError: File is not a JSON list of objects.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"ASR failure report must be a JSON list: {path.name}")
    failed: List[Dict[str, Any]] = []
    seen_ids = set()
    drop_passes = path.name == "asr_stage2.json"
    for item in payload:
        if not isinstance(item, dict):
            continue
        if drop_passes and item.get("passed") is True:
            continue
        chunk_id = _chunk_id_from_row(item)
        if chunk_id is None:
            continue
        if chunk_id in seen_ids:
            continue
        row = dict(item)
        row["chunk_id"] = chunk_id
        failed.append(row)
        seen_ids.add(chunk_id)
    return failed
