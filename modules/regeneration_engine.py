from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.config import (
    MAX_REGENERATION_ATTEMPTS,
    REGEN_TEMPERATURE_ADJUSTMENT,
    REGEN_EXAGGERATION_ADJUSTMENT,
    TTS_PARAM_MIN_TEMPERATURE,
    TTS_PARAM_MIN_EXAGGERATION,
)
from modules.audio_processor import evaluate_chunk_quality

logger = logging.getLogger(__name__)


def compute_composite_score(local_score: Optional[float], asr_result: Optional[dict]) -> float:
    """Averages an already-computed local quality score (MFCC/spectral/VAD, from
    audio_processor.evaluate_chunk_quality) with the ASR daemon's similarity
    score, when both are available; falls back to whichever exists.

    Takes the local score as a plain float rather than re-scoring audio here, so
    callers that already computed it once (decode_tokens does, inline, during
    Phase 2) never redo that work.

    Args:
        local_score: Pre-computed local composite score, or None to skip it.
        asr_result: Daemon result dict with a "score" key, or None/error to skip.

    Returns:
        Composite score in 0.0-1.0; 1.0 if nothing was available to check.
    """
    scores = []
    if local_score is not None:
        scores.append(local_score)
    if asr_result is not None and not asr_result.get("error"):
        scores.append(asr_result.get("score", 0.8))
    # float() here: evaluate_chunk_quality's sub-scores (e.g. librosa/MFCC via
    # detect_spectral_artifacts) can return numpy.float32, which json.dumps
    # can't serialize later when this score lands in the regeneration report.
    return float(sum(scores) / len(scores)) if scores else 1.0


def _adjust_params_for_attempt(base_params: dict, attempt_num: int) -> dict:
    """Progressively lowers temperature/exaggeration for later retry attempts.

    cfg_weight is deliberately left untouched -- it can only be set once per
    vLLM engine load (see Phase 2's cfg_weight fix), so varying it per-attempt
    within the same loaded engine would have no effect.

    Args:
        base_params: The chunk's original tts_params.
        attempt_num: 1-indexed retry number (1 = first retry after the original).

    Returns:
        A new params dict with temperature/exaggeration adjusted, cfg_weight removed.
    """
    adjusted = dict(base_params)
    adjusted["temperature"] = max(
        TTS_PARAM_MIN_TEMPERATURE,
        base_params.get("temperature", 0.8) - REGEN_TEMPERATURE_ADJUSTMENT * attempt_num,
    )
    adjusted["exaggeration"] = max(
        TTS_PARAM_MIN_EXAGGERATION,
        base_params.get("exaggeration", 0.5) - REGEN_EXAGGERATION_ADJUSTMENT * attempt_num,
    )
    adjusted.pop("cfg_weight", None)
    return adjusted


def regenerate_failed_chunks(
    failed: List[dict],
    cond_emb,
    ckpt_dir,
    device: str,
    variant: str,
    decoder_type: str,
    voice_path,
    turbo_ckpt_dir,
    audio_output_dir: Path,
    asr_client,
    asr_threshold: float,
    local_scoring_enabled: bool = True,
    max_attempts: Optional[int] = None,
    report_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
) -> Dict:
    """Regenerates each failed chunk, keeping the max-scoring attempt (including
    the original). Reloads the T3 engine once for ALL failed chunks' retries,
    then reloads the decoder once for ALL attempts' decode -- matching the
    phase-safety design used everywhere else in this pipeline (never load T3 and
    S3Gen as separate heavy models at the same time).

    Args:
        failed: [{"chunk_id", "text", "tts_params", "segments", "pauses",
            "boundary_type", "original_score"}, ...] for chunks below threshold.
        cond_emb: Voice conditioning embedding, reused as-is (same book, same voice).
        ckpt_dir, device, variant: Passed straight to VllmBatchProcessor/VllmDecoder.
        decoder_type, voice_path, turbo_ckpt_dir: Passed straight to VllmDecoder.
        audio_output_dir: Where the book's chunk_NNNNN.wav files already live.
        asr_client: ASRDaemonClient to score attempts with, or None to skip ASR.
        asr_threshold: Composite-score pass/fail threshold.
        local_scoring_enabled: Whether to include the local MFCC/spectral/VAD score.
        max_attempts: Retry attempts per chunk (defaults to config.MAX_REGENERATION_ATTEMPTS,
            itself already overridable via the GUI's max_attempts_spin).
        report_dir: Where to write asr_regeneration_report.json / asr_remaining_failures.json.
        progress_callback: fn(chunk_id, best_score, was_regenerated).

    Returns:
        {"regenerated": int, "still_failed": [...], "report": [...]}.
    """
    if not failed:
        return {"regenerated": 0, "still_failed": [], "report": []}

    max_attempts = max_attempts or MAX_REGENERATION_ATTEMPTS

    from modules.vllm_batch_processor import VllmBatchProcessor
    from modules.vllm_decoder import VllmDecoder

    logger.info("Regenerating %d failed chunk(s), up to %d attempts each", len(failed), max_attempts)
    print(f"[Regen] {len(failed)} chunk(s) below threshold, retrying up to {max_attempts}x each")

    # Phase 1B: reload the T3-only vLLM engine just for these chunks' retries.
    batch_processor = VllmBatchProcessor(ckpt_dir=ckpt_dir, target_device=device, variant=variant)

    attempts_by_chunk: Dict = {}
    for item in failed:
        chunk_id = item["chunk_id"]
        base_params = dict(item["tts_params"])
        segments = item["segments"]
        per_attempt = []
        for attempt_num in range(1, max_attempts + 1):
            adj_params = _adjust_params_for_attempt(base_params, attempt_num)
            token_lists = batch_processor.model.generate_speech_tokens(
                segments, cond_emb=cond_emb, **adj_params,
            )
            per_attempt.append((attempt_num, adj_params, token_lists))
        attempts_by_chunk[chunk_id] = per_attempt

    batch_processor.shutdown()

    # Phase 2B: reload the decoder just for these attempts' decode.
    decoder = VllmDecoder(
        decoder_type=decoder_type, ckpt_dir=ckpt_dir, target_device=device,
        voice_path=voice_path, turbo_ckpt_dir=turbo_ckpt_dir,
    )

    report = []
    still_failed = []
    regenerated = 0
    failed_dir = audio_output_dir / "Failed"

    for item in failed:
        chunk_id = item["chunk_id"]
        chunk_id_str = f"{int(chunk_id):05d}"
        out_path = audio_output_dir / f"chunk_{chunk_id_str}.wav"
        pauses = item.get("pauses", [])
        boundary_type = item.get("boundary_type", "none")

        # Decode every attempt first and submit each to ASR non-blocking, so by
        # the time all attempts are decoded, earlier attempts' ASR results are
        # likely already sitting in the daemon's results dir -- no need to wait
        # per-attempt when nothing depends on an individual attempt's score yet.
        decoded_attempts = []
        for attempt_num, adj_params, token_lists in attempts_by_chunk[chunk_id]:
            attempt_label = f"chunk_{chunk_id_str}_attempt{attempt_num}"
            audio, _debug = decoder.decode_and_assemble_chunk(
                token_lists, pauses, boundary_type, chunk_label=attempt_label
            )
            if audio is None:
                continue
            attempt_path = audio_output_dir / f"{attempt_label}.wav"
            audio.export(str(attempt_path), format="wav")
            asr_key = f"{chunk_id}_r{attempt_num}"
            if asr_client is not None:
                asr_client.submit(asr_key, attempt_path, item["text"], asr_threshold)
            decoded_attempts.append((attempt_num, attempt_path, audio, asr_key))

        best_score = item["original_score"]
        best_attempt_num = None
        best_path = None

        for attempt_num, attempt_path, audio, asr_key in decoded_attempts:
            asr_result = asr_client.get_result(asr_key, timeout=60) if asr_client is not None else None
            local_score = (
                evaluate_chunk_quality(audio, reference_text=None, include_spectral=True)
                if local_scoring_enabled else None
            )
            score = compute_composite_score(local_score, asr_result)
            report.append({"chunk_id": chunk_id, "attempt": attempt_num, "score": score})
            print(f"[Regen] {chunk_id_str} attempt {attempt_num}: score={score:.3f} (original={item['original_score']:.3f})")
            if score > best_score:
                best_score = score
                best_attempt_num = attempt_num
                best_path = attempt_path

        if best_attempt_num is not None:
            failed_dir.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                shutil.move(str(out_path), str(failed_dir / out_path.name))
            shutil.move(str(best_path), str(out_path))
            regenerated += 1
            print(f"[Regen] {chunk_id_str}: attempt {best_attempt_num} wins (score {best_score:.3f})")

        # Clean up every attempt file that wasn't chosen as the winner.
        for attempt_num, attempt_path, _audio, _asr_key in decoded_attempts:
            if attempt_path.exists():
                attempt_path.unlink()

        if best_score < asr_threshold:
            still_failed.append({"chunk_id": chunk_id, "best_score": best_score})

        if progress_callback:
            progress_callback(chunk_id, best_score, best_attempt_num is not None)

    decoder.shutdown()

    if report_dir:
        report_dir = Path(report_dir)
        (report_dir / "asr_regeneration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        (report_dir / "asr_remaining_failures.json").write_text(
            json.dumps(still_failed, indent=2, ensure_ascii=False)
        )

    logger.info(
        "Regeneration complete: %d regenerated, %d still failed", regenerated, len(still_failed)
    )
    print(f"[Regen] Done: {regenerated} regenerated, {len(still_failed)} still below threshold")

    return {"regenerated": regenerated, "still_failed": still_failed, "report": report}
