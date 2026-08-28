from __future__ import annotations

import json
import logging
import shutil
import sys
from collections import defaultdict
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


def _regeneration_param_key(params: dict) -> tuple:
    """Build a stable, type-safe grouping key for one T3 sampling configuration.

    vLLM applies one ``SamplingParams`` configuration to each request in a
    generate call.  ``repr`` keeps values such as ``1`` and ``"1"`` distinct
    while allowing the retry planner to batch only exactly equal parameters.

    Args:
        params: Supported per-request T3 sampling parameters.

    Returns:
        Sorted immutable key suitable for a dictionary.
    """
    return tuple(sorted((str(key), repr(value)) for key, value in params.items()))


def _generate_regeneration_tokens_batched(
    batch_processor,
    failed: List[dict],
    max_attempts: int,
    cond_emb,
) -> Dict:
    """Generate all retry token sets in exact-parameter vLLM batches.

    Each retry must sample fresh T3 tokens, but many failed chunks share the
    same adjusted temperature, exaggeration, and other sampling values.  This
    groups only those equal requests, flattens their pause-split segments into
    vLLM prompt lists, and maps returned token lists back to each attempt.

    Turbo is capped at eight segment prompts per vLLM call because its larger
    conditionals can overflow the encoder cache at whole-book scale.  English
    and multilingual variants retain the normal pipeline's full group batch.

    Args:
        batch_processor: Loaded T3-only ``VllmBatchProcessor``.
        failed: Confirmed failed chunk metadata.
        max_attempts: Retry count per chunk.
        cond_emb: Original voice conditional embedding reused for all retries.

    Returns:
        Mapping of chunk id to ordered ``(attempt, params, token_lists)`` rows.
    """
    grouped_requests = defaultdict(list)
    attempts_by_chunk: Dict = {item["chunk_id"]: [] for item in failed}
    total_attempts = 0

    for item in failed:
        chunk_id = item["chunk_id"]
        base_params = dict(item["tts_params"])
        segments = list(item["segments"])
        for attempt_num in range(1, max_attempts + 1):
            params = _adjust_params_for_attempt(base_params, attempt_num)
            # Language is book-level T3 configuration, never retry sampling state.
            params.pop("language_id", None)
            params.pop("language", None)
            request = {
                "chunk_id": chunk_id,
                "attempt_num": attempt_num,
                "params": params,
                "segments": segments,
            }
            grouped_requests[_regeneration_param_key(params)].append(request)
            total_attempts += 1

    calls = 0
    for requests in grouped_requests.values():
        params = requests[0]["params"]
        flattened = [
            (request, segment)
            for request in requests
            for segment in request["segments"]
        ]
        token_lists_by_request = {id(request): [] for request in requests}
        # Turbo T3 has a small encoder-cache ceiling; match Phase 1's safe batch.
        segment_batch_size = 8 if batch_processor.variant == "turbo" else len(flattened)
        if not flattened:
            for request in requests:
                attempts_by_chunk[request["chunk_id"]].append(
                    (request["attempt_num"], request["params"], [])
                )
            continue

        for start in range(0, len(flattened), segment_batch_size):
            piece = flattened[start : start + segment_batch_size]
            token_lists = batch_processor.model.generate_speech_tokens(
                [segment for _request, segment in piece],
                cond_emb=cond_emb,
                language_id=getattr(batch_processor, "language_id", "en"),
                **params,
            )
            if len(token_lists) != len(piece):
                raise RuntimeError(
                    "T3 returned a token-list count that does not match the "
                    f"regeneration prompt batch ({len(token_lists)} != {len(piece)})"
                )
            calls += 1
            for (request, _segment), tokens in zip(piece, token_lists):
                token_lists_by_request[id(request)].append(tokens)

        for request in requests:
            attempts_by_chunk[request["chunk_id"]].append(
                (
                    request["attempt_num"],
                    request["params"],
                    token_lists_by_request[id(request)],
                )
            )

    for rows in attempts_by_chunk.values():
        rows.sort(key=lambda row: row[0])
    print(
        f"[Regen] T3 batched {total_attempts} retry attempt(s) into "
        f"{calls} vLLM call(s) across {len(grouped_requests)} exact parameter group(s)"
    )
    return attempts_by_chunk


def _candidate_wav_name(chunk_id_str: str, label: str) -> str:
    """Build Failed/Investigation filename for one regen candidate WAV.

    Args:
        chunk_id_str: Zero-padded chunk id, for example 00129.
        label: original or attempt1 / attempt2 / attempt3.

    Returns:
        Filename such as chunk_00129_original.wav.
    """
    return f"chunk_{chunk_id_str}_{label}.wav"


def _copy_if_exists(src: Path, dest: Path) -> bool:
    """Copy a WAV to dest when the source file is present.

    Args:
        src: Source WAV path.
        dest: Destination path; parent dirs are created.

    Returns:
        True when a file was copied.
    """
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _archive_regen_candidates(
    chunk_id_str: str,
    original_path: Path,
    decoded_attempts: list,
    failed_dir: Path,
    investigation_dir: Optional[Path] = None,
) -> None:
    """Copy original plus every regen attempt into Failed/, and Investigation/ if set.

    Live audio_chunks keeps chunk_NNNNN.wav as the chosen WAV. Archives use
    explicit names so all four files can be compared.

    Args:
        chunk_id_str: Zero-padded chunk id.
        original_path: Original failing live WAV.
        decoded_attempts: List of (attempt_num, attempt_path, audio, asr_key).
        failed_dir: audio_chunks/Failed destination.
        investigation_dir: Optional audio_chunks/Investigation destination.
    """
    destinations = [failed_dir]
    if investigation_dir is not None:
        destinations.append(investigation_dir)
    original_name = _candidate_wav_name(chunk_id_str, "original")
    for dest_dir in destinations:
        _copy_if_exists(original_path, dest_dir / original_name)
        for attempt_num, attempt_path, _audio, _asr_key in decoded_attempts:
            label = f"attempt{int(attempt_num)}"
            _copy_if_exists(
                attempt_path,
                dest_dir / _candidate_wav_name(chunk_id_str, label),
            )


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
    asr_batch_config=None,
    asr_batch_work_dir: Optional[Path] = None,
    local_scoring_enabled: bool = True,
    max_attempts: Optional[int] = None,
    report_dir: Optional[Path] = None,
    progress_callback: Optional[Callable] = None,
    asr_start_fn: Optional[Callable] = None,
) -> Dict:
    """Regenerate failed chunks, then score all candidate WAVs and keep the best.

    T3 runs once for every retry token set, then S3Gen once for every candidate
    WAV. ASR is not resident during those loads. After S3Gen unloads, asr_start_fn
    (if given) starts a scorer; every attempt WAV is transcribed and the best
    score including the original is kept. Failed/ always gets original plus
    each attempt WAV (chunk_NNNNN_original.wav, chunk_NNNNN_attempt1.wav, ...).
    Investigation/ gets the same set when the best score is still below threshold.

    Args:
        failed: [{"chunk_id", "text", "tts_params", "segments", "pauses",
            "boundary_type", "original_score"}, ...] for chunks below threshold.
        cond_emb: Voice conditioning embedding, reused as-is (same book, same voice).
        ckpt_dir, device, variant: Passed straight to VllmBatchProcessor/VllmDecoder.
        decoder_type, voice_path, turbo_ckpt_dir: Passed straight to VllmDecoder.
        audio_output_dir: Where the book's chunk_NNNNN.wav files already live.
        asr_client: Optional already-running scorer; normally None so T3/S3Gen
            run without Whisper on the GPU.
        asr_batch_config: Selected backend/model/device configuration for one
            short-lived post-S3Gen batch score of every retry candidate.
        asr_batch_work_dir: Book-local directory for the isolated ASR child job.
        asr_threshold: Spoken-compare pass/fail threshold.
        local_scoring_enabled: Whether to include the local MFCC/spectral/VAD score.
        max_attempts: Retry attempts per chunk (defaults to config.MAX_REGENERATION_ATTEMPTS,
            itself already overridable via the GUI's max_attempts_spin).
        report_dir: Where to write asr_regeneration_report.json / asr_remaining_failures.json.
        progress_callback: fn(chunk_id, best_score, was_regenerated).
        asr_start_fn: Legacy callback retained for callers not yet using the
            batch runner.

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

    # Phase 1B: reload T3-only vLLM (including Turbo GPT2 ChatterboxT3Turbo).
    batch_processor = VllmBatchProcessor(
        ckpt_dir=ckpt_dir, target_device=device, variant=variant
    )

    attempts_by_chunk = _generate_regeneration_tokens_batched(
        batch_processor=batch_processor,
        failed=failed,
        max_attempts=max_attempts,
        cond_emb=cond_emb,
    )

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
    decoded_by_chunk: Dict = {}

    for item in failed:
        chunk_id = item["chunk_id"]
        chunk_id_str = f"{int(chunk_id):05d}"
        pauses = item.get("pauses", [])
        boundary_type = item.get("boundary_type", "none")
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
            decoded_attempts.append((attempt_num, attempt_path, audio, asr_key))
        decoded_by_chunk[chunk_id] = decoded_attempts

    decoder.shutdown()
    try:
        import gc as _gc
        import torch as _torch

        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        _gc.collect()
    except Exception:
        pass
    print("[Regen] S3Gen unloaded; starting ASR scorer for retry WAVs")

    batch_results = {}
    if asr_batch_config is not None:
        from ASR.batch_runner import run_asr_batch_isolated

        retry_tasks = [
            {
                "chunk_id": asr_key,
                "wav_path": str(attempt_path),
                "expected_text": item["text"],
                "threshold": asr_threshold,
            }
            for item in failed
            for _attempt_num, attempt_path, _audio, asr_key in decoded_by_chunk.get(
                item["chunk_id"], []
            )
        ]
        print(
            f"[ASR] Regen batch starting (backend={asr_batch_config.backend}, "
            f"model={asr_batch_config.model_size}, candidates={len(retry_tasks)})"
        )
        batch_run = run_asr_batch_isolated(
            retry_tasks,
            asr_batch_config,
            asr_batch_work_dir or report_dir or audio_output_dir,
            stage_label="regeneration",
        )
        batch_results = batch_run.results
        print(
            f"[ASR] Regen batch done (actual_device="
            f"{','.join(batch_run.actual_devices) or 'unknown'}, "
            f"effective_workers={batch_run.effective_workers})"
        )
    elif asr_client is None and asr_start_fn is not None:
        asr_client = asr_start_fn()

    if asr_client is not None:
        for item in failed:
            for attempt_num, attempt_path, _audio, asr_key in decoded_by_chunk.get(
                item["chunk_id"], []
            ):
                asr_client.submit(asr_key, attempt_path, item["text"], asr_threshold)

    for item in failed:
        chunk_id = item["chunk_id"]
        chunk_id_str = f"{int(chunk_id):05d}"
        out_path = audio_output_dir / f"chunk_{chunk_id_str}.wav"
        decoded_attempts = decoded_by_chunk.get(chunk_id, [])
        best_score = item["original_score"]
        best_attempt_num = None
        best_path = None
        attempt_evidence = []

        for attempt_num, attempt_path, audio, asr_key in decoded_attempts:
            asr_result = batch_results.get(asr_key)
            if asr_result is None and asr_client is not None:
                asr_result = asr_client.get_result(asr_key, timeout=60)
            asr_error = str((asr_result or {}).get("error") or "")
            if asr_result is None or asr_error:
                # An ASR transport/model error is not proof that a retry is good.
                # Never let compute_composite_score's no-input default select it.
                local_score = None
                score = 0.0
            else:
                local_score = (
                    evaluate_chunk_quality(audio, reference_text=None, include_spectral=True)
                    if local_scoring_enabled
                    else None
                )
                score = compute_composite_score(local_score, asr_result)
            attempt_row = {
                "chunk_id": chunk_id,
                "attempt": attempt_num,
                "score": score,
                "asr_backend": (asr_result or {}).get("backend") or "",
                "asr_device": (asr_result or {}).get("device") or "",
                "asr_error": asr_error,
                "asr_text": (asr_result or {}).get("asr_text") or "",
                "classification": (asr_result or {}).get("classification") or "",
                "explanation": (asr_result or {}).get("explanation") or "",
            }
            report.append(attempt_row)
            attempt_evidence.append(dict(attempt_row))
            print(
                f"[Regen] {chunk_id_str} attempt {attempt_num}: "
                f"score={score:.3f} (original={item['original_score']:.3f})"
            )
            if score > best_score:
                best_score = score
                best_attempt_num = attempt_num
                best_path = attempt_path

        still_below = best_score < asr_threshold
        inv_dir = audio_output_dir / "Investigation" if still_below else None
        failed_dir.mkdir(parents=True, exist_ok=True)
        if inv_dir is not None:
            inv_dir.mkdir(parents=True, exist_ok=True)
        _archive_regen_candidates(
            chunk_id_str,
            out_path,
            decoded_attempts,
            failed_dir,
            inv_dir,
        )
        print(
            f"[Regen] {chunk_id_str}: archived original + "
            f"{len(decoded_attempts)} attempt(s) → Failed/"
            + (" and Investigation/" if still_below else "")
        )

        if best_attempt_num is not None and best_path is not None:
            shutil.copy2(best_path, out_path)
            regenerated += 1
            print(
                f"[Regen] {chunk_id_str}: attempt {best_attempt_num} wins "
                f"(score {best_score:.3f})"
            )

        for attempt_num, attempt_path, _audio, _asr_key in decoded_attempts:
            if attempt_path.exists():
                attempt_path.unlink()

        if still_below:
            archive_dir = inv_dir or failed_dir
            attempt_paths = [
                str(archive_dir / _candidate_wav_name(chunk_id_str, f"attempt{attempt_num}"))
                for attempt_num, _attempt_path, _audio, _asr_key in decoded_attempts
            ]
            still_failed.append(
                {
                    "chunk_id": chunk_id,
                    "text": item.get("text") or item.get("expected_text") or "",
                    "expected_text": item.get("expected_text") or item.get("text") or "",
                    "original_score": item.get("original_score"),
                    "best_score": best_score,
                    "threshold": asr_threshold,
                    "status": "failed_after_regeneration",
                    "attempts": attempt_evidence,
                    "original_wav": str(archive_dir / _candidate_wav_name(chunk_id_str, "original")),
                    "attempt_wavs": attempt_paths,
                    "best_attempt": best_attempt_num,
                    "retry_won": best_attempt_num is not None,
                }
            )
            print(f"[Regen] {chunk_id_str}: still failing → Investigation/")

        if progress_callback:
            progress_callback(chunk_id, best_score, best_attempt_num is not None)

    if asr_client is not None:
        asr_client.shutdown()

    if report_dir:
        report_dir = Path(report_dir)
        (report_dir / "asr_regeneration_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        (report_dir / "asr_remaining_failures.json").write_text(
            json.dumps(still_failed, indent=2, ensure_ascii=False)
        )
        # Both names are intentional: remaining_failures preserves the legacy
        # consumer, while investigation_failures is explicit in Repair Tool.
        (report_dir / "asr_failed_regenerations.json").write_text(
            json.dumps(still_failed, indent=2, ensure_ascii=False)
        )
        (report_dir / "asr_investigation_failures.json").write_text(
            json.dumps(still_failed, indent=2, ensure_ascii=False)
        )

    logger.info(
        "Regeneration complete: %d regenerated, %d still failed", regenerated, len(still_failed)
    )
    print(f"[Regen] Done: {regenerated} regenerated, {len(still_failed)} still below threshold")

    return {"regenerated": regenerated, "still_failed": still_failed, "report": report}
