"""
TTS Engine Module
Handles ChatterboxTTS interface, model loading, and chunk processing coordination
"""

import torch
import io
import threading
import gc
import time
import logging
import sys
import numpy as np
import warnings
import subprocess
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config.config import *

from modules.vllm_batch_processor import VllmBatchProcessor
from modules.vllm_decoder import VllmDecoder


def _build_chunk_meta(chunks_data: list) -> dict:
    """Builds {chunk_id: {...}} for VllmDecoder and Phase 3 regeneration.

    Uses the same chunk_id/index fallback as VllmBatchProcessor (id 0 is valid,
    so falsy checks would incorrectly drop it) to key identically to tokens_dict.
    Includes text/tts_params/segments (not just pauses/boundary_type) so a failed
    chunk can be regenerated later without re-reading chunks_info.json.
    """
    chunk_meta = {}
    for item in chunks_data:
        cid = item.get("chunk_id")
        if cid is None:
            cid = item.get("index")
        if cid is None:
            continue
        chunk_meta[cid] = {
            "pauses": item.get("pauses", []),
            "boundary_type": item.get("boundary_type", "none"),
            "text": item.get("text", ""),
            "tts_params": item.get("tts_params", {}),
            "segments": item.get("segments", []),
        }
    return chunk_meta


def _resolve_asr_model_size(asr_level: str) -> str:
    """Maps the GUI's SAFE/MODERATE/INSANE tier to a Whisper model name, using
    the same system-profile-based recommendation the GUI's own preview text
    already computes (modules/system_detector.py), so the daemon's actual
    model matches what the GUI told the user it would use.
    """
    try:
        from modules.system_detector import get_system_profile, recommend_asr_models
        recommendations = recommend_asr_models(get_system_profile())
        return recommendations.get(asr_level, recommendations.get("moderate", {})).get(
            "primary", {}
        ).get("model", "base")
    except Exception as e:
        logging.warning("Could not resolve ASR model tier '%s', defaulting to base: %s", asr_level, e)
        return "base"


def _maybe_start_asr_daemon(tts_dir, enable_asr: bool, asr_level: str = "moderate", asr_device: str = "cpu"):
    """Starts the ASR daemon for Phase 3, or returns None if ASR/regen is off.

    Called BEFORE Phase 2 decode so VllmDecoder.decode_tokens can submit each
    chunk for background transcription as soon as it's written, instead of
    blocking decode on ASR the way the legacy per-chunk backend does.

    asr_device: "cpu" (default, never contends with TTS for VRAM) or "cuda"
        (Tab 2's "Run ASR on GPU" checkbox). The daemon itself falls back to
        CPU on its own if a CUDA load fails, so this is safe to leave on GPU.
    """
    if not (ENABLE_REGENERATION_LOOP and enable_asr):
        return None
    try:
        from tools.asr_client import ASRDaemonClient
        model_size = _resolve_asr_model_size(asr_level)
        asr_client = ASRDaemonClient(base_dir=Path(tts_dir), model_size=model_size, device=asr_device)
        asr_client.start_daemon()
        print(f"✅ [ASR] Daemon loaded (model: {model_size}, device: {asr_device})")
        return asr_client
    except Exception as e:
        logging.error("ASR daemon failed to start, continuing without ASR scoring: %s", e)
        print(f"⚠️ ASR daemon failed to start: {e}")
        return None


def _run_phase3_regen(
    chunk_meta: dict,
    tokens_dict: dict,
    local_scores: dict,
    tts_dir,
    audio_chunks_dir,
    cond_emb,
    ckpt_dir,
    device: str,
    variant: str,
    decoder_type: str,
    voice_path,
    turbo_ckpt_dir,
    asr_client,
    asr_threshold: float,
    true_start_time: float,
):
    """Validate ASR-scored chunks and regenerate anything below asr_threshold.

    Local VAD/MFCC heuristics alone cannot authorize regeneration. When ASR is
    disabled or its daemon is unavailable, the normal token-to-audio pipeline
    ends after Phase 2 and this phase is skipped.
    """
    if not ENABLE_REGENERATION_LOOP or asr_client is None:
        if ENABLE_REGENERATION_LOOP and asr_client is None:
            logging.info("Phase 3 skipped: ASR is disabled or unavailable")
            print("[Regen] Skipped: ASR is disabled or unavailable")
        return

    logging.info("=" * 70)
    logging.info("PHASE 3: ASR + quality validation / regeneration")
    logging.info("=" * 70)
    emit_phase_status(None, "Checking chunk quality (ASR/regeneration)", total_start_time=true_start_time)

    from modules.regeneration_engine import compute_composite_score, regenerate_failed_chunks

    failed = []
    for chunk_id in tokens_dict.keys():
        meta = chunk_meta.get(chunk_id, {})
        asr_result = asr_client.get_result(str(chunk_id), timeout=60) if asr_client is not None else None
        composite = compute_composite_score(local_scores.get(chunk_id), asr_result)
        if composite < asr_threshold:
            failed.append({
                "chunk_id": chunk_id,
                "text": meta.get("text", ""),
                "tts_params": meta.get("tts_params", {}),
                "segments": meta.get("segments", []),
                "pauses": meta.get("pauses", []),
                "boundary_type": meta.get("boundary_type", "none"),
                "original_score": composite,
            })

    logging.info(
        "Phase 3: %d/%d chunks below threshold (%.2f)", len(failed), len(tokens_dict), asr_threshold
    )
    print(f"[Regen] Quality check: {len(failed)}/{len(tokens_dict)} chunks below threshold {asr_threshold:.2f}")

    if failed:
        regenerate_failed_chunks(
            failed=failed,
            cond_emb=cond_emb,
            ckpt_dir=ckpt_dir,
            device=device,
            variant=variant,
            decoder_type=decoder_type,
            voice_path=voice_path,
            turbo_ckpt_dir=turbo_ckpt_dir,
            audio_output_dir=audio_chunks_dir,
            asr_client=asr_client,
            asr_threshold=asr_threshold,
            local_scoring_enabled=bool(local_scores),
            # tts_engine.py's own MAX_REGENERATION_ATTEMPTS (already GUI-overridden
            # above via quality_params) -- regeneration_engine.py's own `from
            # config.config import MAX_REGENERATION_ATTEMPTS` captured the static
            # config.py value at import time and never sees this override otherwise.
            max_attempts=MAX_REGENERATION_ATTEMPTS,
            report_dir=Path(tts_dir),
        )

    if asr_client is not None:
        asr_client.shutdown()

# Suppress noisy syntax warnings emitted by pydub's regex helpers
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub.utils")

# Verbosity control: limit per-chunk console output
import os as _os

_TTS_TIMING = bool(int(_os.getenv("TTS_TIMING", "0") or 0))

# Reusable in-memory WAV buffer to reduce alloc/GC per chunk
_WAV_BUFFER = io.BytesIO()

# Optional async export of CPU post-processing to reduce inter-chunk gaps (disabled by default for stability)
ASYNC_EXPORT = globals().get("ASYNC_EXPORT", False)
_EXPORT_EXECUTOR = None


def _ensure_export_executor(max_workers: int = 2):
    """Ensures an export executor is initialized and returns it.
    Args:
    max_workers (int): The maximum number of worker threads in the executor.
    Returns:
    ThreadPoolExecutor or None: An initialized ThreadPoolExecutor or None if initialization fails.
    """
    global _EXPORT_EXECUTOR
    if _EXPORT_EXECUTOR is None:
        try:
            _EXPORT_EXECUTOR = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="exporter"
            )
        except Exception:
            _EXPORT_EXECUTOR = None
    return _EXPORT_EXECUTOR


def _export_task_fn(
    watermarked_wav,
    sample_rate,
    audio_chunks_dir,
    chunk_id_str,
    boundary_type,
    enable_audio_trimming,
):
    """Exports a watermarked WAV file as an audio chunk.
    Args:
    watermarked_wav (numpy.ndarray): The watermarked audio data.
    sample_rate (int): The sample rate of the audio.
    audio_chunks_dir (str): Directory to save the audio chunks.
    chunk_id_str (str): ID for the audio chunk.
    boundary_type (str): Type of boundary for trimming and silence processing.
    enable_audio_trimming (bool): Whether to enable audio trimming.
    Returns:
    None
    """
    import io
    import soundfile as sf
    from pydub import AudioSegment

    try:
        # Build AudioSegment in-memory
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, watermarked_wav, sample_rate, format="wav")
            wav_buffer.seek(0)
            final_audio = AudioSegment.from_wav(wav_buffer)

        # Trimming and contextual silence
        from modules.audio_processor import (
            process_audio_with_trimming_and_silence,
            trim_audio_endpoint,
        )

        if boundary_type and boundary_type != "none":
            final_audio = process_audio_with_trimming_and_silence(
                final_audio, boundary_type
            )
        elif enable_audio_trimming:
            final_audio = trim_audio_endpoint(final_audio)

        # Final save
        final_path = audio_chunks_dir / f"chunk_{chunk_id_str}.wav"
        final_audio.export(final_path, format="wav")
        logging.info(f"✅ Saved final chunk (async): {final_path.name}")
    except Exception as _e:
        logging.error(f"Async export failed for chunk {chunk_id_str}: {_e}")


from modules.text_processor import (
    smart_punctuate,
    sentence_chunk_text,
    detect_content_boundaries,
)
from modules.punctuation_pauses import add_pause_tags_to_text
from modules.pause_utils import convert_inline_markers_to_pause_tags
from config import config as _cfg

try:
    from src.chatterbox.models.tokenizers import EnTokenizer as _EnTokenizer

    _T3_TOKENIZER = _EnTokenizer()
except Exception:
    _T3_TOKENIZER = None

# ============================================================================
# GLOBAL VOICE CACHE FOR PREWARM OPTIMIZATION
# ============================================================================
# Cache to persist voice embeddings across model reloads within a conversion session
_global_voice_cache = None
_voice_cache_info = None
_GPU_INFER_LOCK = threading.Lock()


def clear_voice_cache():
    """Clear the global voice cache at start of new conversion"""
    global _global_voice_cache, _voice_cache_info
    _global_voice_cache = None
    _voice_cache_info = None
    logging.info("🗑️ Voice cache cleared for new conversion session")


def store_voice_cache(model):
    """Store voice embeddings from model to global cache"""
    global _global_voice_cache, _voice_cache_info
    if hasattr(model, "conds") and model.conds is not None:
        _global_voice_cache = model.conds
        _voice_cache_info = {
            "cached_at": time.time(),
            "cache_size_mb": sys.getsizeof(_global_voice_cache) / 1024 / 1024,
        }
        logging.info(
            f"💾 Voice embeddings cached ({_voice_cache_info['cache_size_mb']:.1f}MB)"
        )
    else:
        logging.warning("⚠️ No voice embeddings found to cache")


def restore_voice_cache(model):
    """Restore voice embeddings from global cache to model"""
    global _global_voice_cache, _voice_cache_info
    if _global_voice_cache is not None:
        model.conds = _global_voice_cache
        logging.info(
            f"🚀 Voice embeddings restored from cache (skipping {2.9:.1f}s warmup)"
        )
        return True
    else:
        logging.debug("📝 No voice cache available - will need fresh prewarm")
        return False


def get_voice_cache_info():
    """Get information about current voice cache"""
    global _voice_cache_info
    return _voice_cache_info


# from modules.performance_integrator import get_performance_integrator, shutdown_performance_system


def find_chunks_json_file(book_name):
    """Find the corresponding chunks JSON file for a book"""
    from config.config import AUDIOBOOK_ROOT

    # Look in the TTS processing directory
    tts_chunks_dir = AUDIOBOOK_ROOT / book_name / "TTS" / "text_chunks"
    json_path = tts_chunks_dir / "chunks_info.json"

    if json_path.exists():
        return json_path

    # Also check old Text_Input location for backwards compatibility
    text_input_dir = Path("Text_Input")
    possible_names = [
        f"{book_name}_chunks.json",
        f"{book_name.lower()}_chunks.json",
        f"{book_name.replace(' ', '_')}_chunks.json",
    ]

    for name in possible_names:
        old_json_path = text_input_dir / name
        if old_json_path.exists():
            return old_json_path

    return None


from modules.audio_processor import (
    pause_for_chunk_review,
    get_chunk_audio_duration,
    has_mid_energy_drop,
)
from modules.terminal_logger import start_terminal_logging
from modules.file_manager import (
    setup_book_directories,
    find_book_files,
    ensure_voice_sample_compatibility,
    combine_audio_chunks,
    get_audio_files_in_directory,
    convert_to_m4b,
    add_metadata_to_m4b,
)
from modules.progress_tracker import (
    setup_logging,
    log_chunk_progress,
    log_run,
    emit_phase_status,
    emit_final_status,
)

# Global shutdown flag and session state
shutdown_requested = False
_SESSION_ACTIVE = False

# ---------------------------------------------------------------------------
# Global model/backend cache to prevent double-loading across conversions
# ---------------------------------------------------------------------------
_GLOBAL_TTS_MODEL = None
_GLOBAL_TTS_MODEL_DEVICE = None
_LAST_RUN_SIGNATURE = None
_VOICE_CONDS_CACHE = {}
_FORCE_MODEL_RELOAD = False
# Persist active backend context across chunks within a session
_SESSION_ACTIVE = False
_SESSION_BACKEND = None

# Prewarm tracking (used by legacy cleanup paths)
# Define defaults so cleanup can always safely reset these.
_GLOBAL_TTS_PREWARMED = False
_PREWARMED_KEYS = set()  # e.g., keys of (voice_sig, core_tts_params_sig)

# Capability probe cache: does current model expose a batch API?
_BATCH_API_SUPPORTED = None


def _release_global_tts_model():
    """Releases global TTS model resources.
    Args:
    None
    Returns:
    None
    """
    global _GLOBAL_TTS_MODEL, _GLOBAL_TTS_MODEL_DEVICE, _FORCE_MODEL_RELOAD

    # Step 1: Explicitly clear model subcomponents before deletion
    if _GLOBAL_TTS_MODEL is not None:
        try:
            # Clear model conditionals if they exist
            if hasattr(_GLOBAL_TTS_MODEL, "conds"):
                _GLOBAL_TTS_MODEL.conds = None

            # Move model to CPU to release GPU memory
            if hasattr(_GLOBAL_TTS_MODEL, "cpu"):
                _GLOBAL_TTS_MODEL.cpu()

            # Clear any cached states
            if hasattr(_GLOBAL_TTS_MODEL, "clear_cache"):
                _GLOBAL_TTS_MODEL.clear_cache()
            elif hasattr(_GLOBAL_TTS_MODEL, "reset_states"):
                _GLOBAL_TTS_MODEL.reset_states()

            print("🧹 Explicitly cleared model subcomponents")

        except Exception as e:
            print(f"⚠️ Warning during model component cleanup: {e}")

        # Step 2: Delete the model object
        try:
            del _GLOBAL_TTS_MODEL
            print("🗑️ Deleted global TTS model object")
        except Exception as e:
            print(f"❌ Failed to delete model: {e}")

    # Step 3: Clear all global variables
    _GLOBAL_TTS_MODEL = None
    _GLOBAL_TTS_MODEL_DEVICE = None
    _FORCE_MODEL_RELOAD = True

    # Step 4: Clear caches and prewarming state
    global _GLOBAL_TTS_PREWARMED
    _GLOBAL_TTS_PREWARMED = False
    global _PREWARMED_KEYS, _LAST_RUN_SIGNATURE
    _PREWARMED_KEYS.clear()
    _LAST_RUN_SIGNATURE = None

    # Step 5: Forcibly clear voice conditionals cache (may contain GPU tensors)
    if _VOICE_CONDS_CACHE:
        for key, cached_conds in _VOICE_CONDS_CACHE.items():
            try:
                if hasattr(cached_conds, "cpu"):
                    cached_conds.cpu()
                del cached_conds
            except Exception:
                pass
        _VOICE_CONDS_CACHE.clear()
        print("🧹 Cleared voice conditionals cache")

    # Step 6: Force Python garbage collection
    import gc

    gc.collect()
    gc.collect()  # Call twice to ensure cleanup

    # Step 7: Aggressive CUDA cleanup
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # Reset memory stats to clear fragmentation tracking
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            print("🧹 Performed aggressive CUDA cleanup")
    except Exception as e:
        print(f"⚠️ CUDA cleanup warning: {e}")

    print("✅ Model release completed - VRAM should be freed")


# Console colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

import random


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        # Check if torch.mps exists before calling
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logging.info(f"Global seed set to: {seed_value}")


# ============================================================================
# MEMORY AND MODEL MANAGEMENT
# ============================================================================


def monitor_gpu_activity(operation_name):
    """Lightweight GPU monitoring for high-speed processing"""
    # Disabled expensive pynvml queries to free up GPU cycles
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        # Skip GPU utilization queries during production runs
        return allocated, 0
    return 0, 0


def optimize_memory_usage():
    """Aggressive memory management for 8GB VRAM"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()


def monitor_vram_usage(operation_name=""):
    """Real-time VRAM monitoring"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        if allocated > VRAM_SAFETY_THRESHOLD:
            logging.warning(
                f"⚠️ High VRAM usage during {operation_name}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved"
            )
            optimize_memory_usage()

        return allocated, reserved
    return 0, 0


def log_vram_checkpoint(operation_name: str) -> tuple[float, float]:
    """Log allocated and reserved CUDA memory at a pipeline boundary."""
    allocated, reserved = monitor_vram_usage(operation_name)
    message = (
        f"[VRAM] {operation_name}: allocated={allocated:.2f} GB, "
        f"reserved={reserved:.2f} GB"
    )
    logging.info(message)
    print(message, flush=True)
    return allocated, reserved


def get_optimal_workers():
    """Dynamic worker allocation based on VRAM usage"""
    if not USE_DYNAMIC_WORKERS:
        return MAX_WORKERS

    allocated_vram = torch.cuda.memory_allocated() / 1024**3

    if allocated_vram < 5.0:
        return min(TEST_MAX_WORKERS, MAX_WORKERS)
    elif allocated_vram < VRAM_SAFETY_THRESHOLD:
        return min(2, MAX_WORKERS)
    else:
        return 1


def _voice_sig(voice_path):
    """Resolves the path of a voice file and returns its absolute path and last modified time in nanoseconds.
    Args:
    voice_path (str): Path to the voice file.
    Returns:
    tuple: A tuple containing the absolute path as a string and the last modified time in nanoseconds, or the original path and None if an error occurs.
    """
    try:
        vpath = Path(voice_path)
        return (str(vpath.resolve()), vpath.stat().st_mtime_ns)
    except Exception:
        return (str(voice_path), None)


def _core_tts_params_sig(tts_params: dict | None):
    """Generates TTS parameters signature based on given dictionary.
    Args:
    tts_params (dict | None): Dictionary containing TTS parameters.
    Returns:
    tuple of four floats representing the configured weight, temperature, minimum probability, and top-p value. Default values are used if corresponding keys are not present in the dictionary.
    """
    tp = tts_params or {}
    return (
        round(float(tp.get("cfg_weight", 0.5)), 3),
        round(float(tp.get("temperature", 0.85)), 3),
        round(float(tp.get("min_p", 0.05)), 3),
        round(float(tp.get("top_p", 0.9)), 3),
    )


def process_chunks_with_pipeline(
    all_chunks,
    batch_chunks,
    chunk_offset,
    text_chunks_dir,
    audio_chunks_dir,
    voice_path,
    tts_params,
    start_time,
    total_chunks,
    punc_norm,
    book_name,
    log_run,
    log_path,
    device,
    model,
    asr_model,
    asr_enabled,
    optimal_workers,
    total_audio_duration,
):
    """Legacy pipeline entry point for resume_handler compatibility.
    Delegates to process_book_folder which now uses VllmBatchProcessor + VllmDecoder.
    """
    logging.warning(
        "process_chunks_with_pipeline called but pipeline was removed; falling back to sequential"
    )
    raise NotImplementedError(
        "Use process_book_folder with vllm/turbo-hybrid backends instead"
    )


def prewarm_model_with_voice(model, voice_path, tts_params=None):
    """
    Pre-warm the TTS model with a voice sample to eliminate cold start quality issues.
    Uses global voice cache to skip prewarming during model reloads.

    Args:
        model: Loaded TTS model
        voice_path: Path to voice sample file
        tts_params: Optional TTS parameters for pre-warming (uses defaults if None)

    Returns:
        model: The pre-warmed model (same object, but with cached conditioning)
    """
    from modules.file_manager import ensure_voice_sample_compatibility

    # Check if we can restore from cache instead of prewarming
    if restore_voice_cache(model):
        print("✅ Model pre-warming skipped - using cached voice embeddings")
        return model

    try:
        print("🔥 Pre-warming model with voice sample...")

        # Prepare voice for TTS
        compatible_voice = ensure_voice_sample_compatibility(voice_path)

        # Set up default TTS parameters if none provided
        if tts_params is None:
            tts_params = {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.9}

        # Prepare voice conditionals
        model.prepare_conditionals(compatible_voice)

        # Generate a short dummy audio to fully warm up the model
        dummy_text = (
            "sixth sick sheik's sixth sheep's sick, and red leather, yellow leather."
        )

        print(f"🎤 Generating warm-up audio: '{dummy_text}'")

        # Generate dummy audio with the voice and parameters
        # Serialize warm-up GPU generation to avoid allocator races
        with _GPU_INFER_LOCK:
            try:
                wav_np = model.generate(
                    dummy_text,
                    exaggeration=tts_params["exaggeration"],
                    cfg_weight=tts_params["cfg_weight"],
                    temperature=tts_params["temperature"],
                    disable_watermark=True,
                )
            except TypeError:
                wav_np = model.generate(
                    dummy_text,
                    exaggeration=tts_params["exaggeration"],
                    cfg_weight=tts_params["cfg_weight"],
                    temperature=tts_params["temperature"],
                )

        print("✅ Model pre-warming completed - first chunk quality optimized")

        # Store voice embeddings in global cache for future model reloads
        store_voice_cache(model)

        # Clean up any temporary audio data (don't save the dummy audio)
        del wav_np

        return model

    except Exception as e:
        print(f"⚠️ Pre-warming failed: {e}")
        print("📝 Model will still work but first chunk may have quality variations")
        return model


def get_best_available_device():
    """Detect and return the best available device with proper fallback"""
    try:
        if torch.cuda.is_available():
            # Test CUDA with a simple operation
            test_tensor = torch.tensor([1.0]).to("cuda")
            del test_tensor
            torch.cuda.empty_cache()
            return "cuda"
    except Exception as e:
        logging.warning(f"CUDA test failed: {e}")

    try:
        if torch.backends.mps.is_available():
            # Test MPS with a simple operation
            test_tensor = torch.tensor([1.0]).to("mps")
            del test_tensor
            return "mps"
    except Exception as e:
        logging.warning(f"MPS test failed: {e}")

    return "cpu"


def load_optimized_model(device, *, force_reload: bool = False):
    """Load TTS model with REAL performance optimizations.

    Priority:
    - If `config.config.CHATTERBOX_CKPT_DIR` (or env var) points to a local checkpoint folder, load from it.
    - Otherwise, fall back to `from_pretrained` (requires network access).
    - Optionally enable ONNX T3 if `ENABLE_T3_ONNX` is True and an ONNX file is present.
    """
    from src.chatterbox.tts import ChatterboxTTS
    from modules.real_tts_optimizer import optimize_chatterbox_model
    from config.config import CHATTERBOX_CKPT_DIR

    # Apply precision/runtime knobs early
    try:
        from config.config import ENABLE_TF32  # bool
    except Exception:
        ENABLE_TF32 = True
    try:
        if device == "cuda" and torch.cuda.is_available():
            # Respect TF32 toggle
            torch.backends.cuda.matmul.allow_tf32 = bool(ENABLE_TF32)
            torch.backends.cudnn.allow_tf32 = bool(ENABLE_TF32)
            # Prefer high-precision matmul policy for speed on Ada
            torch.set_float32_matmul_precision("high" if ENABLE_TF32 else "medium")
    except Exception:
        pass

    logging.info("🚀 Loading ChatterboxTTS with REAL performance optimizations...")

    # Global cache: reuse existing model if same device
    global _GLOBAL_TTS_MODEL, _GLOBAL_TTS_MODEL_DEVICE
    # If a prior Save requested a hard reload, honor it once
    global _FORCE_MODEL_RELOAD
    if _FORCE_MODEL_RELOAD:
        force_reload = True
        _FORCE_MODEL_RELOAD = False

    if not force_reload and _GLOBAL_TTS_MODEL is not None:
        if _GLOBAL_TTS_MODEL_DEVICE == device:
            logging.info("✅ Reusing cached TTS model (no re-load)")
            model = _GLOBAL_TTS_MODEL
            # Ensure eval and return
            try:
                model.eval()
            except Exception:
                pass
            return model

    # Load base model: prefer local if configured
    try:
        ckpt_dir = (CHATTERBOX_CKPT_DIR or "").strip()
        if ckpt_dir:
            ckpt_path = Path(ckpt_dir)
            if ckpt_path.exists():
                logging.info(f"📦 Loading local checkpoints from: {ckpt_path}")
                model = ChatterboxTTS.from_local(ckpt_path, device)
            else:
                logging.warning(
                    f"⚠️ CHATTERBOX_CKPT_DIR set but not found: {ckpt_path}. Falling back to from_pretrained()."
                )
                model = ChatterboxTTS.from_pretrained(device=device)
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
        logging.info("✅ Base ChatterboxTTS model loaded")

    except Exception as e:
        logging.error(f"❌ Failed to load ChatterboxTTS model: {e}")
        raise

    # Apply REAL optimizations that target actual inference bottlenecks
    try:
        logging.info("⚡ Applying REAL TTS optimizations...")
        optimization_count = optimize_chatterbox_model(model)

        if optimization_count > 0:
            logging.info(f"🎯 REAL optimizations applied: {optimization_count}")
            logging.info("🚀 Model ready for HIGH-PERFORMANCE inference")
        else:
            logging.warning("⚠️ No real optimizations could be applied")

    except Exception as e:
        logging.error(f"❌ Real optimization failed: {e}")
        logging.info("📝 Using model without optimizations...")

    _GLOBAL_TTS_MODEL = model
    _GLOBAL_TTS_MODEL_DEVICE = device
    # Warm up cuBLAS handle early to avoid failing later under fragmentation
    try:
        if device == "cuda" and torch.cuda.is_available():
            a = torch.randn(1, 32, device="cuda", dtype=torch.float32)
            b = torch.randn(32, 1, device="cuda", dtype=torch.float32)
            _ = a @ b  # triggers cublasCreate if not already created
            del a, b, _
    except Exception as e:
        logging.warning(f"cuBLAS warm-up skipped: {e}")

    # Basic model setup
    if hasattr(model, "eval"):
        model.eval()

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("✅ Basic CUDNN optimization enabled")

    return model


# ============================================================================
# CHUNK PROCESSING
# ============================================================================


def patch_alignment_layer(tfmr, alignment_layer_idx=12):
    """Patch alignment layer to avoid recursion"""
    from types import MethodType

    target_layer = tfmr.layers[alignment_layer_idx].self_attn
    original_forward = target_layer.forward

    def patched_forward(self, *args, **kwargs):
        """Patches the forward method of a target layer to include output_attentions in kwargs and calls the original forward method.
        Args:
        batch: The batch data to process.
        text_chunks_dir: Directory containing text chunks.
        audio_chunks_dir: Directory containing audio chunks.
        voice_path: Path to the voice file.
        tts_params: Parameters for text-to-speech conversion.
        start_time: Starting time of the processing.
        total_chunks: Total number of chunks.
        punc_norm: Flag for punctuation normalization.
        basename: Base name for chunk files.
        log_run_func: Function to log the run.
        log_path: Path for logging.
        device: Device to run on (e.g., 'cuda' or 'cpu').
        model: Model to use for processing.
        seed: Seed for random operations. Default is 0.
        enable_asr: Flag to enable Automatic Speech Recognition. Default is None.
        backend: Backend to use for processing. Default is None.
        Returns:
        None
        """
        kwargs["output_attentions"] = True
        return original_forward(*args, **kwargs)

    target_layer.forward = MethodType(patched_forward, target_layer)


def process_batch(
    batch,
    text_chunks_dir,
    audio_chunks_dir,
    voice_path,
    tts_params,
    start_time,
    total_chunks,
    punc_norm,
    basename,
    log_run_func,
    log_path,
    device,
    model,
    seed=0,
    enable_asr=None,
    backend=None,
):
    """Process a batch of chunks using the batch-enabled TTS model.
    Args:
    batch: The batch of data to process.
    text_chunks_dir: Directory containing text chunk files.
    audio_chunks_dir: Directory for saving audio chunk files.
    voice_path: Path to the voice file.
    tts_params: Parameters for the text-to-speech conversion.
    start_time: Starting time for processing.
    total_chunks: Total number of chunks.
    punc_norm: Flag to normalize punctuation.
    basename: Base name for output files.
    log_run_func: Function for logging the run process.
    log_path: Path for log file.
    device: Device for computation (e.g., 'cuda' or 'cpu').
    model: The TTS model instance.
    seed: Random seed for reproducibility (default is 0).
    enable_asr: Flag to enable Automatic Speech Recognition.
    backend: Backend for processing.
    Returns:
    None
    """
    if seed != 0:
        set_seed(seed)
    """
    Process a batch of chunks using the batch-enabled TTS model.
    """
    from pydub import AudioSegment
    import io
    import soundfile as sf

    # 1. Prepare batch for TTS
    texts = [chunk_data["text"] for chunk_data in batch]

    # All params are the same, so we take them from the first chunk
    shared_tts_params = batch[0].get("tts_params", tts_params)
    supported_params = {
        "exaggeration",
        "cfg_weight",
        "temperature",
        "min_p",
        "top_p",
        "repetition_penalty",
    }
    tts_args = {k: v for k, v in shared_tts_params.items() if k in supported_params}

    # 2. Generate audio in a batch (heuristic: only if lengths are similar and group size >1)
    try_batch = True
    # Determine once per run whether the model supports a batch API
    global _BATCH_API_SUPPORTED
    if _BATCH_API_SUPPORTED is None:
        _BATCH_API_SUPPORTED = hasattr(model, "generate_batch")
    if not _BATCH_API_SUPPORTED:
        try_batch = False
    # Honor config flag to disable micro-batching completely
    try:
        from config import config as _cfg

        if hasattr(_cfg, "ENABLE_MICRO_BATCHING") and not _cfg.ENABLE_MICRO_BATCHING:
            try_batch = False
    except Exception:
        pass
    try:
        # Heuristic using character lengths as a proxy for token length
        lens = [len(t) for t in texts]
        if len(lens) < 2:
            try_batch = False
        else:
            min_l, max_l = min(lens), max(lens)
            ratio = (max_l / max(1, min_l)) if min_l > 0 else 999.0
            # Threshold can be tuned; start conservative
            threshold = float(os.environ.get("GENTTS_MICROBATCH_LEN_RATIO", "1.8"))
            if ratio > threshold:
                try_batch = False
    except Exception:
        try_batch = True

    if try_batch:
        try:
            with torch.no_grad():
                # Try full batch with OOM backoff
                import gc as _gc

                def gen_with_backoff(text_list):
                    """Generate batches of text using a model with backoff for large inputs.
                    Args:
                    text_list: List of texts to generate audio for.
                    Returns:
                    List of generated audio samples.
                    """
                    size = len(text_list)
                    bs = size
                    results = []
                    while bs >= 1:
                        try:
                            if bs == size:
                                return model.generate_batch(text_list, **tts_args)
                            else:
                                results.clear()
                                for j in range(0, size, bs):
                                    subtexts = text_list[j : j + bs]
                                    subwavs = model.generate_batch(subtexts, **tts_args)
                                    results.extend(subwavs)
                                return results
                        except RuntimeError as _e:
                            msg = str(_e).lower()
                            if "out of memory" in msg or "cuda oom" in msg:
                                try:
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                except Exception:
                                    pass
                                _gc.collect()
                                new_bs = max(1, bs // 2)
                                logging.warning(
                                    f"⚠️ CUDA OOM at microbatch={bs}. Retrying with {new_bs}."
                                )
                                if new_bs == bs:
                                    # Cannot reduce further
                                    raise
                                bs = new_bs
                                continue
                            else:
                                raise

                wavs = gen_with_backoff(texts)
        except AttributeError as e:
            # Model has no batch API; disable for this run and fall back
            _BATCH_API_SUPPORTED = False
            try_batch = False
            logging.warning(
                f"Batch API unavailable on model; falling back to per‑chunk. Reason: {e}"
            )
        except Exception as e:
            # Other failures: fall back this time but keep batch enabled for future groups
            try_batch = False
            logging.warning(
                f"Batch generation failed; using per‑chunk for this group. Reason: {e}"
            )

    if not try_batch:
        # Fallback to individual processing for this batch
        results = []
        for chunk_data in batch:
            i = chunk_data["index"]
            chunk = chunk_data["text"]
            boundary_type = chunk_data.get("boundary_type", "none")
            chunk_tts_params = chunk_data.get("tts_params", tts_params)
            result = process_one_chunk(
                i,
                chunk,
                text_chunks_dir,
                audio_chunks_dir,
                voice_path,
                chunk_tts_params,
                start_time,
                total_chunks,
                punc_norm,
                basename,
                log_run_func,
                log_path,
                device,
                model,
                boundary_type=boundary_type,
                enable_asr=enable_asr,
                backend=backend,
            )
        results.append(result)
        return results

    # 3. Process and save each audio file from the batch
    batch_results = []
    for i, wav_tensor in enumerate(wavs):
        chunk_data = batch[i]
        chunk_index = chunk_data["index"]
        boundary_type = chunk_data.get("boundary_type", "none")
        chunk_id_str = f"{chunk_index + 1:05}"

        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        # Get sample rate from model (default to 24000 if not available)
        sample_rate = getattr(model, "sr", 24000) if model is not None else 24000

        wav_np = wav_tensor.squeeze().cpu().numpy()
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav_np, sample_rate, format="wav")
            wav_buffer.seek(0)
            audio_segment = AudioSegment.from_wav(wav_buffer)

        # Apply trimming and contextual silence
        from modules.audio_processor import (
            process_audio_with_trimming_and_silence,
            trim_audio_endpoint,
        )

        if boundary_type and boundary_type != "none":
            final_audio = process_audio_with_trimming_and_silence(
                audio_segment, boundary_type
            )
        elif ENABLE_AUDIO_TRIMMING:
            final_audio = trim_audio_endpoint(audio_segment)
        else:
            final_audio = audio_segment

        # Final save
        final_path = audio_chunks_dir / f"chunk_{chunk_id_str}.wav"
        final_audio.export(final_path, format="wav")
        logging.info(f"✅ Saved final chunk from batch: {final_path.name}")

        batch_results.append((chunk_index, final_path))

    return batch_results


# Global timestamp for inter-chunk gap measurement
_LAST_CHUNK_END_TS = None


def process_one_chunk(
    i,
    chunk,
    text_chunks_dir,
    audio_chunks_dir,
    voice_path,
    tts_params,
    start_time,
    total_chunks,
    punc_norm,
    basename,
    log_run_func,
    log_path,
    device,
    model,
    seed=0,
    boundary_type="none",
    enable_asr=None,
    fast_tts=None,
    backend=None,
):
    """Enhances chunk processing by applying quality control, contextual silence removal, and deep cleanup.
    Args:
    i (int): Index of the current chunk.
    chunk (str): The text content of the current chunk.
    text_chunks_dir (str): Directory for storing processed text chunks.
    audio_chunks_dir (str): Directory for storing processed audio chunks.
    voice_path (str): Path to the voice file.
    tts_params (dict): Parameters for text-to-speech synthesis.
    start_time (int): Start time of the processing.
    total_chunks (int): Total number of chunks.
    punc_norm (bool): Flag indicating punctuation normalization.
    basename (str): Base name for chunk files.
    log_run_func (func): Function to log runtime information.
    log_path (str): Path to log file.
    device (str): Device type for model inference.
    model (obj): Model object for processing.
    seed (int, optional): Seed value for random operations. Default is 0.
    boundary_type (str, optional): Type of boundaries. Default is "none".
    enable_asr (bool, optional): Flag to enable automatic speech recognition. Default is None.
    fast_tts (bool, optional): Flag to use fast TTS. Default is None.
    backend (str, optional): Backend type for processing
    """
    if seed != 0:
        set_seed(seed)
    """Enhanced chunk processing with quality control, contextual silence, and deep cleanup"""
    from pydub import AudioSegment

    # Debug: inter-chunk gap and backend selection context (only if TTS_TIMING enabled)
    if _TTS_TIMING:
        try:
            import time as _time
            from datetime import datetime as _dt

            now_start = _time.perf_counter()
            try:
                iso_start = _dt.now().isoformat(timespec="milliseconds")
                print(f"[ts] start={iso_start}")
            except Exception:
                pass
            try:
                if _LAST_CHUNK_END_TS is not None:
                    gap_ms = (now_start - _LAST_CHUNK_END_TS) * 1000.0
                    print(
                        f"[engine] gap_between_chunks={gap_ms:.3f}ms since previous end"
                    )
            except Exception:
                pass
            using = "auto"
            if backend is not None:
                if getattr(backend, "fast_tts", None) is not None:
                    using = "fast"
                elif getattr(backend, "model", None) is not None:
                    using = "standard"
            print(
                f"[backend] chunk {i + 1:05} using={using}, model={'set' if model else 'None'}, fast_tts={'set' if fast_tts else 'None'}"
            )
        except Exception:
            pass

    chunk_id_str = f"{i + 1:05}"
    chunk_path = text_chunks_dir / f"chunk_{chunk_id_str}.txt"
    with open(chunk_path, "w", encoding="utf-8") as cf:
        cf.write(chunk)

    chunk_audio_path = audio_chunks_dir / f"chunk_{chunk_id_str}.wav"

    # Spider dry-run: generate a short silent chunk and return quickly to map code paths
    try:
        import os as _os

        if _os.getenv("SPIDER_DRY_RUN", "0") == "1":
            silent = AudioSegment.silent(duration=500)  # 0.5s
            silent.export(chunk_audio_path, format="wav")
            return i, chunk_audio_path
    except Exception:
        pass

    # ============================================================================
    # ENHANCED PERIODIC DEEP CLEANUP
    # ============================================================================
    cleanup_interval = CLEANUP_INTERVAL

    # Skip cleanup on model reinitialization chunks to avoid conflicts
    if (i + 1) % cleanup_interval == 0 and (i + 1) % BATCH_SIZE != 0:
        print(f"\n🧹 {YELLOW}DEEP CLEANUP at chunk {i + 1}/{total_chunks}...{RESET}")

        # Enhanced VRAM monitoring before cleanup
        allocated_before = (
            torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        )
        reserved_before = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )

        print(
            f"   Before: VRAM Allocated: {allocated_before:.1f}GB | Reserved: {reserved_before:.1f}GB"
        )

        # Bulk temp file cleanup
        print("   🗑️ Cleaning bulk temporary files...")
        temp_patterns = [
            "*_try*.wav",
            "*_pre.wav",
            "*_fade*.wav",
            "*_debug*.wav",
            "*_temp*.wav",
            "*_backup*.wav",
        ]
        total_temp_files = 0
        for pattern in temp_patterns:
            temp_files = list(audio_chunks_dir.glob(pattern))
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            total_temp_files += len(temp_files)

        if total_temp_files > 0:
            print(f"   🗑️ Removed {total_temp_files} temporary audio files")

        # Aggressive CUDA context reset
        print("   🔄 Performing aggressive CUDA context reset...")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Force CUDA context reset
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear model cache if it has one
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        elif hasattr(model, "reset_states"):
            model.reset_states()

        # Brief pause to let GPU settle
        time.sleep(1.0)

        # Monitor after cleanup
        allocated_after = (
            torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        )
        reserved_after = (
            torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
        )

        print(
            f"   After:  VRAM Allocated: {allocated_after:.1f}GB | Reserved: {reserved_after:.1f}GB"
        )
        print(
            f"   Freed:  {allocated_before - allocated_after:.1f}GB allocated, {reserved_before - reserved_after:.1f}GB reserved"
        )
        print(f"🧹 {GREEN}Deep cleanup complete!{RESET}\n")

    best_sim, best_asr_text = -1, ""
    wav_path_active = None
    attempt_paths = []
    mid_drop_retries = 0
    max_mid_drop_retries = 2

    # Enhanced regeneration loop with quality validation
    max_attempts = MAX_REGENERATION_ATTEMPTS if ENABLE_REGENERATION_LOOP else 2
    current_tts_params = tts_params.copy()

    # Debug: Log the initial parameters for this chunk
    logging.info(
        f"🎛️ Chunk {chunk_id_str} initial TTS params: exag={current_tts_params.get('exaggeration', 'N/A'):.3f}, cfg={current_tts_params.get('cfg_weight', 'N/A'):.3f}, temp={current_tts_params.get('temperature', 'N/A'):.3f}, min_p={current_tts_params.get('min_p', 'N/A'):.3f}"
    )

    # Run pre-conversion aggressive cleanup once at the start of a session
    global _SESSION_ACTIVE
    if not _SESSION_ACTIVE:
        try:
            if torch.cuda.is_available():
                print("🧹 Performed aggressive CUDA cleanup")
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        _SESSION_ACTIVE = True

    raw_wave_path = None
    backend_result = None

    for attempt_num in range(max_attempts):
        logging.info(
            f"🔁 Starting TTS for chunk {chunk_id_str}, attempt {attempt_num + 1}/{max_attempts}"
        )
        if attempt_num > 0:
            logging.info(
                f"🔧 Adjusted params: exag={current_tts_params.get('exaggeration', 'N/A'):.3f}, cfg={current_tts_params.get('cfg_weight', 'N/A'):.3f}, temp={current_tts_params.get('temperature', 'N/A'):.3f}"
            )

        if attempt_num > 0 and raw_wave_path is not None:
            try:
                raw_wave_path.unlink(missing_ok=True)  # clean previous raw output
            except Exception:
                pass
            raw_wave_path = None

        backend_result = None
        wav = None
        audio_segment = None
        try:
            # Ensure a backend exists even if not threaded by caller
            if backend is None:
                # Reuse session-scoped backend when available
                global _SESSION_BACKEND
                if _SESSION_BACKEND is not None:
                    backend = _SESSION_BACKEND

                try:
                    from config import config as _cfg_local

                    # Choose backend based on config.TTS_BACKEND
                    backend_choice = getattr(_cfg_local, "TTS_BACKEND", "standard")
                    if backend is None and backend_choice == "fast":
                        try:
                            # Avoid dual load only when switching from a different backend
                            # and no session backend is active
                            if torch.cuda.is_available():
                                # light cache trim without full release when continuing same backend
                                torch.cuda.empty_cache()
                            # Skip fast init if free VRAM is too low
                            try:
                                free_bytes, total_bytes = torch.cuda.mem_get_info()
                                free_gb = free_bytes / (1024**3)
                            except Exception:
                                free_gb = 0.0
                            if free_gb and free_gb < 4.5:
                                msg = f"Insufficient free VRAM for fast backend: {free_gb:.2f} GB (< 4.5 GB required)"
                                print(f"[backend] {msg}")
                                raise RuntimeError(msg)
                            from modules.t3_fast_backend import get_fast_tts_cached
                            from modules.backend_context import BackendContext

                            # Suppress HF Llama warnings that break torch.compile/dynamo graphs (set once, outside compiled regions)
                            hf_logger = None
                            _prev_hf_level = None
                            try:
                                hf_logger = logging.getLogger(
                                    "transformers.models.llama.modeling_llama"
                                )
                                _prev_hf_level = hf_logger.level
                                hf_logger.setLevel(logging.ERROR)
                            except Exception:
                                hf_logger = None
                                _prev_hf_level = None
                            # Use float16 on consumer GPUs for better kernel support
                            fast = get_fast_tts_cached(
                                device=str(device), dtype="float16", compile_step=True
                            )
                            # Restore logger level
                            if hf_logger is not None and _prev_hf_level is not None:
                                try:
                                    hf_logger.setLevel(_prev_hf_level)
                                except Exception:
                                    pass
                            backend = BackendContext(model=None, fast_tts=fast)
                            _SESSION_BACKEND = backend
                            print(
                                f"[backend] auto-initialized fast context for chunk {chunk_id_str}"
                            )
                        except Exception as _e:
                            # Hard-fail semantics for fast backend
                            raise RuntimeError(
                                f"Fast backend initialization failed: {_e}"
                            )
                    elif backend is None and backend_choice == "vllm":
                        # Try unified_tts which can provide VLLM or Standard backends
                        try:
                            from modules import unified_tts as _unified

                            uni = _unified.get_tts_backend(_cfg_local)
                            from modules.backend_context import BackendContext

                            backend = BackendContext(unified=uni)
                            _SESSION_BACKEND = backend
                            print(
                                f"[backend] auto-initialized unified context ({uni.__class__.__name__}) for chunk {chunk_id_str}"
                            )
                        except Exception as _e:
                            import traceback as _tb

                            _tb.print_exc()
                            # Fallback to legacy standard model path below
                            print(
                                f"[backend] unified backend unavailable ({_e}), falling back to standard model"
                            )
                            pass
                    if backend is None:
                        # If user selected fast, do not fallback silently
                        if backend_choice == "fast":
                            raise RuntimeError(
                                "Fast backend required but not available (no fallback)"
                            )
                        # Otherwise proceed to standard/unified
                        try:
                            std_model = load_optimized_model(device, force_reload=False)
                            from modules.backend_context import BackendContext

                            backend = BackendContext(model=std_model, fast_tts=None)
                            _SESSION_BACKEND = backend
                            print(
                                f"[backend] auto-initialized standard context for chunk {chunk_id_str}"
                            )
                        except Exception as _e:
                            raise RuntimeError(
                                f"Auto-init standard backend failed: {_e}"
                            )
                except Exception as _e:
                    raise RuntimeError(
                        f"No TTS backend available and auto-init failed: {_e}"
                    )
            # Filter to only supported ChatterboxTTS parameters
            supported_params = {
                "exaggeration",
                "cfg_weight",
                "temperature",
                "min_p",
                "top_p",
                "repetition_penalty",
            }
            tts_args = {
                k: v for k, v in current_tts_params.items() if k in supported_params
            }

            # Smart clamp: avoid CPU tokenization in hot path unless clearly needed
            try:
                max_ctx = int(getattr(_cfg, "MAX_T3_CONTEXT", 640) or 0)
            except Exception:
                max_ctx = 640
            if max_ctx > 0:
                # Use precomputed token_len from JSON if available via nearby scope
                # Fallback to word-count heuristic; only tokenize when necessary
                need_clamp = False
                try:
                    # chunk may come from JSON with a nearby variable or closure providing token_len
                    # Attempt to infer need_clamp cheaply
                    approx_tokens = max(1, int(len(chunk.split())))
                    # Heuristic: only consider clamp if clearly above cap
                    if approx_tokens > max_ctx + 16:
                        need_clamp = True
                except Exception:
                    pass
                if need_clamp and _T3_TOKENIZER is not None:
                    try:
                        ids = _T3_TOKENIZER.encode(chunk)
                        if len(ids) > max_ctx:
                            ids = ids[:max_ctx]
                            chunk = _T3_TOKENIZER.decode(ids)
                    except Exception:
                        pass

            chunk_start_time = time.time()
            try:
                with torch.no_grad():
                    # Unified backend generate via BackendContext
                    # Use GUI-provided cfg_weight; fast sampler now supports CFG properly
                    effective_cfg = current_tts_params.get("cfg_weight", 0.0)
                    # Use GUI-provided repetition_penalty
                    effective_rep_pen = current_tts_params.get(
                        "repetition_penalty", 1.2
                    )
                    sample_rate = 24000
                    backend_params = {
                        "exaggeration": current_tts_params.get("exaggeration", 0.5),
                        "cfg_weight": effective_cfg,
                        "temperature": current_tts_params.get("temperature", 0.85),
                        "min_p": current_tts_params.get("min_p", 0.05),
                        "top_p": current_tts_params.get("top_p", 1.0),
                        "repetition_penalty": effective_rep_pen,
                    }
                    if "language_id" in current_tts_params:
                        backend_params["language_id"] = current_tts_params[
                            "language_id"
                        ]
                    elif "language" in current_tts_params:
                        backend_params["language_id"] = current_tts_params["language"]
                    if "diffusion_steps" in current_tts_params:
                        backend_params["diffusion_steps"] = current_tts_params[
                            "diffusion_steps"
                        ]
                    backend_params["max_tokens"] = max_ctx

                    _gen_t0 = time.perf_counter()
                    if getattr(backend, "unified", None) is not None:
                        backend_result = backend.generate(
                            chunk,
                            str(voice_path) if voice_path else None,
                            backend_params,
                            chunk_output_dir=audio_chunks_dir,
                            chunk_id=chunk_id_str,
                        )
                        wav = backend_result.waveform.detach().cpu()
                        sample_rate = backend_result.sample_rate or 24000
                        raw_wave_path = backend_result.raw_path
                        t3_elapsed = time.perf_counter() - _gen_t0
                    else:
                        # Standard/fast backend path
                        backend_result = backend.generate(
                            chunk,
                            str(voice_path) if voice_path else None,
                            backend_params,
                        )
                        wav = backend_result.waveform.detach().cpu()
                        sample_rate = backend_result.sample_rate or 24000
                        t3_elapsed = time.perf_counter() - _gen_t0
                        # Guaranteed capture: query S3 input tokens from fast backend if accessible; otherwise
                        # capture via a lightweight hook on backend (preferred), and if unavailable, skip silently.
                        try:
                            import json as _json
                            from pathlib import Path as _Path

                            toks = None
                            # Preferred: fast backend exposes last_t3_tokens
                            if hasattr(backend, "fast_tts") and hasattr(
                                backend.fast_tts, "last_t3_tokens"
                            ):
                                toks = backend.fast_tts.last_t3_tokens
                            # Alternate: unified attribute on backend
                            if toks is None and hasattr(backend, "last_t3_tokens"):
                                toks = getattr(backend, "last_t3_tokens")
                            if toks is not None:
                                arr = (
                                    toks.detach().cpu().view(-1).tolist()
                                    if hasattr(toks, "detach")
                                    else list(toks)
                                )
                                if isinstance(arr, list) and len(arr) > 0:
                                    out_dir = _Path("logs")
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    out_fp = out_dir / "t3_tokens.jsonl"
                                    rec = {
                                        "chunk_id": chunk_id_str,
                                        "text": chunk,
                                        "tokens": arr,
                                        "source": "fast_local",
                                    }
                                    with open(out_fp, "a") as f:
                                        f.write(_json.dumps(rec) + "\n")
                                    try:
                                        print(
                                            f"[capture] saved T3 tokens for chunk {chunk_id_str}: n={len(arr)} → {out_fp}"
                                        )
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                    _gen_t1 = time.perf_counter()
                    # CPU post-processing breakdown starts now (wav is ready)
                    import time as _time

                    cpu_start = _time.perf_counter()
                    if _TTS_TIMING:
                        try:
                            print("[engine] cpu_breakdown_ready")
                        except Exception:
                            pass
                    handoff_gap_ms = None
                    if _TTS_TIMING:
                        try:
                            import builtins as _bi

                            s3_end = _bi.__dict__.get("LAST_S3_END_TS", None)
                            if s3_end is not None:
                                handoff_gap_ms = (cpu_start - s3_end) * 1000.0
                        except Exception:
                            pass
                    # Breakdown timers
                    t0 = _time.perf_counter()
                    wav_cpu = wav  # already .detach().cpu() above
                    t1 = _time.perf_counter()
                    wav_np = None
                    # Prefer loading from raw vLLM file if available to exercise Audio Finishing Pipeline on disk output
                    if raw_wave_path is not None and raw_wave_path.exists():
                        try:
                            import soundfile as _sf

                            wav_loaded, file_sr = _sf.read(
                                raw_wave_path, dtype="float32"
                            )
                            if (
                                isinstance(wav_loaded, np.ndarray)
                                and wav_loaded.ndim > 1
                            ):
                                wav_loaded = wav_loaded.mean(axis=1)
                            wav_np = wav_loaded.astype(np.float32, copy=False)
                            sample_rate = int(file_sr)
                        except Exception as _raw_exc:
                            logging.warning(
                                f"⚠️ Failed to read raw vLLM audio from {raw_wave_path}: {_raw_exc}"
                            )
                            wav_np = None
                    # If raw file unavailable, fall back to in-memory tensor
                    if wav_np is None:
                        try:
                            wav_np = wav_cpu.squeeze().numpy()
                        except Exception:
                            wav_np = wav_cpu.squeeze().detach().cpu().numpy()
                    t2 = _time.perf_counter()
                    # watermark
                    try:
                        watermarked_wav = self.watermarker.apply_watermark(
                            wav_np, sample_rate=self.sr
                        )
                    except Exception:
                        watermarked_wav = wav_np
                    t3 = _time.perf_counter()
                    # Build AudioSegment in-memory (no export yet) to time decode path
                    t4 = _time.perf_counter()
                    try:
                        import io as _io
                        import soundfile as _sf
                        from pydub import AudioSegment as _ASeg

                        with _io.BytesIO() as _buf:
                            _sf.write(_buf, watermarked_wav, self.sr, format="wav")
                            _buf.seek(0)
                            _ = _ASeg.from_wav(_buf)
                    except Exception:
                        pass
                    t5 = _time.perf_counter()
                    # Print partial CPU breakdown (timing mode only)
                    if _TTS_TIMING:
                        try:
                            to_numpy_ms = (t2 - t1) * 1000.0
                            watermark_ms = (t3 - t2) * 1000.0
                            audioseg_ms = (t5 - t4) * 1000.0
                            if handoff_gap_ms is not None:
                                print(
                                    f"[engine] handoff_gap={handoff_gap_ms:.3f}ms cpu_post_partial={(to_numpy_ms + watermark_ms + audioseg_ms):.3f}ms parts: to_numpy={to_numpy_ms:.3f} watermark={watermark_ms:.3f} audioseg={audioseg_ms:.3f}"
                                )
                            else:
                                print(
                                    f"[engine] cpu_post_partial={(to_numpy_ms + watermark_ms + audioseg_ms):.3f}ms parts: to_numpy={to_numpy_ms:.3f} watermark={watermark_ms:.3f} audioseg={audioseg_ms:.3f}"
                                )
                        except Exception:
                            pass
            except RuntimeError as e:
                if "probability tensor contains either" in str(e):
                    logging.warning(
                        f"⚠️ Chunk {chunk_id_str} failed in mixed precision. Retrying in FP32..."
                    )
                    from modules.real_tts_optimizer import get_tts_optimizer

                    optimizer = get_tts_optimizer()
                    with optimizer.fp32_fallback_mode():
                        with torch.no_grad():
                            with _GPU_INFER_LOCK:
                                wav = (
                                    model.generate(
                                        chunk, **tts_args, disable_watermark=True
                                    )
                                    .detach()
                                    .cpu()
                                )
                    logging.info(
                        f"✅ Chunk {chunk_id_str} successfully generated in FP32 fallback mode."
                    )
                else:
                    # If fast is selected, do not fallback — hard fail
                    try:
                        from config import config as _cfg_local

                        backend_choice = getattr(_cfg_local, "TTS_BACKEND", "standard")
                    except Exception:
                        backend_choice = "standard"
                    if backend_choice == "fast":
                        raise RuntimeError(f"Fast backend generation failed: {e}")
                    else:
                        raise  # Re-raise other runtime errors

            chunk_processing_time = time.time() - chunk_start_time
            try:
                # Build audio segment and export timings (sync pre-measure for partial)
                import time as _time

                t4 = _time.perf_counter()
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, watermarked_wav, sample_rate, format="wav")
                    wav_buffer.seek(0)
                    audio_segment = AudioSegment.from_wav(wav_buffer)
                t5 = _time.perf_counter()
                # performance.log write timed below
                log_t0 = _time.perf_counter()
                if _TTS_TIMING:
                    print(
                        f"[engine] generate={(_gen_t1 - _gen_t0):.3f}s total={chunk_processing_time:.3f}s for chunk {chunk_id_str}"
                    )
                # Update total audio duration immediately to restore ETA line
                try:
                    duration_sec = 0.0
                    try:
                        duration_sec = float(watermarked_wav.shape[-1]) / float(
                            sample_rate
                        )
                    except Exception:
                        try:
                            duration_sec = float(len(audio_segment)) / 1000.0
                        except Exception:
                            duration_sec = 0.0
                    globals()["total_audio_duration"] = (
                        globals().get("total_audio_duration", 0.0) + duration_sec
                    )
                    from modules.progress_tracker import log_chunk_progress

                    log_chunk_progress(
                        i, total_chunks, start_time, globals()["total_audio_duration"]
                    )
                except Exception:
                    pass
            except Exception:
                pass
            # Mark end timestamp for inter-chunk gap measurement
            try:
                # Assign to module-level end timestamp
                globals()["_LAST_CHUNK_END_TS"] = _time.perf_counter()
                if _TTS_TIMING:
                    from datetime import datetime as _dt

                    iso_end = _dt.now().isoformat(timespec="milliseconds")
                    print(f"[ts] end={iso_end}")
            except Exception:
                pass
            with open("performance.log", "a") as perf_log:
                perf_log.write(f"{i},{len(chunk)},{chunk_processing_time:.4f}\n")
            try:
                log_t1 = time.perf_counter()
                print(f"[engine] log_write={(log_t1 - log_t0) * 1000.0:.3f}ms")
            except Exception:
                pass

            if wav is None:
                raise RuntimeError("Waveform is None after generation attempt.")

            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            # Convert tensor to AudioSegment for in-memory processing
            import io
            import soundfile as sf
            from pydub import AudioSegment

            # Get sample rate from backend or model
            sample_rate = 24000  # Default sample rate
            if backend is not None:
                if (
                    hasattr(backend, "unified")
                    and backend.unified is not None
                    and hasattr(backend.unified, "sr")
                ):
                    sample_rate = backend.unified.sr()
                elif (
                    hasattr(backend, "model")
                    and backend.model is not None
                    and hasattr(backend.model, "sr")
                ):
                    sample_rate = backend.model.sr
                elif (
                    hasattr(backend, "fast_tts")
                    and backend.fast_tts is not None
                    and hasattr(backend.fast_tts, "sr")
                ):
                    sample_rate = backend.fast_tts.sr
            elif model is not None and hasattr(model, "sr"):
                sample_rate = model.sr

            # Convert wav tensor to AudioSegment (in memory)
            wav_np = wav.squeeze().numpy()
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, wav_np, sample_rate, format="wav")
                wav_buffer.seek(0)
                audio_segment = AudioSegment.from_wav(wav_buffer)

            # Enhanced quality validation
            quality_score = 1.0  # Start with perfect score

            # Legacy mid-energy drop check (converted to score)
            if ENABLE_MID_DROP_CHECK and has_mid_energy_drop(wav, sample_rate):
                quality_score *= 0.3  # Significant penalty for mid-drop
                logging.info(f"⚠️ Mid-chunk energy drop detected in {chunk_id_str}")

            # Enhanced quality validation (if enabled)
            if ENABLE_REGENERATION_LOOP:
                from modules.audio_processor import evaluate_chunk_quality

                composite_score = evaluate_chunk_quality(
                    audio_segment, chunk, include_spectral=True
                )
                quality_score *= composite_score
                logging.info(
                    f"📊 Quality score for {chunk_id_str}: {quality_score:.3f} (composite: {composite_score:.3f})"
                )

            # ASR validation via isolated subprocess
            asr_score = 1.0
            asr_text = ""
            asr_enabled = enable_asr if enable_asr is not None else ENABLE_ASR
            if asr_enabled:
                import tempfile
                import json as _json

                temp_audio_path = None
                temp_text_path = None
                try:
                    # Write audio and text to temp files
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        temp_audio_path = Path(tf.name)
                        audio_segment.export(tf, format="wav")
                    with tempfile.NamedTemporaryFile(
                        suffix=".txt", delete=False, mode="w", encoding="utf-8"
                    ) as tf:
                        temp_text_path = Path(tf.name)
                        tf.write(chunk)

                    asr_script = (
                        Path(__file__).parent.parent
                        / "ASR"
                        / "asr_validator_headless.py"
                    )
                    asr_python = sys.executable
                    cmd = [
                        str(asr_python),
                        str(asr_script),
                        "--audio-file",
                        str(temp_audio_path),
                        "--text-file",
                        str(temp_text_path),
                        "--threshold",
                        str(DEFAULT_ASR_THRESHOLD),
                        "--json",
                    ]
                    logging.info(f"Running per-chunk ASR validation for {chunk_id_str}")
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        cwd=str(asr_script.parent),
                    )

                    if proc.returncode in (0, 2):
                        result = _json.loads(proc.stdout)
                        asr_score = result.get("score", 0.0)
                        asr_text = result.get("asr_text", "")
                        logging.info(
                            f"ASR similarity for {chunk_id_str}: {asr_score:.3f} "
                            f"- Expected: '{punc_norm(chunk)}' Got: '{asr_text}'"
                        )
                    else:
                        logging.error(
                            f"ASR subprocess failed for {chunk_id_str} (rc={proc.returncode})"
                        )

                        # Try to parse error JSON from stdout
                        error_logged = False
                        if proc.stdout.strip():
                            try:
                                error_result = _json.loads(proc.stdout)
                                if "error" in error_result:
                                    logging.error(f"ASR error: {error_result['error']}")
                                    if "traceback" in error_result:
                                        logging.error(
                                            f"ASR traceback:\n{error_result['traceback']}"
                                        )
                                    error_logged = True
                            except _json.JSONDecodeError:
                                pass

                        # Fallback: log raw stdout/stderr if JSON parsing failed
                        if not error_logged:
                            if proc.stdout.strip():
                                logging.error(f"ASR stdout: {proc.stdout.strip()}")
                            if proc.stderr.strip():
                                logging.error(f"ASR stderr: {proc.stderr.strip()}")

                        asr_score = 0.8  # Neutral score on subprocess failure
                except Exception as e:
                    logging.error(f"ASR validation error for {chunk_id_str}: {e}")
                    asr_score = 0.8
                finally:
                    for p in (temp_audio_path, temp_text_path):
                        if p and p.exists():
                            try:
                                p.unlink()
                            except Exception:
                                pass

                quality_score *= asr_score

            # Final quality check with all validations
            if quality_score >= QUALITY_THRESHOLD or attempt_num == max_attempts - 1:
                if quality_score >= QUALITY_THRESHOLD:
                    logging.info(
                        f"✅ Quality acceptable for {chunk_id_str} on attempt {attempt_num + 1} (final score: {quality_score:.3f})"
                    )
                else:
                    logging.info(
                        f"⚠️ Max attempts reached for {chunk_id_str}, accepting best effort (final score: {quality_score:.3f})"
                    )

                # Quality acceptable or max attempts reached, continue with processing
                final_audio = audio_segment
                best_sim = asr_score if asr_enabled else 1.0
                best_asr_text = (
                    asr_text if asr_enabled and "asr_text" in locals() else ""
                )
                break
            else:
                # Quality too low, adjust parameters for retry
                logging.info(
                    f"🔄 Quality below threshold ({quality_score:.3f} < {QUALITY_THRESHOLD}), adjusting parameters for retry {attempt_num + 2}"
                )
                from modules.audio_processor import adjust_parameters_for_retry

                current_tts_params = adjust_parameters_for_retry(
                    current_tts_params, quality_score, attempt_num
                )
                continue

        except Exception as e:
            import traceback

            logging.error(
                f"Exception during TTS attempt {attempt_num + 1} for chunk {chunk_id_str}: {e}"
            )
            traceback.print_exc()
            continue

    if "final_audio" not in locals():
        logging.info(f"❌ Chunk {chunk_id_str} failed all attempts.")
        return None, None

    # Apply trimming and contextual silence in memory before final save
    from modules.audio_processor import process_audio_with_trimming_and_silence

    if boundary_type and boundary_type != "none":
        final_audio = process_audio_with_trimming_and_silence(
            final_audio, boundary_type
        )
        # Log silence addition to file only to avoid console noise
        try:
            from modules.terminal_logger import log_only

            log_only(f"🔇 Added {boundary_type} silence to chunk {i + 1:05}")
        except Exception:
            pass
    else:
        # Apply trimming even without boundary type if enabled
        if ENABLE_AUDIO_TRIMMING:
            from modules.audio_processor import trim_audio_endpoint

            final_audio = trim_audio_endpoint(final_audio)

    # Note: ENABLE_CHUNK_END_SILENCE is now handled by punctuation-specific silence
    # The new system provides more precise silence based on actual punctuation

    # Final save – either async or sync depending on flag
    if ASYNC_EXPORT and _ensure_export_executor() is not None:
        try:
            _ensure_export_executor()
            # hand off to background worker; do not block next chunk
            _EXPORT_EXECUTOR.submit(
                _export_task_fn,
                watermarked_wav,
                sample_rate,
                audio_chunks_dir,
                chunk_id_str,
                boundary_type,
                ENABLE_AUDIO_TRIMMING,
            )
            if _TTS_TIMING:
                logging.info(f"🚀 Queued async export for chunk {chunk_id_str}")
        except Exception as _e:
            logging.error(f"Async export queue failed, falling back to sync: {_e}")
            ASYNC_EXPORT_LOCAL = False
        else:
            ASYNC_EXPORT_LOCAL = True
    else:
        ASYNC_EXPORT_LOCAL = False

    if not ASYNC_EXPORT_LOCAL:
        final_path = audio_chunks_dir / f"chunk_{chunk_id_str}.wav"
        _export_t0 = time.perf_counter()
        final_audio.export(final_path, format="wav")
        _export_t1 = time.perf_counter()
        if _TTS_TIMING:
            try:
                print(f"[engine] export={((_export_t1 - _export_t0) * 1000.0):.3f}ms")
            except Exception:
                pass
        logging.info(f"✅ Saved final chunk: {final_path.name}")
        if raw_wave_path is not None:
            try:
                raw_wave_path.unlink(missing_ok=True)
            except Exception as _cleanup_exc:
                logging.debug(f"Raw vLLM audio cleanup skipped ({_cleanup_exc})")

    # Emit one per-chunk sampling summary to console (always on)
    try:
        from modules.terminal_logger import emit_chunk_summary

        emit_chunk_summary()
    except Exception:
        pass

    # Emit ETA/progress line every ~5 chunks using progress_tracker (exact format)
    try:
        from modules.progress_tracker import log_chunk_progress

        # Compute elapsed and total audio duration if available
        # Elapsed from global start_time passed into process_one_chunk
        elapsed = max(0.0, time.time() - start_time)
        # Best-effort total audio duration: sum of durations is not tracked here; pass 0 to let tracker print without audio until known
        total_audio_duration = 0.0
        log_chunk_progress(
            i, total_chunks, elapsed, total_audio_duration, log_chunk_progress
        )
    except Exception:
        pass

    # Progress updates with accurate realtime are handled by batch/micro-batch code
    # which passes measured total_audio_duration. Avoid printing extra partial lines here.

    # No intermediate file cleanup needed - all processing done in memory

    # Avoid duplicate progress updates here as well.

    # Log details - only log ASR failures
    if asr_enabled and best_sim < 0.8:
        log_run_func(
            f"ASR VALIDATION FAILED - Chunk {chunk_id_str}:\nExpected:\n{chunk}\nActual:\n{best_asr_text}\nSimilarity: {best_sim:.3f}\n"
            + "=" * 50,
            log_path,
        )
    elif not asr_enabled:
        log_run_func(f"Chunk {chunk_id_str}: Original text: {chunk}", log_path)

    # Silence already added in memory above - no disk processing needed

    # Enhanced regular cleanup (every chunk)
    del wav
    optimize_memory_usage()

    # Additional per-chunk cleanup for long runs
    if (i + 1) % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()

    if raw_wave_path is not None and raw_wave_path.exists():
        try:
            raw_wave_path.unlink(missing_ok=True)
        except Exception:
            pass

    return i, final_path


# ============================================================================
# MAIN BOOK PROCESSING FUNCTION
# ============================================================================

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wrapper.chunk_loader import save_chunks


def smooth_sentiment_scores(scores, index, method="rolling", window=3):
    """
    Apply sentiment smoothing to prevent harsh emotional transitions.

    Args:
        scores: List of compound sentiment scores
        index: Current chunk index
        method: "rolling" for moving average, "exp_decay" for exponential decay
        window: Number of previous chunks to consider

    Returns:
        float: Smoothed sentiment score
    """
    if index == 0:
        return scores[0]

    start_idx = max(0, index - window + 1)
    window_scores = scores[start_idx : index + 1]

    if method == "rolling":
        return sum(window_scores) / len(window_scores)
    elif method == "exp_decay":
        weights = SENTIMENT_EXP_DECAY_WEIGHTS[: len(window_scores)]
        weighted_sum = sum(w * s for w, s in zip(weights, reversed(window_scores)))
        weight_sum = sum(weights[: len(window_scores)])
        return weighted_sum / weight_sum if weight_sum > 0 else window_scores[-1]
    else:
        return scores[index]  # No smoothing


def generate_enriched_chunks(
    text_file,
    output_dir,
    user_tts_params=None,
    quality_params=None,
    config_params=None,
    voice_name=None,
):
    """Reads a text file, performs VADER sentiment analysis, and returns enriched chunks."""
    analyzer = SentimentIntensityAnalyzer()

    # Extract quality parameters for JSON generation (GUI overrides config)
    if quality_params:
        enable_smoothing = quality_params.get(
            "sentiment_smoothing", ENABLE_SENTIMENT_SMOOTHING
        )
        smoothing_window = quality_params.get(
            "smoothing_window", SENTIMENT_SMOOTHING_WINDOW
        )
        smoothing_method = quality_params.get(
            "smoothing_method", SENTIMENT_SMOOTHING_METHOD
        )
        print(
            f"🔧 JSON Generation: Using GUI smoothing settings - Enabled: {enable_smoothing}, Window: {smoothing_window}, Method: {smoothing_method}"
        )
    else:
        enable_smoothing = ENABLE_SENTIMENT_SMOOTHING
        smoothing_window = SENTIMENT_SMOOTHING_WINDOW
        smoothing_method = SENTIMENT_SMOOTHING_METHOD
        print(
            f"🔧 JSON Generation: Using config smoothing settings - Enabled: {enable_smoothing}"
        )

    # Extract only emotional VADER sensitivities; CFG remains fixed per run.
    if config_params:
        vader_exag_sensitivity = config_params.get(
            "vader_exag_sensitivity", VADER_EXAGGERATION_SENSITIVITY
        )
        vader_temp_sensitivity = config_params.get(
            "vader_temp_sensitivity", VADER_TEMPERATURE_SENSITIVITY
        )
        print(
            f"🔧 JSON Generation: Using GUI VADER sensitivity - Exag: {vader_exag_sensitivity}, Temp: {vader_temp_sensitivity}"
        )
    else:
        vader_exag_sensitivity = VADER_EXAGGERATION_SENSITIVITY
        vader_temp_sensitivity = VADER_TEMPERATURE_SENSITIVITY
        print(
            f"🔧 JSON Generation: Using config VADER sensitivity - Exag: {vader_exag_sensitivity}, Temp: {vader_temp_sensitivity}"
        )

    raw_text = text_file.read_text(encoding="utf-8")
    cleaned = smart_punctuate(raw_text)
    # Allow GUI/runtime overrides for chunk sizing via config_params
    try:
        max_words_override = None
        min_words_override = None
        if config_params:
            max_words_override = int(
                config_params.get("max_chunk_words", MAX_CHUNK_WORDS)
            )
            min_words_override = int(
                config_params.get("min_chunk_words", MIN_CHUNK_WORDS)
            )
        chunks = sentence_chunk_text(
            cleaned,
            max_words=max_words_override
            if max_words_override is not None
            else MAX_CHUNK_WORDS,
            min_words=min_words_override
            if min_words_override is not None
            else MIN_CHUNK_WORDS,
        )
    except Exception:
        # Fallback to config defaults if overrides invalid
        chunks = sentence_chunk_text(cleaned)

    # Use user-provided parameters as base, or fall back to config defaults
    if user_tts_params:
        base_exaggeration = user_tts_params.get("exaggeration", BASE_EXAGGERATION)
        base_cfg_weight = user_tts_params.get("cfg_weight", BASE_CFG_WEIGHT)
        base_temperature = user_tts_params.get("temperature", BASE_TEMPERATURE)
        base_min_p = user_tts_params.get("min_p", DEFAULT_MIN_P)
        base_top_p = user_tts_params.get("top_p", DEFAULT_TOP_P)
        base_repetition_penalty = user_tts_params.get(
            "repetition_penalty", DEFAULT_REPETITION_PENALTY
        )
        use_vader = user_tts_params.get(
            "use_vader", True
        )  # Default to True for backward compatibility

    else:
        base_exaggeration = BASE_EXAGGERATION
        base_cfg_weight = BASE_CFG_WEIGHT
        base_temperature = BASE_TEMPERATURE
        base_min_p = DEFAULT_MIN_P
        base_top_p = DEFAULT_TOP_P
        base_repetition_penalty = DEFAULT_REPETITION_PENALTY
        use_vader = True  # Default behavior

    enriched = []
    chunk_texts = [chunk_text for chunk_text, _ in chunks]

    # First pass: collect all sentiment scores
    raw_sentiment_scores = []
    for chunk_text, _ in chunks:
        sentiment_scores = analyzer.polarity_scores(chunk_text)
        raw_sentiment_scores.append(sentiment_scores["compound"])

    # Second pass: apply smoothing and generate parameters
    token_lengths = []
    for i, (chunk_text, is_para_end) in enumerate(chunks):
        # Get original sentiment score
        raw_compound_score = raw_sentiment_scores[i]

        # Apply sentiment smoothing if enabled (uses GUI settings, not config)
        if use_vader and enable_smoothing:
            compound_score = smooth_sentiment_scores(
                raw_sentiment_scores,
                i,
                method=smoothing_method,
                window=smoothing_window,
            )
            # Debug: Log sentiment changes
            if abs(compound_score - raw_compound_score) > 0.1:
                logging.info(
                    f"📊 Chunk {i + 1:05}: sentiment smoothed {raw_compound_score:.3f} → {compound_score:.3f}"
                )
        else:
            compound_score = raw_compound_score

        if use_vader:
            # VADER controls only emotional intensity and sampling temperature.
            exaggeration = base_exaggeration + (compound_score * vader_exag_sensitivity)
            temperature = base_temperature + (compound_score * vader_temp_sensitivity)
            cfg_weight = base_cfg_weight
            min_p = base_min_p
            repetition_penalty = base_repetition_penalty

            # Clamp values to defined min/max (ensure JSON values respect bounds)
            exaggeration = round(
                max(
                    TTS_PARAM_MIN_EXAGGERATION,
                    min(exaggeration, TTS_PARAM_MAX_EXAGGERATION),
                ),
                2,
            )
            cfg_weight = round(
                max(
                    TTS_PARAM_MIN_CFG_WEIGHT, min(cfg_weight, TTS_PARAM_MAX_CFG_WEIGHT)
                ),
                2,
            )
            temperature = round(
                max(
                    TTS_PARAM_MIN_TEMPERATURE,
                    min(temperature, TTS_PARAM_MAX_TEMPERATURE),
                ),
                2,
            )
            min_p = round(max(TTS_PARAM_MIN_MIN_P, min(min_p, TTS_PARAM_MAX_MIN_P)), 3)
            repetition_penalty = round(
                max(
                    TTS_PARAM_MIN_REPETITION_PENALTY,
                    min(repetition_penalty, TTS_PARAM_MAX_REPETITION_PENALTY),
                ),
                1,
            )

            # Debug: Log VADER-adjusted parameters for significant changes
            if (
                abs(exaggeration - base_exaggeration) > 0.05
                or abs(temperature - base_temperature) > 0.05
            ):
                logging.info(
                    f"🎭 Chunk {i + 1:05}: VADER adjusted params - exag: {base_exaggeration:.2f}→{exaggeration:.2f}, temp: {base_temperature:.2f}→{temperature:.2f}, sentiment: {compound_score:.3f}"
                )
        else:
            # Use fixed base values (no VADER adjustment)
            exaggeration = base_exaggeration
            cfg_weight = base_cfg_weight
            temperature = base_temperature
            min_p = base_min_p
            repetition_penalty = base_repetition_penalty

        boundary_type = detect_content_boundaries(
            chunk_text, i, chunk_texts, is_para_end
        )

        # Manual "~1"/"~2" shorthand -> [pause:Xms] tags (before punctuation-based
        # tags below, so both sources land in the same bracket-tag format).
        try:
            chunk_text = convert_inline_markers_to_pause_tags(chunk_text)
        except Exception as e:
            print(f"⚠️ Failed to convert inline pause markers: {e}")

        # Apply inline pause tags if enabled
        try:
            from config import config as _cfg

            if getattr(_cfg, "ENABLE_PUNCTUATION_PAUSES", False):
                chunk_text, boundary_type = add_pause_tags_to_text(
                    chunk_text, boundary_type
                )
        except Exception as e:
            print(f"⚠️ Failed to add pause tags: {e}")

        entry = {
            "index": i,
            "text": chunk_text,
            "word_count": len(chunk_text.split()),
            "boundary_type": boundary_type if boundary_type else "none",
            "sentiment_compound": compound_score,  # Store smoothed score
            "sentiment_raw": raw_compound_score,  # Store original score for reference
            "tts_params": {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                "min_p": min_p,
                "top_p": base_top_p,  # Top-P remains constant (not adjusted by VADER)
                "repetition_penalty": repetition_penalty,
            },
        }
        try:
            if _T3_TOKENIZER is not None:
                tl = len(_T3_TOKENIZER.encode(chunk_text))
            else:
                tl = int(max(1, len(chunk_text.split())))
            entry["token_len"] = tl
            token_lengths.append(tl)
        except Exception:
            pass
        enriched.append(entry)

    output_json_path = output_dir / "chunks_info.json"

    # Add voice metadata if provided
    if voice_name:
        # Try metadata method first
        try:
            # Create metadata entry as first element
            metadata = {
                "_metadata": True,
                "voice_used": voice_name,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_chunks": len(enriched),
            }
            enriched_with_metadata = [metadata] + enriched
            save_chunks(output_json_path, enriched_with_metadata)
            print(f"✅ Saved voice metadata: {voice_name}")
        except Exception as e:
            # Fallback to comment method if metadata fails
            print(f"⚠️ Metadata method failed, using comment fallback: {e}")
            save_chunks(output_json_path, enriched)

            # Add voice as comment
            from modules.voice_detector import add_voice_to_json

            add_voice_to_json(output_json_path, voice_name, method="comment")
    else:
        save_chunks(output_json_path, enriched)
        try:
            if token_lengths:
                token_lengths.sort()
                p95 = token_lengths[int(0.95 * (len(token_lengths) - 1))]
                import math

                suggested = int(math.ceil(p95 * 1.2))
                print(
                    f"🔧 Suggested MAX_T3_CONTEXT from JSON: P95={p95}, cap={suggested}"
                )
        except Exception:
            pass

    return enriched


def create_parameter_microbatches(chunks):
    """Group chunks by their rounded TTS parameters for micro-batching efficiency."""
    from collections import defaultdict

    # Group chunks by their TTS parameter combination
    parameter_groups = defaultdict(list)

    for chunk in chunks:
        if isinstance(chunk, dict) and "tts_params" in chunk:
            tts_params = chunk["tts_params"]

            # Create parameter key from rounded values
            param_key = (
                tts_params.get("exaggeration", 0.5),
                tts_params.get("cfg_weight", 0.5),
                tts_params.get("temperature", 0.85),
            )
        else:
            # Default parameters for chunks without specific TTS params
            param_key = (0.5, 0.5, 0.85)

        parameter_groups[param_key].append(chunk)

    # Convert groups to list of batches
    chunk_batches = []
    for param_key, chunks_in_group in parameter_groups.items():
        exag, cfg, temp = param_key
        print(
            f"  📦 Micro-batch: {len(chunks_in_group)} chunks with params (exag={exag}, cfg={cfg}, temp={temp})"
        )

        # SORT BY LENGTH - THE MISSING PIECE
        chunks_in_group.sort(key=lambda c: len(c.get("text", "")))

        # Split large groups into smaller batches to avoid memory issues
        # Use smaller microbatch when CFG is enabled (effective 2×B)
        max_microbatch_size = 4 if (float(cfg) > 0.0) else 8
        for i in range(0, len(chunks_in_group), max_microbatch_size):
            batch = chunks_in_group[i : i + max_microbatch_size]
            chunk_batches.append(batch)

    return chunk_batches


def process_book_folder(
    book_dir,
    voice_path,
    tts_params,
    device,
    skip_cleanup=False,
    enable_asr=None,
    quality_params=None,
    config_params=None,
    specific_text_file=None,
    asr_threshold=0.75,
    backend=None,
    fast_tts=None,
    asr_level="moderate",
    asr_device="cpu",
    existing_json_path=None,
):
    """Process a book through configured TTS phases.

    Args:
        existing_json_path: Optional preprocessed ``chunks_info.json`` to use
            verbatim instead of re-chunking the source text.
    """

    existing_json_data = None
    if existing_json_path is not None:
        import json

        source_json = Path(existing_json_path)
        if not source_json.exists():
            raise FileNotFoundError(f"Existing chunks JSON not found: {source_json}")
        with source_json.open("r", encoding="utf-8") as json_file:
            existing_json_data = json.load(json_file)
        if not isinstance(existing_json_data, list):
            raise ValueError("Existing chunks JSON must contain a list of records")
        if not all(isinstance(item, dict) for item in existing_json_data):
            raise ValueError("Existing chunks JSON records must be JSON objects")
        if not any(
            not item.get("_metadata", False)
            and isinstance(item.get("text"), str)
            for item in existing_json_data
        ):
            raise ValueError("Existing chunks JSON contains no text chunk records")

    # Anchor for "Total Elapsed" (button press -> M4B written), independent of
    # start_time below which anchors pure-generation elapsed for realtime-factor math.
    true_start_time = time.time()

    # Hard reset barrier at conversion start: behave like a fresh launch
    # 1) Release any cached model and voice conditionals
    try:
        clear_voice_cache()
        _release_global_tts_model()
    except Exception:
        pass

    # 2) Aggressive CUDA + host cleanup (no extra console noise)
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if hasattr(torch.cuda, "reset_peak_memory_stats"):
                torch.cuda.reset_peak_memory_stats()
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    try:
        import gc

        gc.collect()
        gc.collect()
    except Exception:
        pass

    # Start terminal logging to capture all output
    start_terminal_logging("term.log")

    print(
        f"🔍 DEBUG: Entering process_book_folder with book_dir='{book_dir}', voice_path='{voice_path}'"
    )

    # Apply GUI quality parameters to override config defaults
    if quality_params:
        print(f"🔧 Applying GUI quality parameters: {quality_params}")

        # Override config values with GUI settings
        global \
            ENABLE_REGENERATION_LOOP, \
            ENABLE_SENTIMENT_SMOOTHING, \
            ENABLE_MFCC_VALIDATION, \
            MAX_REGENERATION_ATTEMPTS
        global ENABLE_OUTPUT_VALIDATION, QUALITY_THRESHOLD, OUTPUT_VALIDATION_THRESHOLD
        global \
            SENTIMENT_SMOOTHING_WINDOW, \
            SENTIMENT_SMOOTHING_METHOD, \
            SPECTRAL_ANOMALY_THRESHOLD

        ENABLE_REGENERATION_LOOP = quality_params.get(
            "regeneration_enabled", ENABLE_REGENERATION_LOOP
        )
        MAX_REGENERATION_ATTEMPTS = quality_params.get(
            "max_attempts", MAX_REGENERATION_ATTEMPTS
        )
        ENABLE_SENTIMENT_SMOOTHING = quality_params.get(
            "sentiment_smoothing", ENABLE_SENTIMENT_SMOOTHING
        )
        ENABLE_MFCC_VALIDATION = quality_params.get(
            "mfcc_validation", ENABLE_MFCC_VALIDATION
        )
        ENABLE_OUTPUT_VALIDATION = quality_params.get(
            "output_validation", ENABLE_OUTPUT_VALIDATION
        )
        QUALITY_THRESHOLD = quality_params.get("quality_threshold", QUALITY_THRESHOLD)
        OUTPUT_VALIDATION_THRESHOLD = quality_params.get(
            "output_threshold", OUTPUT_VALIDATION_THRESHOLD
        )
        SENTIMENT_SMOOTHING_WINDOW = quality_params.get(
            "smoothing_window", SENTIMENT_SMOOTHING_WINDOW
        )
        SENTIMENT_SMOOTHING_METHOD = quality_params.get(
            "smoothing_method", SENTIMENT_SMOOTHING_METHOD
        )
        SPECTRAL_ANOMALY_THRESHOLD = quality_params.get(
            "spectral_threshold", SPECTRAL_ANOMALY_THRESHOLD
        )

        print(
            f"✅ Quality settings applied - Regeneration: {ENABLE_REGENERATION_LOOP}, MFCC: {ENABLE_MFCC_VALIDATION}, Output Validation: {ENABLE_OUTPUT_VALIDATION}"
        )

    # Apply GUI config parameters that impact runtime without editing file
    if config_params:
        try:
            # Global worker and batching overrides
            global MAX_WORKERS, BATCH_SIZE, ENABLE_MID_DROP_CHECK, ENABLE_HUM_DETECTION
            if "max_workers" in config_params:
                MAX_WORKERS = int(config_params["max_workers"])
            if "batch_size" in config_params:
                BATCH_SIZE = int(config_params["batch_size"])
            if "enable_mid_drop_check" in config_params:
                ENABLE_MID_DROP_CHECK = bool(config_params["enable_mid_drop_check"])
            if "enable_hum_detection" in config_params:
                ENABLE_HUM_DETECTION = bool(config_params["enable_hum_detection"])

            # Apply overrides to dependent modules
            try:
                from modules import file_manager as fm

                if "enable_normalization" in config_params:
                    fm.ENABLE_NORMALIZATION = bool(
                        config_params["enable_normalization"]
                    )
                if "normalization_type" in config_params:
                    fm.NORMALIZATION_TYPE = str(config_params["normalization_type"])
                if "target_lufs" in config_params:
                    fm.TARGET_LUFS = float(config_params["target_lufs"])
                if "target_peak_db" in config_params:
                    fm.TARGET_PEAK_DB = float(config_params["target_peak_db"])
                if "m4b_sample_rate" in config_params:
                    fm.M4B_SAMPLE_RATE = int(config_params["m4b_sample_rate"])
                if "playback_speed" in config_params:
                    fm.ATEMPO_SPEED = float(config_params["playback_speed"])
            except Exception as _e:
                print(f"⚠️ Failed to apply file_manager overrides: {_e}")

            try:
                from modules import audio_processor as ap

                if "enable_audio_trimming" in config_params:
                    ap.ENABLE_AUDIO_TRIMMING = bool(
                        config_params["enable_audio_trimming"]
                    )
                if "speech_threshold" in config_params:
                    ap.SPEECH_ENDPOINT_THRESHOLD = float(
                        config_params["speech_threshold"]
                    )
                if "trimming_buffer" in config_params:
                    ap.TRIMMING_BUFFER_MS = int(config_params["trimming_buffer"])
                if "silence_chapter_start" in config_params:
                    ap.SILENCE_CHAPTER_START = int(
                        config_params["silence_chapter_start"]
                    )
                if "silence_chapter_end" in config_params:
                    ap.SILENCE_CHAPTER_END = int(config_params["silence_chapter_end"])
                if "silence_section" in config_params:
                    ap.SILENCE_SECTION_BREAK = int(config_params["silence_section"])
                if "silence_paragraph" in config_params:
                    ap.SILENCE_PARAGRAPH_END = int(config_params["silence_paragraph"])
                if "silence_comma" in config_params:
                    ap.SILENCE_COMMA = int(config_params["silence_comma"])
                if "silence_period" in config_params:
                    ap.SILENCE_PERIOD = int(config_params["silence_period"])
                if "silence_question" in config_params:
                    ap.SILENCE_QUESTION_MARK = int(config_params["silence_question"])
                if "silence_exclamation" in config_params:
                    ap.SILENCE_EXCLAMATION = int(config_params["silence_exclamation"])
                if "enable_chunk_silence" in config_params:
                    ap.ENABLE_CHUNK_END_SILENCE = bool(
                        config_params["enable_chunk_silence"]
                    )
                if "chunk_silence_duration" in config_params:
                    ap.CHUNK_END_SILENCE_MS = int(
                        config_params["chunk_silence_duration"]
                    )

                # Inline pause settings
                if "enable_punctuation_pauses" in config_params:
                    from config import config as _cfg

                    _cfg.ENABLE_PUNCTUATION_PAUSES = bool(
                        config_params["enable_punctuation_pauses"]
                    )
                    print(
                        f"🎵 Inline pauses {'ENABLED' if config_params['enable_punctuation_pauses'] else 'DISABLED'} via GUI"
                    )

                # Update PUNCTUATION_PAUSE_MAPPING with GUI values
                if any(
                    k in config_params
                    for k in [
                        "inline_comma_ms",
                        "inline_period_ms",
                        "inline_question_ms",
                        "inline_exclamation_ms",
                    ]
                ):
                    from config import config as _cfg

                    if "inline_comma_ms" in config_params:
                        _cfg.PUNCTUATION_PAUSE_MAPPING[","] = config_params[
                            "inline_comma_ms"
                        ]
                    if "inline_period_ms" in config_params:
                        _cfg.PUNCTUATION_PAUSE_MAPPING["."] = config_params[
                            "inline_period_ms"
                        ]
                    if "inline_question_ms" in config_params:
                        _cfg.PUNCTUATION_PAUSE_MAPPING["?"] = config_params[
                            "inline_question_ms"
                        ]
                    if "inline_exclamation_ms" in config_params:
                        _cfg.PUNCTUATION_PAUSE_MAPPING["!"] = config_params[
                            "inline_exclamation_ms"
                        ]
                    print("🎵 Updated inline pause mapping from GUI values")
            except Exception as _e:
                print(f"⚠️ Failed to apply audio_processor overrides: {_e}")

            # Blunt micro-batching switch (propagate to config module for consistency)
            if "enable_micro_batching" in config_params:
                emb = bool(config_params["enable_micro_batching"])
                from config import config as _cfg

                _cfg.ENABLE_MICRO_BATCHING = emb
                _cfg.ENABLE_VADER_MICRO_BATCHING = emb
                print(
                    f"🧩 Micro-batching globally {'ENABLED' if emb else 'DISABLED'} via GUI"
                )
        except Exception as _e:
            print(f"⚠️ Failed to apply GUI runtime overrides: {_e}")

    from src.chatterbox.tts import punc_norm

    print("🔍 DEBUG: Successfully imported punc_norm")

    # Setup directories
    print("🔍 DEBUG: Calling setup_book_directories...")
    output_root, tts_dir, text_chunks_dir, audio_chunks_dir = setup_book_directories(
        book_dir
    )
    print("🔍 DEBUG: Directory setup complete")

    # ============================================================================
    # CLEAN PROCESSING - REAL OPTIMIZATIONS ONLY
    # ============================================================================
    print("🚀 Using optimized TTS model with REAL performance improvements")

    # Initialize fast backend here if enabled and not provided from GUI
    try:
        from config import config as _cfg_local

        if (
            getattr(_cfg_local, "TTS_BACKEND", "standard") == "fast"
            and fast_tts is None
        ):
            try:
                # Ensure standard model is released
                try:
                    _release_global_tts_model()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                from modules.t3_fast_backend import get_fast_tts_cached
                from modules.backend_context import BackendContext

                fast = get_fast_tts_cached(
                    device=str(device), dtype="bfloat16", compile_step=True
                )
                backend = BackendContext(model=None, fast_tts=fast)
                fast_tts = fast
                print("⚡ Fast backend initialized in engine")
            except Exception as _e:
                print(f"⚠️ Fast backend initialization failed in engine: {_e}")
    except Exception:
        pass

    # Clean previous processing files (but skip for resume operations)
    if skip_cleanup:
        print("🔄 RESUME MODE: Skipping cleanup to preserve existing chunks")
        print(f"📁 Preserving: {text_chunks_dir}, {audio_chunks_dir}")
    else:
        print("🧹 FRESH PROCESSING: Cleaning previous processing files...")

        # Clear text chunks
        for txt_file in text_chunks_dir.glob("*.txt"):
            txt_file.unlink(missing_ok=True)
        for json_file in text_chunks_dir.glob("*.json"):
            json_file.unlink(missing_ok=True)

        # Clear audio chunks
        for wav_file in audio_chunks_dir.glob("*.wav"):
            wav_file.unlink(missing_ok=True)

        # Clear logs
        for log_file in output_root.glob("*.log"):
            log_file.unlink(missing_ok=True)

        print("✅ Cleanup complete")

    # Find book files
    print("🔍 DEBUG: Calling find_book_files...")
    book_files = find_book_files(book_dir)

    # Use specific text file if provided (GUI selection), otherwise use auto-detected file
    if specific_text_file:
        text_file_to_use = Path(specific_text_file)
        print(f"🎯 DEBUG: Using GUI-selected text file: {text_file_to_use}")
        if not text_file_to_use.exists():
            logging.error(
                f"[{book_dir.name}] ERROR: Selected text file not found: {text_file_to_use}"
            )
            return None, None, []
    else:
        text_file_to_use = book_files["text"]
        print(f"🔍 DEBUG: Using auto-detected text file: {text_file_to_use}")
        if not text_file_to_use:
            logging.info(
                f"[{book_dir.name}] ERROR: No .txt files found in the book folder."
            )
            return None, None, []

    cover_file = book_files["cover"]
    nfo_file = book_files["nfo"]

    setup_logging(output_root)

    # Extract voice name for logging and JSON metadata
    voice_name_for_log = (
        voice_path.stem if hasattr(voice_path, "stem") else Path(voice_path).stem
    )

    if existing_json_data is not None:
        # Cleanup may have removed the source when it already lives in this
        # book's output directory, so restore the already-loaded records now.
        import json

        with (text_chunks_dir / "chunks_info.json").open("w", encoding="utf-8") as json_file:
            json.dump(existing_json_data, json_file, indent=2, ensure_ascii=False)
        all_chunks = [
            item
            for item in existing_json_data
            if not item.get("_metadata", False)
        ]
        print(f"📄 Using existing chunks JSON unchanged: {len(all_chunks)} chunks")
    else:
        # Generate enriched chunks with VADER analysis using user parameters and GUI quality settings.
        print(
            f"🔍 DEBUG: About to call generate_enriched_chunks with quality_params: {quality_params}"
        )
        print(
            f"🔍 DEBUG: About to call generate_enriched_chunks with config_params: {config_params}"
        )
        print(f"🔍 DEBUG: Using voice: {voice_name_for_log}")
        all_chunks = generate_enriched_chunks(
            text_file_to_use,
            text_chunks_dir,
            tts_params,
            quality_params,
            config_params,
            voice_name_for_log,
        )

    # Ensure per-chunk text files exist for downstream repair/combine workflows
    for default_idx, chunk_data in enumerate(all_chunks):
        chunk_text = chunk_data.get("text", "")
        chunk_index = chunk_data.get("index", default_idx)
        chunk_id_str = f"{chunk_index + 1:05d}"
        chunk_path = text_chunks_dir / f"chunk_{chunk_id_str}.txt"
        try:
            chunk_path.write_text(chunk_text, encoding="utf-8")
        except Exception as exc:
            logging.warning("Failed to write %s: %s", chunk_path, exc)

    print(f"🎯 Processing {len(all_chunks)} chunks with REAL optimized inference")

    # Create run_log_lines
    print("🔍 DEBUG: Creating run_log_lines...")
    print(f"🔍 DEBUG: voice_path type: {type(voice_path)}, value: {voice_path}")

    run_log_lines = [
        f"\n===== Processing: {book_dir.name} =====",
        f"Voice: {voice_name_for_log}",
        f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Text file processed: {text_file_to_use.name}",
        f"Total chunks generated: {len(all_chunks)}",
    ]

    start_time = time.time()
    total_chunks = len(all_chunks)
    log_path = output_root / "chunk_validation.log"
    total_audio_duration = 0.0

    # Process isolation pipeline has been removed; proceed with standard processing.

    # Standard batch processing (fallback or when isolation disabled)
    print(f"📊 Processing {total_chunks} chunks with intelligent reload decisions")

    # Reset smart reload manager for new session
    if ENABLE_SMART_RELOAD:
        reset_reload_manager()
        print("🧠 Smart reload manager initialized")

    all_results = []

    # Detect changes that require model reload using EFFECTIVE values (runtime overrides first)
    try:
        from config import config as _cfg

        eff_workers = int(
            (config_params or {}).get("max_workers", getattr(_cfg, "MAX_WORKERS", 0))
        )
        eff_batch = int(
            (config_params or {}).get("batch_size", getattr(_cfg, "BATCH_SIZE", 0))
        )
        eff_tts_batch = int(
            (config_params or {}).get(
                "tts_batch_size", getattr(_cfg, "TTS_BATCH_SIZE", 16)
            )
        )
        eff_micro = bool(
            (config_params or {}).get(
                "enable_micro_batching", getattr(_cfg, "ENABLE_MICRO_BATCHING", True)
            )
        )
        run_sig = (device, eff_workers, eff_batch, eff_tts_batch, eff_micro)
    except Exception:
        run_sig = (device, 0, 0, 0, True)

    global _LAST_RUN_SIGNATURE
    if _LAST_RUN_SIGNATURE is not None and run_sig != _LAST_RUN_SIGNATURE:
        print("🧹 Config change detected. Releasing cached model.")
        _release_global_tts_model()
    _LAST_RUN_SIGNATURE = run_sig

    # Prepare voice sample compatibility once; reload model per-batch below
    compatible_voice = ensure_voice_sample_compatibility(voice_path, output_dir=tts_dir)
    backend_choice = getattr(_cfg, "TTS_BACKEND", "standard").lower()
    logging.info("🔧 Effective TTS backend: %s", backend_choice)
    if backend_choice == "vllm":
        logging.info("🎯 TTS backend set to vLLM. Running three-phase pipeline.")
        json_path = text_chunks_dir / "chunks_info.json"

        ckpt_dir = Path(_cfg.CHATTERBOX_CKPT_DIR)

        # CFG scale is fixed once at vLLM engine load (T3VllmModel reads it from this
        # env var in __init__) and applied identically to every chunk until the engine
        # reloads — it cannot vary per-chunk like VADER's other params. Apply the GUI's
        # base cfg_weight here, fresh each run, instead of the silent hardcoded 0.5
        # default that every prior run actually used regardless of this setting.
        import os
        os.environ["CHATTERBOX_CFG_SCALE"] = str(tts_params.get("cfg_weight", 0.5))

        # ============================================================
        # PHASE 0: Compute conditionals (loads VE + S3Gen fp16, frees cleanly)
        # ============================================================
        logging.info("=" * 70)
        logging.info("PHASE 0: Computing voice conditionals")
        logging.info("=" * 70)
        emit_phase_status(0, "Computing voice conditionals", total_start_time=true_start_time)
        from chatterbox_vllm.tts import compute_conditionals

        cond_emb = compute_conditionals(
            ckpt_dir=ckpt_dir,
            target_device=device,
            variant=getattr(_cfg, "VLLM_MODEL_VARIANT", "english"),
            audio_prompt_path=str(compatible_voice) if compatible_voice else None,
        )
        logging.info("Conditionals computed (shape: %s)", cond_emb.shape)

        # ============================================================
        # PHASE 1: Generate speech tokens (T3-only vLLM)
        # ============================================================
        logging.info("=" * 70)
        logging.info("PHASE 1: Generating tokens (T3-only vLLM)")
        logging.info("=" * 70)
        phase1_start = time.time()

        def phase1_progress(current, total):
            """Reports Phase 1 (token generation) progress to the GUI status panel."""
            emit_phase_status(
                1, "Generating speech tokens", current, total,
                phase_start_time=phase1_start, total_start_time=true_start_time,
            )

        # Emit immediately, before the (potentially slow) engine load, so the panel
        # leaves "Phase 0" the instant Phase 1 begins rather than staying stuck on
        # Phase 0's text through engine load + the first token-generation batch.
        emit_phase_status(
            1, "Loading T3 model (vLLM engine)",
            phase_start_time=phase1_start, total_start_time=true_start_time,
        )

        batch_processor = VllmBatchProcessor(
            ckpt_dir=ckpt_dir,
            target_device=device,
            variant=getattr(_cfg, "VLLM_MODEL_VARIANT", "english"),
        )
        tokens_dict, chunks_data = batch_processor.process_chunks(
            json_path=json_path,
            cond_emb=cond_emb,
            progress_callback=phase1_progress,
        )
        log_vram_checkpoint("before vLLM shutdown")
        batch_processor.shutdown()
        batch_processor = None
        gc.collect()
        log_vram_checkpoint("after vLLM shutdown, before Turbo S3Gen")
        chunk_meta = _build_chunk_meta(chunks_data)

        logging.info("=" * 70)
        logging.info("PHASE 2: Generating audio with standard S3Gen decoder")
        logging.info("=" * 70)
        phase2_start = time.time()

        def phase2_progress(current, total, message):
            """Reports Phase 2 (S3Gen decode) progress to the GUI status panel."""
            emit_phase_status(
                2, "Decoding audio (S3Gen)", current, total,
                phase_start_time=phase2_start, total_start_time=true_start_time,
            )

        variant = getattr(_cfg, "VLLM_MODEL_VARIANT", "english")
        asr_enabled_resolved = enable_asr if enable_asr is not None else ENABLE_ASR
        asr_client = _maybe_start_asr_daemon(tts_dir, asr_enabled_resolved, asr_level, asr_device)

        decoder = VllmDecoder(
            decoder_type="standard",
            ckpt_dir=ckpt_dir,
            target_device=device,
            voice_path=compatible_voice,
        )
        generated, skipped, local_scores = decoder.decode_tokens(
            tokens_dict=tokens_dict,
            audio_output_dir=audio_chunks_dir,
            chunk_meta=chunk_meta,
            progress_callback=phase2_progress,
            asr_client=asr_client,
            asr_threshold=asr_threshold,
            enable_quality_scoring=ENABLE_REGENERATION_LOOP and asr_client is not None,
        )
        # Captured HERE, before decoder.shutdown()/Phase 3's ASR wait -- not after --
        # so the "Elapsed Time"/"Realtime Factor" stat below reflects pure generation
        # time, not however long Phase 3 took to collect ASR results.
        phase2_end_time = time.time()
        decoder.shutdown()

        logging.info("=" * 70)
        logging.info(f"PHASE 2 complete: {generated} generated, {skipped} skipped")
        logging.info("=" * 70)

        _run_phase3_regen(
            chunk_meta=chunk_meta,
            tokens_dict=tokens_dict,
            local_scores=local_scores,
            tts_dir=tts_dir,
            audio_chunks_dir=audio_chunks_dir,
            cond_emb=cond_emb,
            ckpt_dir=ckpt_dir,
            device=device,
            variant=variant,
            decoder_type="standard",
            voice_path=compatible_voice,
            turbo_ckpt_dir=None,
            asr_client=asr_client,
            asr_threshold=asr_threshold,
            true_start_time=true_start_time,
        )

        # ============================================================
        # FINALIZE: Combine audio chunks and create M4B
        # ============================================================
        # start_time=phase1_start (not the outer start_time, which predates Phase 0):
        # "Elapsed Time"/"Realtime Factor" should span Phase 1 start -> Phase 2 finish
        # only, excluding Phase 0's conditional-computation time. generation_elapsed
        # pins that span explicitly so Phase 3's ASR wait (which happens after this
        # point, before _finalize_book_output runs) can't inflate it.
        return _finalize_book_output(
            audio_chunks_dir=audio_chunks_dir,
            output_root=output_root,
            voice_path=voice_path,
            book_dir=book_dir,
            cover_file=cover_file,
            nfo_file=nfo_file,
            run_log_lines=run_log_lines,
            start_time=phase1_start,
            generation_elapsed=phase2_end_time - phase1_start,
            total_start_time=true_start_time,
            tts_params=tts_params,
        )

    elif backend_choice == "turbo-hybrid":
        logging.info(
            "🎯 TTS backend set to Turbo-Hybrid. Running three-phase pipeline."
        )
        json_path = text_chunks_dir / "chunks_info.json"

        ckpt_dir = Path(_cfg.CHATTERBOX_CKPT_DIR)

        # See the "vllm" branch above for why this must be set fresh each run,
        # before the engine loads, from the GUI's base cfg_weight.
        import os
        os.environ["CHATTERBOX_CFG_SCALE"] = str(tts_params.get("cfg_weight", 0.5))

        # ============================================================
        # PHASE 0: Compute conditionals (loads VE + S3Gen fp16, frees cleanly)
        # ============================================================
        logging.info("=" * 70)
        logging.info("PHASE 0: Computing voice conditionals")
        logging.info("=" * 70)
        emit_phase_status(0, "Computing voice conditionals", total_start_time=true_start_time)
        from chatterbox_vllm.tts import compute_conditionals

        cond_emb = compute_conditionals(
            ckpt_dir=ckpt_dir,
            target_device=device,
            variant=getattr(_cfg, "VLLM_MODEL_VARIANT", "english"),
            audio_prompt_path=str(compatible_voice) if compatible_voice else None,
        )
        logging.info("Conditionals computed (shape: %s)", cond_emb.shape)

        # ============================================================
        # PHASE 1: Generate speech tokens (T3-only vLLM)
        # ============================================================
        logging.info("=" * 70)
        logging.info("PHASE 1: Generating tokens (T3-only vLLM)")
        logging.info("=" * 70)
        phase1_start = time.time()

        def phase1_progress(current, total):
            """Reports Phase 1 (token generation) progress to the GUI status panel."""
            emit_phase_status(
                1, "Generating speech tokens", current, total,
                phase_start_time=phase1_start, total_start_time=true_start_time,
            )

        # Emit immediately, before the (potentially slow) engine load, so the panel
        # leaves "Phase 0" the instant Phase 1 begins rather than staying stuck on
        # Phase 0's text through engine load + the first token-generation batch.
        emit_phase_status(
            1, "Loading T3 model (vLLM engine)",
            phase_start_time=phase1_start, total_start_time=true_start_time,
        )

        batch_processor = VllmBatchProcessor(
            ckpt_dir=ckpt_dir,
            target_device=device,
            variant=getattr(_cfg, "VLLM_MODEL_VARIANT", "english"),
        )
        tokens_dict, chunks_data = batch_processor.process_chunks(
            json_path=json_path,
            cond_emb=cond_emb,
            progress_callback=phase1_progress,
        )
        log_vram_checkpoint("before vLLM shutdown")
        batch_processor.shutdown()
        batch_processor = None
        gc.collect()
        log_vram_checkpoint("after vLLM shutdown, before Turbo S3Gen")
        chunk_meta = _build_chunk_meta(chunks_data)

        # ============================================================
        # PHASE 2: Convert tokens to audio using TURBO S3Gen (5x faster)
        # ============================================================
        logging.info("=" * 70)
        logging.info("PHASE 2: Converting tokens to audio (Turbo S3Gen - 5x faster)")
        logging.info("=" * 70)
        phase2_start = time.time()

        variant = getattr(_cfg, "VLLM_MODEL_VARIANT", "english")
        asr_enabled_resolved = enable_asr if enable_asr is not None else ENABLE_ASR
        asr_client = _maybe_start_asr_daemon(tts_dir, asr_enabled_resolved, asr_level, asr_device)

        try:
            def audio_progress(current, total, message):
                """Reports Phase 2 (Turbo S3Gen decode) progress to the GUI status panel."""
                logging.info(f"[Turbo S3Gen] {message}")
                print(f"[Turbo S3Gen Audio] {message}")
                sys.stdout.flush()
                emit_phase_status(
                    2, "Decoding audio (Turbo S3Gen)", current, total,
                    phase_start_time=phase2_start, total_start_time=true_start_time,
                )

            turbo_ckpt_dir = getattr(_cfg, "TURBO_CKPT_DIR", None)
            decoder = VllmDecoder(
                decoder_type="turbo",
                ckpt_dir=ckpt_dir,
                target_device=device,
                voice_path=compatible_voice,
                turbo_ckpt_dir=turbo_ckpt_dir,
            )
            generated, skipped, local_scores = decoder.decode_tokens(
                tokens_dict=tokens_dict,
                audio_output_dir=audio_chunks_dir,
                chunk_meta=chunk_meta,
                progress_callback=audio_progress,
                asr_client=asr_client,
                asr_threshold=asr_threshold,
                enable_quality_scoring=ENABLE_REGENERATION_LOOP and asr_client is not None,
            )
            # Captured HERE, before decoder.shutdown()/Phase 3's ASR wait -- not after --
            # so the "Elapsed Time"/"Realtime Factor" stat below reflects pure generation
            # time, not however long Phase 3 took to collect ASR results.
            phase2_end_time = time.time()
            decoder.shutdown()

            logging.info(
                f"Phase 2 complete: {generated} chunks generated, {skipped} skipped"
            )

        except Exception as e:
            logging.error(f"Phase 2 (token-to-audio) failed: {e}", exc_info=True)
            raise RuntimeError(
                f"Turbo-Hybrid Phase 2 audio generation failed: {e}"
            ) from e

        _run_phase3_regen(
            chunk_meta=chunk_meta,
            tokens_dict=tokens_dict,
            local_scores=local_scores,
            tts_dir=tts_dir,
            audio_chunks_dir=audio_chunks_dir,
            cond_emb=cond_emb,
            ckpt_dir=ckpt_dir,
            device=device,
            variant=variant,
            decoder_type="turbo",
            voice_path=compatible_voice,
            turbo_ckpt_dir=turbo_ckpt_dir,
            asr_client=asr_client,
            asr_threshold=asr_threshold,
            true_start_time=true_start_time,
        )

        # ============================================================
        # FINALIZE: Combine audio chunks and create M4B
        # ============================================================
        # start_time=phase1_start (not the outer start_time, which predates Phase 0):
        # "Elapsed Time"/"Realtime Factor" should span Phase 1 start -> Phase 2 finish
        # only, excluding Phase 0's conditional-computation time. generation_elapsed
        # pins that span explicitly so Phase 3's ASR wait (which happens after this
        # point, before _finalize_book_output runs) can't inflate it.
        return _finalize_book_output(
            audio_chunks_dir=audio_chunks_dir,
            output_root=output_root,
            voice_path=voice_path,
            book_dir=book_dir,
            cover_file=cover_file,
            nfo_file=nfo_file,
            run_log_lines=run_log_lines,
            start_time=phase1_start,
            generation_elapsed=phase2_end_time - phase1_start,
            total_start_time=true_start_time,
            tts_params=tts_params,
        )

    elif backend_choice == "turbo":
        logging.info(
            "🎯 TTS backend set to Turbo. Running full ChatterboxTurboTTS pipeline."
        )

        # Import full Turbo TTS class
        try:
            from src.chatterbox_turbo.tts_turbo import ChatterboxTurboTTS
        except ImportError as e:
            raise RuntimeError(
                f"Turbo backend unavailable: {e}. Ensure Turbo models are installed."
            )

        # Determine checkpoint directory (same as turbo-hybrid)
        turbo_ckpt_dir = getattr(_cfg, "TURBO_CKPT_DIR", None)
        if turbo_ckpt_dir is None:
            # Auto-detect from HF cache
            hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
            candidates = list(
                hf_cache.glob("models--ResembleAI--chatterbox-turbo/snapshots/*")
            )
            if candidates:
                turbo_ckpt_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[
                    -1
                ]  # Latest
            else:
                raise RuntimeError(
                    "Turbo model not found. Set TURBO_CKPT_DIR in config or download chatterbox-turbo from HuggingFace."
                )

        turbo_ckpt_dir = Path(turbo_ckpt_dir)
        if not turbo_ckpt_dir.exists():
            raise RuntimeError(
                f"Turbo checkpoint directory not found: {turbo_ckpt_dir}"
            )

        # VRAM requirement check
        if device == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                if free_gb < 4.5:
                    raise RuntimeError(
                        f"Insufficient VRAM for Turbo backend: {free_gb:.2f} GB free (< 4.5 GB required)"
                    )
            except Exception as e:
                logging.warning(f"Could not check VRAM: {e}")

        # Voice length validation (Turbo requires >=5s)
        try:
            import librosa

            audio, sr = librosa.load(str(compatible_voice), sr=None)
            duration = len(audio) / sr
            if duration < 5.0:
                raise RuntimeError(
                    f"Turbo requires ≥5 second voice samples. Current voice duration: {duration:.1f}s"
                )
            logging.info(
                f"✅ Voice sample validated: {duration:.1f}s (meets Turbo ≥5s requirement)"
            )
        except Exception as e:
            if "Turbo requires" in str(e):
                raise  # Re-raise validation errors
            logging.warning(f"Could not validate voice duration: {e}")

        # Load full Turbo model
        logging.info(f"📦 Loading ChatterboxTurboTTS from: {turbo_ckpt_dir}")
        turbo_model = ChatterboxTurboTTS.from_local(str(turbo_ckpt_dir), device)
        logging.info("✅ ChatterboxTurboTTS model loaded successfully")

        # Override tts_params for Turbo (always use exaggeration=0.0)
        turbo_tts_params = tts_params.copy()
        turbo_tts_params["exaggeration"] = 0.0
        logging.info(
            "🔧 Turbo voice conditioning: exaggeration=0.0 (Turbo requirement)"
        )

        # Prepare voice conditioning
        logging.info(f"🎤 Preparing voice conditioning from: {compatible_voice}")
        try:
            # Use Turbo's prepare_conditionals method (takes wav_fpath parameter)
            turbo_model.prepare_conditionals(
                wav_fpath=str(compatible_voice),
                exaggeration=0.0,  # Turbo requirement
                norm_loudness=True,  # Turbo requirement
            )
            logging.info("✅ Voice conditioning prepared for Turbo model")
        except Exception as e:
            raise RuntimeError(f"Failed to prepare Turbo voice conditioning: {e}")

        # Prepare token capture structure (same format as vLLM for easy comparison)
        token_capture_data = []

        # Process all chunks with Turbo model
        all_results = []
        for chunk_idx, chunk_data in enumerate(all_chunks):
            chunk_text = chunk_data.get("text", "")
            chunk_id = chunk_data.get("index", chunk_idx)
            chunk_id_str = f"{chunk_id + 1:05d}"
            boundary_type = chunk_data.get("boundary_type", "none")

            # Generate audio using Turbo model (with token capture)
            try:
                logging.info(f"🔊 Generating audio for chunk {chunk_id_str} with Turbo")

                # Call generate with token capture enabled
                result = turbo_model.generate(
                    text=chunk_text,
                    exaggeration=0.0,  # Always 0.0 for Turbo (ignored by Turbo but kept for API compatibility)
                    cfg_weight=turbo_tts_params.get(
                        "cfg_weight", 3.0
                    ),  # Ignored by Turbo
                    temperature=turbo_tts_params.get("temperature", 1.0),
                    repetition_penalty=turbo_tts_params.get("repetition_penalty", 1.2),
                    top_p=turbo_tts_params.get("top_p", 0.95),
                    min_p=turbo_tts_params.get("min_p", 0.0),  # Ignored by Turbo
                    chunk_text_enabled=False,  # Disable internal chunking since we already chunk
                    return_tokens=True,  # Capture tokens for diagnostic comparison
                    boundary_type=boundary_type,  # NEW: Apply post-processing based on boundary type
                )

                # Unpack result (audio and tokens)
                audio_output, captured_tokens = result

                # Store tokens in same format as vLLM for easy comparison
                token_capture_data.append(
                    {
                        "index": chunk_id,
                        "text": chunk_text,
                        "word_count": len(chunk_text.split()),
                        "boundary_type": chunk_data.get("boundary_type", "period"),
                        "tts_params": turbo_tts_params,
                        "token_len": len(captured_tokens),
                        "speech_tokens": [
                            captured_tokens
                        ],  # Wrapped in list to match vLLM format [[tokens]]
                    }
                )

                # Save audio chunk
                audio_path = audio_chunks_dir / f"chunk_{chunk_id_str}.wav"
                import torchaudio

                # audio_output is already a tensor, save directly
                torchaudio.save(
                    str(audio_path),
                    audio_output,  # Already has correct shape from generate()
                    turbo_model.sr,
                )

                all_results.append((chunk_id, audio_path))

                # Progress logging
                if (chunk_idx + 1) % 5 == 0 or chunk_idx == len(all_chunks) - 1:
                    progress_pct = int((chunk_idx + 1) / total_chunks * 100)
                    logging.info(
                        f"📊 Progress: {chunk_idx + 1}/{total_chunks} chunks ({progress_pct}%)"
                    )

            except Exception as e:
                logging.error(
                    f"❌ Failed to generate chunk {chunk_id_str} with Turbo: {e}"
                )
                raise RuntimeError(
                    f"Turbo generation failed for chunk {chunk_id_str}: {e}"
                )

        logging.info(
            f"✅ Turbo processing complete: {len(all_results)} chunks generated"
        )

        # Save captured tokens to JSON (same format as vLLM for comparison)
        if token_capture_data:
            tokens_json_path = text_chunks_dir / "chunks_tokens_turbo.json"
            import json

            with open(tokens_json_path, "w", encoding="utf-8") as f:
                json.dump(token_capture_data, f, ensure_ascii=False, indent=2)
            logging.info(f"💾 Saved Turbo tokens to {tokens_json_path} for comparison")

        # Cleanup Turbo model to free VRAM
        logging.info("🧹 Cleaning up Turbo model...")
        del turbo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

        # Finalize output
        return _finalize_book_output(
            audio_chunks_dir=audio_chunks_dir,
            output_root=output_root,
            voice_path=voice_path,
            book_dir=book_dir,
            cover_file=cover_file,
            nfo_file=nfo_file,
            run_log_lines=run_log_lines,
            start_time=start_time,
            tts_params=turbo_tts_params,
        )

    elif backend_choice == "original_vllm":
        logging.info(
            "🎯 TTS backend set to Original vLLM. Using benchmark-style pipeline."
        )

        # Use benchmark-style pipeline: load model, batch generate all chunks, concatenate
        # This matches benchmark.py exactly but integrated with GUI inputs

        import json
        import subprocess

        # Extract text chunks from chunks_info.json
        chunks_info_path = text_chunks_dir / "chunks_info.json"
        if not chunks_info_path.exists():
            raise FileNotFoundError(f"chunks_info.json not found at {chunks_info_path}")

        with open(chunks_info_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        # Extract text chunks (skip metadata entries)
        text_chunks = []
        for chunk in chunks_data:
            if not chunk.get("_metadata"):  # Skip metadata entries
                text_chunks.append(chunk["text"])

        if not text_chunks:
            raise ValueError("No text chunks found in chunks_info.json")

        logging.info(f"📝 Loaded {len(text_chunks)} text chunks for batch processing")

        # Get TTS parameters
        exaggeration = getattr(_cfg, "exaggeration", 0.5)
        temperature = getattr(_cfg, "temperature", 0.8)
        max_tokens = getattr(_cfg, "MAX_T3_CONTEXT", 1000)
        top_p = getattr(_cfg, "top_p", 0.8)
        repetition_penalty = getattr(_cfg, "repetition_penalty", 2.0)
        language_id = getattr(_cfg, "language", "en")

        # Hardcode batch size for 8GB VRAM (from benchmark.py)
        batch_size = 15

        # Create the benchmark-style generation script
        text_chunks_json = json.dumps(text_chunks)
        vllm_script = (
            '''
import os
import sys
import json
import torch
import torchaudio as ta
from pathlib import Path

# Force in-process EngineCore — must be set before vLLM is imported
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

try:
    from chatterbox_vllm.tts import ChatterboxTTS
except ImportError as e:
    print(f"Failed to import ChatterboxTTS: {e}", file=sys.stderr)
    sys.exit(1)

def main():
    text_chunks = json.loads("""'''
            + text_chunks_json
            + '''""")
    voice_path = r"'''
            + str(compatible_voice)
            + '''"
    audio_chunks_dir = r"'''
            + str(audio_chunks_dir)
            + """"
    exaggeration = """
            + str(exaggeration)
            + """
    temperature = """
            + str(temperature)
            + """
    max_tokens = """
            + str(max_tokens)
            + """
    top_p = """
            + str(top_p)
            + """
    repetition_penalty = """
            + str(repetition_penalty)
            + '''
    language_id = "'''
            + language_id
            + """"
    batch_size = """
            + str(batch_size)
            + """

    print(f"[Original vLLM] Processing {len(text_chunks)} chunks with batch_size={batch_size}")

    # Load model (benchmark style)
    model = ChatterboxTTS.from_pretrained(
        max_batch_size=batch_size,
        max_model_len=max_tokens,
    )
    print("[Original vLLM] Model loaded successfully")

    # Generate all audio in one batched call (benchmark style)
    audios = model.generate(
        text_chunks,
        audio_prompt_path=voice_path,
        exaggeration=exaggeration,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        language_id=language_id,
    )
    print(f"[Original vLLM] Generated {len(audios)} audio chunks")

    # Concatenate all audio chunks
    full_audio = torch.cat(audios, dim=-1)
    print("[Original vLLM] Concatenated audio: shape={}, duration={:.1f}s".format(tuple(full_audio.shape), full_audio.shape[-1] / model.sr))

    # Save combined audio
    combined_path = Path(audio_chunks_dir) / "combined_audio.wav"
    ta.save(str(combined_path), full_audio, model.sr)
    print(f"[Original vLLM] Saved combined audio to {combined_path}")

    # Split into individual chunk files for compatibility with post-processing
    # (This maintains compatibility with existing finalize_book_output logic)
    for i, audio in enumerate(audios):
        chunk_path = Path(audio_chunks_dir) / f"chunk_{i:03d}.wav"
        ta.save(str(chunk_path), audio, model.sr)
        if i % 10 == 0:
            print(f"[Original vLLM] Saved chunk {i+1}/{len(audios)}")

    print(f"[Original vLLM] Saved {len(audios)} individual chunk files")

    # Cleanup
    model.shutdown()
    torch.cuda.empty_cache()
    print("[Original vLLM] Processing complete")

if __name__ == "__main__":
    main()
"""
        )

        # Execute the script in the vLLM environment
        project_root = Path(__file__).resolve().parent.parent
        vllm_python = project_root / "chatterbox-vllm" / ".venv" / "bin" / "python"
        if not vllm_python.exists():
            raise FileNotFoundError(
                f"vLLM python executable not found at {vllm_python}"
            )

        # Set up clean environment for subprocess
        env = _os.environ.copy()
        env["PATH"] = f"{vllm_python.parent}{_os.pathsep}{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(vllm_python.parent.parent)
        env["PYTHONUNBUFFERED"] = "1"
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)

        logging.info("🚀 Launching Original vLLM benchmark-style generation...")

        process = subprocess.Popen(
            [str(vllm_python), "-c", vllm_script],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=env,
        )

        assert process.stdout is not None
        output_lines = []
        for line in process.stdout:
            stripped = line.rstrip()
            if stripped:
                logging.info("[Original vLLM] %s", stripped)
                print(stripped)
                sys.stdout.flush()
                output_lines.append(stripped)

        return_code = process.wait()
        if return_code != 0:
            output_tail = "\n".join(output_lines[-50:])
            raise RuntimeError(
                f"Original vLLM generation failed (exit code {return_code}):\n{output_tail}"
            )

        logging.info("✅ Original vLLM processing completed successfully")

        return _finalize_book_output(
            audio_chunks_dir=audio_chunks_dir,
            output_root=output_root,
            voice_path=voice_path,
            book_dir=book_dir,
            cover_file=cover_file,
            nfo_file=nfo_file,
            run_log_lines=run_log_lines,
            start_time=start_time,
            tts_params=tts_params,
        )

        logging.info(
            f"✅ Original vLLM processing complete: {len(all_results)} chunks generated"
        )

        # Cleanup
        model.shutdown()

        return _finalize_book_output(
            audio_chunks_dir=audio_chunks_dir,
            output_root=output_root,
            voice_path=voice_path,
            book_dir=book_dir,
            cover_file=cover_file,
            nfo_file=nfo_file,
            run_log_lines=run_log_lines,
            start_time=start_time,
            tts_params=tts_params,
        )

    for batch_start in range(0, total_chunks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_chunks)
        batch_chunks = all_chunks[batch_start:batch_end]

        # Smart reload decision logic
        if ENABLE_SMART_RELOAD and batch_start > 0:
            remaining_chunks = total_chunks - batch_start
            reload_decision = should_reload_model(remaining_chunks)

            if reload_decision["should_reload"]:
                print(f"\n🧠 Smart reload triggered: {reload_decision['reason']}")
                print(
                    f"   📊 Performance degradation: {reload_decision['degradation_pct']:.1f}%"
                )
                print(
                    f"   💰 Economics: {reload_decision['economics']['roi']:.1f}x ROI"
                )
                record_model_reload(batch_start)
            else:
                print(f"\n🧠 Smart reload analysis: {reload_decision['reason']}")

        print(f"\n🔄 Processing batch: chunks {batch_start + 1}-{batch_end}")
        # Inform logger about current batch size for per-chunk summaries
        try:
            from modules.terminal_logger import set_batch_size

            set_batch_size(len(batch_chunks))
        except Exception:
            pass

        # Optional light cleanup between batches without destroying caches (disabled by default)
        if batch_start > 0 and os.environ.get("GENTTS_LIGHT_BATCH_CLEANUP", "0") == "1":
            print(f"🧹 Light cleanup before batch {batch_start + 1}-{batch_end}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                gc.collect()
            except Exception:
                pass

        # ASR validation is handled per-chunk via subprocess (ASR/asr_validator_headless.py)
        asr_model = None
        # Use parameter if provided, otherwise fall back to config
        asr_enabled = enable_asr if enable_asr is not None else ENABLE_ASR

        # Reload TTS model at the top of each batch (honor BATCH_SIZE semantics)
        model = None
        if backend is None or getattr(backend, "fast_tts", None) is None:
            model = load_optimized_model(device, force_reload=True)
            # Pre-warm model for selected voice
            model = prewarm_model_with_voice(model, compatible_voice, tts_params)

        futures = []
        batch_results = []

        # Dynamic worker allocation
        optimal_workers = get_optimal_workers()
        print(
            f"🔧 Using {optimal_workers} workers for batch {batch_start + 1}-{batch_end}"
        )

        use_vader = tts_params.get("use_vader", True)

        # ============================================================================
        # CLEAN PROCESSING WITH REAL OPTIMIZATIONS
        # ============================================================================
        batch_start_time = time.time()
        print(
            "🚀 Processing with REAL TTS optimizations (mixed precision, torch.compile)"
        )

        # MEASURE BATCH-BINNING EFFECTIVENESS
        from config.config import ENABLE_BATCH_BINNING

        if ENABLE_BATCH_BINNING:
            print(
                "📊 PERFORMANCE MEASUREMENT: Batch-binning enabled - measuring actual speed impact"
            )

        batch_timing_start = time.time()

        if not use_vader:
            # --- BATCH MODE ---
            print("🚀 VADER disabled. Running in high-performance batch mode.")

            # Check if batch-binning is enabled for micro-batching by parameters
            from config.config import ENABLE_BATCH_BINNING

            if ENABLE_BATCH_BINNING:
                try:
                    from modules.terminal_logger import log_only

                    log_only(
                        "🔗 BATCH-BINNING: Grouping chunks by rounded TTS parameters for micro-batching"
                    )
                except Exception:
                    pass
                chunk_batches = create_parameter_microbatches(batch_chunks)
                try:
                    from modules.terminal_logger import log_only

                    log_only(
                        f"📊 Processing {len(batch_chunks)} chunks in {len(chunk_batches)} parameter-grouped micro-batches"
                    )
                except Exception:
                    pass
            else:
                # Standard fixed-size batching
                tts_batch_size = config_params.get("tts_batch_size", 16)
                chunk_batches = [
                    batch_chunks[i : i + tts_batch_size]
                    for i in range(0, len(batch_chunks), tts_batch_size)
                ]
                print(
                    f"📊 Processing {len(batch_chunks)} chunks in {len(chunk_batches)} fixed batches of size {tts_batch_size}"
                )

            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                for batch in chunk_batches:
                    if shutdown_requested:
                        break
                    futures.append(
                        executor.submit(
                            process_batch,
                            batch,
                            text_chunks_dir,
                            audio_chunks_dir,
                            voice_path,
                            tts_params,
                            start_time,
                            total_chunks,
                            punc_norm,
                            book_dir.name,
                            log_run,
                            log_path,
                            device,
                            model,
                            0,
                            asr_enabled,
                        )
                    )

                # Wait for batches to complete
                for fut in as_completed(futures):
                    try:
                        # process_batch returns a list of (idx, wav_path) tuples
                        results_list = fut.result()
                        for idx, wav_path in results_list:
                            if wav_path and wav_path.exists():
                                chunk_duration = get_chunk_audio_duration(wav_path)
                                total_audio_duration += chunk_duration
                                batch_results.append((idx, wav_path))
                        # Throttle ETA printing to avoid console spam; status layer still receives updates
                        if (
                            len(batch_results) == 1
                            or (len(batch_results) % 5) == 0
                            or len(batch_results) == len(batch_chunks)
                        ):
                            log_chunk_progress(
                                batch_start + len(batch_results) - 1,
                                total_chunks,
                                start_time,
                                total_audio_duration,
                            )
                    except Exception as e:
                        logging.error(f"Future failed in batch: {e}")

            # Calculate performance with DETAILED debugging
            batch_end_time = time.time()
            total_batch_time = batch_end_time - batch_start_time
            actual_processing_time = batch_end_time - batch_timing_start

            print("🔍 PERFORMANCE CALCULATION DEBUG:")
            print(f"   Batch start time: {batch_start_time}")
            print(f"   Batch end time: {batch_end_time}")
            print(f"   Total batch time: {total_batch_time:.2f} seconds")
            print(f"   Actual processing time: {actual_processing_time:.2f} seconds")
            print(f"   Chunks processed: {len(batch_chunks)}")
            print(f"   Batch range: {batch_start + 1}-{batch_end}")

            # BATCH-BINNING PERFORMANCE MEASUREMENT
            from config.config import ENABLE_BATCH_BINNING

            if ENABLE_BATCH_BINNING:
                chunks_per_sec = (
                    len(batch_chunks) / actual_processing_time
                    if actual_processing_time > 0
                    else 0
                )
                print(
                    f"📊 BATCH-BINNING PERFORMANCE: {chunks_per_sec:.2f} chunks/sec with parameter rounding"
                )

            if total_batch_time > 0:
                its_performance = len(batch_chunks) / total_batch_time
                print(f"📊 CALCULATED PERFORMANCE: {its_performance:.2f} it/s")
                print(
                    f"   Formula: {len(batch_chunks)} chunks ÷ {total_batch_time:.2f} seconds = {its_performance:.2f} it/s"
                )
            else:
                print("⚠️ Zero or negative processing time detected")

        else:
            # --- VADER-ENABLED MODE ---
            from config.config import ENABLE_VADER_MICRO_BATCHING

            if ENABLE_VADER_MICRO_BATCHING:
                try:
                    from modules.terminal_logger import log_only

                    log_only(
                        "🎨 VADER enabled. Running in nuanced mode with micro-batching."
                    )
                except Exception:
                    pass
            else:
                try:
                    from modules.terminal_logger import log_only

                    log_only(
                        "🎨 VADER enabled. Micro-batching disabled by config; processing per-chunk."
                    )
                except Exception:
                    pass

            # Apply parameter rounding for micro-batching
            rounded_chunks = []
            for chunk_data in batch_chunks:
                if isinstance(chunk_data, dict):
                    rounded_chunk = chunk_data.copy()
                    if "tts_params" in rounded_chunk and rounded_chunk["tts_params"]:
                        tts_params_copy = rounded_chunk["tts_params"].copy()
                        # Round VADER-influenced parameters to enable groupings
                        for param in ["exaggeration", "cfg_weight", "temperature"]:
                            if param in tts_params_copy:
                                original_value = tts_params_copy[param]
                                steps = round(original_value / BATCH_BIN_PRECISION)
                                binned_value = steps * BATCH_BIN_PRECISION
                                tts_params_copy[param] = round(binned_value, 3)
                        rounded_chunk["tts_params"] = tts_params_copy
                    rounded_chunks.append(rounded_chunk)
                else:
                    rounded_chunks.append(chunk_data)

            # Create micro-batches by parameter groupings, or force per-chunk
            if ENABLE_VADER_MICRO_BATCHING:
                micro_batches = create_parameter_microbatches(rounded_chunks)
                try:
                    from modules.terminal_logger import log_only

                    log_only(
                        f"🔗 VADER MICRO-BATCHING: Created {len(micro_batches)} micro-batches from {len(rounded_chunks)} chunks"
                    )
                except Exception:
                    pass
            else:
                micro_batches = [[ch] for ch in rounded_chunks]

            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                for microbatch_idx, microbatch in enumerate(micro_batches):
                    if ENABLE_VADER_MICRO_BATCHING:
                        try:
                            from modules.terminal_logger import log_only

                            log_only(
                                f"🎯 Processing micro-batch {microbatch_idx + 1}/{len(micro_batches)} ({len(microbatch)} chunks)"
                            )
                        except Exception:
                            pass

                    # Process all chunks in this micro-batch
                    microbatch_futures = []
                    for i, chunk_data in enumerate(microbatch):
                        # Check for shutdown request
                        if shutdown_requested:
                            print(
                                f"\n⏹️ {YELLOW}Stopping submission of new chunks...{RESET}"
                            )
                            break

                        # Handle both dictionary and tuple formats for chunk data
                        if isinstance(chunk_data, dict):
                            chunk = chunk_data["text"]
                            boundary_type = chunk_data.get("boundary_type", "none")
                            # Use chunk-specific TTS params if available, otherwise fall back to global
                            chunk_tts_params = chunk_data.get("tts_params", tts_params)
                            # Use the chunk's original index from JSON instead of calculated position
                            global_chunk_index = chunk_data.get(
                                "index",
                                batch_start
                                + sum(len(mb) for mb in micro_batches[:microbatch_idx])
                                + i,
                            )
                        else:
                            # Handle old tuple format (text, is_para_end) - convert to boundary_type
                            chunk = (
                                chunk_data[0]
                                if len(chunk_data) > 0
                                else str(chunk_data)
                            )
                            # Convert old is_paragraph_end to boundary_type
                            is_old_para_end = (
                                chunk_data[1] if len(chunk_data) > 1 else False
                            )
                            boundary_type = (
                                "paragraph_end" if is_old_para_end else "none"
                            )
                            chunk_tts_params = tts_params  # Fallback for old format
                            # Fallback calculation for old tuple format
                            global_chunk_index = (
                                batch_start
                                + sum(len(mb) for mb in micro_batches[:microbatch_idx])
                                + i
                            )

                        microbatch_futures.append(
                            executor.submit(
                                process_one_chunk,
                                global_chunk_index,
                                chunk,
                                text_chunks_dir,
                                audio_chunks_dir,
                                voice_path,
                                chunk_tts_params,
                                start_time,
                                total_chunks,
                                punc_norm,
                                book_dir.name,
                                log_run,
                                log_path,
                                device,
                                model,
                                boundary_type=boundary_type,
                                enable_asr=asr_enabled,
                            )
                        )

                    # Wait for micro-batch to complete
                    try:
                        from modules.terminal_logger import log_only

                        log_only(
                            f"🔄 Waiting for micro-batch {microbatch_idx + 1} to complete..."
                        )
                    except Exception:
                        pass
                    completed_count = 0

                    for fut in as_completed(microbatch_futures):
                        try:
                            idx, wav_path = fut.result()
                            if wav_path and wav_path.exists():
                                # Measure actual audio duration for this chunk
                                chunk_duration = get_chunk_audio_duration(wav_path)
                                total_audio_duration += chunk_duration
                                batch_results.append((idx, wav_path))

                                # Track chunk performance for smart reload
                                if ENABLE_SMART_RELOAD:
                                    chunk_idx = (
                                        batch_start
                                        + sum(
                                            len(mb)
                                            for mb in micro_batches[:microbatch_idx]
                                        )
                                        + completed_count
                                    )
                                    elapsed_time = time.time() - start_time
                                    processing_time_per_chunk = (
                                        elapsed_time / chunk_idx
                                        if chunk_idx > 0
                                        else 1.0
                                    )
                                    estimated_tokens = estimate_tokens_in_text(
                                        chunk.get("text", "")
                                    )
                                    track_chunk_performance(
                                        chunk_idx,
                                        processing_time_per_chunk,
                                        estimated_tokens,
                                    )

                                # Update progress on every completed chunk; terminal logger throttles display
                                completed_count += 1
                                log_chunk_progress(
                                    batch_start
                                    + sum(
                                        len(mb) for mb in micro_batches[:microbatch_idx]
                                    )
                                    + completed_count
                                    - 1,
                                    total_chunks,
                                    start_time,
                                    total_audio_duration,
                                )

                        except Exception as e:
                            logging.error(f"Future failed in micro-batch: {e}")

                    futures.extend(microbatch_futures)

        # Clean up model after batch
        print(f"🧹 Cleaning up after batch {batch_start + 1}-{batch_end}")
        del model
        del asr_model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

        all_results.extend(batch_results)
        print(
            f"✅ Batch {batch_start + 1}-{batch_end} completed ({len(batch_results)} chunks)"
        )

    # Final backend cleanup at end of book processing
    try:
        if backend is not None:
            backend.shutdown()
            print("🔌 Backend shutdown completed after book processing")
    except Exception as e:
        print(f"⚠️ Warning during final backend cleanup: {e}")

    return _finalize_book_output(
        audio_chunks_dir=audio_chunks_dir,
        output_root=output_root,
        voice_path=voice_path,
        book_dir=book_dir,
        cover_file=cover_file,
        nfo_file=nfo_file,
        run_log_lines=run_log_lines,
        start_time=start_time,
        tts_params=tts_params,
    )


def _finalize_book_output(
    audio_chunks_dir: Path,
    output_root: Path,
    voice_path: Path,
    book_dir: Path,
    cover_file: Path | None,
    nfo_file: Path | None,
    run_log_lines: list[str],
    start_time: float,
    tts_params: dict,
    total_start_time: float | None = None,
    generation_elapsed: float | None = None,
):
    """Shared concatenation + export logic for both legacy and vLLM backends."""
    quarantine_dir = audio_chunks_dir / "quarantine"
    pause_for_chunk_review(quarantine_dir)

    chunk_paths = get_audio_files_in_directory(audio_chunks_dir)

    if not chunk_paths:
        logging.info(
            f"{RED}❌ No valid audio chunks found. Skipping concatenation and conversion.{RESET}"
        )
        return None, None, []

    # elapsed_total is the realtime-factor window: pure generation time, excluding
    # WAV-combine/M4B-conversion below AND (for vllm/turbo-hybrid callers) excluding
    # Phase 3's ASR-collection wait, which happens between Phase 2 finishing and this
    # function being called. generation_elapsed pins that window explicitly (captured
    # by the caller right after Phase 2, before Phase 3 runs) so the ASR wait can't
    # leak into it; legacy callers that don't pass it keep the old start_time-based
    # measurement (function-entry time is still correct for them, no Phase 3 in between).
    elapsed_total = (
        generation_elapsed if generation_elapsed is not None else time.time() - start_time
    )
    elapsed_td = timedelta(seconds=int(elapsed_total))

    total_audio_duration_final = sum(
        get_chunk_audio_duration(chunk_path) for chunk_path in chunk_paths
    )
    audio_duration_td = timedelta(seconds=int(total_audio_duration_final))
    realtime_factor = (
        total_audio_duration_final / elapsed_total if elapsed_total > 0 else 0.0
    )

    print("\n⏱️ TTS Processing Complete:")
    print(f"   Elapsed Time: {CYAN}{str(elapsed_td)}{RESET}")
    print(f"   Audio Duration: {GREEN}{str(audio_duration_td)}{RESET}")
    print(f"   Realtime Factor: {YELLOW}{realtime_factor:.2f}x{RESET}")

    # Transition the GUI panel out of "Phase 2" while combine/M4B conversion run,
    # instead of leaving it stuck on the last Phase 2 message with no feedback.
    if total_start_time is not None:
        emit_phase_status(
            None, "Finalizing: combining audio & converting to M4B",
            total_start_time=total_start_time,
        )

    voice_name = (
        voice_path.stem if hasattr(voice_path, "stem") else Path(voice_path).stem
    )
    combined_wav_path = output_root / f"{book_dir.name} [{voice_name}].wav"
    print("\n💾 Saving WAV file...")
    combine_audio_chunks(chunk_paths, combined_wav_path)

    temp_m4b_path = output_root / "output.m4b"
    final_m4b_path = output_root / f"{book_dir.name}[{voice_name}].m4b"
    convert_to_m4b(combined_wav_path, temp_m4b_path)
    add_metadata_to_m4b(temp_m4b_path, final_m4b_path, cover_file, nfo_file)

    logging.info(f"Audiobook created: {final_m4b_path}")

    run_log_lines.extend(
        [
            f"Combined WAV: {combined_wav_path}",
            "--- Generation Settings ---",
            f"Batch Processing: Enabled ({BATCH_SIZE} chunks per batch)",
            f"ASR Enabled: {ENABLE_ASR}",
            f"Hum Detection: {ENABLE_HUM_DETECTION}",
            f"Dynamic Workers: {USE_DYNAMIC_WORKERS}",
            f"Voice used: {voice_name}",
            f"Exaggeration: {tts_params['exaggeration']}",
            f"CFG weight: {tts_params['cfg_weight']}",
            f"Temperature: {tts_params['temperature']}",
            f"Processing Time: {str(elapsed_td)}",
            f"Audio Duration: {str(audio_duration_td)}",
            f"Realtime Factor: {realtime_factor:.2f}x",
            f"Total Chunks: {len(chunk_paths)}",
        ]
    )

    # total_start_time anchors button-press -> file-actually-written, so this has
    # to be measured HERE (after WAV combine + M4B conversion above), not earlier.
    if total_start_time is not None:
        total_elapsed_td = timedelta(seconds=int(time.time() - total_start_time))
        run_log_lines.append(f"Total Elapsed (start → output): {str(total_elapsed_td)}")
        print(f"   Total Elapsed (start → output): {CYAN}{str(total_elapsed_td)}{RESET}")
        emit_final_status(
            elapsed=str(elapsed_td),
            audio=str(audio_duration_td),
            realtime=f"{realtime_factor:.2f}x",
            total_elapsed=str(total_elapsed_td),
        )

    log_run("\n".join(run_log_lines), output_root / "run.log")
    print(f"📝 Run log written to: {output_root / 'run.log'}")

    return final_m4b_path, combined_wav_path, run_log_lines


def process_single_batch(
    batch_chunks,
    text_chunks_dir,
    audio_chunks_dir,
    voice_path,
    tts_params,
    start_time,
    total_chunks,
    basename,
    log_path,
    device,
    enable_asr,
    seed=0,
    fast_tts=None,
):
    """
    Loads models and processes a single batch of chunks.
    Designed to be called from a separate worker process.
    """
    import torch
    import gc
    from pathlib import Path
    from src.chatterbox.tts import punc_norm
    from modules.file_manager import ensure_voice_sample_compatibility

    # A simple logger function to satisfy the dependency of process_one_chunk (fallback)
    def log_run(message, path):
        """Writes a message to a file.
        Args:
        message (str): The message to write.
        path (str): The file path to write to.
        Returns: None
        """
        with open(path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    # Prepare voice
    compatible_voice = ensure_voice_sample_compatibility(voice_path)

    # Load models
    model = load_optimized_model(device, force_reload=True)
    model = prewarm_model_with_voice(model, compatible_voice, tts_params)

    # Get the punc_norm function - assuming 'en'
    punc_normalizer = punc_norm("en")

    # Call the existing process_batch function
    results = process_batch(
        batch=batch_chunks,
        text_chunks_dir=Path(text_chunks_dir),
        audio_chunks_dir=Path(audio_chunks_dir),
        voice_path=Path(voice_path),
        tts_params=tts_params,
        start_time=start_time,
        total_chunks=total_chunks,
        punc_norm=punc_normalizer,
        basename=basename,
        log_run_func=log_run,
        log_path=Path(log_path),
        device=device,
        model=model,
        seed=seed,
        enable_asr=enable_asr,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"✅ Worker process finished batch. Results: {len(results)} chunks processed."
    )

    return results
