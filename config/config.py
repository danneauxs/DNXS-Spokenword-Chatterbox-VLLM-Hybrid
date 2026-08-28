"""
GenTTS Configuration Module copy
Central location for all settings, paths, and feature toggles
"""

import os
from pathlib import Path


# ============================================================================
# CORE DIRECTORIES
# ============================================================================
TEXT_INPUT_ROOT = Path("Text_Input")
AUDIOBOOK_ROOT = Path("Audiobook")
VOICE_SAMPLES_DIR = Path("Voice_Samples")

# Optional: Local checkpoint directory for ChatterboxTTS weights
# If set, the engine will load from this path instead of downloading.
# You can also override via environment variable `CHATTERBOX_CKPT_DIR`.
_APP_ROOT = Path(__file__).resolve().parents[1]
CHATTERBOX_CKPT_DIR = os.environ.get(
    "CHATTERBOX_CKPT_DIR",
    str(_APP_ROOT / "models" / "chatterbox"),
)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TRANSFORMERS_NO_PROGRESS_BAR"] = "1"
os.environ["HF_TRANSFORMERS_NO_TQDM"] = "1"
# Cache handling is now done by launcher scripts:
# - launch_gradio_local.sh: Sets shared cache for development
# - launch_gradio.sh: Uses PyTorch defaults for containers/deployment

# ============================================================================
# COLOR CODES FOR TERMINAL OUTPUT
# ============================================================================
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"


# ============================================================================
# TEXT PROCESSING SETTINGS
# ============================================================================
MAX_CHUNK_WORDS = 28
MIN_CHUNK_WORDS = 4
# word_cap = current behavior (honor max words, may split long sentences).
# sentence_pack = Pocket-style: pack full sentences until min words, never
# split mid-sentence, ignore max words (a sentence may overshoot min).
CHUNKING_MODE = "word_cap"
USE_TOKEN_CHUNKING = False  # If True, use token-based chunking instead of word-based to avoid EOS triggers
USE_ORIGINAL_CHUNKING = True  # True = sentence/punctuation boundaries with boundary detection, False = 50-word simple chunking



# ============================================================================
# WORKER AND PERFORMANCE SETTINGS
# ============================================================================
MAX_WORKERS = 1
TEST_MAX_WORKERS = 2  # For experimentation
USE_DYNAMIC_WORKERS = False  # Toggle for testing
VRAM_SAFETY_THRESHOLD = 6.5  # GB

# ============================================================================
# VLLM PIPELINE WORKER SETTINGS
# ============================================================================
# Post-processing workers handle trimming, silence addition, and disk writes
# More workers = faster processing but more RAM usage

# Auto-select worker count based on CPU cores (recommended)
VLLM_PP_USE_AUTO = True

# Worker count range when auto mode enabled (will use: min(CPU_count-1, AUTO_MAX))
VLLM_PP_AUTO_MIN = 2   # Minimum workers (even on low-core systems)
VLLM_PP_AUTO_MAX = 12  # Maximum workers (caps high-core systems)

# Fixed worker count (only used when USE_AUTO=False)
VLLM_PP_FIXED_WORKERS = 0  # 0 = disabled

# Hard limits
VLLM_PP_MAX_WORKERS = 16      # Absolute maximum worker count
VLLM_PP_DEFAULT_WORKERS = 0   # Force specific count (0 = use auto logic)

# Audio queue settings (buffers between GPU generation and CPU post-processing)
# Base queue size = workers × multiplier, but capped by MAX_SIZE to prevent RAM exhaustion
VLLM_AUDIO_QUEUE_MULTIPLIER = 3   # Base multiplier per worker

# Maximum audio queue size (prevents RAM exhaustion on long audiobook runs)
# Each queued chunk ≈ 2-10 MB depending on length. Cap of 24 = max ~240 MB queued audio.
# Lower = less RAM usage, but GPU may idle waiting for queue space. Range: 10-50
VLLM_AUDIO_QUEUE_MAX_SIZE = 24

# T3 batch size and processing mode
VLLM_MAX_BATCH_SIZE = 15          # Upper bound for T3 batch size
VLLM_PP_USE_PROCESS_POOL = True   # Use process pool for CPU post-processing (vs thread pool)

# ============================================================================
# ASR VALIDATION SETTINGS
# ============================================================================
ASR_WORKERS = 4  # Parallel ASR on CPU threads
DEFAULT_ASR_THRESHOLD = 0.65

# ============================================================================
# AUDIO QUALITY SETTINGS
# ============================================================================
ENABLE_MID_DROP_CHECK = False
ENABLE_ASR = False  # Disabled by default due to tensor dimension errors
ASR_WORKERS = 4  # Parallel ASR on CPU threads
DEFAULT_ASR_MODEL = "base"  # Default Whisper model for ASR validation
# Two-stage ASR: Stage 1 scores every chunk; Stage 2 re-scores only Stage 1 fails.
# Stage 2 "disabled" skips the independent verifier. Not written unless the user
# enables ASR in the GUI; ENABLE_ASR stays False by default.
ASR_STAGE1_MODEL = "parakeet-tdt-0.6b-v3"
ASR_STAGE2_MODEL = "medium"
# Stage 1: faster_whisper | whisper_cpp | parakeet. Stage 2: faster_whisper | whisper_cpp.
ASR_STAGE1_BACKEND = "parakeet"
ASR_STAGE2_BACKEND = "whisper_cpp"
ASR_STAGE1_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v3",
)
# Tab 2 "Run ASR on GPU" checkbox default; off = safe CPU default. Saved by
# save_config_to_file, which strips inline comments on this line -- keep the
# explanation here, above the line, instead.
ASR_USE_GPU = True

# ASR_USE_GPU requests CUDA. Each ASR result records the resolved runtime device,
# because a backend can fall back to CPU. whisper.cpp specifically requires a
# pywhispercpp build that contains libggml-cuda (ASR/install_pywhispercpp_cuda.sh).

# ASR Model Memory Requirements (approximate)
ASR_MODEL_VRAM_MB = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550,
    "large-v2": 1550,
    "large-v3": 1550,
}

ASR_MODEL_RAM_MB = {
    "tiny": 150,
    "base": 300,
    "small": 800,
    "medium": 2000,
    "large": 4000,
    "large-v2": 4000,
    "large-v3": 4000,
}

# ============================================================================
# TTS HUM DETECTION SETTINGS
# ============================================================================
ENABLE_HUM_DETECTION = False
HUM_FREQ_MIN = 50  # Hz - Lower frequency bound for hum detection
HUM_FREQ_MAX = 200  # Hz - Upper frequency bound for hum detection
HUM_ENERGY_THRESHOLD = 0.3  # Ratio of hum energy to total energy (0.1-0.5 range)
HUM_STEADY_THRESHOLD = 0.6  # Ratio of segments with steady amplitude (0.5-0.8 range)
HUM_AMPLITUDE_MIN = 0.005  # Minimum RMS for steady hum detection
HUM_AMPLITUDE_MAX = 0.1  # Maximum RMS for steady hum detection

# ============================================================================
# SILERO VAD SETTINGS
# ============================================================================
# Enable Silero VAD for various audio processing tasks
# VAD provides more accurate speech detection than RMS-based methods

# Use VAD for audio trimming (more accurate speech endpoint detection)
USE_VAD_FOR_TRIMMING = True

# Use VAD for gap detection (detects TTS failures and unnatural pauses)
USE_VAD_FOR_GAP_DETECTION = True

# Use VAD for quality scoring (evaluates speech continuity and coverage)
USE_VAD_FOR_QUALITY_SCORING = True

# Maximum natural gap duration in milliseconds
# Gaps longer than this are flagged as potential TTS failures
VAD_MAX_NATURAL_GAP_MS = 1000

# VAD speech probability threshold (0.0-1.0)
# Higher values = more conservative (only high-confidence speech detected)
VAD_THRESHOLD = 0.5

# Minimum speech duration to register as valid speech segment (milliseconds)
VAD_MIN_SPEECH_DURATION_MS = 250

# Minimum silence duration between speech segments (milliseconds)
VAD_MIN_SILENCE_DURATION_MS = 100

# ============================================================================
# AUDIO TRIMMING SETTINGS
# ============================================================================
ENABLE_AUDIO_TRIMMING = True

# RMS-based trimming settings (fallback when VAD is disabled)
SPEECH_ENDPOINT_THRESHOLD = 0.005
TRIMMING_BUFFER_MS = 75

# ============================================================================
# SILENCE DURATION SETTINGS (milliseconds)
# ============================================================================
SILENCE_CHAPTER_START = 1195
SILENCE_CHAPTER_END = 1200
SILENCE_SECTION_BREAK = 600
SILENCE_PARAGRAPH_END = 1000

# Punctuation-specific silence settings (milliseconds)
SILENCE_COMMA = 0
SILENCE_SEMICOLON = 0  # Mid-clause; keep 0 unless you accept a T3 split
SILENCE_COLON = 0
SILENCE_PERIOD = 300
SILENCE_QUESTION_MARK = 350
SILENCE_EXCLAMATION = 598
SILENCE_DASH = 0  # Mid-clause; 0 = no T3 split
SILENCE_ELLIPSIS = 0
SILENCE_QUOTE_END = 150  # End of quoted speech

# Chunk-level silence settings
ENABLE_CHUNK_END_SILENCE = False
CHUNK_END_SILENCE_MS = 200

# Content boundary silence settings (milliseconds)
SILENCE_PARAGRAPH_FALLBACK = 500  # Original paragraph logic fallback


# Punctuation-to-pause mapping (milliseconds)
# Add new punctuation types here to enable pause insertion
PUNCTUATION_PAUSE_MAPPING = {
    '.': 'SILENCE_PERIOD',           # 1500ms
    '?': 'SILENCE_QUESTION_MARK',    # 500ms
    '!': 'SILENCE_EXCLAMATION',      # 200ms
    ';': 'SILENCE_SEMICOLON',        # 150ms
    ':': 'SILENCE_COLON',            # 150ms
    '—': 'SILENCE_DASH',             # 200ms (em dash)
    '...': 'SILENCE_ELLIPSIS',       # 80ms
}

# Enable/disable automatic insertion of pause tags for punctuation
ENABLE_PUNCTUATION_PAUSES = False

# ============================================================================
# AUDIO NORMALIZATION SETTINGS
# ============================================================================
ENABLE_NORMALIZATION = True
NORMALIZATION_TYPE = "peak"
TARGET_LUFS = -16
TARGET_PEAK_DB = -1.5
TARGET_LRA = 11  # Target loudness range for consistency

# ============================================================================
# AUDIO PLAYBACK SPEED SETTINGS
# ============================================================================
ATEMPO_SPEED = 1.0

# ============================================================================
# M4B OUTPUT SETTINGS
# ============================================================================
M4B_SAMPLE_RATE = 24000
# Encode only checked formats. M4B does not require a full-book WAV.
WRITE_M4B = True
WRITE_MP3 = False
WRITE_WAV = False
CHAPTERIZE = False
# headings_only | headings_or_minutes | headings_with_max
CHAPTER_MODE = "headings_only"
MAX_CHAPTER_MINUTES = 0

# ============================================================================
# VLLM BACKEND SETTINGS
# ============================================================================
# Device for vLLM backend. Set VLLM_DEVICE env var to override auto-detection.
# Options: "cuda", "mps" (Apple Silicon), "cpu", or VLLM_DEVICE None/unset for auto-detect (cuda → mps → cpu)
VLLM_DEVICE = os.getenv("VLLM_DEVICE")

# T3 checkpoint source for turbo-hybrid Phase 1. Vocoder is always Turbo S3Gen.
# Options:
# - "english" / "standard" → t3_cfg.safetensors (original English T3, vLLM)
# - "multilingual-v2"      → t3_mtl23ls_v2.safetensors (vLLM)
# - "multilingual-v3"      → t3_mtl23ls_v3.safetensors (vLLM)
# - "turbo"                → t3_turbo_v1.safetensors (native Turbo T3, not vLLM)
T3_SOURCE = os.environ.get("CHATTERBOX_T3_SOURCE", "multilingual-v3").lower()

# ISO language tag prepended as <lang> for multilingual T3. Ignored for english T3.
T3_LANGUAGE = os.environ.get(
    "CHATTERBOX_T3_LANGUAGE",
    os.environ.get("VLLM_DEFAULT_LANGUAGE", "en"),
).lower()

T3_SOURCE_FILES = {
    "english": "t3_cfg.safetensors",
    "multilingual-v2": "t3_mtl23ls_v2.safetensors",
    "multilingual-v3": "t3_mtl23ls_v3.safetensors",
    "turbo": "t3_turbo_v1.safetensors",
}

_T3_SOURCE_ALIASES = {
    "en": "english",
    "eng": "english",
    "english": "english",
    "standard": "english",
    "v2": "multilingual-v2",
    "multilingual-v2": "multilingual-v2",
    "mtl-v2": "multilingual-v2",
    "v3": "multilingual-v3",
    "multilingual": "multilingual-v3",
    "multilingual-v3": "multilingual-v3",
    "mtl-v3": "multilingual-v3",
    "mtl": "multilingual-v3",
    "turbo": "turbo",
}

# Directory holding multilingual V3 T3 weights (not VE/S3Gen).
CHATTERBOX_MTL_CKPT_DIR = os.environ.get(
    "CHATTERBOX_MTL_CKPT_DIR",
    str(_APP_ROOT / "models" / "chatterbox-mtl-v3"),
)

# Directory holding multilingual V2 T3 weights when T3_SOURCE=multilingual-v2.
CHATTERBOX_MTL_V2_CKPT_DIR = os.environ.get(
    "CHATTERBOX_MTL_V2_CKPT_DIR",
    str(_APP_ROOT / "models" / "chatterbox-mtl-v2"),
)


def resolve_t3_source(source=None):
    """Normalize a T3 source string to english | multilingual-v2 | multilingual-v3 | turbo.

    Args:
        source: Optional raw source name or alias. None uses T3_SOURCE.

    Returns:
        Canonical T3 source name.

    Raises:
        ValueError: If the source is not a known alias.
    """
    raw = (source if source is not None else T3_SOURCE) or "multilingual-v3"
    raw = str(raw).strip().lower()
    resolved = _T3_SOURCE_ALIASES.get(raw)
    if resolved is None:
        known = ", ".join(sorted(set(_T3_SOURCE_ALIASES)))
        raise ValueError(f"Unknown T3_SOURCE '{raw}'. Expected one of: {known}")
    return resolved


def t3_vllm_variant(source=None):
    """Return the vLLM from_local variant string for a T3 source.

    Args:
        source: Optional T3 source. None uses T3_SOURCE.

    Returns:
        "english" or "multilingual".
    """
    src = resolve_t3_source(source)
    if src == "english":
        return "english"
    if src == "turbo":
        return "turbo"
    return "multilingual"


def t3_weights_filename(source=None):
    """Return the T3 safetensors filename for a T3 source.

    Args:
        source: Optional T3 source. None uses T3_SOURCE.

    Returns:
        Checkpoint filename such as t3_mtl23ls_v3.safetensors.
    """
    return T3_SOURCE_FILES[resolve_t3_source(source)]


def t3_checkpoint_dir(source=None):
    """Return the directory that stores the T3 safetensors file for a source.

    English T3 lives with VE/S3Gen under CHATTERBOX_CKPT_DIR. Multilingual T3
    files live in their own dirs so Phase 0 can still load VE/S3Gen from the
    English chatterbox folder.

    Args:
        source: Optional T3 source. None uses T3_SOURCE.

    Returns:
        Path to the T3 checkpoint directory.
    """
    src = resolve_t3_source(source)
    if src == "english":
        return Path(CHATTERBOX_CKPT_DIR)
    if src == "multilingual-v2":
        return Path(CHATTERBOX_MTL_V2_CKPT_DIR)
    if src == "turbo":
        return Path(os.environ.get("TURBO_CKPT_DIR", str(_APP_ROOT / "models" / "chatterbox-turbo")))
    return Path(CHATTERBOX_MTL_CKPT_DIR)


def vllm_t3_model_dir(source=None):
    """Return the isolated one-weight vLLM directory for a T3 source.

    Turbo T3 is not a vLLM model; callers must not use this for source=turbo.

    Args:
        source: Optional T3 source. None uses T3_SOURCE.

    Returns:
        Path to config.json + model.safetensors for vLLM.

    Raises:
        ValueError: If the source is turbo.
    """
    src = resolve_t3_source(source)
    if src == "turbo":
        return Path(
            os.environ.get(
                "VLLM_TURBO_CKPT_DIR",
                str(_APP_ROOT / "models" / "vllm-t3-turbo"),
            )
        )
    if src == "english":
        return Path(VLLM_ENGLISH_CKPT_DIR or str(_APP_ROOT / "models" / "vllm-t3"))
    if src == "multilingual-v2":
        return Path(
            os.environ.get(
                "VLLM_MULTILINGUAL_V2_CKPT_DIR",
                str(_APP_ROOT / "models" / "vllm-t3-mtl-v2"),
            )
        )
    return Path(VLLM_MULTILINGUAL_CKPT_DIR or str(_APP_ROOT / "models" / "vllm-t3-mtl-v3"))


def t3_weights_path(source=None):
    """Return the full path to the T3 safetensors file for a source.

    Args:
        source: Optional T3 source. None uses T3_SOURCE.

    Returns:
        Path to the T3 weights file.
    """
    return t3_checkpoint_dir(source) / t3_weights_filename(source)


# Derived from T3_SOURCE so existing variant=="english"|"multilingual" call sites
# keep working. Override CHATTERBOX_T3_SOURCE rather than this name.
VLLM_MODEL_VARIANT = t3_vllm_variant()

# Default language code (e.g., "en", "es", "fr", "de", "zh")
VLLM_DEFAULT_LANGUAGE = T3_LANGUAGE

# S3Gen diffusion steps - quality/speed tradeoff. Range: 1-50
# Performance guide (per chunk):
#   10 steps (default): Best quality, ~0.5-2s per chunk
#   7 steps: Faster (1.4x), minimal quality loss
#   5 steps: Faster (2x), slight quality loss  ← OPTIMIZED SETTING
#   3 steps: Much faster, noticeable quality loss
#   2 steps: Very fast, substantial quality degradation
VLLM_DIFFUSION_STEPS = 4
# S3Gen mini-batch processing - process N chunks at once through decoder
# Higher = better GPU utilization but more VRAM usage
# Memory: ~250MB for 5 chunks, ~500MB for 10 chunks
# Speed gain: ~2-3x when enabled (conservative default: disabled)
VLLM_ENABLE_S3GEN_BATCHING = True
VLLM_S3GEN_MINI_BATCH_SIZE = 5  # Only used if VLLM_ENABLE_S3GEN_BATCHING = True

# Enable FP16 precision for S3Gen decoder
# Benefit: ~2-3× faster generation, 50% less VRAM
# Risk: Potential minor quality loss (test before enabling)
# ENABLED: Testing FP16 for performance boost (dtype issues should be fixed with dtype-aware casting)
VLLM_S3GEN_USE_FP16 = False

# Parallel S3Gen processing settings
# Enable parallel processing with multiple worker threads (shares FP16 model across workers)
# Benefit: ~1.7-1.9× faster S3Gen generation for batch processing
# Memory: Threads share the same FP16 model (~800MB), minimal overhead per worker (~800MB activations)
VLLM_S3GEN_ENABLE_PARALLEL = False #changed
VLLM_S3GEN_NUM_WORKERS = 0  # Number of worker threads (2 is safe for 8GB VRAM)

# Override path to English model checkpoints (defaults to ./t3-model or ./chatterbox-vllm/t3-model)
VLLM_ENGLISH_CKPT_DIR = os.environ.get("VLLM_ENGLISH_CKPT_DIR")

# Override path to multilingual model checkpoints (isolated one-weight vLLM dir)
VLLM_MULTILINGUAL_CKPT_DIR = os.environ.get(
    "VLLM_MULTILINGUAL_CKPT_DIR",
    str(_APP_ROOT / "models" / "vllm-t3-mtl-v3"),
)

# ============================================================================
# STANDARD TTS S3GEN SETTINGS (for token-to-audio conversion)
# ============================================================================
# S3Gen diffusion steps for standard TTS backend when converting pre-generated tokens
# This is used after vLLM tokenization completes, when main process generates audio
# from pre-computed speech tokens using the standard ChatterboxTTS model.
# Range: 1-10 (lower = faster, higher = better quality)
# Default: 5 (good balance of speed/quality for production use)
# Performance guide (per chunk):
#   3 steps: ~1.0s/chunk (fast but lower quality)
#   5 steps: ~2.0s/chunk (balanced - RECOMMENDED)
#   10 steps: ~4.0s/chunk (best quality)
S3GEN_DIFFUSION_STEPS = 3

# Enable FP16 (half precision) for S3Gen in standard TTS backend
# When True, uses FP16 for faster inference; if dtype mismatch occurs,
# automatically falls back to FP32 with notification.
# Benefit: ~2-3× faster generation, less VRAM usage
# Risk: Potential minor quality loss (mitigated by automatic FP32 fallback)
# Note: vLLM backend has its own separate FP16 setting above (VLLM_S3GEN_USE_FP16)
ENABLE_FP16_S3GEN = False


# ============================================================================
# Effective max context length for T3 (tokens). Used to clamp prompt length.
MAX_T3_CONTEXT = 1200
# vLLM forbids speech-stop until this many tokens per word (~half of 25Hz speech).
# Stops T3 from ending a prompt at the first period. Do not disable stop tokens.
T3_MIN_SPEECH_TOKENS_PER_WORD = 5

# Enable Flash-Attention 2 for T3 backbone if available
USE_FA2 = False

# Backend selection
# Options:
# - "standard"     → src.chatterbox.tts (original unified ChatterboxTTS)
# - "turbo"        → Full Turbo T3/S3Gen model (fastest single model)
# - "vllm"         → modules.unified_tts.VLLMBackend (VLLM T3 + VLLM S3Gen)
# - "turbo-hybrid" → VLLM T3 + Turbo S3Gen (best performance combination)
TTS_BACKEND = os.environ.get("CHATTERBOX_TTS_BACKEND", "standard").lower()

# Turbo model checkpoint directory (for turbo-hybrid and turbo backends)
TURBO_CKPT_DIR = os.environ.get(
    "TURBO_CKPT_DIR",
    None  # Auto-detect from ~/.cache/huggingface/hub/
)

# Turbo S3Gen diffusion steps (always 2 for meanflow models - 5x faster than standard)
TURBO_S3GEN_DIFFUSION_STEPS = 2

# Turbo S3Gen scheduler. Capacity values are empirical safe padded lengths for
# this checkpoint, dtype, diffusion-step count, and 8 GB RTX 4060 Ti sweep.
TURBO_S3GEN_ENABLE_BATCHING = True
TURBO_S3GEN_BATCH_SIZE = 3
TURBO_S3GEN_BATCH_CAPACITY = {2: 1400, 3: 1000}

DEFAULT_EXAGGERATION = 0.5
DEFAULT_CFG_WEIGHT = 0.4
DEFAULT_TEMPERATURE = 0.85
DEFAULT_SEED = 0  # Random seed for generation. 0 means random.

# Advanced Sampling Parameters (Min_P Sampler Support)
DEFAULT_MIN_P = 0.05  # Min probability threshold (0.0 disables)
DEFAULT_TOP_P = 1.0  # Top-p sampling (1.0 disables)
DEFAULT_REPETITION_PENALTY = 1.2  # Repetition penalty (1.0 = no penalty)
DEFAULT_VADER_ENABLED = os.environ.get("CHATTERBOX_VADER_ENABLED", "0").lower() not in {"0", "false", "no", "off"}

# ============================================================================
# VADER SENTIMENT TO TTS PARAMETER MAPPING
# ============================================================================
# These settings control how VADER sentiment analysis dynamically adjusts TTS parameters.
# The formula used is: new_param = base_param + (compound_score * sensitivity)
# The result is then clamped within the defined MIN/MAX range.

# --- Base TTS Parameters (used as the starting point) ---
# These are the same as the main defaults, but listed here for clarity.
BASE_EXAGGERATION = DEFAULT_EXAGGERATION  # Default: 1.0
BASE_CFG_WEIGHT = DEFAULT_CFG_WEIGHT  # Default: 0.7
BASE_TEMPERATURE = DEFAULT_TEMPERATURE  # Default: 0.7

# --- Sensitivity ---
# How much VADER's compound score affects emotional parameters.
# Higher values mean more dramatic changes based on sentiment.
VADER_EXAGGERATION_SENSITIVITY = 0.33
VADER_TEMPERATURE_SENSITIVITY = 0.3
# Retained as zero-valued compatibility keys; VADER no longer changes these.
VADER_CFG_WEIGHT_SENSITIVITY = 0.0
VADER_MIN_P_SENSITIVITY = 0.0
VADER_REPETITION_PENALTY_SENSITIVITY = 0.0

# --- Min/Max Clamps ---
# Hard limits to prevent extreme, undesirable audio artifacts.
TTS_PARAM_MIN_EXAGGERATION = 0.1
TTS_PARAM_MAX_EXAGGERATION = 0.65
TTS_PARAM_MIN_CFG_WEIGHT = 0.0
TTS_PARAM_MAX_CFG_WEIGHT = 1.0

TTS_PARAM_MIN_TEMPERATURE = 0.1
TTS_PARAM_MAX_TEMPERATURE = 2.35

TTS_PARAM_MIN_MIN_P = 0.02  # Increased from 0.0 to prevent sampling issues
TTS_PARAM_MAX_MIN_P = 0.3  # Reduced from MAX 0.5 to prevent over-restriction
TTS_PARAM_MIN_TOP_P = 0.5  # Too low causes repetition
TTS_PARAM_MAX_TOP_P = 1.0  # MAX 1.0 disables top_p
TTS_PARAM_MIN_REPETITION_PENALTY = 1.0  # 1.0 = no penalty
TTS_PARAM_MAX_REPETITION_PENALTY = 2.0  # Higher values too restrictive MAX 2

# ============================================================================
# TTS_PRESETS
# ============================================================================
TTS_PRESETS = {
    "Narration": {
        "exaggeration": 0.55,
        "cfg_weight": 0.5,
        "temperature": 0.85,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "vader_enabled": True,  # Default to VADER on for nuanced presets
        "sentiment_smoothing": True,
        "smoothing_window": 3,
        "smoothing_method": "rolling",
        "seed": 12345,  # Unique seed for Narration preset
    },
    "Expressive Mod": {
        "exaggeration": 0.50,
        "cfg_weight": 0.4,
        "temperature": 0.60,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "vader_enabled": True,
        "sentiment_smoothing": True,
        "smoothing_window": 3,
        "smoothing_method": "rolling",
        "seed": 67890,  # Unique seed for Expressive preset
    },
    "Expressive": {
        "exaggeration": 0.65,
        "cfg_weight": 0.8,
        "temperature": 0.95,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "vader_enabled": True,
        "sentiment_smoothing": True,
        "smoothing_window": 3,
        "smoothing_method": "rolling",
        "seed": 67890,  # Unique seed for Expressive preset
    },
    "Exposition": {
        "exaggeration": 0.4,
        "cfg_weight": 0.3,
        "temperature": 0.55,
        "min_p": 0.05,
        "top_p": 1.0,
        "repetition_penalty": 1.2,
        "vader_enabled": False,  # VADER off for consistent, clear delivery
        "sentiment_smoothing": False,
        "seed": 98765,  # Unique seed for Exposition preset
    },
}


# ============================================================================
# BATCH-BINNING SETTINGS FOR VADER PARAMETER OPTIMIZATION
# ============================================================================
# Enable batch-binning: rounds VADER parameters (exaggeration, cfg_weight, temperature)
# to nearest 0.05 for better microbatching when VADER is enabled
ENABLE_BATCH_BINNING = False  # Enable automatic parameter rounding for batch optimization (NEW) - Testing actual impact
BATCH_BIN_PRECISION = 0.05  # Rounding precision (0.05 = round to nearest 0.05)

# (Removed) Process isolation + CUDA MPS settings were pruned along with the
# performance integrator/pipeline. No replacement flags are needed.


# ============================================================================
# BATCH PROCESSING SETTINGS
# ============================================================================
# (Legacy) Batch sizing kept for GUI compatibility; consider removing after GUI update
BATCH_SIZE = 50000
TTS_BATCH_SIZE = 32
CLEANUP_INTERVAL = 500000  # Deep cleanup every N chunks (reduced frequency for speed)

# ============================================================================
# SMART RELOAD SETTINGS
# ============================================================================
ENABLE_SMART_RELOAD = False  # Feature appears incomplete, disable by default

# ============================================================================
# QUALITY ENHANCEMENT SETTINGS (Phase 1)
# ============================================================================

# --- Regeneration Loop Settings ---
ENABLE_REGENERATION_LOOP = (
    True  # Enable automatic chunk regeneration on quality failure
)
MAX_REGENERATION_ATTEMPTS = 3  # Maximum retry attempts per chunk
QUALITY_THRESHOLD = (
    0.30  # TEMPORARILY LOWERED - Composite quality score threshold (0.0-1.0)
)

# --- Sentiment Smoothing Settings ---
ENABLE_SENTIMENT_SMOOTHING = True  # Re-enabled - GUI controls now working properly
SENTIMENT_SMOOTHING_WINDOW = 3  # Number of previous chunks to consider
SENTIMENT_SMOOTHING_METHOD = "rolling"  # "rolling" or "exp_decay"

# Exponential decay weights for smoothing (used if method is "exp_decay")
SENTIMENT_EXP_DECAY_WEIGHTS = [0.5, 0.3, 0.2]  # Most recent to oldest

# --- Enhanced Anomaly Detection ---
SPECTRAL_ANOMALY_THRESHOLD = 0.6  # Spectral anomaly score threshold (0.0-1.0)
# MFCC analysis remains available as an explicit diagnostic, but is excluded
# from automatic regeneration by default until its metric is calibrated against
# speech-quality reference data.
ENABLE_MFCC_VALIDATION = False
SPECTRAL_VARIANCE_LIMIT = 100.0  # Maximum spectral variance before flagging as artifact

# --- Output Validation Settings ---
ENABLE_OUTPUT_VALIDATION = (
    True  # Enable quality control clearinghouse (runs individual checks when enabled)
)
OUTPUT_VALIDATION_THRESHOLD = (
    0.6  # Minimum F1 score for output validation (reduced for punctuation tolerance)
)

# --- Parameter Adjustment for Regeneration ---
REGEN_TEMPERATURE_ADJUSTMENT = (
    0.1  # How much to adjust temperature per retry (increased for visibility)
)
REGEN_EXAGGERATION_ADJUSTMENT = (
    0.15  # How much to adjust exaggeration per retry (increased for visibility)
)
REGEN_CFG_ADJUSTMENT = (
    0.1  # How much to adjust cfg_weight per retry (increased for visibility)
)

# ============================================================================
# TORCH.COMPILE OPTIMIZATION SETTINGS
# ============================================================================

ENABLE_TORCH_COMPILE = False  # Master enable/disable for torch.compile

# Component-specific compilation flags
COMPILE_VOICE_ENCODER = False  # Compile voice encoder
COMPILE_TTS_DECODER = False  # Compile T3 text-to-speech decoder
COMPILE_VOCODER = True  # Compile S3Gen vocoder (start conservative)
COMPILE_TEXT_PROCESSOR = True  # Compile text processing components

# Compilation settings
TORCH_COMPILE_MODE = "default"  # default|reduce-overhead|max-autotune
TORCH_COMPILE_BACKEND = "inductor"  # inductor|nvfuser|aot_eager|eager
TORCH_COMPILE_DYNAMIC_SHAPES = (
    False  # Enable dynamic shape support (slower compilation)
)
TORCH_COMPILE_FALLBACK_TO_CPU = (
    False  # Fall back to CPU backend if GPU compilation fails
)

# Text chunk size bucketing for compilation optimization
ENABLE_CHUNK_SIZE_BUCKETING = True  # Group similar chunk sizes for better compilation
CHUNK_BUCKET_SHORT_RANGE = [50, 200]  # Character range for short chunks
CHUNK_BUCKET_MEDIUM_RANGE = [200, 500]  # Character range for medium chunks
CHUNK_BUCKET_LONG_RANGE = [500, 1000]  # Character range for long chunks

# Cache and warmup settings
TORCH_COMPILE_CACHE_DIR = "venv/.cache/torch_compile"  # Compilation cache directory
ENABLE_COMPILATION_WARMUP = True  # Pre-compile models during initialization
COMPILATION_WARMUP_SAMPLES = 3  # Number of warmup inference passes

# ============================================================================
# PROGRESS DISPLAY SETTINGS (Sep 14 Performance Optimization)
# ============================================================================
ENABLE_PROGRESS_DISPLAY = True  # Enable/disable chunk progress display
# (Removed) Progress/tqdm flags unused by current terminal logger
ENABLE_AVERAGE_ITS_DISPLAY = True  # Show average it/s in progress display
ENABLE_DEBUG_TIMING = False  # Per-chunk timing (debug mode only)

# ============================================================================
# GUI SETTINGS
# ============================================================================
START_SIZE = "FULL"  # Options: "FULL" for maximized window, or integer (50-100) for percentage of screen

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
ENABLE_LOG_APPEND = False  # Set to False for overwrite mode, True for append mode

# ============================================================================
# FEATURE TOGGLES
# ============================================================================
shutdown_requested = False  # Global shutdown flag


# ============================================================================
# CUDA OPTIMIZATION FOUNDATION SETTINGS
# ============================================================================

# FP16 + TF32 Foundation (biggest performance win)
ENABLE_FP16_PRECISION = True  # Use half-precision (FP16) for models (NEW)
ENABLE_TF32_MATMUL = True  # Enable TF32 for matrix operations (NEW)
ENABLE_MIXED_PRECISION = True  # Use torch.cuda.amp for automatic mixed precision (NEW)

# CUDA Graphs (eliminate kernel launch overhead)
# (Removed) CUDA Graphs controls are unused

# Memory Management (prevent fragmentation on RTX 4060 Ti 8GB)
PYTORCH_CUDA_ALLOC_CONF = (
    "max_split_size_mb:128"  # Disable expandable_segments for allocator stability
)

# Micro-batching (disable to run per-chunk only)
ENABLE_MICRO_BATCHING = False
ENABLE_VADER_MICRO_BATCHING = False  # Disable micro-batching even when VADER is enabled
