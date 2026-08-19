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
DEFAULT_ASR_THRESHOLD = 0.75  # Default similarity threshold for ASR validation

# ============================================================================
# AUDIO QUALITY SETTINGS
# ============================================================================
ENABLE_MID_DROP_CHECK = False
ENABLE_ASR = False  # Disabled by default due to tensor dimension errors
ASR_WORKERS = 4  # Parallel ASR on CPU threads
DEFAULT_ASR_MODEL = "base"  # Default Whisper model for ASR validation
# Tab 2 "Run ASR on GPU" checkbox default; off = safe CPU default. Saved by
# save_config_to_file, which strips inline comments on this line -- keep the
# explanation here, above the line, instead.
ASR_USE_GPU = True

# NOTE: ASR runs on CPU only (hardcoded in asr_daemon.py)
# This prevents VRAM contention with TTS models and avoids OOM errors
# CPU mode uses system RAM (~74 MB per worker) instead of competing for GPU VRAM
# GPU mode is intentionally disabled during audiobook generation

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
SILENCE_COMMA = 120
SILENCE_SEMICOLON = 150  # Medium pause after semicolons
SILENCE_COLON = 150  # Pause after colons
SILENCE_PERIOD = 400
SILENCE_QUESTION_MARK = 650
SILENCE_EXCLAMATION = 200
SILENCE_DASH = 200  # Em dash pause
SILENCE_ELLIPSIS = 80  # Ellipsis pause (suspense)
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
# INLINE PAUSES (Option A)
# ============================================================================
# Enable inline pause markers embedded in text (e.g., "~1", "~2").
# When enabled, the TTS wrapper will parse these markers and insert configured
# silences at those positions without splitting the chunk.
ENABLE_INLINE_PAUSES = True
INLINE_PAUSE_1_MS = 150  # "~1" → 150 ms
INLINE_PAUSE_2_MS = 700  # "~2" → 300 ms


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

# ============================================================================
# VLLM BACKEND SETTINGS
# ============================================================================
# Device for vLLM backend. Set VLLM_DEVICE env var to override auto-detection.
# Options: "cuda", "mps" (Apple Silicon), "cpu", or VLLM_DEVICE None/unset for auto-detect (cuda → mps → cpu)
VLLM_DEVICE = os.getenv("VLLM_DEVICE")

# Model variant. Options: "english" (faster), "multilingual" (supports multiple languages)
VLLM_MODEL_VARIANT = "english"

# Default language code (e.g., "en", "es", "fr", "de", "zh")
VLLM_DEFAULT_LANGUAGE = "en"

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

# Override path to multilingual model checkpoints (defaults to ./t3-model-multilingual)
VLLM_MULTILINGUAL_CKPT_DIR = os.environ.get("VLLM_MULTILINGUAL_CKPT_DIR")

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
DEFAULT_CFG_WEIGHT = 0.5
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
