"""Download and prepare model files required by the portable Pipeline 4 runtime."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = APP_ROOT / "models"
CHATTERBOX_DIR = MODEL_ROOT / "chatterbox"
TURBO_DIR = MODEL_ROOT / "chatterbox-turbo"
VLLM_DIR = MODEL_ROOT / "vllm-t3"
MTL_V3_DIR = MODEL_ROOT / "chatterbox-mtl-v3"
MTL_V2_DIR = MODEL_ROOT / "chatterbox-mtl-v2"
VLLM_MTL_DIR = MODEL_ROOT / "vllm-t3-mtl-v3"
VLLM_MTL_V2_DIR = MODEL_ROOT / "vllm-t3-mtl-v2"
CHATTERBOX_REPO = "ResembleAI/chatterbox"
CHATTERBOX_REVISION = "1b475dffa71fb191cb6d5901215eb6f55635a9b6"
TURBO_REPO = "ResembleAI/chatterbox-turbo"
CHATTERBOX_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",
)
# V3 T3 is on repo main, not the pinned English revision used for t3_cfg.
MTL_V3_T3_FILE = "t3_mtl23ls_v3.safetensors"
MTL_V2_T3_FILE = "t3_mtl23ls_v2.safetensors"
MTL_V3_SIDE_FILES = (
    "grapheme_mtl_merged_expanded_v1.json",
    "Cangjie5_TC.json",
)
TURBO_T3_FILE = "t3_turbo_v1.safetensors"
TURBO_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
VLLM_CONFIG = {
    "architectures": ["ChatterboxT3"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_implementation": "sdpa",
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 30,
    "num_key_value_heads": 16,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": {
        "factor": 8.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 8,
}
VLLM_TURBO_CONFIG = {
    "architectures": ["ChatterboxT3Turbo"],
    "model_type": "gpt2",
    "n_embd": 1024,
    "hidden_size": 1024,
    "n_head": 16,
    "num_attention_heads": 16,
    "n_layer": 24,
    "num_hidden_layers": 24,
    "n_inner": 4096,
    "n_positions": 8196,
    "max_position_embeddings": 8196,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.0,
    "embd_pdrop": 0.0,
    "attn_pdrop": 0.0,
    "layer_norm_epsilon": 1e-05,
    "vocab_size": 50276,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
    "add_cross_attention": False,
    "scale_attn_by_inverse_layer_idx": False,
    "reorder_and_upcast_attn": False,
    "torch_dtype": "bfloat16",
}


def _download_file(repo_id: str, filename: str, destination: Path, revision: str | None = None) -> None:
    """Download one missing model file directly into the portable model store.

    Args:
        repo_id: Hugging Face repository containing the file.
        filename: Repository-relative filename to retrieve.
        destination: Directory where the file must be available at runtime.
        revision: Optional immutable repository revision.
    """
    target = destination / filename
    if target.is_file():
        return
    print(f"Downloading {repo_id}/{filename}...")
    kwargs = {"repo_id": repo_id, "filename": filename, "local_dir": str(destination)}
    if revision is not None:
        kwargs["revision"] = revision
    hf_hub_download(**kwargs)


def _link_or_copy(source: Path, destination: Path) -> None:
    """Create runtime model view without duplicating weights when filesystem allows it.

    Args:
        source: Downloaded checkpoint file.
        destination: Single-weight vLLM model-file location.
    """
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source)
        except OSError:
            shutil.copy2(source, destination)


def _write_vllm_config(destination: Path, config: dict | None = None) -> None:
    """Write a vLLM config.json into an isolated one-weight model directory.

    Args:
        destination: vLLM one-weight model directory.
        config: HuggingFace-style config dict. Defaults to English Llama ChatterboxT3.
    """
    destination.mkdir(parents=True, exist_ok=True)
    payload = VLLM_CONFIG if config is None else config
    (destination / "config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_vllm_directory() -> None:
    """Build vLLM's isolated one-weight model directory for English T3."""
    _write_vllm_config(VLLM_DIR)
    tokenizer_source = CHATTERBOX_DIR / "tokenizer.json"
    tokenizer_target = VLLM_DIR / "tokenizer.json"
    if not tokenizer_target.exists() and tokenizer_source.is_file():
        shutil.copy2(tokenizer_source, tokenizer_target)
    _link_or_copy(CHATTERBOX_DIR / "t3_cfg.safetensors", VLLM_DIR / "model.safetensors")


def _prepare_vllm_multilingual_directory() -> None:
    """Build vLLM's isolated one-weight directory for multilingual V3 T3.

    MtlTokenizer loads grapheme_mtl_merged_expanded_v1.json from the package, so
    this dir only needs config.json plus a single model.safetensors.
    """
    _write_vllm_config(VLLM_MTL_DIR)
    _link_or_copy(MTL_V3_DIR / MTL_V3_T3_FILE, VLLM_MTL_DIR / "model.safetensors")


def _prepare_vllm_multilingual_v2_directory() -> None:
    """Build vLLM's isolated one-weight directory for multilingual V2 T3."""
    _write_vllm_config(VLLM_MTL_V2_DIR)
    _link_or_copy(MTL_V2_DIR / MTL_V2_T3_FILE, VLLM_MTL_V2_DIR / "model.safetensors")


def _ensure_multilingual_v3() -> None:
    """Download multilingual V3 T3 plus tokenizer sidecars and populate HF cache.
    """
    MTL_V3_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(CHATTERBOX_REPO, MTL_V3_T3_FILE, MTL_V3_DIR)
    for filename in MTL_V3_SIDE_FILES:
        _download_file(CHATTERBOX_REPO, filename, MTL_V3_DIR)
    _prepare_vllm_multilingual_directory()


def _ensure_multilingual_v2() -> None:
    """Download multilingual V2 T3 and build its isolated vLLM directory."""
    MTL_V2_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(CHATTERBOX_REPO, MTL_V2_T3_FILE, MTL_V2_DIR)
    for filename in MTL_V3_SIDE_FILES:
        _download_file(CHATTERBOX_REPO, filename, MTL_V2_DIR)
    _prepare_vllm_multilingual_v2_directory()


def _ensure_turbo_t3() -> None:
    """Download Turbo T3 weights and tokenizer files into the turbo model dir.

    Turbo T3 needs the GPT2 tokenizer JSON next to t3_turbo_v1.safetensors.
    S3Gen meanflow is already downloaded for Phase 2.
    """
    TURBO_DIR.mkdir(parents=True, exist_ok=True)
    _download_file(TURBO_REPO, TURBO_T3_FILE, TURBO_DIR)
    _download_file(TURBO_REPO, "ve.safetensors", TURBO_DIR)
    for filename in TURBO_TOKENIZER_FILES:
        try:
            _download_file(TURBO_REPO, filename, TURBO_DIR)
        except Exception as exc:
            print(f"Warning: turbo tokenizer file {filename} not downloaded: {exc}")
    _prepare_vllm_turbo_directory()


def _prepare_vllm_turbo_directory() -> None:
    """Build vLLM's isolated one-weight directory for Turbo GPT2 T3."""
    VLLM_TURBO_DIR = MODEL_ROOT / "vllm-t3-turbo"
    _write_vllm_config(VLLM_TURBO_DIR, VLLM_TURBO_CONFIG)
    _link_or_copy(TURBO_DIR / TURBO_T3_FILE, VLLM_TURBO_DIR / "model.safetensors")


def ensure_pipeline4_models() -> None:
    """Ensure all model assets needed by Pipeline 4 exist before GUI startup."""
    CHATTERBOX_DIR.mkdir(parents=True, exist_ok=True)
    TURBO_DIR.mkdir(parents=True, exist_ok=True)
    for filename in CHATTERBOX_FILES:
        _download_file(CHATTERBOX_REPO, filename, CHATTERBOX_DIR, CHATTERBOX_REVISION)
    _download_file(TURBO_REPO, "s3gen_meanflow.safetensors", TURBO_DIR)
    _prepare_vllm_directory()
    _ensure_multilingual_v3()
    _ensure_multilingual_v2()
    _ensure_turbo_t3()
    os.environ["CHATTERBOX_CKPT_DIR"] = str(CHATTERBOX_DIR)
    os.environ["TURBO_CKPT_DIR"] = str(TURBO_DIR)
    os.environ["VLLM_ENGLISH_CKPT_DIR"] = str(VLLM_DIR)
    os.environ["VLLM_MULTILINGUAL_CKPT_DIR"] = str(VLLM_MTL_DIR)
    os.environ["VLLM_MULTILINGUAL_V2_CKPT_DIR"] = str(VLLM_MTL_V2_DIR)
    os.environ["VLLM_TURBO_CKPT_DIR"] = str(MODEL_ROOT / "vllm-t3-turbo")
    os.environ["CHATTERBOX_MTL_CKPT_DIR"] = str(MTL_V3_DIR)
    os.environ["CHATTERBOX_MTL_V2_CKPT_DIR"] = str(MTL_V2_DIR)
    os.environ["CHATTERBOX_T3_SOURCE"] = os.environ.get(
        "CHATTERBOX_T3_SOURCE", "multilingual-v3"
    )
    os.environ["CHATTERBOX_T3_LANGUAGE"] = os.environ.get(
        "CHATTERBOX_T3_LANGUAGE", "en"
    )


def main() -> None:
    """Prepare model files when the Pipeline 4 launcher starts."""
    ensure_pipeline4_models()
    print("Pipeline 4 model files ready.")


if __name__ == "__main__":
    main()
