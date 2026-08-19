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


def _prepare_vllm_directory() -> None:
    """Build vLLM's isolated one-weight model directory for Pipeline 4 Phase 1."""
    VLLM_DIR.mkdir(parents=True, exist_ok=True)
    (VLLM_DIR / "config.json").write_text(
        json.dumps(VLLM_CONFIG, indent=2) + "\n",
        encoding="utf-8",
    )
    tokenizer_source = CHATTERBOX_DIR / "tokenizer.json"
    tokenizer_target = VLLM_DIR / "tokenizer.json"
    if not tokenizer_target.exists() and tokenizer_source.is_file():
        shutil.copy2(tokenizer_source, tokenizer_target)
    _link_or_copy(CHATTERBOX_DIR / "t3_cfg.safetensors", VLLM_DIR / "model.safetensors")


def ensure_pipeline4_models() -> None:
    """Ensure all model assets needed by Pipeline 4 exist before GUI startup."""
    CHATTERBOX_DIR.mkdir(parents=True, exist_ok=True)
    TURBO_DIR.mkdir(parents=True, exist_ok=True)
    for filename in CHATTERBOX_FILES:
        _download_file(CHATTERBOX_REPO, filename, CHATTERBOX_DIR, CHATTERBOX_REVISION)
    _download_file(TURBO_REPO, "s3gen_meanflow.safetensors", TURBO_DIR)
    _prepare_vllm_directory()
    os.environ["CHATTERBOX_CKPT_DIR"] = str(CHATTERBOX_DIR)
    os.environ["TURBO_CKPT_DIR"] = str(TURBO_DIR)
    os.environ["VLLM_ENGLISH_CKPT_DIR"] = str(VLLM_DIR)


def main() -> None:
    """Prepare model files when the Pipeline 4 launcher starts."""
    ensure_pipeline4_models()
    print("Pipeline 4 model files ready.")


if __name__ == "__main__":
    main()
