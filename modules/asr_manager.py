"""ASR model manager built on top of faster-whisper."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

# Try to import required modules, fallback if not available
try:
    import torch
    from faster_whisper import WhisperModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    WhisperModel = Any

from config.config import DEFAULT_ASR_MODEL, ASR_MODEL_VRAM_MB

DEFAULT_SAFETY_BUFFER_MB = 500


def get_real_time_vram_status() -> Dict[str, int | bool]:
    """Returns a dictionary containing real-time VRAM status, including total, allocated, reserved, cached, and available memory in MB, as well as whether a GPU is present. If PyTorch is not available, returns default values for all metrics with `has_gpu` set to False."""
    if not TORCH_AVAILABLE:
        return {
            "total_mb": 0,
            "allocated_mb": 0,
            "reserved_mb": 0,
            "cached_mb": 0,
            "available_mb": 0,
            "has_gpu": False,
        }
        
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            total_vram = props.total_memory
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            available = total_vram - allocated
            return {
                "total_mb": total_vram // 1024 // 1024,
                "allocated_mb": allocated // 1024 // 1024,
                "reserved_mb": reserved // 1024 // 1024,
                "cached_mb": cached // 1024 // 1024,
                "available_mb": available // 1024 // 1024,
                "has_gpu": True,
            }
    except Exception as exc:  # pragma: no cover - diagnostic only
        logging.warning("Failed to compute VRAM status: %s", exc)
    return {
        "total_mb": 0,
        "allocated_mb": 0,
        "reserved_mb": 0,
        "cached_mb": 0,
        "available_mb": 0,
        "has_gpu": False,
    }


def calculate_available_vram_for_asr(safety_buffer_mb: int = DEFAULT_SAFETY_BUFFER_MB) -> int:
    """Calculates available VRAM for ASR after subtracting a safety buffer.
    Args:
    safety_buffer_mb (int): Safety buffer in MB to be subtracted from available VRAM.
    Returns:
    int: Available VRAM for ASR after accounting for the safety buffer.
    ---
    Checks if a model can fit into the GPU based on its required VRAM.
    Args:
    model_name (str): The name of the model.
    available_vram_mb (int): Available VRAM in MB on the GPU.
    Returns:
    bool: True if the model can fit into the GPU, False otherwise.
    ---
    Loads a Whisper model for ASR.
    Args:
    model_name (str): The name of the model to load.
    device (str): The device to load the model onto.
    compute_type (str): The compute type to use for the model.
    Raises:
    RuntimeError: If PyTorch/faster-whisper is not available.
    """
    vram_status = get_real_time_vram_status()
    if not vram_status["has_gpu"]:
        return 0
    return max(0, vram_status["available_mb"] - safety_buffer_mb)


def can_model_fit_gpu(model_name: str, available_vram_mb: int) -> bool:
    """Determines if a model can fit on a GPU based on available VRAM.
    Args:
    model_name (str): The name of the model.
    available_vram_mb (int): Available VRAM in megabytes.
    Returns:
    bool: True if the model can fit, False otherwise.
    """
    required_vram = ASR_MODEL_VRAM_MB.get(model_name, 0)
    return available_vram_mb >= required_vram


def _load_whisper_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    """Loads a Whisper model with specified device and compute type.
    Args:
    model_name (str): The name of the model to load.
    primary_device (str): The preferred device for loading the model.
    fallback_device (str, optional): The fallback device if the primary one is unavailable. Defaults to "cpu".
    primary_compute_type (Optional[str], optional): The compute type for the primary device. If None, determined automatically. Defaults to None.
    fallback_compute_type (Optional[str], optional): The compute type for the fallback device. If None, determined automatically. Defaults to None.
    Returns:
    Tuple[WhisperModel, str]: A tuple containing the loaded Whisper model and the effective device used.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch/faster-whisper not available")
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def try_load_model_with_fallback(
    model_name: str,
    primary_device: str,
    fallback_device: str = "cpu",
    primary_compute_type: Optional[str] = None,
    fallback_compute_type: Optional[str] = None,
) -> Tuple[WhisperModel, str]:
    """Attempts to load a Whisper model on a specified device. If it fails, falls back to another device.
    Args:
    - model_name (str): The name of the model to load.
    - primary_device (str): The preferred device for loading the model.
    - fallback_device (str, optional): The device to fall back to if the primary device fails. Defaults to "cpu".
    - primary_compute_type (Optional[str], optional): Compute type for the primary device. If None, defaults based on the device.
    - fallback_compute_type (Optional[str], optional): Compute type for the fallback device. If None, defaults based on the device.
    Returns:
    Tuple[WhisperModel, str]: The loaded model and the device it was loaded on.
    """
    primary_compute_type = primary_compute_type or _compute_type_for_device(primary_device)
    fallback_compute_type = fallback_compute_type or _compute_type_for_device(fallback_device)
    try:
        print(f"🎯 Attempting to load {model_name} on {primary_device.upper()} ({primary_compute_type})")
        model = _load_whisper_model(model_name, device=primary_device, compute_type=primary_compute_type)
        print(f"✅ Successfully loaded {model_name} on {primary_device.upper()}")
        return model, primary_device
    except Exception as exc:
        print(f"⚠️ {model_name} failed on {primary_device} ({exc})")
        if fallback_device != primary_device:
            try:
                print(f"🔄 Trying {model_name} on {fallback_device.upper()} ({fallback_compute_type})")
                model = _load_whisper_model(model_name, device=fallback_device, compute_type=fallback_compute_type)
                print(f"✅ Successfully loaded {model_name} on {fallback_device.upper()}")
                return model, fallback_device
            except Exception as fallback_exc:
                print(f"❌ {model_name} also failed on {fallback_device} ({fallback_exc})")
        raise RuntimeError(
            f"Model {model_name} failed on both {primary_device} and {fallback_device}"
        ) from exc


def _compute_type_for_device(device: str) -> str:
    """Computes and returns the type for a given device.
    Args:
    device (str): The device name, expected to be "cuda".
    Returns:
    str: The computed type ("float16" if device is "cuda", otherwise "int8").
    This function checks if the provided device name is "cuda" and returns "float16" accordingly. For other devices, it defaults to "int8".
    """
    if device.lower() == "cuda":
        return "float16"
    return "int8"


def load_asr_model_adaptive(
    asr_config: Optional[Dict[str, Any]] = None,
    force_cpu: bool = False,
) -> Tuple[Optional[WhisperModel], Optional[str]]:
    """Loads an adaptive ASR model using faster-whisper. Args: asr_config (Optional[Dict[str, Any]]): Configuration for the ASR model. force_cpu (bool): If True, forces loading the model on CPU regardless of VRAM availability. Returns: Tuple containing the loaded WhisperModel and a status message if applicable."""
    print("🔍 Starting adaptive ASR model loading (faster-whisper)...")
    vram_status = get_real_time_vram_status()
    available_vram = calculate_available_vram_for_asr()
    print("🖥️ Real-time VRAM status:")
    print(f"   Total: {vram_status['total_mb']:,}MB")
    print(f"   Allocated: {vram_status['allocated_mb']:,}MB")
    print(f"   Reserved: {vram_status['reserved_mb']:,}MB")
    print(f"   Available: {available_vram:,}MB (after {DEFAULT_SAFETY_BUFFER_MB}MB safety buffer)")

    if force_cpu:
        print("🔄 force_cpu=True → loading on CPU regardless of availability")
        try:
            model = _load_whisper_model(DEFAULT_ASR_MODEL, device="cpu", compute_type="int8")
            print(f"✅ Successfully loaded {DEFAULT_ASR_MODEL} on CPU")
            return model, "cpu"
        except Exception as exc:
            print(f"❌ Failed to load {DEFAULT_ASR_MODEL} on CPU: {exc}")
            return None, None

    if asr_config and asr_config.get("enabled") and "primary_model" in asr_config:
        primary_model = asr_config["primary_model"]
        primary_device = asr_config["primary_device"].lower()
        fallback_model = asr_config.get("fallback_model", primary_model)
        fallback_device = asr_config.get("fallback_device", "cpu").lower()

        print("🧠 Using intelligent ASR configuration:")
        print(f"   Primary: {primary_model} on {primary_device.upper()}")
        print(f"   Fallback: {fallback_model} on {fallback_device.upper()}")

        if primary_device == "gpu":
            if not vram_status["has_gpu"]:
                print("⚠️ No GPU detected – forcing CPU")
                primary_device = "cpu"
            elif not can_model_fit_gpu(primary_model, available_vram):
                required = ASR_MODEL_VRAM_MB.get(primary_model, 0)
                print(
                    f"⚠️ Insufficient VRAM for {primary_model}: need {required}MB, have {available_vram}MB"
                )
                primary_device = "cpu"

        try:
            return try_load_model_with_fallback(
                primary_model,
                "cuda" if primary_device == "gpu" else primary_device,
                "cuda" if fallback_device == "gpu" else fallback_device,
            )
        except RuntimeError:
            print("🔁 Falling back to default configuration…")

    device = "cuda"
    compute_type = "float16"
    if not vram_status["has_gpu"] or not can_model_fit_gpu(DEFAULT_ASR_MODEL, available_vram):
        device = "cpu"
        compute_type = "int8"
    print(f"🔧 Loading {DEFAULT_ASR_MODEL} on {device.upper()} ({compute_type})")

    try:
        model = _load_whisper_model(DEFAULT_ASR_MODEL, device=device, compute_type=compute_type)
        print(f"✅ Successfully loaded {DEFAULT_ASR_MODEL} on {device.upper()}")
        return model, device
    except Exception as exc:
        print(f"❌ Failed to load {DEFAULT_ASR_MODEL} on {device.upper()}: {exc}")
        if device == "cuda":
            try:
                print("🆘 Retrying on CPU…")
                model = _load_whisper_model(DEFAULT_ASR_MODEL, device="cpu", compute_type="int8")
                print(f"✅ Successfully loaded {DEFAULT_ASR_MODEL} on CPU")
                return model, "cpu"
            except Exception as cpu_exc:
                print(f"💀 Complete failure: {cpu_exc}")
        return None, None


def cleanup_asr_model(asr_model: Optional[WhisperModel]) -> None:
    """Cleans up an ASR model by deleting it and emptying the CUDA cache if available.
    Args:
    asr_model (Optional[WhisperModel]): The ASR model to clean up.
    Returns: None
    ---
    Retrieves memory information for ASR processing.
    Returns:
    Dict[str, int | bool]: A dictionary containing VRAM status and available VRAM for ASR.
    """
    if asr_model is not None:
        try:
            del asr_model
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🧹 ASR model cleaned up")
        except Exception as exc:  # pragma: no cover - cleanup best-effort
            logging.warning("Failed to cleanup ASR model: %s", exc)


def get_asr_memory_info() -> Dict[str, int | bool]:
    """Get memory information for ASR processes.
    Args:
    - None
    Returns:
    - Dict[str, int | bool]: A dictionary containing VRAM status and available memory for ASR in MB.
    """
    status = get_real_time_vram_status()
    info = dict(status)
    info["available_after_buffer_mb"] = calculate_available_vram_for_asr()
    return info


__all__ = [
    "get_real_time_vram_status",
    "calculate_available_vram_for_asr",
    "can_model_fit_gpu",
    "try_load_model_with_fallback",
    "load_asr_model_adaptive",
    "cleanup_asr_model",
    "get_asr_memory_info",
]
