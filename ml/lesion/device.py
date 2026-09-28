"""Device selection and hardware inspection utilities for StenoTrace."""

from typing import Optional
import torch


def get_device(verbose: bool = True) -> torch.device:
    """Automatically select CUDA if available, otherwise CPU.

    Args:
        verbose: If True, prints device details (GPU name and VRAM if CUDA).

    Returns:
        torch.device: Selected compute device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[Device] Selected: {device} ({device_name}, {vram_gb:.2f} GB VRAM)")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[Device] Selected: cpu (CUDA unavailable)")

    return device


def get_device_info() -> dict:
    """Return dictionary with device hardware details."""
    cuda_avail = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_avail,
        "device_type": "cuda" if cuda_avail else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if cuda_avail else None,
        "vram_gb": (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)) if cuda_avail else 0.0,
        "cuda_version": torch.version.cuda if cuda_avail else None,
    }
    return info
