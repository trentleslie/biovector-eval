"""Device detection utilities."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_best_device() -> str:
    """Detect and return the best available compute device.

    Returns:
        'cuda' if GPU is available and working, otherwise 'cpu'.
    """
    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return "cpu"

    try:
        # Test that CUDA actually works by creating a small tensor
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        torch.cuda.empty_cache()

        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Using GPU: {device_name} ({memory_gb:.1f} GB)")
        return "cuda"
    except Exception as e:
        logger.warning(f"CUDA available but failed to initialize: {e}")
        logger.info("Falling back to CPU")
        return "cpu"


def check_gpu_status() -> dict:
    """Get detailed GPU status information.

    Returns:
        Dictionary with GPU status details.
    """
    status = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "recommended_device": "cpu",
    }

    if not status["cuda_available"]:
        return status

    try:
        status["cuda_version"] = torch.version.cuda
        status["device_count"] = torch.cuda.device_count()

        for i in range(status["device_count"]):
            props = torch.cuda.get_device_properties(i)
            status["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
            })

        # Try to actually use CUDA
        test_tensor = torch.zeros(1, device="cuda")
        del test_tensor
        torch.cuda.empty_cache()
        status["recommended_device"] = "cuda"

    except Exception as e:
        status["error"] = str(e)
        status["recommended_device"] = "cpu"

    return status
