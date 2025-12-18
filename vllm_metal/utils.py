# SPDX-License-Identifier: Apache-2.0
"""Utility functions for vLLM Metal backend."""

import platform
import re
import subprocess
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    if platform.system() != "Darwin":
        return False

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "Apple" in result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to architecture check
        return platform.machine() == "arm64"


@lru_cache(maxsize=1)
def get_apple_chip_name() -> str:
    """Get the Apple chip name (e.g., 'M1', 'M2 Pro', 'M3 Max')."""
    if not is_apple_silicon():
        return "Unknown"

    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
        )
        brand = result.stdout.strip()
        # Extract chip name from brand string
        match = re.search(r"Apple\s+(M\d+(?:\s+(?:Pro|Max|Ultra))?)", brand)
        if match:
            return match.group(1)
        return brand
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Apple Silicon"


@lru_cache(maxsize=1)
def get_metal_device_info() -> dict:
    """Get Metal device information."""
    info = {
        "name": get_apple_chip_name(),
        "is_apple_silicon": is_apple_silicon(),
        "metal_available": is_apple_silicon() and platform.system() == "Darwin",
    }

    if info["metal_available"]:
        # Get memory info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True,
            )
            info["total_memory"] = int(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            info["total_memory"] = 0

        # Get GPU cores
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                check=True,
            )
            match = re.search(r"Total Number of Cores:\s*(\d+)", result.stdout)
            if match:
                info["gpu_cores"] = int(match.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    return info


def check_metal_availability() -> tuple[bool, str | None]:
    """Check if Metal is available and return status with error message if not.

    Returns:
        Tuple of (is_available, error_message)
    """
    if platform.system() != "Darwin":
        return False, "Metal is only available on macOS"

    if not is_apple_silicon():
        return False, "Metal GPU acceleration requires Apple Silicon"

    return True, None


def get_metal_memory_info() -> tuple[int, int]:
    """Get Metal memory usage information.

    Returns:
        Tuple of (allocated_bytes, total_bytes)
    """
    available, _ = check_metal_availability()
    if not available:
        return 0, 0

    try:
        # Metal uses unified memory, so we report system memory
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        )
        total = int(result.stdout.strip())

        # Get current allocated memory from PyTorch
        # Note: This is an approximation based on PyTorch's tracking
        try:
            allocated = torch.mps.current_allocated_memory()
        except Exception:
            allocated = 0

        return allocated, total
    except Exception:
        return 0, 0


def metal_synchronize() -> None:
    """Synchronize Metal operations."""
    try:
        torch.mps.synchronize()
    except Exception:
        pass


def metal_empty_cache() -> None:
    """Empty Metal cache to free memory."""
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


def get_optimal_dtype() -> torch.dtype:
    """Get the optimal dtype for Metal inference.

    Metal works best with float16 or bfloat16 on newer chips.
    """
    # bfloat16 support was added in later Metal versions
    # For now, default to float16 which is well-supported
    return torch.float16


def check_model_compatibility(model_config) -> tuple[bool, str | None]:
    """Check if a model configuration is compatible with Metal backend.

    Args:
        model_config: vLLM model configuration

    Returns:
        Tuple of (is_compatible, error_message)
    """
    # Metal doesn't support all quantization methods
    if hasattr(model_config, "quantization") and model_config.quantization:
        quant = model_config.quantization
        supported_quant = {"awq", "gptq", None}
        if quant not in supported_quant:
            return False, f"Quantization method '{quant}' not supported on Metal"

    return True, None


def format_memory_size(size_bytes: int | float) -> str:
    """Format memory size in human-readable format."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"
