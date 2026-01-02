# SPDX-License-Identifier: Apache-2.0
"""Tensor bridge between MLX and PyTorch.

Provides zero-copy conversion when possible using Apple Silicon's unified memory.
"""

from typing import Literal

import mlx.core as mx
import torch

# MLX to PyTorch dtype mapping
MLX_TO_TORCH_DTYPE: dict[mx.Dtype, torch.dtype] = {
    mx.float32: torch.float32,
    mx.float16: torch.float16,
    mx.bfloat16: torch.bfloat16,
    mx.int32: torch.int32,
    mx.int64: torch.int64,
    mx.int16: torch.int16,
    mx.int8: torch.int8,
    mx.uint8: torch.uint8,
    mx.bool_: torch.bool,
}

# PyTorch to MLX dtype mapping
TORCH_TO_MLX_DTYPE: dict[torch.dtype, mx.Dtype] = {
    v: k for k, v in MLX_TO_TORCH_DTYPE.items()
}


def get_torch_device() -> torch.device:
    """Get the PyTorch device for Metal/MPS.

    Returns:
        torch.device for MPS if available, else CPU
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def torch_to_mlx(tensor: torch.Tensor) -> mx.array:
    """Convert PyTorch tensor to MLX array.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        tensor: PyTorch tensor (can be on any device)

    Returns:
        MLX array with the same data
    """
    # Move to CPU if on MPS for numpy conversion
    if tensor.device.type == "mps":
        tensor = tensor.cpu()

    # Convert via numpy for zero-copy on unified memory
    np_array = tensor.detach().numpy()
    return mx.array(np_array)


def mlx_to_torch(
    array: mx.array,
    device: torch.device | Literal["mps", "cpu"] | None = None,
    already_contiguous: bool = False,
) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor.

    Uses numpy as an intermediate to enable zero-copy on unified memory.

    Args:
        array: MLX array
        device: Target PyTorch device (default: MPS if available)
        already_contiguous: Skip contiguity check if array is known contiguous

    Returns:
        PyTorch tensor with the same data
    """
    if device is None:
        device = get_torch_device()
    elif isinstance(device, str):
        device = torch.device(device)

    # Use memoryview for zero-copy conversion (bypasses numpy for bfloat16)
    # reference: https://github.com/ml-explore/mlx/issues/403
    torch_dtype = MLX_TO_TORCH_DTYPE.get(array.dtype)
    if torch_dtype is not None:
        if already_contiguous:
            # Fast path: skip contiguity check, single eval
            mx.eval(array)
            buffer = memoryview(array)
        else:
            # MLX views / non-contiguous arrays expose a non-contiguous buffer (or
            # sometimes no usable buffer), which `torch.frombuffer` can't consume.
            # Make contiguous first, then eval once
            array = mx.contiguous(array)
            mx.eval(array)
            buffer = memoryview(array)

        tensor = torch.frombuffer(buffer, dtype=torch_dtype).reshape(array.shape)
    else:
        # Fallback to numpy path for unsupported dtypes
        raise ValueError(f"Unsupported MLX dtype: {array.dtype}")

    # Move to target device
    if device.type != "cpu":
        tensor = tensor.to(device)

    return tensor


def sync_mlx() -> None:
    """Synchronize MLX operations.

    Call this before converting MLX arrays to ensure all operations complete.
    """
    mx.eval([])


def sync_torch() -> None:
    """Synchronize PyTorch MPS operations.

    Call this before converting PyTorch tensors to ensure all operations complete.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
