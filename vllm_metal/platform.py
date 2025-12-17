# SPDX-License-Identifier: Apache-2.0
"""Metal Platform implementation for vLLM."""

import platform
from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional, Tuple, Type

import torch

from vllm_metal._compat import Platform, PlatformEnum, init_logger

from vllm_metal.utils import (
    check_mps_availability,
    get_apple_chip_name,
    get_metal_device_info,
    get_mps_memory_info,
    is_apple_silicon,
    mps_empty_cache,
    mps_synchronize,
)
from vllm_metal.envs import (
    VLLM_METAL_DEVICE_ID,
    VLLM_METAL_EAGER_MODE,
    VLLM_METAL_MEMORY_FRACTION,
)

if TYPE_CHECKING:
    from vllm_metal._compat import VllmConfig

logger = init_logger(__name__)


class MetalPlatform(Platform):
    """Platform implementation for Apple Metal/MPS backend."""

    _enum = PlatformEnum.OOT  # Out-of-tree platform
    device_name: str = "mps"
    device_type: str = "mps"
    dispatch_key: str = "MPS"

    supported_quantization = ["awq", "gptq", "compressed-tensors"]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of the Metal device."""
        return get_apple_chip_name()

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        """Get a unique identifier for the Metal device."""
        # MPS doesn't have a PCI bus ID like CUDA
        # Use chip name + device_id as identifier
        chip_name = get_apple_chip_name().replace(" ", "_")
        return f"mps:{chip_name}:{device_id}"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get total memory available to the device in bytes.

        On Apple Silicon, GPU uses unified memory shared with CPU.
        """
        info = get_metal_device_info()
        total_mem = info.get("total_memory", 0)
        # Apply memory fraction limit
        return int(total_mem * VLLM_METAL_MEMORY_FRACTION)

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        """Get device capability.

        Returns None as MPS doesn't have compute capability like CUDA.
        """
        return None

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool) -> bool:
        """Check if async output is supported."""
        # MPS supports async operations
        return not enforce_eager

    @classmethod
    def inference_mode(cls):
        """Return the appropriate inference mode context manager."""
        return torch.inference_mode()

    @classmethod
    def seed_everything(cls, seed: int) -> None:
        """Seed all random number generators."""
        torch.manual_seed(seed)
        if torch.backends.mps.is_available():
            # MPS uses the same seed as CPU for now
            torch.mps.manual_seed(seed)

    @classmethod
    def set_device(cls, device) -> None:
        """Set the current device.

        MPS only has one device, so this is mostly a no-op.
        """
        # Ensure MPS is the default device for new tensors
        if isinstance(device, int):
            device = torch.device("mps", device)
        elif isinstance(device, str):
            device = torch.device(device)

    @classmethod
    def get_current_memory_usage(cls, device=None) -> Tuple[int, int]:
        """Get current memory usage.

        Returns:
            Tuple of (used_bytes, total_bytes)
        """
        return get_mps_memory_info()

    @classmethod
    def empty_cache(cls) -> None:
        """Empty the MPS memory cache."""
        mps_empty_cache()

    @classmethod
    def synchronize(cls) -> None:
        """Synchronize MPS operations."""
        mps_synchronize()

    @classmethod
    def mem_get_info(cls) -> Tuple[int, int]:
        """Get memory info (free, total).

        Note: MPS uses unified memory, so 'free' is estimated.
        """
        allocated, total = get_mps_memory_info()
        free = total - allocated
        return free, total

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Check and update vLLM configuration for Metal backend."""
        from vllm.config import CacheConfig

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        # Validate platform availability
        available, error = check_mps_availability()
        if not available:
            raise RuntimeError(f"Metal/MPS backend not available: {error}")

        # MPS doesn't support tensor parallelism
        if parallel_config.tensor_parallel_size > 1:
            raise ValueError(
                "Metal backend does not support tensor parallelism. "
                "Please set tensor_parallel_size=1"
            )

        # MPS doesn't support pipeline parallelism
        if parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "Metal backend does not support pipeline parallelism. "
                "Please set pipeline_parallel_size=1"
            )

        # Set the worker class for Metal platform
        # Use CPU worker as base since MPS uses similar patterns
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.cpu_worker.CPUWorker"

        # Force eager mode if configured
        if VLLM_METAL_EAGER_MODE:
            model_config.enforce_eager = True
            logger.info("Metal backend: Using eager mode")

        # Adjust KV cache dtype if needed
        if cache_config.cache_dtype == "auto":
            # Default to float16 for MPS
            cache_config.cache_dtype = "float16"
            logger.info("Metal backend: Using float16 for KV cache")

        # Log configuration
        logger.info(
            f"Metal backend initialized: device={cls.get_device_name()}, "
            f"memory={cls.get_device_total_memory() / 1e9:.1f}GB"
        )

    @classmethod
    def verify_quantization(cls, quant: str) -> None:
        """Verify that the quantization method is supported."""
        if quant and quant not in cls.supported_quantization:
            raise ValueError(
                f"Quantization method '{quant}' not supported on Metal. "
                f"Supported: {cls.supported_quantization}"
            )

    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None:
        """Verify that the model architecture is supported."""
        # Most transformer architectures are supported via PyTorch MPS
        # Log a warning for potentially unsupported architectures
        unsupported = {"mamba", "rwkv", "xlnet"}
        if model_arch.lower() in unsupported:
            logger.warning(
                f"Model architecture '{model_arch}' may not be fully "
                "supported on Metal backend"
            )

    @classmethod
    def get_attn_backend_cls(cls):
        """Get the attention backend class for Metal."""
        from vllm_metal.attention import MetalAttentionBackend
        return MetalAttentionBackend

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """Check if pin memory is available.

        MPS uses unified memory, so pinned memory isn't applicable.
        """
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> bool:
        """Check if the dtype is supported on Metal."""
        supported = {
            torch.float32,
            torch.float16,
            torch.bfloat16,  # Supported on newer chips
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        }
        return dtype in supported

    @classmethod
    def supports_fp8(cls) -> bool:
        """Check if FP8 is supported."""
        return False

    @classmethod
    def supports_mx(cls) -> bool:
        """Check if MX formats are supported."""
        return False

    @classmethod
    def get_punica_wrapper(cls):
        """Get the Punica wrapper for LoRA.

        Returns None as Punica is CUDA-specific.
        """
        return None

    @classmethod
    def can_update_inplace(cls) -> bool:
        """Check if in-place updates are supported."""
        return True

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """Check if hybrid KV cache is supported."""
        return False

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        """Check if static graph mode is supported."""
        # MPS doesn't have graph capture like CUDA
        return False

    @classmethod
    @contextmanager
    def device_scope(cls, device_id: int = 0):
        """Context manager for device scope."""
        # MPS only has one device
        yield

    @classmethod
    def get_device_communicator_cls(cls):
        """Get the device communicator class.

        Returns None as MPS doesn't support distributed.
        """
        return None

    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        timeout,
    ):
        """Initialize torch distributed process group.

        MPS doesn't support distributed training.
        """
        raise NotImplementedError(
            "Metal backend does not support distributed operations"
        )

    @classmethod
    def import_kernels(cls) -> None:
        """Import Metal-specific kernels."""
        from vllm_metal import ops
        # Trigger kernel registration
        ops.register_metal_ops()
        logger.debug("Metal kernels imported")
