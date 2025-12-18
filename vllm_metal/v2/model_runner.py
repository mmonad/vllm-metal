# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner V2 - High-performance inference with custom Metal kernels.

This is a thin Python wrapper around the Rust Metal kernels.
Key features:
- Custom Metal kernels for attention, GEMV, RoPE, RMS norm
- Zero-copy data transfer via unified memory
- vLLM-compatible interface
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# Import Rust extensions
try:
    from vllm_metal_rust import (
        BatchStateManager,
        BlockTableManager,
        MetalBuffer,
    )

    RUST_METAL_AVAILABLE = True
    logger.info("Rust Metal V2 extensions loaded")
except ImportError as e:
    RUST_METAL_AVAILABLE = False
    MetalBuffer = None
    BatchStateManager = None  # type: ignore[misc, assignment]
    BlockTableManager = None  # type: ignore[misc, assignment]
    logger.warning(f"Rust Metal V2 extensions not available: {e}")


class MetalModelRunner:
    """Metal Model Runner V2 with custom Metal kernels.

    This runner uses Rust-based Metal kernels for maximum performance on Apple Silicon.
    It maintains compatibility with vLLM's model runner interface while using
    custom Metal kernels for attention and other critical operations.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        assert device.type == "mps", (
            f"MetalModelRunner requires Metal device (mps), got {device}"
        )
        assert RUST_METAL_AVAILABLE, "Rust Metal extensions required for V2 runner"

        self.vllm_config = vllm_config
        self.device = device
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config

        # Model parameters
        self.hidden_size = self.model_config.get_hidden_size()
        self.num_heads = self.model_config.get_num_attention_heads()
        self.num_kv_heads = self.model_config.get_num_kv_heads()
        self.head_dim = self.hidden_size // self.num_heads
        self.block_size = self.cache_config.block_size

        # GQA ratio
        self.gqa_ratio = self.num_heads // self.num_kv_heads

        # Initialize Rust managers
        max_num_reqs = self.scheduler_config.max_num_seqs
        max_model_len = self.model_config.max_model_len
        max_num_blocks = (max_model_len + self.block_size - 1) // self.block_size

        self._batch_state = BatchStateManager(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            block_size=self.block_size,
            max_num_blocks_per_req=max_num_blocks,
        )

        self._block_table = BlockTableManager(
            num_kv_cache_groups=1,
            block_size=self.block_size,
            max_num_reqs=max_num_reqs,
            max_num_blocks_per_req=max_num_blocks,
        )

        # Model will be loaded later
        self.model: nn.Module | None = None

        # Statistics
        self._decode_count = 0
        self._prefill_count = 0

        logger.info(
            f"MetalModelRunner V2 initialized: "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_heads}, "
            f"num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"block_size={self.block_size}"
        )

    def load_model(self) -> nn.Module:
        """Load the model onto the Metal device."""
        from vllm.model_executor.model_loader import get_model

        self.model = get_model(
            vllm_config=self.vllm_config,
        )

        # Move to Metal device
        self.model = self.model.to(self.device)

        # Verify
        try:
            first_param = next(iter(self.model.parameters()))
            logger.info(f"Model loaded on device: {first_param.device}")
        except StopIteration:
            logger.warning("Model has no parameters")

        return self.model

    def get_model(self) -> nn.Module | None:
        """Get the loaded model."""
        return self.model

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, list[int]]:
        """Execute model forward pass.

        This is a simplified interface for the V2 runner.
        For full vLLM integration, this would need to match
        the GPUModelRunner interface more closely.
        """
        # For now, delegate to PyTorch for model forward
        # The Metal kernels will be used for attention within the model
        raise NotImplementedError(
            "Full execute_model not yet implemented. "
            "V2 runner requires further vLLM integration."
        )

    def profile_run(self) -> None:
        """Profile run for memory estimation."""
        logger.info("Running Metal V2 profiling...")
        # Run a dummy forward pass to warm up Metal
        if self.model is not None:
            # Create dummy input
            dummy_input = torch.randint(
                0, 1000, (1, 16), dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.mps.synchronize()
        logger.info("Metal V2 profiling complete")

    def capture_model(self) -> int:
        """Capture model for graph execution.

        Metal doesn't support CUDA graphs, so this is a no-op.
        """
        logger.debug("Metal does not support graph capture, skipping")
        return 0

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.model_config.get_vocab_size()


def create_metal_v2_runner(
    vllm_config: VllmConfig,
    device: torch.device,
) -> MetalModelRunner:
    """Factory function to create a Metal V2 model runner."""
    return MetalModelRunner(vllm_config, device)
