# SPDX-License-Identifier: Apache-2.0
"""Metal worker implementation for vLLM."""

import torch

from vllm_metal._compat import (
    ExecuteModelRequest,
    WorkerBase,
    init_logger,
)
from vllm_metal.utils import (
    check_metal_availability,
    get_metal_device_info,
    metal_empty_cache,
)
from vllm_metal.worker.metal_model_runner import MetalModelRunner

logger = init_logger(__name__)


class MetalWorker(WorkerBase):
    """Worker for executing models on Apple Metal.

    This worker manages model execution on the Metal device,
    including memory management and KV cache handling.
    """

    def __init__(
        self,
        model_config,
        parallel_config,
        scheduler_config,
        device_config,
        cache_config,
        load_config,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = True,
        model_runner_cls: type[MetalModelRunner] | None = None,
        **kwargs,
    ):
        """Initialize the Metal worker.

        Args:
            model_config: Model configuration
            parallel_config: Parallel execution configuration
            scheduler_config: Scheduler configuration
            device_config: Device configuration
            cache_config: Cache configuration
            load_config: Model loading configuration
            local_rank: Local rank (always 0 for Metal)
            rank: Global rank (always 0 for Metal)
            distributed_init_method: Distributed init (not supported)
            is_driver_worker: Whether this is the driver worker
            model_runner_cls: Optional model runner class override
        """
        # Validate Metal availability
        available, error = check_metal_availability()
        if not available:
            raise RuntimeError(f"Metal not available: {error}")

        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Set device (PyTorch uses "mps" for Metal)
        self.device = torch.device("mps")

        # Initialize model runner
        if model_runner_cls is None:
            model_runner_cls = MetalModelRunner

        self.model_runner = model_runner_cls(
            model_config=model_config,
            parallel_config=parallel_config,
            scheduler_config=scheduler_config,
            device_config=device_config,
            cache_config=cache_config,
            load_config=load_config,
            is_driver_worker=is_driver_worker,
        )

        # KV cache
        self.gpu_cache: list[torch.Tensor] | None = None
        self.cpu_cache: list[torch.Tensor] | None = None

        logger.info(
            f"Initialized MetalWorker: device={self.device}, driver={is_driver_worker}"
        )

    def init_device(self) -> None:
        """Initialize the Metal device."""
        # Set the default device
        torch.set_default_device(self.device)

        # Log device info
        info = get_metal_device_info()
        logger.info(
            f"Metal device: {info['name']}, Metal available: {info['metal_available']}"
        )

    def load_model(self) -> None:
        """Load the model onto the Metal device."""
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        """Determine the number of available KV cache blocks.

        Returns:
            Tuple of (num_gpu_blocks, num_cpu_blocks)
        """
        # Get available memory
        info = get_metal_device_info()
        total_memory = info.get("total_memory", 0)

        # Reserve memory for model and activations
        # This is a rough estimate
        model_memory = self.model_runner.get_model_memory_usage()
        available_memory = total_memory * self.cache_config.gpu_memory_utilization
        cache_memory = available_memory - model_memory

        if cache_memory <= 0:
            logger.warning(
                "Insufficient memory for KV cache. "
                "Consider reducing model size or batch size."
            )
            return 0, 0

        # Calculate number of blocks
        block_size = self.cache_config.block_size
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        # Memory per block (K + V for all layers)
        bytes_per_block = (
            2  # K and V
            * block_size
            * num_layers
            * num_kv_heads
            * head_size
            * 2  # float16
        )

        num_gpu_blocks = int(cache_memory // bytes_per_block)

        # CPU blocks (for swapping)
        # Use a fraction of system memory for CPU cache
        cpu_memory = total_memory * 0.1  # 10% of system memory
        num_cpu_blocks = int(cpu_memory // bytes_per_block)

        logger.info(
            f"KV cache: {num_gpu_blocks} GPU blocks, {num_cpu_blocks} CPU blocks"
        )

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache.

        Args:
            num_gpu_blocks: Number of GPU blocks
            num_cpu_blocks: Number of CPU blocks
        """
        self.gpu_cache = self.model_runner.initialize_kv_cache(
            num_gpu_blocks,
            self.device,
        )

        if num_cpu_blocks > 0:
            self.cpu_cache = self.model_runner.initialize_kv_cache(
                num_cpu_blocks,
                torch.device("cpu"),
            )

        logger.info(
            f"Initialized KV cache: "
            f"{len(self.gpu_cache)} layers on Metal, "
            f"{len(self.cpu_cache) if self.cpu_cache else 0} layers on CPU"
        )

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest | None = None,
    ):
        """Execute model forward pass.

        Args:
            execute_model_req: Model execution request

        Returns:
            Model outputs or None
        """
        if execute_model_req is None:
            return None

        assert self.gpu_cache is not None, "GPU cache must be initialized"
        return self.model_runner.execute_model(
            execute_model_req,
            self.gpu_cache,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a cache block in bytes."""
        return self.model_runner.get_cache_block_size_bytes()

    def do_metadata_broadcast(self) -> bool:
        """Check if metadata broadcast is needed.

        Metal doesn't support distributed execution.
        """
        return False

    def kv_cache_dtype(self) -> torch.dtype:
        """Get the KV cache data type."""
        return self.model_runner.kv_cache_dtype

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.model_runner.vocab_size

    def _warm_up_model(self) -> None:
        """Warm up the model with a dummy run."""
        self.model_runner.warm_up()

    def swap_in(self, blocks_to_swap_in: dict[int, int]) -> None:
        """Swap blocks from CPU to GPU.

        Args:
            blocks_to_swap_in: Mapping from CPU block to GPU block
        """
        if not blocks_to_swap_in or self.cpu_cache is None or self.gpu_cache is None:
            return

        for cpu_block, gpu_block in blocks_to_swap_in.items():
            for layer_idx in range(len(self.gpu_cache)):
                self.gpu_cache[layer_idx][gpu_block].copy_(
                    self.cpu_cache[layer_idx][cpu_block]
                )

    def swap_out(self, blocks_to_swap_out: dict[int, int]) -> None:
        """Swap blocks from GPU to CPU.

        Args:
            blocks_to_swap_out: Mapping from GPU block to CPU block
        """
        if not blocks_to_swap_out or self.cpu_cache is None or self.gpu_cache is None:
            return

        for gpu_block, cpu_block in blocks_to_swap_out.items():
            for layer_idx in range(len(self.gpu_cache)):
                self.cpu_cache[layer_idx][cpu_block].copy_(
                    self.gpu_cache[layer_idx][gpu_block]
                )

    def copy_blocks(self, blocks_to_copy: list[tuple[int, int]]) -> None:
        """Copy blocks within GPU cache.

        Args:
            blocks_to_copy: List of (src_block, dst_block) tuples
        """
        if not blocks_to_copy or self.gpu_cache is None:
            return

        from vllm_metal.ops import copy_blocks as copy_blocks_op

        src_to_dsts = torch.tensor(
            blocks_to_copy,
            dtype=torch.int64,
            device=self.device,
        )

        copy_blocks_op(self.gpu_cache, src_to_dsts)

    def __del__(self):
        """Cleanup when worker is destroyed."""
        metal_empty_cache()
