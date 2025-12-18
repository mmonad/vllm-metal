# SPDX-License-Identifier: Apache-2.0
"""Model runner for Metal backend."""

import torch
import torch.nn as nn

from vllm_metal._compat import (
    ExecuteModelRequest,
    SequenceGroupMetadata,
    init_logger,
)
from vllm_metal.attention import MetalAttentionBackend, MetalAttentionMetadata
from vllm_metal.ops.cache import allocate_unified_kv_cache
from vllm_metal.utils import get_optimal_dtype, metal_synchronize

logger = init_logger(__name__)


class MetalModelRunner:
    """Model runner for executing models on Apple Metal.

    This class handles model execution, KV cache management,
    and input preparation for the Metal backend.
    """

    def __init__(
        self,
        model_config,
        parallel_config,
        scheduler_config,
        device_config,
        cache_config,
        load_config,
        is_driver_worker: bool = True,
    ):
        """Initialize the model runner.

        Args:
            model_config: Model configuration
            parallel_config: Parallel configuration
            scheduler_config: Scheduler configuration
            device_config: Device configuration
            cache_config: Cache configuration
            load_config: Load configuration
            is_driver_worker: Whether this is the driver worker
        """
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        self.device = torch.device("mps")  # PyTorch uses "mps" for Metal
        self.model: nn.Module | None = None

        # Determine dtype
        if model_config.dtype == "auto":
            self.dtype = get_optimal_dtype()
        else:
            self.dtype = getattr(torch, model_config.dtype)

        # KV cache dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = getattr(torch, cache_config.cache_dtype)

        # Block size for paged attention
        self.block_size = cache_config.block_size

        # Attention backend
        self.attn_backend = MetalAttentionBackend

    def load_model(self) -> None:
        """Load the model onto Metal device."""
        from vllm.model_executor.model_loader import get_model

        logger.info(f"Loading model: {self.model_config.model}")

        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
        )

        # Move to Metal and convert dtype
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        logger.info(f"Model loaded: device={self.device}, dtype={self.dtype}")

    def get_model_memory_usage(self) -> int:
        """Get model memory usage in bytes."""
        if self.model is None:
            return 0

        total_bytes = 0
        for param in self.model.parameters():
            total_bytes += param.numel() * param.element_size()

        return total_bytes

    def initialize_kv_cache(
        self,
        num_blocks: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Initialize KV cache for all layers.

        Args:
            num_blocks: Number of blocks to allocate
            device: Device to allocate on

        Returns:
            List of KV cache tensors, one per layer
        """
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()

        kv_caches = []
        for _ in range(num_layers):
            kv_cache = allocate_unified_kv_cache(
                num_blocks=num_blocks,
                block_size=self.block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=self.kv_cache_dtype,
                device=device,
            )
            kv_caches.append(kv_cache)

        logger.info(
            f"Initialized KV cache: {num_layers} layers, {num_blocks} blocks each"
        )

        return kv_caches

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a cache block in bytes."""
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        head_size = self.model_config.get_head_size()
        num_layers = self.model_config.get_num_layers(self.parallel_config)

        # 2 for K and V
        elements_per_block = 2 * self.block_size * num_kv_heads * head_size

        bytes_per_element = 2 if self.kv_cache_dtype == torch.float16 else 4

        return int(elements_per_block * bytes_per_element * num_layers)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return int(self.model_config.get_vocab_size())

    def execute_model(
        self,
        execute_model_req: ExecuteModelRequest,
        kv_caches: list[torch.Tensor],
    ):
        """Execute model forward pass.

        Args:
            execute_model_req: Model execution request
            kv_caches: KV cache tensors

        Returns:
            Model outputs
        """
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        # Prepare inputs
        input_tokens, input_positions, attn_metadata = self._prepare_inputs(
            seq_group_metadata_list
        )

        # Execute model
        assert self.model is not None, "Model must be loaded before execution"
        with torch.inference_mode():
            hidden_states = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
            )

        # Synchronize before returning
        metal_synchronize()

        return hidden_states

    def _prepare_inputs(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
    ) -> tuple[torch.Tensor, torch.Tensor, MetalAttentionMetadata]:
        """Prepare inputs for model execution.

        Args:
            seq_group_metadata_list: List of sequence group metadata

        Returns:
            Tuple of (input_tokens, input_positions, attn_metadata)
        """
        input_tokens: list[int] = []
        input_positions: list[int] = []
        seq_lens: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        block_tables: list[list[int]] = []

        num_prefill_tokens = 0
        num_decode_tokens = 0

        for seq_group_metadata in seq_group_metadata_list:
            is_prompt = seq_group_metadata.is_prompt

            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                if is_prompt:
                    # Prefill: process all tokens
                    tokens = seq_data.get_token_ids()
                    seq_len = len(tokens)
                    positions = list(range(seq_len))

                    input_tokens.extend(tokens)
                    input_positions.extend(positions)
                    seq_lens.append(seq_len)
                    num_prefill_tokens += seq_len

                    # Compute slot mapping for prefill
                    if seq_id in seq_group_metadata.block_tables:
                        block_table = seq_group_metadata.block_tables[seq_id]
                        for pos in range(seq_len):
                            block_idx = pos // self.block_size
                            block_offset = pos % self.block_size
                            if block_idx < len(block_table):
                                slot = (
                                    block_table[block_idx] * self.block_size
                                    + block_offset
                                )
                                slot_mapping.append(slot)
                else:
                    # Decode: process only the last token
                    tokens = seq_data.get_token_ids()
                    context_len = len(tokens) - 1
                    last_token = tokens[-1]
                    position = context_len

                    input_tokens.append(last_token)
                    input_positions.append(position)
                    context_lens.append(context_len)
                    num_decode_tokens += 1

                    # Block table and slot mapping for decode
                    if seq_id in seq_group_metadata.block_tables:
                        block_table = seq_group_metadata.block_tables[seq_id]
                        block_tables.append(list(block_table))

                        block_idx = context_len // self.block_size
                        block_offset = context_len % self.block_size
                        if block_idx < len(block_table):
                            slot = (
                                block_table[block_idx] * self.block_size + block_offset
                            )
                            slot_mapping.append(slot)

        # Convert to tensors
        input_tokens_tensor = torch.tensor(
            input_tokens, dtype=torch.long, device=self.device
        )
        input_positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )

        # Build attention metadata
        attn_metadata = self._build_attn_metadata(
            seq_lens=seq_lens,
            context_lens=context_lens,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
        )

        return input_tokens_tensor, input_positions_tensor, attn_metadata

    def _build_attn_metadata(
        self,
        seq_lens: list[int],
        context_lens: list[int],
        block_tables: list[list[int]],
        slot_mapping: list[int],
        num_prefill_tokens: int,
        num_decode_tokens: int,
    ) -> MetalAttentionMetadata:
        """Build attention metadata.

        Args:
            seq_lens: Sequence lengths for prefill
            context_lens: Context lengths for decode
            block_tables: Block tables for decode
            slot_mapping: Slot mapping for KV cache
            num_prefill_tokens: Number of prefill tokens
            num_decode_tokens: Number of decode tokens

        Returns:
            MetalAttentionMetadata
        """
        slot_mapping_tensor = (
            torch.tensor(slot_mapping, dtype=torch.long, device=self.device)
            if slot_mapping
            else None
        )

        seq_lens_tensor = (
            torch.tensor(seq_lens, dtype=torch.int, device=self.device)
            if seq_lens
            else None
        )

        context_lens_tensor = (
            torch.tensor(context_lens, dtype=torch.int, device=self.device)
            if context_lens
            else None
        )

        block_tables_tensor = None
        if block_tables:
            max_blocks = max(len(bt) for bt in block_tables)
            padded = [bt + [0] * (max_blocks - len(bt)) for bt in block_tables]
            block_tables_tensor = torch.tensor(
                padded, dtype=torch.int, device=self.device
            )

        return MetalAttentionMetadata(
            seq_lens=seq_lens if seq_lens else None,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=max(seq_lens) if seq_lens else 0,
            max_decode_seq_len=max(context_lens) + 1 if context_lens else 0,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=len(seq_lens),
            block_tables=block_tables_tensor,
            slot_mapping=slot_mapping_tensor,
            context_lens_tensor=context_lens_tensor,
            is_prompt=num_prefill_tokens > 0,
        )

    def warm_up(self) -> None:
        """Warm up the model with a dummy forward pass."""
        logger.info("Warming up model...")

        # Create dummy inputs
        dummy_tokens = torch.zeros(1, dtype=torch.long, device=self.device)
        dummy_positions = torch.zeros(1, dtype=torch.long, device=self.device)

        # Run a dummy forward pass
        if self.model is None:
            logger.warning("Model not loaded, skipping warmup")
            return
        with torch.inference_mode():
            try:
                # This may fail for some models, but that's okay for warmup
                _ = self.model(
                    input_ids=dummy_tokens,
                    positions=dummy_positions,
                    kv_caches=None,
                    attn_metadata=None,
                )
            except Exception as e:
                logger.debug(f"Warmup forward pass failed (expected): {e}")

        metal_synchronize()
        logger.info("Model warm-up complete")
