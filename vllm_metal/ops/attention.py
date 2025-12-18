# SPDX-License-Identifier: Apache-2.0
"""Paged attention operations for Metal backend."""

import torch
from torch.nn import functional


def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Paged attention v1 for Metal.

    This is a Python implementation that works on Metal.

    Args:
        out: Output tensor [num_seqs, num_heads, head_size]
        query: Query tensor [num_seqs, num_heads, head_size]
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size]
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size]
        num_kv_heads: Number of KV heads
        scale: Attention scale
        block_tables: Block table [num_seqs, max_blocks]
        seq_lens: Sequence lengths [num_seqs]
        block_size: Block size
        max_seq_len: Maximum sequence length
        alibi_slopes: Optional ALiBi slopes
        kv_cache_dtype: KV cache data type
        k_scale: Key scaling factor
        v_scale: Value scaling factor
    """
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        if seq_len == 0:
            continue

        # Get query for this sequence
        q = query[seq_idx]  # [num_heads, head_size]

        # Gather keys and values from cache
        num_blocks_needed = int((seq_len + block_size - 1) // block_size)
        block_table = block_tables[seq_idx]

        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []

        tokens_gathered = 0
        for block_idx in range(num_blocks_needed):
            physical_block = int(block_table[block_idx].item())
            tokens_in_block = int(min(block_size, seq_len - tokens_gathered))

            k_block = key_cache[physical_block, :tokens_in_block]
            v_block = value_cache[physical_block, :tokens_in_block]

            keys.append(k_block)
            values.append(v_block)
            tokens_gathered += tokens_in_block

        # Concatenate: [seq_len, num_kv_heads, head_size]
        k = torch.cat(keys, dim=0)
        v = torch.cat(values, dim=0)

        # Apply scaling
        k = k * k_scale
        v = v * v_scale

        # Expand KV for GQA
        if num_queries_per_kv > 1:
            k = k.repeat_interleave(num_queries_per_kv, dim=1)
            v = v.repeat_interleave(num_queries_per_kv, dim=1)

        # Compute attention
        # q: [num_heads, head_size]
        # k: [seq_len, num_heads, head_size]
        # v: [seq_len, num_heads, head_size]

        # Attention scores: [num_heads, seq_len]
        attn_weights = torch.einsum("hd,shd->hs", q, k) * scale

        # Apply ALiBi if provided
        if alibi_slopes is not None:
            positions = torch.arange(seq_len, device=query.device)
            alibi_bias = alibi_slopes.unsqueeze(1) * (positions - seq_len + 1)
            attn_weights = attn_weights + alibi_bias

        # Softmax
        attn_weights = functional.softmax(attn_weights, dim=-1)

        # Weighted sum: [num_heads, head_size]
        output = torch.einsum("hs,shd->hd", attn_weights, v)

        out[seq_idx] = output


def paged_attention_v2(
    out: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Paged attention v2 for Metal.

    This version is designed for longer sequences with partition-based
    softmax computation. On Metal, we use the same implementation as v1
    since the Python overhead is similar.

    Args:
        out: Output tensor [num_seqs, num_heads, head_size]
        exp_sums: Exponential sums tensor (unused in this implementation)
        max_logits: Max logits tensor (unused in this implementation)
        tmp_out: Temporary output tensor (unused in this implementation)
        query: Query tensor [num_seqs, num_heads, head_size]
        key_cache: Key cache
        value_cache: Value cache
        num_kv_heads: Number of KV heads
        scale: Attention scale
        block_tables: Block table
        seq_lens: Sequence lengths
        block_size: Block size
        max_seq_len: Maximum sequence length
        alibi_slopes: Optional ALiBi slopes
        kv_cache_dtype: KV cache data type
        k_scale: Key scaling factor
        v_scale: Value scaling factor
    """
    # For Metal, we use the same implementation as v1
    # The partitioned approach of v2 is mainly beneficial for CUDA
    paged_attention_v1(
        out=out,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        num_kv_heads=num_kv_heads,
        scale=scale,
        block_tables=block_tables,
        seq_lens=seq_lens,
        block_size=block_size,
        max_seq_len=max_seq_len,
        alibi_slopes=alibi_slopes,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale,
        v_scale=v_scale,
    )
