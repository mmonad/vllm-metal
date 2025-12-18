# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal attention backend."""

import pytest
import torch
from torch.nn import functional

from vllm_metal.attention.backend import (
    MetalAttentionBackend,
    MetalAttentionMetadata,
)
from vllm_metal.attention.metal_attention import MetalAttentionImpl


class TestMetalAttentionBackend:
    """Tests for MetalAttentionBackend."""

    def test_get_name(self):
        """Test backend name."""
        assert MetalAttentionBackend.get_name() == "METAL"

    def test_get_impl_cls(self):
        """Test getting implementation class."""
        impl_cls = MetalAttentionBackend.get_impl_cls()
        assert impl_cls == MetalAttentionImpl

    def test_get_metadata_cls(self):
        """Test getting metadata class."""
        meta_cls = MetalAttentionBackend.get_metadata_cls()
        assert meta_cls == MetalAttentionMetadata

    def test_get_kv_cache_shape(self):
        """Test KV cache shape calculation."""
        num_blocks = 100
        block_size = 16
        num_kv_heads = 8
        head_size = 64

        shape = MetalAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

        assert shape == (num_blocks, 2, block_size, num_kv_heads, head_size)

    def test_get_supported_head_sizes(self):
        """Test supported head sizes."""
        sizes = MetalAttentionBackend.get_supported_head_sizes()

        assert isinstance(sizes, list)
        assert 64 in sizes
        assert 128 in sizes


class TestMetalAttentionMetadata:
    """Tests for MetalAttentionMetadata."""

    def test_empty_metadata(self):
        """Test creating empty metadata."""
        meta = MetalAttentionMetadata()

        assert meta.num_prefill_tokens == 0
        assert meta.num_decode_tokens == 0
        assert meta.num_prefills == 0

    def test_prefill_metadata_property(self):
        """Test prefill metadata extraction."""
        meta = MetalAttentionMetadata(
            seq_lens=[10, 15],
            num_prefill_tokens=25,
            num_decode_tokens=0,
            num_prefills=2,
            is_prompt=True,
        )

        prefill_meta = meta.prefill_metadata
        assert prefill_meta is not None
        assert prefill_meta.num_prefill_tokens == 25
        assert prefill_meta.is_prompt is True

    def test_decode_metadata_property(self):
        """Test decode metadata extraction."""
        meta = MetalAttentionMetadata(
            num_prefill_tokens=0,
            num_decode_tokens=4,
            num_prefills=0,
            is_prompt=False,
        )

        decode_meta = meta.decode_metadata
        assert decode_meta is not None
        assert decode_meta.num_decode_tokens == 4
        assert decode_meta.is_prompt is False


class TestMetalAttentionImpl:
    """Tests for Metal attention implementation."""

    @pytest.fixture
    def attention_impl(self):
        """Create attention implementation for testing."""
        return MetalAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=1.0 / (64**0.5),
            num_kv_heads=8,
        )

    @pytest.mark.metal
    def test_init(self, attention_impl):
        """Test attention initialization."""
        assert attention_impl.num_heads == 8
        assert attention_impl.head_size == 64
        assert attention_impl.num_kv_heads == 8

    @pytest.mark.metal
    def test_forward_prefill(self, attention_impl, metal_device):
        """Test forward pass for prefill."""
        seq_len = 16
        num_heads = 8
        head_size = 64

        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        key = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        value = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )

        metadata = MetalAttentionMetadata(
            seq_lens=[seq_len],
            num_prefill_tokens=seq_len,
            num_decode_tokens=0,
            num_prefills=1,
            is_prompt=True,
        )

        output = attention_impl.forward(
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        assert output.shape == (seq_len, num_heads * head_size)

    @pytest.mark.metal
    def test_gqa_support(self, metal_device):
        """Test grouped query attention support."""
        num_heads = 32
        num_kv_heads = 8  # GQA with 4 query groups
        head_size = 64

        impl = MetalAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size**0.5),
            num_kv_heads=num_kv_heads,
        )

        seq_len = 16
        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        key = torch.randn(
            seq_len, num_kv_heads * head_size, device=metal_device, dtype=torch.float16
        )
        value = torch.randn(
            seq_len, num_kv_heads * head_size, device=metal_device, dtype=torch.float16
        )

        metadata = MetalAttentionMetadata(
            seq_lens=[seq_len],
            num_prefill_tokens=seq_len,
            num_decode_tokens=0,
            num_prefills=1,
            is_prompt=True,
        )

        output = impl.forward(
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        assert output.shape == (seq_len, num_heads * head_size)


class TestAttentionCorrectness:
    """Tests for attention correctness."""

    @pytest.mark.metal
    def test_attention_matches_reference(self, metal_device):
        """Test that attention output matches reference implementation."""
        num_heads = 4
        head_size = 32
        seq_len = 8

        impl = MetalAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=1.0 / (head_size**0.5),
        )

        # Create inputs
        query = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )
        key = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )
        value = torch.randn(
            seq_len, num_heads * head_size, device=metal_device, dtype=torch.float32
        )

        metadata = MetalAttentionMetadata(
            seq_lens=[seq_len],
            num_prefill_tokens=seq_len,
            num_decode_tokens=0,
            num_prefills=1,
            is_prompt=True,
        )

        # Our implementation
        output = impl.forward(
            query=query,
            key=key,
            value=value,
            kv_cache=None,
            attn_metadata=metadata,
        )

        # Reference: manual scaled dot product attention
        q = query.view(seq_len, num_heads, head_size).transpose(0, 1)
        k = key.view(seq_len, num_heads, head_size).transpose(0, 1)
        v = value.view(seq_len, num_heads, head_size).transpose(0, 1)

        scale = 1.0 / (head_size**0.5)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=metal_device),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = functional.softmax(attn_weights, dim=-1)
        expected = torch.matmul(attn_weights, v)
        expected = expected.transpose(0, 1).reshape(seq_len, num_heads * head_size)

        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
