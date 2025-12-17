# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for vLLM Metal backend."""

import pytest
import torch

# Mark all tests in this module as requiring Metal and being slow
pytestmark = [pytest.mark.metal, pytest.mark.slow]


class TestBasicInference:
    """Basic inference tests."""

    @pytest.fixture
    def small_model_name(self):
        """Provide a small model name for testing."""
        # Using a small model for faster testing
        return "gpt2"

    def test_mps_tensor_operations(self, mps_device):
        """Test basic MPS tensor operations work."""
        # Create tensors
        a = torch.randn(32, 64, device=mps_device, dtype=torch.float16)
        b = torch.randn(64, 32, device=mps_device, dtype=torch.float16)

        # Matrix multiplication
        c = torch.matmul(a, b)
        assert c.shape == (32, 32)
        assert c.device.type == "mps"

        # Softmax
        d = torch.softmax(c, dim=-1)
        assert d.shape == c.shape
        assert torch.allclose(
            d.sum(dim=-1),
            torch.ones(32, device=mps_device, dtype=torch.float16),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_attention_computation(self, mps_device):
        """Test attention computation on MPS."""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 32

        # Create Q, K, V
        q = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=mps_device,
            dtype=torch.float16,
        )
        k = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=mps_device,
            dtype=torch.float16,
        )
        v = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=mps_device,
            dtype=torch.float16,
        )

        # Use scaled dot product attention
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert out.shape == (batch_size, num_heads, seq_len, head_dim)
        assert out.device.type == "mps"

    def test_layer_norm_on_mps(self, mps_device):
        """Test layer normalization on MPS."""
        batch_size = 4
        seq_len = 16
        hidden_size = 64

        x = torch.randn(
            batch_size, seq_len, hidden_size, device=mps_device, dtype=torch.float16
        )

        layer_norm = torch.nn.LayerNorm(
            hidden_size, device=mps_device, dtype=torch.float16
        )
        out = layer_norm(x)

        assert out.shape == x.shape
        assert out.device.type == "mps"

    def test_feed_forward_on_mps(self, mps_device):
        """Test feed-forward computation on MPS."""
        batch_size = 4
        seq_len = 16
        hidden_size = 64
        intermediate_size = 256

        x = torch.randn(
            batch_size, seq_len, hidden_size, device=mps_device, dtype=torch.float16
        )

        # Create linear layers
        up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, device=mps_device, dtype=torch.float16
        )
        down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, device=mps_device, dtype=torch.float16
        )

        # Feed forward with SiLU activation
        hidden = torch.nn.functional.silu(up_proj(x))
        out = down_proj(hidden)

        assert out.shape == x.shape
        assert out.device.type == "mps"


class TestKVCache:
    """Tests for KV cache operations."""

    def test_kv_cache_storage(self, mps_device):
        """Test storing to KV cache."""
        from vllm_metal.ops.cache import (
            allocate_unified_kv_cache,
            reshape_and_cache_flash,
        )

        num_blocks = 10
        block_size = 16
        num_kv_heads = 4
        head_size = 32

        # Allocate cache
        kv_cache = allocate_unified_kv_cache(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            dtype=torch.float16,
            device=mps_device,
        )

        assert kv_cache.shape == (num_blocks, 2, block_size, num_kv_heads, head_size)

        # Store some values
        num_tokens = 8
        key = torch.randn(
            num_tokens, num_kv_heads, head_size, device=mps_device, dtype=torch.float16
        )
        value = torch.randn(
            num_tokens, num_kv_heads, head_size, device=mps_device, dtype=torch.float16
        )
        slot_mapping = torch.arange(num_tokens, device=mps_device)

        reshape_and_cache_flash(key, value, kv_cache, slot_mapping)

        # Verify storage
        for i in range(num_tokens):
            block_idx = i // block_size
            block_offset = i % block_size
            torch.testing.assert_close(kv_cache[block_idx, 0, block_offset], key[i])
            torch.testing.assert_close(kv_cache[block_idx, 1, block_offset], value[i])


class TestMetalBackendIntegration:
    """Integration tests for Metal backend."""

    def test_platform_registration(self):
        """Test that the Metal platform registers correctly."""
        from vllm_metal import register

        platform_cls = register()
        assert platform_cls == "vllm_metal.platform.MetalPlatform"

    def test_ops_registration(self):
        """Test that Metal ops register correctly."""
        from vllm_metal import register_ops

        # Should not raise
        register_ops()

    def test_attention_backend_instantiation(self, mps_device):
        """Test attention backend can be instantiated."""
        from vllm_metal.attention import MPSAttentionImpl

        impl = MPSAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
        )

        assert impl.num_heads == 8
        assert impl.head_size == 64
        assert impl.num_kv_heads == 8


class TestTransformerBlock:
    """Test a complete transformer block on MPS."""

    @pytest.fixture
    def transformer_config(self):
        """Provide transformer configuration."""
        return {
            "hidden_size": 256,
            "num_heads": 4,
            "head_dim": 64,
            "intermediate_size": 512,
            "num_kv_heads": 4,
        }

    def test_transformer_block(self, mps_device, transformer_config):
        """Test a complete transformer block."""
        batch_size = 2
        seq_len = 16

        hidden_size = transformer_config["hidden_size"]
        num_heads = transformer_config["num_heads"]
        head_dim = transformer_config["head_dim"]
        intermediate_size = transformer_config["intermediate_size"]

        # Input
        x = torch.randn(
            batch_size, seq_len, hidden_size, device=mps_device, dtype=torch.float16
        )

        # Layer norm
        ln1 = torch.nn.LayerNorm(hidden_size, device=mps_device, dtype=torch.float16)

        # Attention projections
        q_proj = torch.nn.Linear(
            hidden_size, num_heads * head_dim, device=mps_device, dtype=torch.float16
        )
        k_proj = torch.nn.Linear(
            hidden_size, num_heads * head_dim, device=mps_device, dtype=torch.float16
        )
        v_proj = torch.nn.Linear(
            hidden_size, num_heads * head_dim, device=mps_device, dtype=torch.float16
        )
        o_proj = torch.nn.Linear(
            num_heads * head_dim, hidden_size, device=mps_device, dtype=torch.float16
        )

        # FFN
        ln2 = torch.nn.LayerNorm(hidden_size, device=mps_device, dtype=torch.float16)
        up_proj = torch.nn.Linear(
            hidden_size, intermediate_size, device=mps_device, dtype=torch.float16
        )
        down_proj = torch.nn.Linear(
            intermediate_size, hidden_size, device=mps_device, dtype=torch.float16
        )

        # Forward pass
        residual = x
        x = ln1(x)

        # Attention
        q = q_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v_proj(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, seq_len, num_heads * head_dim
        )
        attn_out = o_proj(attn_out)

        x = residual + attn_out

        # FFN
        residual = x
        x = ln2(x)
        x = down_proj(torch.nn.functional.silu(up_proj(x)))
        x = residual + x

        assert x.shape == (batch_size, seq_len, hidden_size)
        assert x.device.type == "mps"
        assert not torch.isnan(x).any()
