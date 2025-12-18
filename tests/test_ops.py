# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal operations."""

import pytest
import torch
from torch.nn import functional

from vllm_metal.ops.activation import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from vllm_metal.ops.layernorm import (
    fused_add_rms_norm,
    rms_norm,
)
from vllm_metal.ops.rotary import (
    create_cos_sin_cache,
    rotary_embedding,
)
from vllm_metal.ops.sampling import (
    greedy_sampling,
    sampling_from_probs,
    top_k_sampling,
    top_p_sampling,
)


class TestActivations:
    """Tests for activation functions."""

    @pytest.mark.metal
    def test_silu_and_mul(self, metal_device):
        """Test fused SiLU and multiplication."""
        batch_size, hidden_size = 4, 64
        x = torch.randn(
            batch_size, hidden_size * 2, device=metal_device, dtype=torch.float16
        )
        out = torch.empty(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )

        silu_and_mul(out, x)

        # Verify against reference
        gate = x[..., :hidden_size]
        up = x[..., hidden_size:]
        expected = functional.silu(gate) * up

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.metal
    def test_gelu_and_mul(self, metal_device):
        """Test fused GELU and multiplication."""
        batch_size, hidden_size = 4, 64
        x = torch.randn(
            batch_size, hidden_size * 2, device=metal_device, dtype=torch.float16
        )
        out = torch.empty(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )

        gelu_and_mul(out, x)

        # Verify against reference
        gate = x[..., :hidden_size]
        up = x[..., hidden_size:]
        expected = functional.gelu(gate) * up

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.metal
    def test_gelu_tanh_and_mul(self, metal_device):
        """Test fused GELU (tanh) and multiplication."""
        batch_size, hidden_size = 4, 64
        x = torch.randn(
            batch_size, hidden_size * 2, device=metal_device, dtype=torch.float16
        )
        out = torch.empty(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )

        gelu_tanh_and_mul(out, x)

        # Verify against reference
        gate = x[..., :hidden_size]
        up = x[..., hidden_size:]
        expected = functional.gelu(gate, approximate="tanh") * up

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


class TestLayerNorm:
    """Tests for layer normalization operations."""

    @pytest.mark.metal
    def test_rms_norm(self, metal_device):
        """Test RMS normalization."""
        batch_size, hidden_size = 4, 64
        epsilon = 1e-6

        x = torch.randn(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )
        weight = torch.ones(hidden_size, device=metal_device, dtype=torch.float16)
        out = torch.empty_like(x)

        rms_norm(out, x, weight, epsilon)

        # Verify against reference
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(variance + epsilon) * weight

        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.metal
    def test_fused_add_rms_norm(self, metal_device):
        """Test fused residual add and RMS normalization."""
        batch_size, hidden_size = 4, 64
        epsilon = 1e-6

        x = torch.randn(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )
        residual = torch.randn(
            batch_size, hidden_size, device=metal_device, dtype=torch.float16
        )
        weight = torch.ones(hidden_size, device=metal_device, dtype=torch.float16)

        # Reference computation
        x_ref = x.clone()
        x_ref.add_(residual)
        variance = x_ref.pow(2).mean(dim=-1, keepdim=True)
        expected = x_ref * torch.rsqrt(variance + epsilon) * weight

        # Fused operation
        fused_add_rms_norm(x, residual, weight, epsilon)

        torch.testing.assert_close(x, expected, rtol=1e-2, atol=1e-2)


class TestRotaryEmbedding:
    """Tests for rotary positional embeddings."""

    def test_create_cos_sin_cache(self):
        """Test creating cos/sin cache."""
        max_seq_len = 128
        head_size = 64

        cache = create_cos_sin_cache(max_seq_len, head_size)

        assert cache.shape == (max_seq_len, head_size)
        assert cache.dtype == torch.float32

    @pytest.mark.metal
    def test_rotary_embedding(self, metal_device):
        """Test rotary embedding application."""
        num_tokens = 8
        num_heads = 4
        num_kv_heads = 4
        head_size = 64

        positions = torch.arange(num_tokens, device=metal_device)
        query = torch.randn(
            num_tokens, num_heads * head_size, device=metal_device, dtype=torch.float16
        )
        key = torch.randn(
            num_tokens,
            num_kv_heads * head_size,
            device=metal_device,
            dtype=torch.float16,
        )

        cache = create_cos_sin_cache(
            128, head_size, device=metal_device, dtype=torch.float16
        )

        q_out, k_out = rotary_embedding(positions, query, key, head_size, cache)

        # Verify shapes
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape


class TestSampling:
    """Tests for sampling operations."""

    @pytest.mark.metal
    def test_greedy_sampling(self, metal_device):
        """Test greedy (argmax) sampling."""
        batch_size, vocab_size = 4, 1000
        logits = torch.randn(
            batch_size, vocab_size, device=metal_device, dtype=torch.float16
        )

        samples = greedy_sampling(logits)

        assert samples.shape == (batch_size,)
        assert samples.dtype == torch.int64

        # Verify against reference
        expected = logits.argmax(dim=-1)
        torch.testing.assert_close(samples, expected)

    @pytest.mark.metal
    def test_sampling_from_probs(self, metal_device):
        """Test sampling from probability distribution."""
        batch_size, vocab_size = 4, 100
        probs = functional.softmax(
            torch.randn(batch_size, vocab_size, device=metal_device), dim=-1
        )
        random_numbers = torch.rand(batch_size, device=metal_device)

        samples = sampling_from_probs(probs, random_numbers)

        assert samples.shape == (batch_size,)
        assert (samples >= 0).all()
        assert (samples < vocab_size).all()

    @pytest.mark.metal
    def test_sampling_from_probs_deterministic(self, metal_device):
        """Test deterministic sampling (argmax mode)."""
        batch_size, vocab_size = 4, 100
        probs = functional.softmax(
            torch.randn(batch_size, vocab_size, device=metal_device), dim=-1
        )
        random_numbers = torch.rand(batch_size, device=metal_device)

        samples = sampling_from_probs(probs, random_numbers, deterministic=True)

        expected = probs.argmax(dim=-1)
        torch.testing.assert_close(samples, expected)

    @pytest.mark.metal
    def test_top_k_sampling(self, metal_device):
        """Test top-k sampling."""
        batch_size, vocab_size = 4, 1000
        logits = torch.randn(
            batch_size, vocab_size, device=metal_device, dtype=torch.float16
        )

        samples = top_k_sampling(logits, top_k=50)

        assert samples.shape == (batch_size,)
        assert (samples >= 0).all()
        assert (samples < vocab_size).all()

    @pytest.mark.metal
    def test_top_p_sampling(self, metal_device):
        """Test top-p (nucleus) sampling."""
        batch_size, vocab_size = 4, 1000
        logits = torch.randn(
            batch_size, vocab_size, device=metal_device, dtype=torch.float16
        )

        samples = top_p_sampling(logits, top_p=0.9)

        assert samples.shape == (batch_size,)
        assert (samples >= 0).all()
        assert (samples < vocab_size).all()
