# SPDX-License-Identifier: Apache-2.0
"""Tests for utility functions."""

import platform

import pytest
import torch

from vllm_metal.utils import (
    check_metal_availability,
    format_memory_size,
    get_apple_chip_name,
    get_metal_device_info,
    get_metal_memory_info,
    get_optimal_dtype,
    is_apple_silicon,
    metal_empty_cache,
    metal_synchronize,
)


class TestPlatformDetection:
    """Tests for platform detection utilities."""

    def test_is_apple_silicon(self):
        """Test Apple Silicon detection."""
        result = is_apple_silicon()
        # Should return True on Apple Silicon, False elsewhere
        assert isinstance(result, bool)

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            assert result is True

    def test_get_apple_chip_name(self):
        """Test getting Apple chip name."""
        name = get_apple_chip_name()
        assert isinstance(name, str)

        if is_apple_silicon():
            # Should contain "M" for Apple Silicon chips
            assert "M" in name or "Apple" in name

    def test_get_metal_device_info(self):
        """Test getting Metal device information."""
        info = get_metal_device_info()

        assert isinstance(info, dict)
        assert "name" in info
        assert "is_apple_silicon" in info
        assert "metal_available" in info


class TestMetalAvailability:
    """Tests for Metal availability checking."""

    def test_check_metal_availability(self):
        """Test Metal availability check."""
        available, error = check_metal_availability()

        assert isinstance(available, bool)
        assert error is None or isinstance(error, str)

        if platform.system() != "Darwin":
            assert available is False
            assert error is not None

    @pytest.mark.metal
    def test_metal_synchronize(self, metal_device):
        """Test Metal synchronization."""
        # Should not raise
        metal_synchronize()

    @pytest.mark.metal
    def test_metal_empty_cache(self, metal_device):
        """Test Metal cache clearing."""
        # Create some tensors
        _ = torch.randn(100, 100, device=metal_device)

        # Should not raise
        metal_empty_cache()

    @pytest.mark.metal
    def test_get_metal_memory_info(self, metal_device):
        """Test getting Metal memory information."""
        allocated, total = get_metal_memory_info()

        assert isinstance(allocated, int)
        assert isinstance(total, int)
        assert allocated >= 0
        assert total >= 0


class TestDtypeUtils:
    """Tests for dtype utilities."""

    def test_get_optimal_dtype(self):
        """Test getting optimal dtype."""
        dtype = get_optimal_dtype()
        assert dtype in (torch.float16, torch.bfloat16)


class TestFormatting:
    """Tests for formatting utilities."""

    def test_format_memory_size_bytes(self):
        """Test formatting bytes."""
        assert format_memory_size(512) == "512.00 B"

    def test_format_memory_size_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_memory_size(2048)
        assert "KB" in result

    def test_format_memory_size_megabytes(self):
        """Test formatting megabytes."""
        result = format_memory_size(2 * 1024 * 1024)
        assert "MB" in result

    def test_format_memory_size_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_memory_size(8 * 1024 * 1024 * 1024)
        assert "GB" in result
