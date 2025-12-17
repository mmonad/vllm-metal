# SPDX-License-Identifier: Apache-2.0
"""
vLLM Metal Backend - Hardware plugin for Apple Silicon

This module provides Metal/MPS backend support for vLLM,
enabling high-performance LLM inference on Apple Silicon devices.
"""

__version__ = "0.1.0"


def register() -> str:
    """Register the Metal platform with vLLM.

    Returns:
        The fully qualified class name of the MetalPlatform.
    """
    return "vllm_metal.platform.MetalPlatform"


def register_ops() -> None:
    """Register Metal-specific operations with vLLM."""
    from vllm_metal.ops import register_metal_ops
    register_metal_ops()
