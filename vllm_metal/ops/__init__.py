# SPDX-License-Identifier: Apache-2.0
"""Metal operations for vLLM."""

from vllm_metal.ops.activation import (
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from vllm_metal.ops.attention import (
    paged_attention_v1,
    paged_attention_v2,
)
from vllm_metal.ops.cache import (
    copy_blocks,
    reshape_and_cache,
    swap_blocks,
)
from vllm_metal.ops.layernorm import (
    fused_add_rms_norm,
    rms_norm,
)
from vllm_metal.ops.rotary import (
    rotary_embedding,
)
from vllm_metal.ops.sampling import (
    sampling_from_probs,
)

_registered = False


def register_metal_ops() -> None:
    """Register Metal-specific operations with vLLM.

    This function patches vLLM's ops module to use Metal-optimized
    implementations where available.
    """
    global _registered
    if _registered:
        return

    # The operations are already implemented as Python functions
    # that use PyTorch Metal-compatible operations.
    # Registration is mainly for tracking purposes.
    _registered = True


__all__ = [
    "paged_attention_v1",
    "paged_attention_v2",
    "copy_blocks",
    "reshape_and_cache",
    "swap_blocks",
    "gelu_and_mul",
    "gelu_tanh_and_mul",
    "silu_and_mul",
    "rms_norm",
    "fused_add_rms_norm",
    "rotary_embedding",
    "sampling_from_probs",
    "register_metal_ops",
]
