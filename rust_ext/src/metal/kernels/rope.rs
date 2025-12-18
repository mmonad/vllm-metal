//! RoPE kernel dispatch for rotary position embeddings.

use crate::metal::buffer::MetalBuffer;
use crate::metal::device::MetalContext;
use std::sync::Arc;

/// RoPE kernel for rotary position embeddings.
pub struct RopeKernel {
    ctx: Arc<MetalContext>,
}

impl RopeKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Apply RoPE for decode (single position per sequence).
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn decode_forward(
        &self,
        _q: &MetalBuffer,
        _k: &MetalBuffer,
        _cos: &MetalBuffer,
        _sin: &MetalBuffer,
        _positions: &MetalBuffer,
        _batch_size: i32,
        _num_heads: i32,
        _num_kv_heads: i32,
        _head_dim: i32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("RoPE kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for RopeKernel {
    fn default() -> Self {
        Self::new()
    }
}
