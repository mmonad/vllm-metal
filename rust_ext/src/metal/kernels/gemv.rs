//! GEMV kernel dispatch for matrix-vector multiply.

use crate::metal::buffer::{MetalBuffer, DType};
use crate::metal::device::MetalContext;
use std::sync::Arc;

/// GEMV kernel for matrix-vector multiply.
pub struct GemvKernel {
    ctx: Arc<MetalContext>,
}

impl GemvKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Run batched GEMV: Y = AX
    /// A: [M, K], X: [batch, K], Y: [batch, M]
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn batched_forward(
        &self,
        _a: &MetalBuffer,
        _x: &MetalBuffer,
        _y: &MetalBuffer,
        _m: i32,
        _k: i32,
        _batch: i32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("GEMV kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for GemvKernel {
    fn default() -> Self {
        Self::new()
    }
}
