//! RMS normalization kernel dispatch.

use crate::metal::buffer::MetalBuffer;
use crate::metal::device::MetalContext;
use std::sync::Arc;

/// RMS normalization kernel.
pub struct RmsNormKernel {
    ctx: Arc<MetalContext>,
}

impl RmsNormKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Apply RMS normalization.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn forward(
        &self,
        _input: &MetalBuffer,
        _weight: &MetalBuffer,
        _output: &MetalBuffer,
        _batch_size: i32,
        _hidden_size: i32,
        _eps: f32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("RMS norm kernel not yet implemented - requires shader compilation".to_string())
    }

    /// Apply fused add + RMS normalization.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn fused_add_forward(
        &self,
        _input: &MetalBuffer,
        _residual: &MetalBuffer,
        _weight: &MetalBuffer,
        _output: &MetalBuffer,
        _batch_size: i32,
        _hidden_size: i32,
        _eps: f32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("Fused add RMS norm kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for RmsNormKernel {
    fn default() -> Self {
        Self::new()
    }
}
