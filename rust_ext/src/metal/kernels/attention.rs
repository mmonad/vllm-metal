//! Attention kernel dispatch for vLLM Metal.
//!
//! Provides high-level API for running SDPA and paged attention kernels.

use crate::metal::buffer::{MetalBuffer, DType};
use crate::metal::device::MetalContext;
use pyo3::prelude::*;
use std::sync::Arc;

/// Attention parameters for kernel dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AttentionParams {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub seq_len: i32,
    pub num_queries: i32,
    pub scale: f32,
    pub gqa_ratio: i32,
}

/// Paged attention parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PagedAttentionParams {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub block_size: i32,
    pub num_blocks: i32,
    pub scale: f32,
    pub gqa_ratio: i32,
}

/// SDPA kernel dispatcher.
pub struct SdpaKernel {
    ctx: Arc<MetalContext>,
}

impl SdpaKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Get kernel name for given dtype and head_dim.
    fn kernel_name(dtype: DType, head_dim: i32) -> String {
        let dtype_str = match dtype {
            DType::Float16 | DType::BFloat16 => "f16",
            DType::Float32 => "f32",
            _ => panic!("Unsupported dtype for attention"),
        };
        format!("sdpa_vector_{}_{}", dtype_str, head_dim)
    }

    /// Run SDPA kernel for decode.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn forward(
        &self,
        _queries: &MetalBuffer,
        _keys: &MetalBuffer,
        _values: &MetalBuffer,
        _output: &MetalBuffer,
        _params: AttentionParams,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("SDPA kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for SdpaKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Paged attention kernel dispatcher.
pub struct PagedAttentionKernel {
    ctx: Arc<MetalContext>,
}

impl PagedAttentionKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Get kernel name for given dtype, head_dim, and block_size.
    fn kernel_name(dtype: DType, head_dim: i32, block_size: i32) -> String {
        let dtype_str = match dtype {
            DType::Float16 | DType::BFloat16 => "f16",
            DType::Float32 => "f32",
            _ => panic!("Unsupported dtype for attention"),
        };
        format!("paged_attention_{}_{}_{}", dtype_str, head_dim, block_size)
    }

    /// Run paged attention kernel for decode.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn forward(
        &self,
        _queries: &MetalBuffer,
        _key_cache: &MetalBuffer,
        _value_cache: &MetalBuffer,
        _block_tables: &MetalBuffer,
        _seq_lens: &MetalBuffer,
        _output: &MetalBuffer,
        _params: PagedAttentionParams,
        _max_blocks_per_seq: i32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("Paged attention kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for PagedAttentionKernel {
    fn default() -> Self {
        Self::new()
    }
}

/// Python-exposed attention function.
#[pyfunction]
pub fn metal_sdpa(
    _py: Python<'_>,
    _queries_ptr: usize,
    _keys_ptr: usize,
    _values_ptr: usize,
    _output_ptr: usize,
    _num_queries: i32,
    _num_heads: i32,
    _num_kv_heads: i32,
    _head_dim: i32,
    _seq_len: i32,
    _scale: f32,
) -> PyResult<()> {
    // Placeholder - actual implementation would:
    // 1. Wrap PyTorch tensor pointers in MetalBuffers (zero-copy)
    // 2. Call the kernel
    // 3. Return
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_params_size() {
        assert_eq!(std::mem::size_of::<AttentionParams>(), 28);
        assert_eq!(std::mem::size_of::<PagedAttentionParams>(), 28);
    }
}
