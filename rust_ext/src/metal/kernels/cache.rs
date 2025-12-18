//! KV cache operations kernel dispatch.

use crate::metal::buffer::MetalBuffer;
use crate::metal::device::MetalContext;
use std::sync::Arc;

/// KV cache kernel for paged attention.
pub struct CacheKernel {
    ctx: Arc<MetalContext>,
}

impl CacheKernel {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
        }
    }

    /// Store new K/V into cache.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn reshape_and_cache(
        &self,
        _key: &MetalBuffer,
        _value: &MetalBuffer,
        _key_cache: &MetalBuffer,
        _value_cache: &MetalBuffer,
        _slot_mapping: &MetalBuffer,
        _num_tokens: i32,
        _num_kv_heads: i32,
        _head_dim: i32,
        _block_size: i32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("Reshape and cache kernel not yet implemented - requires shader compilation".to_string())
    }

    /// Copy blocks between sequences.
    /// TODO: Implement actual kernel dispatch when shaders are compiled.
    pub fn copy_blocks(
        &self,
        _key_cache: &MetalBuffer,
        _value_cache: &MetalBuffer,
        _block_mapping: &MetalBuffer,
        _num_pairs: i32,
        _block_size: i32,
        _num_kv_heads: i32,
        _head_dim: i32,
    ) -> Result<(), String> {
        // Stub - actual implementation requires shader compilation
        Err("Copy blocks kernel not yet implemented - requires shader compilation".to_string())
    }
}

impl Default for CacheKernel {
    fn default() -> Self {
        Self::new()
    }
}
