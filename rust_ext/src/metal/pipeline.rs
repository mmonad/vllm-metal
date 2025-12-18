//! Compute pipeline management with caching.
//!
//! Provides efficient pipeline state caching to avoid expensive
//! recompilation of Metal compute pipelines.

use super::device::{MetalContext, MTLDeviceTrait, MTLLibraryTrait};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLFunction, MTLSize};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Compute pipeline wrapper with cached state.
pub struct ComputePipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    max_threads_per_group: usize,
    threadgroup_memory_length: usize,
}

// SAFETY: Metal pipeline states are thread-safe when accessed through command buffers
unsafe impl Send for ComputePipeline {}
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Create a pipeline from a function.
    pub fn new(
        ctx: &MetalContext,
        function: &ProtocolObject<dyn MTLFunction>,
    ) -> Result<Self, String> {
        let state = unsafe {
            ctx.device().newComputePipelineStateWithFunction_error(function)
        }
        .map_err(|e| format!("Failed to create pipeline: {:?}", e))?;

        let max_threads = state.maxTotalThreadsPerThreadgroup() as usize;
        let tg_mem = state.staticThreadgroupMemoryLength() as usize;

        Ok(Self {
            state,
            max_threads_per_group: max_threads,
            threadgroup_memory_length: tg_mem,
        })
    }

    /// Get the underlying pipeline state.
    pub fn state(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }

    /// Get max threads per threadgroup.
    pub fn max_threads_per_group(&self) -> usize {
        self.max_threads_per_group
    }

    /// Get static threadgroup memory length.
    pub fn threadgroup_memory_length(&self) -> usize {
        self.threadgroup_memory_length
    }

    /// Calculate optimal threadgroup size for 1D dispatch.
    pub fn optimal_1d_threadgroup(&self, total_threads: usize) -> MTLSize {
        let width = self.max_threads_per_group.min(total_threads);
        MTLSize {
            width,
            height: 1,
            depth: 1,
        }
    }

    /// Calculate grid size for 1D dispatch.
    pub fn grid_1d(&self, total_threads: usize) -> MTLSize {
        MTLSize {
            width: total_threads,
            height: 1,
            depth: 1,
        }
    }

    /// Calculate optimal threadgroup size for 2D dispatch.
    pub fn optimal_2d_threadgroup(&self, width: usize, height: usize) -> MTLSize {
        // Common pattern: 16x16 or 32x8 threadgroups
        let max = self.max_threads_per_group;
        let w = 32.min(width).min(max);
        let h = (max / w).min(height);
        MTLSize {
            width: w,
            height: h,
            depth: 1,
        }
    }

    /// Calculate grid size for 2D dispatch.
    pub fn grid_2d(&self, width: usize, height: usize) -> MTLSize {
        MTLSize {
            width,
            height,
            depth: 1,
        }
    }
}

/// Pipeline cache for efficient kernel reuse.
pub struct PipelineCache {
    ctx: Arc<MetalContext>,
    pipelines: RwLock<HashMap<String, Arc<ComputePipeline>>>,
}

impl PipelineCache {
    pub fn new() -> Self {
        Self {
            ctx: MetalContext::get(),
            pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a pipeline for the given function name.
    pub fn get(&self, function_name: &str) -> Result<Arc<ComputePipeline>, String> {
        // Fast path: check if pipeline exists
        {
            let cache = self.pipelines.read();
            if let Some(pipeline) = cache.get(function_name) {
                return Ok(pipeline.clone());
            }
        }

        // Slow path: create pipeline
        let library = self.ctx.library()
            .ok_or("No shader library loaded")?;

        let ns_name = unsafe { NSString::from_str(function_name) };
        let function = library.newFunctionWithName(&ns_name)
            .ok_or_else(|| format!("Function not found: {}", function_name))?;

        let pipeline = Arc::new(ComputePipeline::new(&self.ctx, &function)?);

        // Store in cache
        let mut cache = self.pipelines.write();
        cache.insert(function_name.to_string(), pipeline.clone());

        Ok(pipeline)
    }

    /// Get a pipeline with function constants (for specialization).
    pub fn get_specialized(
        &self,
        function_name: &str,
        constants: &[(usize, u32)],  // (index, value) pairs
    ) -> Result<Arc<ComputePipeline>, String> {
        // Create unique key for this specialization
        let mut key = function_name.to_string();
        for (idx, val) in constants {
            key.push_str(&format!("_{}_{}", idx, val));
        }

        // Check cache
        {
            let cache = self.pipelines.read();
            if let Some(pipeline) = cache.get(&key) {
                return Ok(pipeline.clone());
            }
        }

        // Create specialized function
        let library = self.ctx.library()
            .ok_or("No shader library loaded")?;

        // For now, fall back to non-specialized function
        // Full specialization requires MTLFunctionConstantValues
        let ns_name = unsafe { NSString::from_str(function_name) };
        let function = library.newFunctionWithName(&ns_name)
            .ok_or_else(|| format!("Function not found: {}", function_name))?;

        let pipeline = Arc::new(ComputePipeline::new(&self.ctx, &function)?);

        let mut cache = self.pipelines.write();
        cache.insert(key, pipeline.clone());

        Ok(pipeline)
    }

    /// Clear all cached pipelines.
    pub fn clear(&self) {
        let mut cache = self.pipelines.write();
        cache.clear();
    }

    /// Get number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.read().len()
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Global pipeline cache singleton.
static PIPELINE_CACHE: once_cell::sync::OnceCell<PipelineCache> = once_cell::sync::OnceCell::new();

/// Get the global pipeline cache.
pub fn global_pipeline_cache() -> &'static PipelineCache {
    PIPELINE_CACHE.get_or_init(PipelineCache::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_cache_creation() {
        let cache = PipelineCache::new();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_optimal_threadgroup() {
        // This test requires an actual Metal device and pipeline
        // Skip if not available
    }
}
