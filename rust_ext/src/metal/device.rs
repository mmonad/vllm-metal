//! Metal Device singleton for GPU management.
//!
//! Provides thread-safe access to the Metal device and command queue.
//! Uses Apple's unified memory architecture for zero-copy data sharing.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
};
use pyo3::prelude::*;
// Re-export traits for use in other modules
pub use objc2_metal::MTLDevice as MTLDeviceTrait;
pub use objc2_metal::MTLLibrary as MTLLibraryTrait;
pub use objc2_metal::MTLCommandQueue as MTLCommandQueueTrait;
pub use objc2_metal::MTLCommandBuffer as MTLCommandBufferTrait;
#[allow(unused_imports)]
pub use objc2_metal::MTLCommandEncoder as MTLCommandEncoderTrait;
pub use objc2_metal::MTLComputeCommandEncoder as MTLComputeCommandEncoderTrait;
pub use objc2_metal::MTLComputePipelineState as MTLComputePipelineStateTrait;
pub use objc2_metal::MTLBuffer as MTLBufferTrait;
pub use objc2_metal::MTLResource as MTLResourceTrait;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use std::path::Path;
use std::sync::Arc;

/// Global Metal context singleton.
static METAL_CONTEXT: OnceCell<Arc<MetalContext>> = OnceCell::new();

/// Metal context holding device, command queue, and shader library.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Mutex<Option<Retained<ProtocolObject<dyn MTLLibrary>>>>,
}

// SAFETY: Metal objects are thread-safe when accessed through command buffers
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

impl MetalContext {
    /// Get or create the global Metal context.
    pub fn get() -> Arc<MetalContext> {
        METAL_CONTEXT
            .get_or_init(|| {
                Arc::new(MetalContext::new().expect("Failed to initialize Metal"))
            })
            .clone()
    }

    /// Create a new Metal context.
    fn new() -> Result<Self, String> {
        // Get the default Metal device (Apple GPU)
        let device_ptr = unsafe { MTLCreateSystemDefaultDevice() };
        let device = unsafe { Retained::from_raw(device_ptr) }
            .ok_or("No Metal device available")?;

        // Create command queue for submitting work
        let command_queue = device
            .newCommandQueue()
            .ok_or("Failed to create command queue")?;

        Ok(Self {
            device,
            command_queue,
            library: Mutex::new(None),
        })
    }

    /// Get the Metal device.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Get the command queue.
    pub fn command_queue(&self) -> &ProtocolObject<dyn MTLCommandQueue> {
        &self.command_queue
    }

    /// Load shader library from a .metallib file.
    pub fn load_library(&self, path: &Path) -> Result<(), String> {
        let path_str = path.to_str().ok_or("Invalid path")?;
        let ns_path = unsafe { NSString::from_str(path_str) };

        let library = unsafe {
            self.device.newLibraryWithFile_error(&ns_path)
        }.map_err(|e| format!("Failed to load library: {:?}", e))?;

        *self.library.lock() = Some(library);
        Ok(())
    }

    /// Load shader library from source code (for development).
    pub fn compile_library(&self, source: &str) -> Result<(), String> {
        let ns_source = unsafe { NSString::from_str(source) };

        let library = unsafe {
            self.device.newLibraryWithSource_options_error(&ns_source, None)
        }.map_err(|e| format!("Failed to compile library: {:?}", e))?;

        *self.library.lock() = Some(library);
        Ok(())
    }

    /// Get a reference to the loaded library.
    pub fn library(&self) -> Option<Retained<ProtocolObject<dyn MTLLibrary>>> {
        self.library.lock().clone()
    }

    /// Get device name for debugging.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Get recommended threadgroup memory size.
    pub fn max_threadgroup_memory(&self) -> usize {
        self.device.maxThreadgroupMemoryLength() as usize
    }

    /// Get maximum threads per threadgroup.
    pub fn max_threads_per_threadgroup(&self) -> usize {
        let size = self.device.maxThreadsPerThreadgroup();
        (size.width * size.height * size.depth) as usize
    }
}

/// Python-exposed Metal context wrapper.
#[pyclass(name = "MetalContext")]
pub struct PyMetalContext {
    ctx: Arc<MetalContext>,
}

#[pymethods]
impl PyMetalContext {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            ctx: MetalContext::get(),
        })
    }

    /// Get device name.
    fn device_name(&self) -> String {
        self.ctx.device_name()
    }

    /// Get max threads per threadgroup.
    fn max_threads_per_threadgroup(&self) -> usize {
        self.ctx.max_threads_per_threadgroup()
    }

    /// Get max threadgroup memory.
    fn max_threadgroup_memory(&self) -> usize {
        self.ctx.max_threadgroup_memory()
    }

    /// Load shader library from path.
    fn load_library(&self, path: &str) -> PyResult<()> {
        self.ctx.load_library(Path::new(path))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Compile shader library from source.
    fn compile_library(&self, source: &str) -> PyResult<()> {
        self.ctx.compile_library(source)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
    }

    /// Check if library is loaded.
    fn has_library(&self) -> bool {
        self.ctx.library().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_context_creation() {
        let ctx = MetalContext::get();
        println!("Metal device: {}", ctx.device_name());
        println!("Max threadgroup memory: {} bytes", ctx.max_threadgroup_memory());
        println!("Max threads per threadgroup: {}", ctx.max_threads_per_threadgroup());
    }
}
