//! Metal buffer management with zero-copy support.
//!
//! Leverages Apple Silicon's unified memory architecture for efficient
//! data sharing between CPU and GPU without explicit copies.

use super::device::{MetalContext, MTLDeviceTrait};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLResourceOptions};
use pyo3::prelude::*;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Data type for Metal buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    Float16,
    BFloat16,
    Float32,
    Int32,
    Int64,
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float16 | DType::BFloat16 => 2,
            DType::Float32 | DType::Int32 => 4,
            DType::Int64 => 8,
        }
    }
}

/// Metal buffer wrapper with shape and dtype information.
#[pyclass]
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    shape: Vec<usize>,
    dtype: DType,
    ctx: Arc<MetalContext>,
}

// SAFETY: Metal buffers are thread-safe when properly synchronized
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    /// Create a new Metal buffer with the given shape and dtype.
    pub fn new(shape: Vec<usize>, dtype: DType) -> Result<Self, String> {
        let ctx = MetalContext::get();
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype.size_bytes();

        // Use shared storage mode for unified memory (zero-copy)
        // MTLResourceStorageModeShared = 0
        let options = MTLResourceOptions::empty();

        let buffer = ctx
            .device()
            .newBufferWithLength_options(size_bytes, options)
            .ok_or("Failed to allocate Metal buffer")?;

        Ok(Self {
            buffer,
            shape,
            dtype,
            ctx,
        })
    }

    /// Create a Metal buffer from existing data (copies data).
    pub fn from_slice<T: Copy>(data: &[T], shape: Vec<usize>, dtype: DType) -> Result<Self, String> {
        let ctx = MetalContext::get();
        let size_bytes = data.len() * std::mem::size_of::<T>();
        let options = MTLResourceOptions::empty();

        let ptr = NonNull::new(data.as_ptr() as *mut c_void)
            .ok_or("Null pointer")?;
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                ptr,
                size_bytes,
                options,
            )
        }
        .ok_or("Failed to create Metal buffer from data")?;

        Ok(Self {
            buffer,
            shape,
            dtype,
            ctx,
        })
    }

    /// Create a Metal buffer wrapping existing memory (zero-copy).
    ///
    /// # Safety
    /// The caller must ensure the memory remains valid for the buffer's lifetime.
    /// Note: This currently copies data since newBufferWithBytesNoCopy isn't available.
    pub unsafe fn from_ptr(
        ptr: *mut c_void,
        size_bytes: usize,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, String> {
        let ctx = MetalContext::get();
        let options = MTLResourceOptions::empty();

        // Fall back to copy - true zero-copy needs different API
        let nn_ptr = NonNull::new(ptr)
            .ok_or("Null pointer")?;
        let buffer = ctx
            .device()
            .newBufferWithBytes_length_options(
                nn_ptr,
                size_bytes,
                options,
            )
            .ok_or("Failed to create Metal buffer from pointer")?;

        Ok(Self {
            buffer,
            shape,
            dtype,
            ctx,
        })
    }

    /// Get the underlying Metal buffer.
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Get the raw MTLBuffer for kernel dispatch.
    pub fn raw_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Get buffer contents as a mutable pointer.
    pub fn contents_ptr(&self) -> *mut c_void {
        self.buffer.contents().as_ptr()
    }

    /// Get buffer size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.buffer.length() as usize
    }

    /// Get number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Copy data to slice (for reading back results).
    pub fn to_slice<T: Copy>(&self, output: &mut [T]) {
        assert!(
            output.len() * std::mem::size_of::<T>() <= self.size_bytes(),
            "Output buffer too small"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.contents_ptr() as *const T,
                output.as_mut_ptr(),
                output.len(),
            );
        }
    }

    /// Copy data from slice (for uploading data).
    pub fn from_slice_mut<T: Copy>(&self, input: &[T]) {
        assert!(
            input.len() * std::mem::size_of::<T>() <= self.size_bytes(),
            "Input data too large"
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.contents_ptr() as *mut T,
                input.len(),
            );
        }
    }
}

#[pymethods]
impl MetalBuffer {
    /// Create a new buffer (Python API).
    #[new]
    fn py_new(shape: Vec<usize>, dtype_str: &str) -> PyResult<Self> {
        let dtype = match dtype_str {
            "float16" | "f16" => DType::Float16,
            "bfloat16" | "bf16" => DType::BFloat16,
            "float32" | "f32" => DType::Float32,
            "int32" | "i32" => DType::Int32,
            "int64" | "i64" => DType::Int64,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown dtype: {}", dtype_str)
            )),
        };
        MetalBuffer::new(shape, dtype)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))
    }

    /// Get shape as Python list.
    #[getter]
    fn get_shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get size in bytes.
    #[getter]
    fn nbytes(&self) -> usize {
        self.size_bytes()
    }

    /// Get number of elements.
    #[getter]
    fn get_numel(&self) -> usize {
        self.numel()
    }

    /// Get data pointer as integer (for interop with PyTorch).
    fn data_ptr(&self) -> usize {
        self.contents_ptr() as usize
    }
}

/// Buffer pool for reusing allocations.
pub struct BufferPool {
    ctx: Arc<MetalContext>,
    buffers: parking_lot::Mutex<Vec<MetalBuffer>>,
    max_size: usize,
}

impl BufferPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            ctx: MetalContext::get(),
            buffers: parking_lot::Mutex::new(Vec::new()),
            max_size,
        }
    }

    /// Get or create a buffer with the given shape and dtype.
    pub fn get(&self, shape: Vec<usize>, dtype: DType) -> Result<MetalBuffer, String> {
        let required_bytes: usize = shape.iter().product::<usize>() * dtype.size_bytes();

        let mut buffers = self.buffers.lock();

        // Try to find a suitable existing buffer
        if let Some(idx) = buffers.iter().position(|b| b.size_bytes() >= required_bytes) {
            let mut buf = buffers.swap_remove(idx);
            // Update shape (buffer is reused)
            buf.shape = shape;
            buf.dtype = dtype;
            return Ok(buf);
        }

        // No suitable buffer found, create new one
        drop(buffers);
        MetalBuffer::new(shape, dtype)
    }

    /// Return a buffer to the pool for reuse.
    pub fn put(&self, buffer: MetalBuffer) {
        let mut buffers = self.buffers.lock();
        if buffers.len() < self.max_size {
            buffers.push(buffer);
        }
        // If pool is full, buffer is dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buf = MetalBuffer::new(vec![4, 64, 128], DType::Float16).unwrap();
        assert_eq!(buf.shape(), &[4, 64, 128]);
        assert_eq!(buf.numel(), 4 * 64 * 128);
        assert_eq!(buf.size_bytes(), 4 * 64 * 128 * 2);
    }

    #[test]
    fn test_buffer_read_write() {
        let buf = MetalBuffer::new(vec![10], DType::Float32).unwrap();
        let input: Vec<f32> = (0..10).map(|i| i as f32).collect();
        buf.from_slice_mut(&input);

        let mut output = vec![0.0f32; 10];
        buf.to_slice(&mut output);
        assert_eq!(input, output);
    }
}
