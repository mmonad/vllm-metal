//! Metal integration module for vLLM on Apple Silicon.
//!
//! Provides direct Metal GPU access for high-performance inference:
//! - Device singleton for GPU management
//! - Zero-copy buffer management with unified memory
//! - Compute pipeline caching
//! - Custom kernel dispatch

pub mod device;
pub mod buffer;
pub mod pipeline;
pub mod kernels;
pub mod dispatch;

pub use device::{MetalContext, PyMetalContext};
pub use buffer::MetalBuffer;
pub use pipeline::ComputePipeline;
pub use dispatch::{dispatch_sdpa, dispatch_paged_attention};
