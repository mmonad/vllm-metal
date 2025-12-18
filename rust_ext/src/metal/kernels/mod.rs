//! Metal kernel wrappers for high-performance inference.
//!
//! Each kernel follows Apple Metal best practices:
//! - Simdgroup-based threading (32 threads/simdgroup)
//! - Online softmax with max tracking
//! - Compile-time specialization for head dimensions
//! - Minimal memory bandwidth through careful data layout

pub mod attention;
pub mod gemv;
pub mod rope;
pub mod rms_norm;
pub mod cache;

pub use attention::*;
pub use gemv::*;
pub use rope::*;
pub use rms_norm::*;
pub use cache::*;
