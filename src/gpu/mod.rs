//! GPU rendering module using wgpu
//!
//! Provides GPU-accelerated display and (eventually) compute shaders
//! for the lighting algorithm.

pub mod context;
pub mod pipelines;

pub use context::GpuContext;
pub use pipelines::DisplayPipeline;
