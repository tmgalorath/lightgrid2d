//! GPU rendering module using wgpu
//!
//! Provides GPU-accelerated display and compute shaders
//! for the lighting algorithm.

pub mod blur;
pub mod compute;
pub mod compute_texture;
pub mod context;
pub mod pipelines;

pub use blur::{BlurPipeline, WallOverlayPipeline};
pub use compute::{BlendPipeline, BlendUniforms};
pub use compute_texture::BlendToTexturePipeline;
pub use context::GpuContext;
pub use pipelines::DisplayPipeline;
