//! Interactive visualization module for real-time lighting testing

mod viewer;
pub mod gpu_viewer;

pub use viewer::{InteractiveViewer, ViewerConfig};
pub use gpu_viewer::{run_gpu_viewer, GpuViewerConfig};
