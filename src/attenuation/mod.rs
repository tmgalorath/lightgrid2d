//! Layer 1: Attenuation Calculation (pure geometry/physics)
//!
//! This module provides different algorithms for calculating light attenuation
//! through a decay grid. All algorithms implement the `AttenuationAlgorithm` trait.

pub mod sweeping;

pub use sweeping::{Sweeping, flatten_grid};

/// Trait for light attenuation algorithms.
///
/// Implementations calculate how light propagates through a grid of decay values,
/// returning attenuation percentages (0.0 to 1.0) for each cell.
pub trait AttenuationAlgorithm {
    /// Calculate light attenuation from a point light source.
    ///
    /// # Arguments
    /// * `decay_grid` - 2D grid of decay values (0.0 = transparent, 1.0 = fully opaque)
    /// * `light_pos` - Position of the light source (x, y)
    ///
    /// # Returns
    /// Attenuation percentages (0.0 to 1.0) where 1.0 = full light, 0.0 = no light
    fn calculate(&self, decay_grid: &Vec<Vec<f32>>, light_pos: (usize, usize)) -> Vec<Vec<f32>>;
}

/// Converts an attenuation grid to a formatted string for debugging
pub fn attenuation_to_string(attenuation: &Vec<Vec<f32>>) -> String {
    let width = attenuation.len();
    let height = if width > 0 { attenuation[0].len() } else { 0 };
    let mut result = String::new();

    // Print y going down (top to bottom)
    for y in 0..height {
        for x in 0..width {
            result.push_str(&format!("{:5.2} ", attenuation[x][y]));
        }
        result.push('\n');
    }
    result
}
