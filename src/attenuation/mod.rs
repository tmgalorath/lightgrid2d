//! Layer 1: Attenuation Calculation (pure geometry/physics)
//!
//! This module provides the sweeping algorithm for calculating light attenuation
//! through a decay grid using a flat memory layout for optimal performance.

pub mod sweeping;

pub use sweeping::{Sweeping, flatten_grid};

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
