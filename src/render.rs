//! Rendering and normalization functions for converting light grids to displayable formats

use crate::color::RGBA;
use std::fs::File;
use std::io::{self, Write};

/// Normalization mode for converting HDR light values to displayable range
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMode {
    /// Standard global max scaling - divides all values by the maximum
    Standard,
    /// Per-pixel brightness limiting (OpenStarbound-style)
    BrightnessLimit(f32),
    /// Perceptual luminance-based normalization
    PerceptualLuminance(f32),
}

/// Convert a float value (0.0-1.0) to a byte (0-255)
#[inline]
pub fn to_byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

/// Normalize an RGBA grid using the specified mode
pub fn normalize_grid_with_mode(grid: &Vec<Vec<RGBA>>, mode: NormalizationMode) -> Vec<Vec<RGBA>> {
    match mode {
        NormalizationMode::Standard => normalize_grid(grid),
        NormalizationMode::BrightnessLimit(limit) => normalize_grid_osb(grid, limit),
        NormalizationMode::PerceptualLuminance(target) => normalize_grid_perceptual(grid, target),
    }
}

/// Standard normalization: scale all values by global maximum
pub fn normalize_grid(grid: &Vec<Vec<RGBA>>) -> Vec<Vec<RGBA>> {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    
    // Find global maximum across all channels
    let mut max_val = 0.0f32;
    for x in 0..width {
        for y in 0..height {
            let pixel = &grid[x][y];
            max_val = max_val.max(pixel.r).max(pixel.g).max(pixel.b);
        }
    }
    
    if max_val <= 0.0 {
        return grid.clone();
    }
    
    // Scale all values
    let scale = 1.0 / max_val;
    let mut result = vec![vec![RGBA::black(); height]; width];
    
    for x in 0..width {
        for y in 0..height {
            let pixel = &grid[x][y];
            result[x][y] = RGBA::new(
                pixel.r * scale,
                pixel.g * scale,
                pixel.b * scale,
                pixel.a,
            );
        }
    }
    
    result
}

/// OpenStarbound-style normalization: per-pixel brightness limiting
pub fn normalize_grid_osb(grid: &Vec<Vec<RGBA>>, limit: f32) -> Vec<Vec<RGBA>> {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    
    let mut result = vec![vec![RGBA::black(); height]; width];
    
    for x in 0..width {
        for y in 0..height {
            let pixel = &grid[x][y];
            let max_component = pixel.r.max(pixel.g).max(pixel.b);
            
            if max_component > limit {
                let scale = limit / max_component;
                result[x][y] = RGBA::new(
                    pixel.r * scale,
                    pixel.g * scale,
                    pixel.b * scale,
                    pixel.a,
                );
            } else {
                result[x][y] = RGBA::new(
                    pixel.r / limit,
                    pixel.g / limit,
                    pixel.b / limit,
                    pixel.a,
                );
            }
        }
    }
    
    result
}

/// Perceptual luminance-based normalization
pub fn normalize_grid_perceptual(grid: &Vec<Vec<RGBA>>, target_luminance: f32) -> Vec<Vec<RGBA>> {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    
    // Standard perceptual luminance weights (Rec. 709)
    const LUM_R: f32 = 0.2126;
    const LUM_G: f32 = 0.7152;
    const LUM_B: f32 = 0.0722;
    
    let mut result = vec![vec![RGBA::black(); height]; width];
    
    for x in 0..width {
        for y in 0..height {
            let pixel = &grid[x][y];
            let luminance = pixel.r * LUM_R + pixel.g * LUM_G + pixel.b * LUM_B;
            
            if luminance > target_luminance {
                let scale = target_luminance / luminance;
                result[x][y] = RGBA::new(
                    pixel.r * scale,
                    pixel.g * scale,
                    pixel.b * scale,
                    pixel.a,
                );
            } else if luminance > 0.0 {
                // Scale to fit within 0-1 range based on target
                let scale = 1.0 / target_luminance.max(1.0);
                result[x][y] = RGBA::new(
                    (pixel.r * scale).min(1.0),
                    (pixel.g * scale).min(1.0),
                    (pixel.b * scale).min(1.0),
                    pixel.a,
                );
            } else {
                result[x][y] = RGBA::black();
            }
        }
    }
    
    result
}

/// Save an RGBA grid to a PPM file
pub fn save_ppm(grid: &Vec<Vec<RGBA>>, filename: &str, scale: usize) -> io::Result<()> {
    save_ppm_with_walls(grid, None, 0.5, filename, scale)
}

/// Save an RGBA grid to a PPM file, optionally showing walls
pub fn save_ppm_with_walls(
    grid: &Vec<Vec<RGBA>>,
    decay_grid: Option<&Vec<Vec<f32>>>,
    wall_threshold: f32,
    filename: &str,
    scale: usize,
) -> io::Result<()> {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    
    let img_width = width * scale;
    let img_height = height * scale;
    
    let mut file = File::create(filename)?;
    writeln!(file, "P3")?;
    writeln!(file, "{} {}", img_width, img_height)?;
    writeln!(file, "255")?;
    
    // Normalize the grid first
    let normalized = normalize_grid(grid);
    
    for img_y in 0..img_height {
        for img_x in 0..img_width {
            let x = img_x / scale;
            let y = img_y / scale;
            
            // Check if this is a wall
            let is_wall = decay_grid
                .map(|dg| dg[x][y] >= wall_threshold)
                .unwrap_or(false);
            
            let (r, g, b) = if is_wall {
                (64u8, 64u8, 64u8) // Gray for walls
            } else {
                let pixel = &normalized[x][y];
                (to_byte(pixel.r), to_byte(pixel.g), to_byte(pixel.b))
            };
            
            write!(file, "{} {} {} ", r, g, b)?;
        }
        writeln!(file)?;
    }
    
    Ok(())
}

/// Save a pre-normalized RGBA grid to a PPM file
pub fn save_ppm_normalized(grid: &Vec<Vec<RGBA>>, filename: &str, scale: usize) -> io::Result<()> {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    
    let img_width = width * scale;
    let img_height = height * scale;
    
    let mut file = File::create(filename)?;
    writeln!(file, "P3")?;
    writeln!(file, "{} {}", img_width, img_height)?;
    writeln!(file, "255")?;
    
    for img_y in 0..img_height {
        for img_x in 0..img_width {
            let x = img_x / scale;
            let y = img_y / scale;
            let pixel = &grid[x][y];
            write!(file, "{} {} {} ", to_byte(pixel.r), to_byte(pixel.g), to_byte(pixel.b))?;
        }
        writeln!(file)?;
    }
    
    Ok(())
}
