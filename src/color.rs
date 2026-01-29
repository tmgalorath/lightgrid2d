//! Layer 2 & 3: Color Application and Multi-light Blending
//! 
//! This module handles applying colors to attenuation grids and blending
//! multiple colored light contributions together.

/// RGBA color with floating point components
#[derive(Debug, Clone, Copy)]
pub struct RGBA {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl RGBA {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        RGBA { r, g, b, a }
    }

    pub fn black() -> Self {
        RGBA { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }
    }
}

/// A colored point light with position, color, and intensity
#[derive(Debug, Clone)]
pub struct ColoredLight {
    pub color: (f32, f32, f32), // RGB 0.0-1.0
    pub intensity: f32,
    pub position: (usize, usize),
}

/// Applies light color and intensity to an attenuation grid.
/// 
/// # Arguments
/// * `attenuation` - 2D grid of attenuation values (0.0 to 1.0)
/// * `light` - The colored light to apply
/// 
/// # Returns
/// RGBA contribution grid for each cell
pub fn apply_light_color(
    attenuation: &Vec<Vec<f32>>,
    light: &ColoredLight,
) -> Vec<Vec<RGBA>> {
    let width = attenuation.len();
    let height = if width > 0 { attenuation[0].len() } else { 0 };

    let mut result = vec![vec![RGBA::black(); height]; width];

    for x in 0..width {
        for y in 0..height {
            let att = attenuation[x][y];
            result[x][y] = RGBA::new(
                light.color.0 * light.intensity * att,
                light.color.1 * light.intensity * att,
                light.color.2 * light.intensity * att,
                1.0,
            );
        }
    }

    result
}

/// Blends multiple light contributions using additive blending.
/// 
/// # Arguments
/// * `contributions` - Slice of RGBA grids from different lights
/// 
/// # Returns
/// Combined RGBA grid with all lights blended together
pub fn blend_lights(contributions: &[Vec<Vec<RGBA>>]) -> Vec<Vec<RGBA>> {
    if contributions.is_empty() {
        return vec![];
    }

    let width = contributions[0].len();
    let height = if width > 0 { contributions[0][0].len() } else { 0 };

    let mut result = vec![vec![RGBA::black(); height]; width];

    for contribution in contributions {
        for x in 0..width {
            for y in 0..height {
                result[x][y].r += contribution[x][y].r;
                result[x][y].g += contribution[x][y].g;
                result[x][y].b += contribution[x][y].b;
            }
        }
    }

    result
}

/// Converts an RGBA grid to a formatted string for debugging
pub fn rgba_grid_to_string(grid: &Vec<Vec<RGBA>>) -> String {
    let width = grid.len();
    let height = if width > 0 { grid[0].len() } else { 0 };
    let mut result = String::new();

    for y in 0..height {
        for x in 0..width {
            let c = &grid[x][y];
            result.push_str(&format!("({:.1},{:.1},{:.1}) ", c.r, c.g, c.b));
        }
        result.push('\n');
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgba_basics() {
        let color = RGBA::new(0.5, 0.75, 0.25, 0.9);
        assert_eq!((color.r, color.g, color.b, color.a), (0.5, 0.75, 0.25, 0.9));

        let black = RGBA::black();
        assert_eq!((black.r, black.g, black.b, black.a), (0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_apply_light_color() {
        // 2x2 grid: full, zero, and partial attenuation
        let attenuation = vec![
            vec![1.0_f32, 0.0],
            vec![0.5, 0.25],
        ];
        let light = ColoredLight {
            color: (1.0, 0.5, 0.0),
            intensity: 10.0,
            position: (0, 0),
        };

        let result = apply_light_color(&attenuation, &light);

        // Full attenuation: color * intensity * 1.0
        assert_eq!((result[0][0].r, result[0][0].g), (10.0, 5.0));
        // Zero attenuation: should be black
        assert_eq!((result[0][1].r, result[0][1].g), (0.0, 0.0));
        // Partial: 50% and 25%
        assert_eq!((result[1][0].r, result[1][1].r), (5.0, 2.5));
    }

    #[test]
    fn test_blend_lights_additive() {
        // Red + Blue = Magenta
        let red = vec![vec![RGBA::new(5.0, 0.0, 0.0, 1.0)]];
        let blue = vec![vec![RGBA::new(0.0, 0.0, 5.0, 1.0)]];

        let result = blend_lights(&[red, blue]);

        assert_eq!((result[0][0].r, result[0][0].g, result[0][0].b), (5.0, 0.0, 5.0));
    }

    #[test]
    fn test_empty_inputs() {
        // Empty grids should return empty results
        let empty_att: Vec<Vec<f32>> = vec![];
        let light = ColoredLight { color: (1.0, 1.0, 1.0), intensity: 10.0, position: (0, 0) };
        assert!(apply_light_color(&empty_att, &light).is_empty());
        assert!(blend_lights(&[]).is_empty());
    }
}
