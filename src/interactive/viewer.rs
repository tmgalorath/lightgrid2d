//! Interactive light viewer - mouse controls light position in real-time

use minifb::{Key, Window, WindowOptions, MouseMode, MouseButton};
use rayon::prelude::*;
use crate::attenuation::Sweeping;
use crate::render::{NormalizationMode, to_byte};

/// Configuration for the interactive viewer
#[derive(Clone)]
pub struct ViewerConfig {
    /// Grid size (width x height in cells)
    pub grid_size: (usize, usize),
    /// Pixel scale factor (each cell = scale x scale pixels)
    pub scale: usize,
    /// Base decay rate for empty cells
    pub base_decay: f32,
    /// Wall decay rate (1.0 = fully opaque)
    pub wall_decay: f32,
    /// Initial light color (r, g, b)
    pub light_color: (f32, f32, f32),
    /// Initial normalization mode
    pub normalization_mode: NormalizationMode,
}

impl Default for ViewerConfig {
    fn default() -> Self {
        Self {
            grid_size: (100, 100),
            scale: 16,
            base_decay: 0.1,
            wall_decay: 0.6,
            light_color: (1.0, 0.8, 0.4), // Warm torch color
            normalization_mode: NormalizationMode::PerceptualLuminance(1.0),
        }
    }
}

/// Interactive viewer for testing lighting algorithms
pub struct InteractiveViewer {
    config: ViewerConfig,
    decay_flat: Vec<f32>,
    wall_flat: Vec<bool>,
    window: Window,
    buffer: Vec<u32>,
}

impl InteractiveViewer {
    /// Create a new interactive viewer with the given configuration
    pub fn new(config: ViewerConfig) -> Result<Self, String> {
        let (grid_w, grid_h) = config.grid_size;
        let window_w = grid_w * config.scale;
        let window_h = grid_h * config.scale;
        
        let window = Window::new(
            "Lighting Test - Interactive Viewer (ESC to exit)",
            window_w,
            window_h,
            WindowOptions {
                resize: false,
                ..WindowOptions::default()
            },
        ).map_err(|e| e.to_string())?;
        
        // Initialize flat grids
        let decay_flat = vec![config.base_decay; grid_w * grid_h];
        let wall_flat = vec![false; grid_w * grid_h];
        let buffer = vec![0u32; window_w * window_h];
        
        Ok(Self {
            config,
            decay_flat,
            wall_flat,
            window,
            buffer,
        })
    }
    
    /// Run the interactive viewer loop
    pub fn run(&mut self) -> Result<(), String> {
        let (grid_w, grid_h) = self.config.grid_size;
        let scale = self.config.scale;
        
        // Limit to ~60fps
        self.window.set_target_fps(60);
        
        let mut current_color = self.config.light_color;
        let mut current_mode = self.config.normalization_mode;
        
        // Subpixel blending for smooth light movement
        let mut subpixel_enabled = true;
        
        println!("=== Interactive Light Viewer ===");
        println!("Controls:");
        println!("  Mouse      - Move light source");
        println!("  Left Click - Toggle wall");
        println!("  Right Click- Clear all walls");
        println!("  1/2/3      - Normalization: Standard/OSB/Perceptual");
        println!("  R/G/B/Y/W  - Color: Red/Green/Blue/Yellow/White");
        println!("  +/-        - Adjust decay rate");
        println!("  T          - Toggle subpixel blending ON/OFF");
        println!("  C          - Clear walls");
        println!("  ESC        - Exit");
        println!();
        
        while self.window.is_open() && !self.window.is_key_down(Key::Escape) {
            // Handle keyboard input
            if self.window.is_key_pressed(Key::Key1, minifb::KeyRepeat::No) {
                current_mode = NormalizationMode::Standard;
                println!("Mode: Standard");
            }
            if self.window.is_key_pressed(Key::Key2, minifb::KeyRepeat::No) {
                current_mode = NormalizationMode::BrightnessLimit(1.0);
                println!("Mode: OSB (BrightnessLimit 1.0)");
            }
            if self.window.is_key_pressed(Key::Key3, minifb::KeyRepeat::No) {
                current_mode = NormalizationMode::PerceptualLuminance(1.0);
                println!("Mode: Perceptual Luminance");
            }
            
            // Color keys
            if self.window.is_key_pressed(Key::R, minifb::KeyRepeat::No) {
                current_color = (1.0, 0.0, 0.0);
                println!("Color: Red");
            }
            if self.window.is_key_pressed(Key::G, minifb::KeyRepeat::No) {
                current_color = (0.0, 1.0, 0.0);
                println!("Color: Green");
            }
            if self.window.is_key_pressed(Key::B, minifb::KeyRepeat::No) {
                current_color = (0.0, 0.0, 1.0);
                println!("Color: Blue");
            }
            if self.window.is_key_pressed(Key::Y, minifb::KeyRepeat::No) {
                current_color = (1.0, 1.0, 0.0);
                println!("Color: Yellow");
            }
            if self.window.is_key_pressed(Key::W, minifb::KeyRepeat::No) {
                current_color = (1.0, 1.0, 1.0);
                println!("Color: White");
            }
            
            // Decay adjustment
            if self.window.is_key_pressed(Key::Equal, minifb::KeyRepeat::Yes) 
               || self.window.is_key_pressed(Key::NumPadPlus, minifb::KeyRepeat::Yes) {
                self.config.base_decay = (self.config.base_decay + 0.02).min(0.5);
                self.update_decay_grid();
                println!("Decay: {:.2}", self.config.base_decay);
            }
            if self.window.is_key_pressed(Key::Minus, minifb::KeyRepeat::Yes)
               || self.window.is_key_pressed(Key::NumPadMinus, minifb::KeyRepeat::Yes) {
                self.config.base_decay = (self.config.base_decay - 0.02).max(0.01);
                self.update_decay_grid();
                println!("Decay: {:.2}", self.config.base_decay);
            }
            
            // Clear walls
            if self.window.is_key_pressed(Key::C, minifb::KeyRepeat::No) {
                self.clear_walls();
                println!("Walls cleared");
            }
            
            // Toggle bilinear blending
            if self.window.is_key_pressed(Key::T, minifb::KeyRepeat::No) {
                subpixel_enabled = !subpixel_enabled;
                if subpixel_enabled {
                    println!("Subpixel blending: ON (smooth movement)");
                } else {
                    println!("Subpixel blending: OFF (snappy grid)");
                }
            }
            
            // Handle mouse input for walls
            if let Some((mx, my)) = self.window.get_mouse_pos(MouseMode::Discard) {
                let grid_x = (mx as usize / scale).min(grid_w - 1);
                let grid_y = (my as usize / scale).min(grid_h - 1);
                
                // Toggle wall on left click
                if self.window.get_mouse_down(MouseButton::Left) {
                    // Only toggle once per cell (simple debounce by checking if just pressed)
                    static mut LAST_WALL_POS: (usize, usize) = (usize::MAX, usize::MAX);
                    unsafe {
                        if LAST_WALL_POS != (grid_x, grid_y) {
                            self.toggle_wall(grid_x, grid_y);
                            LAST_WALL_POS = (grid_x, grid_y);
                        }
                    }
                } else {
                    unsafe {
                        static mut LAST_WALL_POS: (usize, usize) = (usize::MAX, usize::MAX);
                        LAST_WALL_POS = (usize::MAX, usize::MAX);
                    }
                }
                
                // Clear walls on right click
                if self.window.get_mouse_down(MouseButton::Right) {
                    self.clear_walls();
                }
                
                // Calculate subpixel position (0.0 to grid_size with fractional part)
                let subpixel_x = mx / scale as f32;
                let subpixel_y = my / scale as f32;
                
                if subpixel_enabled {
                    // 4-grid bilinear (smooth movement, fast with rayon)
                    self.render_lighting_bilinear(subpixel_x, subpixel_y, current_color, current_mode);
                } else {
                    // Snap to nearest cell
                    let render_x = (subpixel_x.round() as usize).min(grid_w - 1);
                    let render_y = (subpixel_y.round() as usize).min(grid_h - 1);
                    self.render_lighting(render_x, render_y, current_color, current_mode);
                }
            }
            
            // Update window
            self.window.update_with_buffer(&self.buffer, grid_w * scale, grid_h * scale)
                .map_err(|e| e.to_string())?;
        }
        
        Ok(())
    }
    
    /// Toggle wall at grid position
    fn toggle_wall(&mut self, x: usize, y: usize) {
        let (grid_w, _grid_h) = self.config.grid_size;
        let idx = y * grid_w + x;
        self.wall_flat[idx] = !self.wall_flat[idx];
        self.decay_flat[idx] = if self.wall_flat[idx] {
            self.config.wall_decay
        } else {
            self.config.base_decay
        };
    }
    
    /// Clear all walls
    fn clear_walls(&mut self) {
        self.wall_flat.fill(false);
        self.decay_flat.fill(self.config.base_decay);
    }
    
    /// Update decay grid after base_decay change (preserves walls)
    fn update_decay_grid(&mut self) {
        for (i, is_wall) in self.wall_flat.iter().enumerate() {
            if !is_wall {
                self.decay_flat[i] = self.config.base_decay;
            }
        }
    }
    
    /// Render lighting from the given integer position (fully flat pipeline)
    fn render_lighting(&mut self, light_x: usize, light_y: usize, color: (f32, f32, f32), mode: NormalizationMode) {
        let (grid_w, grid_h) = self.config.grid_size;
        
        // Calculate attenuation using sweeping algorithm
        let sweeping = Sweeping::new();
        let attenuation = sweeping.calculate_flat(&self.decay_flat, grid_w, grid_h, light_x, light_y);
        
        // Render directly to buffer (fused color + normalize + write)
        self.render_flat_to_buffer(&attenuation, color, mode);
    }
    
    /// Render lighting with bilinear blending for subpixel positions
    /// Calculates light from 4 neighboring cells and blends based on fractional position
    /// Uses rayon thread pool and flat memory layout for best performance
    fn render_lighting_bilinear(&mut self, subpixel_x: f32, subpixel_y: f32, color: (f32, f32, f32), mode: NormalizationMode) {
        let (grid_w, grid_h) = self.config.grid_size;
        let size = grid_w * grid_h;
        
        // Get the 4 corner cell positions
        let x0 = (subpixel_x.floor() as usize).min(grid_w - 1);
        let y0 = (subpixel_y.floor() as usize).min(grid_h - 1);
        let x1 = (x0 + 1).min(grid_w - 1);
        let y1 = (y0 + 1).min(grid_h - 1);
        
        // Fractional part determines blend weights
        let fx = subpixel_x.fract().max(0.0);
        let fy = subpixel_y.fract().max(0.0);
        
        // Bilinear weights for each corner
        let w00 = (1.0 - fx) * (1.0 - fy);  // top-left
        let w10 = fx * (1.0 - fy);          // top-right
        let w01 = (1.0 - fx) * fy;          // bottom-left
        let w11 = fx * fy;                  // bottom-right
        
        // Calculate all 4 corners in parallel using rayon
        let positions = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
        let decay_flat = &self.decay_flat;
        
        let grids: Vec<Vec<f32>> = positions
            .par_iter()
            .map(|&(lx, ly)| {
                let sweeping = Sweeping::new();
                sweeping.calculate_flat(decay_flat, grid_w, grid_h, lx, ly)
            })
            .collect();
        
        // Blend all 4 grids into a single flat buffer
        let weights = [w00, w10, w01, w11];
        let mut blended = vec![0.0f32; size];
        
        for i in 0..size {
            blended[i] = grids[0][i] * weights[0] 
                       + grids[1][i] * weights[1] 
                       + grids[2][i] * weights[2] 
                       + grids[3][i] * weights[3];
        }
        
        // Render directly to buffer (fused color + normalize + write)
        self.render_flat_to_buffer(&blended, color, mode);
    }
    
    /// Render flat attenuation directly to pixel buffer
    /// Fuses color application, normalization, and buffer write in one pass
    fn render_flat_to_buffer(&mut self, attenuation: &[f32], color: (f32, f32, f32), mode: NormalizationMode) {
        let (grid_w, grid_h) = self.config.grid_size;
        let scale = self.config.scale;
        
        // Calculate normalization factor based on mode
        let norm_factor = match mode {
            NormalizationMode::Standard => {
                // Find max colored value
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max);
                let max_colored = max_att * color.0.max(color.1).max(color.2);
                if max_colored > 0.0 { 1.0 / max_colored } else { 1.0 }
            }
            NormalizationMode::BrightnessLimit(limit) => {
                // OSB-style: clamp to limit
                1.0 / limit
            }
            NormalizationMode::PerceptualLuminance(target) => {
                // Perceptual: based on luminance
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max);
                let max_lum = max_att * (0.2126 * color.0 + 0.7152 * color.1 + 0.0722 * color.2);
                if max_lum > 0.0 { target / max_lum } else { 1.0 }
            }
        };
        
        // Render each cell
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let idx = gy * grid_w + gx;
                let att = attenuation[idx];
                
                // Apply color, intensity, and normalization
                let r_f = (color.0 * att * norm_factor).min(1.0);
                let g_f = (color.1 * att * norm_factor).min(1.0);
                let b_f = (color.2 * att * norm_factor).min(1.0);
                
                let mut r = to_byte(r_f) as u32;
                let mut g = to_byte(g_f) as u32;
                let mut b = to_byte(b_f) as u32;
                
                // If it's a wall, add a slight tint to make it visible
                if self.wall_flat[idx] {
                    r = r.max(30);
                    g = g.max(30);
                    b = b.max(30);
                }
                
                let color_u32 = (r << 16) | (g << 8) | b;
                
                // Fill scaled pixels
                for sy in 0..scale {
                    for sx in 0..scale {
                        let px = gx * scale + sx;
                        let py = gy * scale + sy;
                        let buf_idx = py * (grid_w * scale) + px;
                        self.buffer[buf_idx] = color_u32;
                    }
                }
            }
        }
    }
}
