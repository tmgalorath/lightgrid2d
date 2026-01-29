//! GPU-accelerated interactive light viewer using wgpu + winit

use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};
use rayon::prelude::*;

use crate::attenuation::Sweeping;
use crate::gpu::{GpuContext, DisplayPipeline, BlendPipeline, BlendToTexturePipeline, BlendUniforms, BlurPipeline, WallOverlayPipeline};
use crate::render::NormalizationMode;

/// Configuration for the GPU viewer
#[derive(Clone)]
pub struct GpuViewerConfig {
    /// Grid size (width x height in cells)
    pub grid_size: (usize, usize),
    /// Base decay rate for empty cells
    pub base_decay: f32,
    /// Wall decay rate (1.0 = fully opaque)
    pub wall_decay: f32,
    /// Initial light color (r, g, b)
    pub light_color: (f32, f32, f32),
    /// Initial normalization mode
    pub normalization_mode: NormalizationMode,
    /// Window title
    pub title: String,
}

impl Default for GpuViewerConfig {
    fn default() -> Self {
        Self {
            grid_size: (100, 100),
            base_decay: 0.1,
            wall_decay: 0.6,
            light_color: (1.0, 0.8, 0.4), // Warm torch color
            normalization_mode: NormalizationMode::PerceptualLuminance(1.0),
            title: "Lighting Test - GPU Viewer (ESC to exit)".to_string(),
        }
    }
}

/// GPU-accelerated viewer state
struct ViewerState {
    config: GpuViewerConfig,
    gpu_ctx: GpuContext,
    display_pipeline: DisplayPipeline,
    blend_pipeline: BlendPipeline,
    blend_to_texture_pipeline: BlendToTexturePipeline,
    blur_pipeline: BlurPipeline,
    wall_overlay_pipeline: WallOverlayPipeline,
    texture_bind_group: wgpu::BindGroup,
    
    // Lighting state
    decay_flat: Vec<f32>,
    wall_flat: Vec<bool>,
    pixel_buffer: Vec<u8>, // RGBA8 for CPU fallback
    
    // Interaction state
    current_color: (f32, f32, f32),
    current_mode: NormalizationMode,
    subpixel_enabled: bool,
    use_gpu_blend: bool,
    use_srgb: bool,
    use_linear_filter: bool,
    blur_level: u32,  // 0 = off, 1-3 = blur iterations
    mouse_pos: Option<(f32, f32)>,
    left_mouse_down: bool,
    last_wall_pos: Option<(usize, usize)>,
    
    // Placed lights
    placed_lights: Vec<(usize, usize)>,
    mouse_light_enabled: bool,
    
    // Light source intensity
    source_intensity: f32,
}

impl ViewerState {
    fn new(window: Arc<Window>, config: GpuViewerConfig) -> Result<Self, String> {
        let gpu_ctx = GpuContext::new(window)?;
        let display_pipeline = DisplayPipeline::new(&gpu_ctx);
        let blend_pipeline = BlendPipeline::new(
            &gpu_ctx,
            config.grid_size.0 as u32,
            config.grid_size.1 as u32,
        );
        let blend_to_texture_pipeline = BlendToTexturePipeline::new(
            &gpu_ctx,
            config.grid_size.0 as u32,
            config.grid_size.1 as u32,
        );
        
        let mut blur_pipeline = BlurPipeline::new(
            &gpu_ctx,
            config.grid_size.0 as u32,
            config.grid_size.1 as u32,
        );
        
        // Set up blur to work on the blend output texture
        blur_pipeline.setup_for_texture(&gpu_ctx, blend_to_texture_pipeline.output_texture_view());
        
        // Set up wall overlay to work on the blend output texture
        let mut wall_overlay_pipeline = WallOverlayPipeline::new(
            &gpu_ctx,
            config.grid_size.0 as u32,
            config.grid_size.1 as u32,
        );
        wall_overlay_pipeline.setup_for_texture(&gpu_ctx, blend_to_texture_pipeline.output_texture_view());
        
        // Create bind group for the compute output texture
        let texture_bind_group = display_pipeline.create_bind_group_for_texture(
            &gpu_ctx,
            blend_to_texture_pipeline.output_texture_view(),
            false, // Start with nearest (crisp pixels)
        );
        
        let (grid_w, grid_h) = config.grid_size;
        let decay_flat = vec![config.base_decay; grid_w * grid_h];
        let wall_flat = vec![false; grid_w * grid_h];
        let pixel_buffer = vec![0u8; grid_w * grid_h * 4];
        
        let current_color = config.light_color;
        let current_mode = config.normalization_mode;
        
        Ok(Self {
            config,
            gpu_ctx,
            display_pipeline,
            blend_pipeline,
            blend_to_texture_pipeline,
            blur_pipeline,
            wall_overlay_pipeline,
            texture_bind_group,
            decay_flat,
            wall_flat,
            pixel_buffer,
            current_color,
            current_mode,
            subpixel_enabled: true,
            use_gpu_blend: true,
            use_srgb: false,
            use_linear_filter: false,
            blur_level: 0,
            mouse_pos: None,
            left_mouse_down: false,
            last_wall_pos: None,
            placed_lights: Vec::new(),
            mouse_light_enabled: true,
            source_intensity: 1.0,
        })
    }
    
    fn update_texture_bind_group(&mut self) {
        self.texture_bind_group = self.display_pipeline.create_bind_group_for_texture(
            &self.gpu_ctx,
            self.blend_to_texture_pipeline.output_texture_view(),
            self.use_linear_filter,
        );
    }
    
    fn toggle_wall(&mut self, x: usize, y: usize) {
        let (grid_w, _) = self.config.grid_size;
        let idx = y * grid_w + x;
        self.wall_flat[idx] = !self.wall_flat[idx];
        self.decay_flat[idx] = if self.wall_flat[idx] {
            self.config.wall_decay
        } else {
            self.config.base_decay
        };
    }
    
    fn clear_walls(&mut self) {
        self.wall_flat.fill(false);
        self.decay_flat.fill(self.config.base_decay);
    }
    
    fn place_light(&mut self, x: usize, y: usize) {
        // Don't place lights in walls
        let (grid_w, _) = self.config.grid_size;
        let idx = y * grid_w + x;
        if !self.wall_flat[idx] {
            self.placed_lights.push((x, y));
            println!("Placed light at ({}, {}). Total lights: {}", x, y, self.placed_lights.len());
        } else {
            println!("Cannot place light in a wall!");
        }
    }
    
    fn clear_lights(&mut self) {
        self.placed_lights.clear();
        println!("Cleared all placed lights");
    }
    
    /// Generate a cave pattern using cellular automata
    fn generate_cave(&mut self) {
        let (width, height) = self.config.grid_size;
        
        // Step 1: Initialize with random noise (~45% walls)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        use std::time::{SystemTime, UNIX_EPOCH};
        
        // Simple seeded random using time
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let mut hasher = DefaultHasher::new();
        for i in 0..self.wall_flat.len() {
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            hasher = DefaultHasher::new();
            self.wall_flat[i] = (hash % 100) < 45; // 45% chance of wall
        }
        
        // Step 2: Apply cellular automata rules (5 iterations)
        let mut temp = vec![false; width * height];
        
        for _ in 0..5 {
            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    
                    // Count walls in 3x3 neighborhood (including self)
                    let mut wall_count = 0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            
                            // Treat out-of-bounds as walls (creates solid borders)
                            if nx < 0 || nx >= width as i32 || ny < 0 || ny >= height as i32 {
                                wall_count += 1;
                            } else {
                                let ni = ny as usize * width + nx as usize;
                                if self.wall_flat[ni] {
                                    wall_count += 1;
                                }
                            }
                        }
                    }
                    
                    // Rule: become wall if >= 5 neighbors are walls
                    temp[idx] = wall_count >= 5;
                }
            }
            
            // Swap buffers
            std::mem::swap(&mut self.wall_flat, &mut temp);
        }
        
        // Step 3: Ensure borders are walls (for a contained cave)
        for x in 0..width {
            self.wall_flat[x] = true;                           // Top row
            self.wall_flat[(height - 1) * width + x] = true;    // Bottom row
        }
        for y in 0..height {
            self.wall_flat[y * width] = true;                   // Left column
            self.wall_flat[y * width + (width - 1)] = true;     // Right column
        }
        
        // Step 4: Update decay grid to match walls
        for (i, &is_wall) in self.wall_flat.iter().enumerate() {
            self.decay_flat[i] = if is_wall {
                self.config.wall_decay
            } else {
                self.config.base_decay
            };
        }
    }
    
    fn update_decay_grid(&mut self) {
        for (i, is_wall) in self.wall_flat.iter().enumerate() {
            if !is_wall {
                self.decay_flat[i] = self.config.base_decay;
            }
        }
    }
    
    fn render_lighting(&mut self, light_x: usize, light_y: usize) {
        let (grid_w, grid_h) = self.config.grid_size;
        
        let sweeping = Sweeping::new();
        let attenuation = sweeping.calculate_flat(&self.decay_flat, grid_w, grid_h, light_x, light_y, self.source_intensity);
        
        if self.use_gpu_blend {
            self.render_single_gpu(&attenuation);
        } else {
            self.render_attenuation_to_buffer(&attenuation);
        }
    }
    
    fn render_lighting_bilinear(&mut self, subpixel_x: f32, subpixel_y: f32) {
        let (grid_w, grid_h) = self.config.grid_size;
        
        let x0 = (subpixel_x.floor() as usize).min(grid_w - 1);
        let y0 = (subpixel_y.floor() as usize).min(grid_h - 1);
        let x1 = (x0 + 1).min(grid_w - 1);
        let y1 = (y0 + 1).min(grid_h - 1);
        
        let fx = subpixel_x.fract().max(0.0);
        let fy = subpixel_y.fract().max(0.0);
        
        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;
        
        let positions = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)];
        let decay_flat = &self.decay_flat;
        
        // Calculate 4 grids in parallel (CPU)
        let grids: Vec<Vec<f32>> = positions
            .par_iter()
            .map(|&(lx, ly)| {
                let sweeping = Sweeping::new();
                sweeping.calculate_flat(decay_flat, grid_w, grid_h, lx, ly, 1.0)
            })
            .collect();
        
        if self.use_gpu_blend {
            // GPU path: upload grids and run compute shader
            self.render_bilinear_gpu(&grids, [w00, w10, w01, w11]);
        } else {
            // CPU path: blend on CPU
            let weights = [w00, w10, w01, w11];
            let size = grid_w * grid_h;
            let mut blended = vec![0.0f32; size];
            
            for i in 0..size {
                blended[i] = grids[0][i] * weights[0]
                           + grids[1][i] * weights[1]
                           + grids[2][i] * weights[2]
                           + grids[3][i] * weights[3];
            }
            
            self.render_attenuation_to_buffer(&blended);
        }
    }
    
    fn render_bilinear_gpu(&mut self, grids: &[Vec<f32>], weights: [f32; 4]) {
        let (grid_w, grid_h) = self.config.grid_size;
        let color = self.current_color;
        let mode = self.current_mode;
        
        // Calculate normalization factor (still need max from grids)
        let max_blended: f32 = (0..grid_w * grid_h)
            .map(|i| {
                grids[0][i] * weights[0]
                + grids[1][i] * weights[1]
                + grids[2][i] * weights[2]
                + grids[3][i] * weights[3]
            })
            .fold(0.0f32, f32::max);
        
        let norm_factor = match mode {
            NormalizationMode::Standard => {
                let max_colored = max_blended * color.0.max(color.1).max(color.2);
                if max_colored > 0.0 { 1.0 / max_colored } else { 1.0 }
            }
            NormalizationMode::BrightnessLimit(limit) => 1.0 / limit,
            NormalizationMode::PerceptualLuminance(target) => {
                let max_lum = max_blended * (0.2126 * color.0 + 0.7152 * color.1 + 0.0722 * color.2);
                if max_lum > 0.0 { target / max_lum } else { 1.0 }
            }
        };
        
        // Upload to GPU
        self.blend_pipeline.upload_grids(
            &self.gpu_ctx,
            &[&grids[0], &grids[1], &grids[2], &grids[3]],
        );
        self.blend_pipeline.upload_walls(&self.gpu_ctx, &self.wall_flat);
        
        let uniforms = BlendUniforms {
            weights,
            color: [color.0, color.1, color.2],
            norm_factor,
            grid_width: grid_w as u32,
            grid_height: grid_h as u32,
            apply_srgb: if self.use_srgb { 1 } else { 0 },
            _padding: 0,
        };
        
        // Use optimized texture pipeline (no readback!)
        self.blend_to_texture_pipeline.upload_grids(
            &self.gpu_ctx,
            &[&grids[0], &grids[1], &grids[2], &grids[3]],
        );
        self.blend_to_texture_pipeline.upload_walls(&self.gpu_ctx, &self.wall_flat);
        self.blend_to_texture_pipeline.upload_uniforms(&self.gpu_ctx, &uniforms);
        
        // Run compute shader → writes directly to texture
        self.blend_to_texture_pipeline.dispatch_blend(&self.gpu_ctx);
        
        // Apply blur if enabled
        if self.blur_level > 0 {
            self.blur_pipeline.dispatch_iterations(&self.gpu_ctx, self.blur_level);
        }
        
        // Apply wall overlay (always, to show walls after blur or when blur is off)
        self.wall_overlay_pipeline.upload_walls(&self.gpu_ctx, &self.wall_flat);
        self.wall_overlay_pipeline.dispatch(&self.gpu_ctx);
        
        // No readback needed! Display will use texture_bind_group
    }
    
    fn render_single_gpu(&mut self, attenuation: &[f32]) {
        let (grid_w, grid_h) = self.config.grid_size;
        let color = self.current_color;
        let mode = self.current_mode;
        
        // Calculate normalization factor
        // Adjust max by source_intensity so dimmer lights actually appear dimmer
        let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max) / self.source_intensity;
        let norm_factor = match mode {
            NormalizationMode::Standard => {
                let max_colored = max_att * color.0.max(color.1).max(color.2);
                if max_colored > 0.0 { 1.0 / max_colored } else { 1.0 }
            }
            NormalizationMode::BrightnessLimit(limit) => 1.0 / limit,
            NormalizationMode::PerceptualLuminance(target) => {
                let max_lum = max_att * (0.2126 * color.0 + 0.7152 * color.1 + 0.0722 * color.2);
                if max_lum > 0.0 { target / max_lum } else { 1.0 }
            }
        };
        
        let uniforms = BlendUniforms {
            weights: [1.0, 0.0, 0.0, 0.0],
            color: [color.0, color.1, color.2],
            norm_factor,
            grid_width: grid_w as u32,
            grid_height: grid_h as u32,
            apply_srgb: if self.use_srgb { 1 } else { 0 },
            _padding: 0,
        };
        
        // Use optimized texture pipeline (no readback!)
        self.blend_to_texture_pipeline.upload_single_grid(&self.gpu_ctx, attenuation);
        self.blend_to_texture_pipeline.upload_walls(&self.gpu_ctx, &self.wall_flat);
        self.blend_to_texture_pipeline.upload_uniforms(&self.gpu_ctx, &uniforms);
        
        // Run compute shader → writes directly to texture
        self.blend_to_texture_pipeline.dispatch_single(&self.gpu_ctx);
        
        // Apply blur if enabled
        if self.blur_level > 0 {
            self.blur_pipeline.dispatch_iterations(&self.gpu_ctx, self.blur_level);
        }
        
        // Apply wall overlay (always, to show walls after blur or when blur is off)
        self.wall_overlay_pipeline.upload_walls(&self.gpu_ctx, &self.wall_flat);
        self.wall_overlay_pipeline.dispatch(&self.gpu_ctx);
        
        // No readback needed!
    }
    
    fn read_compute_output_to_pixel_buffer(&mut self) {
        let (grid_w, grid_h) = self.config.grid_size;
        let num_pixels = grid_w * grid_h;
        let buffer_size = (num_pixels * std::mem::size_of::<u32>()) as u64;
        
        // Create staging buffer for readback
        let staging_buffer = self.gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from output buffer to staging buffer
        let mut encoder = self.gpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.blend_pipeline.output_buffer(),
            0,
            &staging_buffer,
            0,
            buffer_size,
        );
        self.gpu_ctx.queue.submit(std::iter::once(encoder.finish()));
        
        // Map and read the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        self.gpu_ctx.device.poll(wgpu::Maintain::Wait);
        
        {
            let data = buffer_slice.get_mapped_range();
            let pixels: &[u32] = bytemuck::cast_slice(&data);
            
            // Convert 0xAARRGGBB to RGBA8
            for (i, &pixel) in pixels.iter().enumerate() {
                let a = ((pixel >> 24) & 0xFF) as u8;
                let r = ((pixel >> 16) & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = (pixel & 0xFF) as u8;
                
                let idx = i * 4;
                self.pixel_buffer[idx] = r;
                self.pixel_buffer[idx + 1] = g;
                self.pixel_buffer[idx + 2] = b;
                self.pixel_buffer[idx + 3] = a;
            }
        }
        
        staging_buffer.unmap();
    }
    
    fn render_attenuation_to_buffer(&mut self, attenuation: &[f32]) {
        let (grid_w, grid_h) = self.config.grid_size;
        let color = self.current_color;
        let mode = self.current_mode;
        
        // Calculate normalization factor
        // Adjust max by source_intensity so dimmer lights actually appear dimmer
        let norm_factor = match mode {
            NormalizationMode::Standard => {
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max) / self.source_intensity;
                let max_colored = max_att * color.0.max(color.1).max(color.2);
                if max_colored > 0.0 { 1.0 / max_colored } else { 1.0 }
            }
            NormalizationMode::BrightnessLimit(limit) => 1.0 / limit,
            NormalizationMode::PerceptualLuminance(target) => {
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max) / self.source_intensity;
                let max_lum = max_att * (0.2126 * color.0 + 0.7152 * color.1 + 0.0722 * color.2);
                if max_lum > 0.0 { target / max_lum } else { 1.0 }
            }
        };
        
        // Render to pixel buffer
        for y in 0..grid_h {
            for x in 0..grid_w {
                let idx = y * grid_w + x;
                let att = attenuation[idx];
                
                let mut r = ((color.0 * att * norm_factor).min(1.0) * 255.0) as u8;
                let mut g = ((color.1 * att * norm_factor).min(1.0) * 255.0) as u8;
                let mut b = ((color.2 * att * norm_factor).min(1.0) * 255.0) as u8;
                
                // Wall tint
                if self.wall_flat[idx] {
                    r = r.max(30);
                    g = g.max(30);
                    b = b.max(30);
                }
                
                let pixel_idx = idx * 4;
                self.pixel_buffer[pixel_idx] = r;
                self.pixel_buffer[pixel_idx + 1] = g;
                self.pixel_buffer[pixel_idx + 2] = b;
                self.pixel_buffer[pixel_idx + 3] = 255;
            }
        }
    }
    
    /// Render multiple light sources, combining with max to avoid over-saturation
    fn render_multi_lights(&mut self, lights: &[(usize, usize)]) {
        let (grid_w, grid_h) = self.config.grid_size;
        let size = grid_w * grid_h;
        let decay_flat = &self.decay_flat;
        let source_intensity = self.source_intensity;
        
        // Calculate attenuation for each light in parallel
        let grids: Vec<Vec<f32>> = lights
            .par_iter()
            .map(|&(lx, ly)| {
                let sweeping = Sweeping::new();
                sweeping.calculate_flat(decay_flat, grid_w, grid_h, lx, ly, source_intensity)
            })
            .collect();
        
        // Combine all grids using max (prevents over-saturation)
        let mut combined = vec![0.0f32; size];
        for grid in &grids {
            for i in 0..size {
                combined[i] = combined[i].max(grid[i]);
            }
        }
        
        // Render the combined result
        if self.use_gpu_blend {
            self.render_single_gpu(&combined);
        } else {
            self.render_attenuation_to_buffer(&combined);
        }
    }
    
    /// Render with no light sources (just walls visible)
    fn render_no_lights(&mut self) {
        let (grid_w, grid_h) = self.config.grid_size;
        let size = grid_w * grid_h;
        
        // All-zero attenuation
        let combined = vec![0.0f32; size];
        
        if self.use_gpu_blend {
            self.render_single_gpu(&combined);
        } else {
            self.render_attenuation_to_buffer(&combined);
        }
    }
    
    fn update_and_render(&mut self) {
        let (grid_w, grid_h) = self.config.grid_size;
        let window_size = self.gpu_ctx.size;
        
        // Collect all light sources
        let mut all_lights: Vec<(usize, usize)> = self.placed_lights.clone();
        
        if let Some((mx, my)) = self.mouse_pos {
            // Convert window coords to grid coords
            let scale_x = window_size.0 as f32 / grid_w as f32;
            let scale_y = window_size.1 as f32 / grid_h as f32;
            
            let subpixel_x = mx / scale_x;
            let subpixel_y = my / scale_y;
            
            let grid_x = (subpixel_x as usize).min(grid_w - 1);
            let grid_y = (subpixel_y as usize).min(grid_h - 1);
            
            // Handle wall toggling
            if self.left_mouse_down {
                if self.last_wall_pos != Some((grid_x, grid_y)) {
                    self.toggle_wall(grid_x, grid_y);
                    self.last_wall_pos = Some((grid_x, grid_y));
                }
            }
            
            // Add mouse light if enabled
            if self.mouse_light_enabled {
                all_lights.push((grid_x, grid_y));
            }
        } else if self.mouse_light_enabled && all_lights.is_empty() {
            // No mouse position, no placed lights, render from center
            all_lights.push((grid_w / 2, grid_h / 2));
        }
        
        // Render lighting with all light sources
        if !all_lights.is_empty() {
            self.render_multi_lights(&all_lights);
        } else {
            // No lights at all - render a dark scene
            self.render_no_lights();
        }
        
        // Render to screen
        if self.use_gpu_blend {
            // GPU path: render directly from compute output texture (no readback!)
            if let Err(e) = self.display_pipeline.render_with_bind_group(&self.gpu_ctx, &self.texture_bind_group) {
                log::error!("Render error: {:?}", e);
            }
        } else {
            // CPU path: upload pixel buffer to texture
            self.display_pipeline.update_texture(
                &self.gpu_ctx,
                grid_w as u32,
                grid_h as u32,
                &self.pixel_buffer,
            );
            
            if let Err(e) = self.display_pipeline.render(&self.gpu_ctx) {
                log::error!("Render error: {:?}", e);
            }
        }
    }
}

/// Application handler for winit event loop
struct GpuViewerApp {
    config: GpuViewerConfig,
    state: Option<ViewerState>,
}

impl GpuViewerApp {
    fn new(config: GpuViewerConfig) -> Self {
        Self { config, state: None }
    }
}

impl ApplicationHandler for GpuViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        
        let (grid_w, grid_h) = self.config.grid_size;
        let window_attrs = Window::default_attributes()
            .with_title(&self.config.title)
            .with_inner_size(winit::dpi::LogicalSize::new(
                grid_w as f64 * 8.0,
                grid_h as f64 * 8.0,
            ));
        
        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );
        
        match ViewerState::new(window, self.config.clone()) {
            Ok(state) => {
                println!("=== GPU Interactive Light Viewer ===");
                println!("Controls:");
                println!("  Mouse      - Move light source");
                println!("  Left Click - Toggle wall");
                println!("  Space      - Place light at mouse position");
                println!("  M          - Toggle mouse light ON/OFF");
                println!("  X          - Clear all placed lights");
                println!("  1/2/3      - Normalization: Standard/OSB/Perceptual");
                println!("  R/G/B/Y/W  - Color: Red/Green/Blue/Yellow/White");
                println!("  +/-        - Adjust decay rate");
                println!("  T          - Toggle subpixel blending ON/OFF");
                println!("  P          - Toggle GPU compute ON/OFF");
                println!("  S          - Toggle sRGB/Linear output");
                println!("  F          - Toggle filter: Nearest/Linear (smooth)");
                println!("  L          - Cycle blur: OFF/Light/Medium/Heavy");
                println!("  [/]        - Adjust source intensity");
                println!("  V          - Generate random cave");
                println!("  C          - Clear walls");
                println!("  ESC        - Exit");
                println!();
                
                self.state = Some(state);
            }
            Err(e) => {
                log::error!("Failed to create viewer state: {}", e);
                event_loop.exit();
            }
        }
    }
    
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match &mut self.state {
            Some(s) => s,
            None => return,
        };
        
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            
            WindowEvent::Resized(size) => {
                state.gpu_ctx.resize((size.width, size.height));
            }
            
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(key),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => {
                match key {
                    KeyCode::Escape => event_loop.exit(),
                    
                    KeyCode::Digit1 => {
                        state.current_mode = NormalizationMode::Standard;
                        println!("Mode: Standard");
                    }
                    KeyCode::Digit2 => {
                        state.current_mode = NormalizationMode::BrightnessLimit(1.0);
                        println!("Mode: OSB (BrightnessLimit 1.0)");
                    }
                    KeyCode::Digit3 => {
                        state.current_mode = NormalizationMode::PerceptualLuminance(1.0);
                        println!("Mode: Perceptual Luminance");
                    }
                    
                    KeyCode::KeyR => {
                        state.current_color = (1.0, 0.0, 0.0);
                        println!("Color: Red");
                    }
                    KeyCode::KeyG => {
                        state.current_color = (0.0, 1.0, 0.0);
                        println!("Color: Green");
                    }
                    KeyCode::KeyB => {
                        state.current_color = (0.0, 0.0, 1.0);
                        println!("Color: Blue");
                    }
                    KeyCode::KeyY => {
                        state.current_color = (1.0, 1.0, 0.0);
                        println!("Color: Yellow");
                    }
                    KeyCode::KeyW => {
                        state.current_color = (1.0, 1.0, 1.0);
                        println!("Color: White");
                    }
                    
                    KeyCode::Equal | KeyCode::NumpadAdd => {
                        state.config.base_decay = (state.config.base_decay + 0.02).min(0.5);
                        state.update_decay_grid();
                        println!("Decay: {:.2}", state.config.base_decay);
                    }
                    KeyCode::Minus | KeyCode::NumpadSubtract => {
                        state.config.base_decay = (state.config.base_decay - 0.02).max(0.01);
                        state.update_decay_grid();
                        println!("Decay: {:.2}", state.config.base_decay);
                    }
                    
                    KeyCode::KeyT => {
                        state.subpixel_enabled = !state.subpixel_enabled;
                        if state.subpixel_enabled {
                            println!("Subpixel blending: ON");
                        } else {
                            println!("Subpixel blending: OFF");
                        }
                    }
                    
                    KeyCode::KeyP => {
                        state.use_gpu_blend = !state.use_gpu_blend;
                        if state.use_gpu_blend {
                            println!("GPU compute: ON");
                        } else {
                            println!("GPU compute: OFF (CPU fallback)");
                        }
                    }
                    
                    KeyCode::KeyS => {
                        state.use_srgb = !state.use_srgb;
                        if state.use_srgb {
                            println!("Output: sRGB (gamma encoded, tighter falloff)");
                        } else {
                            println!("Output: Linear (physically correct, wider falloff)");
                        }
                    }
                    
                    KeyCode::KeyF => {
                        state.use_linear_filter = !state.use_linear_filter;
                        state.update_texture_bind_group();
                        if state.use_linear_filter {
                            println!("Filter: Linear (smooth)");
                        } else {
                            println!("Filter: Nearest (crisp pixels)");
                        }
                    }
                    
                    KeyCode::KeyL => {
                        // Cycle blur level: 0 -> 1 -> 2 -> 3 -> 0
                        state.blur_level = (state.blur_level + 1) % 4;
                        match state.blur_level {
                            0 => println!("Blur: OFF"),
                            1 => println!("Blur: Light (1 pass)"),
                            2 => println!("Blur: Medium (2 passes)"),
                            3 => println!("Blur: Heavy (3 passes)"),
                            _ => {}
                        }
                    }
                    
                    KeyCode::KeyC => {
                        state.clear_walls();
                        println!("Walls cleared");
                    }
                    
                    KeyCode::KeyV => {
                        state.generate_cave();
                        println!("Generated cave pattern");
                    }
                    
                    KeyCode::Space => {
                        // Place a light at current mouse position
                        if let Some((mx, my)) = state.mouse_pos {
                            let (grid_w, grid_h) = state.config.grid_size;
                            let window_size = state.gpu_ctx.size;
                            let scale_x = window_size.0 as f32 / grid_w as f32;
                            let scale_y = window_size.1 as f32 / grid_h as f32;
                            let grid_x = (mx / scale_x) as usize;
                            let grid_y = (my / scale_y) as usize;
                            let grid_x = grid_x.min(grid_w - 1);
                            let grid_y = grid_y.min(grid_h - 1);
                            state.place_light(grid_x, grid_y);
                        }
                    }
                    
                    KeyCode::KeyM => {
                        state.mouse_light_enabled = !state.mouse_light_enabled;
                        if state.mouse_light_enabled {
                            println!("Mouse light: ON");
                        } else {
                            println!("Mouse light: OFF");
                        }
                    }
                    
                    KeyCode::KeyX => {
                        state.clear_lights();
                    }
                    
                    KeyCode::BracketLeft => {
                        state.source_intensity = (state.source_intensity - 0.1).max(0.1);
                        println!("Source intensity: {:.1}", state.source_intensity);
                    }
                    
                    KeyCode::BracketRight => {
                        state.source_intensity = (state.source_intensity + 0.1).min(2.0);
                        println!("Source intensity: {:.1}", state.source_intensity);
                    }
                    
                    _ => {}
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_pos = Some((position.x as f32, position.y as f32));
            }
            
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                match button {
                    MouseButton::Left => {
                        state.left_mouse_down = btn_state == ElementState::Pressed;
                        if !state.left_mouse_down {
                            state.last_wall_pos = None;
                        }
                    }
                    MouseButton::Right if btn_state == ElementState::Pressed => {
                        state.clear_walls();
                    }
                    _ => {}
                }
            }
            
            WindowEvent::RedrawRequested => {
                state.update_and_render();
            }
            
            _ => {}
        }
    }
    
    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            // Request continuous redraw for smooth updates
            state.gpu_ctx.request_redraw();
        }
    }
}

/// Run the GPU viewer
pub fn run_gpu_viewer(config: GpuViewerConfig) -> Result<(), String> {
    env_logger::init();
    
    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {}", e))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    
    let mut app = GpuViewerApp::new(config);
    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Event loop error: {}", e))?;
    
    Ok(())
}
