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
use crate::gpu::{GpuContext, DisplayPipeline};
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
    
    // Lighting state
    decay_flat: Vec<f32>,
    wall_flat: Vec<bool>,
    pixel_buffer: Vec<u8>, // RGBA8 for GPU upload
    
    // Interaction state
    current_color: (f32, f32, f32),
    current_mode: NormalizationMode,
    subpixel_enabled: bool,
    mouse_pos: Option<(f32, f32)>,
    left_mouse_down: bool,
    last_wall_pos: Option<(usize, usize)>,
}

impl ViewerState {
    fn new(window: Arc<Window>, config: GpuViewerConfig) -> Result<Self, String> {
        let gpu_ctx = GpuContext::new(window)?;
        let display_pipeline = DisplayPipeline::new(&gpu_ctx);
        
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
            decay_flat,
            wall_flat,
            pixel_buffer,
            current_color,
            current_mode,
            subpixel_enabled: true,
            mouse_pos: None,
            left_mouse_down: false,
            last_wall_pos: None,
        })
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
        let attenuation = sweeping.calculate_flat(&self.decay_flat, grid_w, grid_h, light_x, light_y);
        
        self.render_attenuation_to_buffer(&attenuation);
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
        
        let grids: Vec<Vec<f32>> = positions
            .par_iter()
            .map(|&(lx, ly)| {
                let sweeping = Sweeping::new();
                sweeping.calculate_flat(decay_flat, grid_w, grid_h, lx, ly)
            })
            .collect();
        
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
    
    fn render_attenuation_to_buffer(&mut self, attenuation: &[f32]) {
        let (grid_w, grid_h) = self.config.grid_size;
        let color = self.current_color;
        let mode = self.current_mode;
        
        // Calculate normalization factor
        let norm_factor = match mode {
            NormalizationMode::Standard => {
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max);
                let max_colored = max_att * color.0.max(color.1).max(color.2);
                if max_colored > 0.0 { 1.0 / max_colored } else { 1.0 }
            }
            NormalizationMode::BrightnessLimit(limit) => 1.0 / limit,
            NormalizationMode::PerceptualLuminance(target) => {
                let max_att = attenuation.iter().cloned().fold(0.0f32, f32::max);
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
    
    fn update_and_render(&mut self) {
        let (grid_w, grid_h) = self.config.grid_size;
        let window_size = self.gpu_ctx.size;
        
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
            
            // Render lighting
            if self.subpixel_enabled {
                self.render_lighting_bilinear(subpixel_x, subpixel_y);
            } else {
                let render_x = (subpixel_x.round() as usize).min(grid_w - 1);
                let render_y = (subpixel_y.round() as usize).min(grid_h - 1);
                self.render_lighting(render_x, render_y);
            }
        } else {
            // No mouse position, render from center
            self.render_lighting(grid_w / 2, grid_h / 2);
        }
        
        // Upload to GPU and render
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
                println!("  1/2/3      - Normalization: Standard/OSB/Perceptual");
                println!("  R/G/B/Y/W  - Color: Red/Green/Blue/Yellow/White");
                println!("  +/-        - Adjust decay rate");
                println!("  T          - Toggle subpixel blending ON/OFF");
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
                    
                    KeyCode::KeyC => {
                        state.clear_walls();
                        println!("Walls cleared");
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
