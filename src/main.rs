mod attenuation;
mod color;
mod gpu;
mod interactive;
mod render;

#[cfg(test)]
mod tests;

// Re-export public API
pub use attenuation::{Sweeping, flatten_grid, attenuation_to_string};
pub use color::{RGBA, ColoredLight, apply_light_color, blend_lights, rgba_grid_to_string};
pub use render::{save_ppm, save_ppm_with_walls, normalize_grid, normalize_grid_osb, normalize_grid_perceptual, NormalizationMode};
pub use interactive::{InteractiveViewer, ViewerConfig, run_gpu_viewer, GpuViewerConfig};

fn main() {
    // Check for command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--interactive" {
        run_interactive();
    } else if args.len() > 1 && args[1] == "--gpu" {
        run_gpu_interactive();
    } else if args.len() > 1 && args[1] == "--benchmark" {
        run_benchmark();
    } else if args.len() > 1 && args[1] == "--benchmark-blend" {
        run_blend_benchmark();
    } else {
        println!("Lighting Test");
        println!("Run with --interactive for minifb viewer (CPU)");
        println!("Run with --gpu for wgpu viewer (GPU display)");
        println!("Run with --benchmark to test sweeping performance");
        println!("Run with --benchmark-blend to compare CPU vs GPU blending");
    }
}

fn run_benchmark() {
    use std::time::Instant;
    use rayon::prelude::*;
    
    println!("=== Sweeping Algorithm Benchmark ===\n");
    
    // Test parameters
    let sizes = [(50, 50), (100, 100), (200, 200)];
    let iterations = 20;
    let decay_value = 0.1f32;
    
    for (width, height) in sizes {
        println!("Grid size: {}x{}", width, height);
        println!("-----------------------");
        
        // Create decay grid
        let decay_nested: Vec<Vec<f32>> = vec![vec![decay_value; height]; width];
        let decay_flat: Vec<f32> = flatten_grid(&decay_nested);
        
        let light_x = width / 2;
        let light_y = height / 2;
        
        // Benchmark flat memory API
        let sweeping = Sweeping::new();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sweeping.calculate_flat(&decay_flat, width, height, light_x, light_y, 1.0);
        }
        let elapsed_flat = start.elapsed();
        let avg_flat_ms = elapsed_flat.as_secs_f64() * 1000.0 / iterations as f64;
        
        println!("  Time: {:.3} ms/iter", avg_flat_ms);
        println!();
    }
    
    // Benchmark 4-grid bilinear scenario (the real use case)
    println!("=== 4-Grid Bilinear Scenario ===");
    println!("(Simulates subpixel light movement)\n");
    
    let (width, height) = (100, 100);
    let decay_nested: Vec<Vec<f32>> = vec![vec![decay_value; height]; width];
    let decay_flat: Vec<f32> = flatten_grid(&decay_nested);
    
    let positions = [(49, 49), (50, 49), (49, 50), (50, 50)];
    
    // Parallel with rayon + flat memory
    let start = Instant::now();
    for _ in 0..iterations {
        let _grids: Vec<_> = positions.par_iter().map(|&(x, y)| {
            let sweeping = Sweeping::new();
            sweeping.calculate_flat(&decay_flat, width, height, x, y, 1.0)
        }).collect();
    }
    let elapsed_parallel = start.elapsed();
    let avg_parallel_ms = elapsed_parallel.as_secs_f64() * 1000.0 / iterations as f64;
    
    println!("Grid size: {}x{}, 4 grids (parallel)", width, height);
    println!("-----------------------");
    println!("  Time: {:.3} ms/iter", avg_parallel_ms);
    println!();
    
    // FPS estimate
    let fps_parallel = 1000.0 / avg_parallel_ms;
    println!("Estimated max FPS (lighting only): {:.1} FPS", fps_parallel);
}

fn run_interactive() {
    let config = ViewerConfig::default();

    match InteractiveViewer::new(config) {
        Ok(mut viewer) => {
            if let Err(e) = viewer.run() {
                eprintln!("Error: {}", e);
            }
        }
        Err(e) => {
            eprintln!("Failed to create viewer: {}", e);
        }
    }
}

fn run_gpu_interactive() {
    let config = GpuViewerConfig::default();
    
    if let Err(e) = run_gpu_viewer(config) {
        eprintln!("GPU viewer error: {}", e);
    }
}

fn run_blend_benchmark() {
    use std::time::Instant;
    use rayon::prelude::*;
    use gpu::{GpuContext, BlendPipeline, BlendToTexturePipeline, BlendUniforms};
    use std::sync::Arc;
    use winit::event_loop::EventLoop;
    use winit::window::Window;
    
    println!("=== CPU vs GPU Blending Benchmark ===\n");
    
    // Test parameters
    let sizes = [(100, 100), (200, 200), (400, 400)];
    let iterations = 50;
    let warmup = 10;
    let decay_value = 0.1f32;
    let weights = [0.25f32, 0.25, 0.25, 0.25]; // Equal bilinear weights
    let color = (1.0f32, 0.8, 0.4);
    
    // Create a hidden window for GPU context
    println!("Initializing GPU...");
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = Arc::new(
        event_loop
            .create_window(
                Window::default_attributes()
                    .with_visible(false)
                    .with_inner_size(winit::dpi::PhysicalSize::new(100, 100)),
            )
            .expect("Failed to create window"),
    );
    
    let gpu_ctx = GpuContext::new(window).expect("Failed to create GPU context");
    println!("GPU initialized: using device\n");
    
    for (width, height) in sizes {
        println!("Grid size: {}x{} ({} cells)", width, height, width * height);
        println!("============================================");
        
        // Create test data - 4 attenuation grids
        let decay_flat: Vec<f32> = vec![decay_value; width * height];
        let positions = [
            (width / 2, height / 2),
            (width / 2 + 1, height / 2),
            (width / 2, height / 2 + 1),
            (width / 2 + 1, height / 2 + 1),
        ];
        
        // Pre-calculate the 4 grids
        let grids: Vec<Vec<f32>> = positions
            .par_iter()
            .map(|&(x, y)| {
                Sweeping::new().calculate_flat(&decay_flat, width, height, x, y, 1.0)
            })
            .collect();
        
        let walls: Vec<bool> = vec![false; width * height];
        let size = width * height;
        
        // ==================== CPU Blend ====================
        // Warmup
        for _ in 0..warmup {
            let mut blended = vec![0.0f32; size];
            for i in 0..size {
                blended[i] = grids[0][i] * weights[0]
                           + grids[1][i] * weights[1]
                           + grids[2][i] * weights[2]
                           + grids[3][i] * weights[3];
            }
            // Color + normalize
            let max_val = blended.iter().cloned().fold(0.0f32, f32::max);
            let norm = if max_val > 0.0 { 1.0 / max_val } else { 1.0 };
            let mut pixels = vec![0u8; size * 4];
            for i in 0..size {
                let att = blended[i];
                pixels[i * 4] = ((color.0 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 1] = ((color.1 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 2] = ((color.2 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 3] = 255;
            }
            std::hint::black_box(&pixels);
        }
        
        // Benchmark CPU
        let start = Instant::now();
        for _ in 0..iterations {
            let mut blended = vec![0.0f32; size];
            for i in 0..size {
                blended[i] = grids[0][i] * weights[0]
                           + grids[1][i] * weights[1]
                           + grids[2][i] * weights[2]
                           + grids[3][i] * weights[3];
            }
            let max_val = blended.iter().cloned().fold(0.0f32, f32::max);
            let norm = if max_val > 0.0 { 1.0 / max_val } else { 1.0 };
            let mut pixels = vec![0u8; size * 4];
            for i in 0..size {
                let att = blended[i];
                pixels[i * 4] = ((color.0 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 1] = ((color.1 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 2] = ((color.2 * att * norm).min(1.0) * 255.0) as u8;
                pixels[i * 4 + 3] = 255;
            }
            std::hint::black_box(&pixels);
        }
        let cpu_elapsed = start.elapsed();
        let cpu_avg_us = cpu_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        
        // ==================== GPU Blend (with readback - old way) ====================
        let blend_pipeline = BlendPipeline::new(&gpu_ctx, width as u32, height as u32);
        
        // Pre-calculate norm_factor (same as we do in viewer)
        let max_blended: f32 = (0..size)
            .map(|i| {
                grids[0][i] * weights[0]
                + grids[1][i] * weights[1]
                + grids[2][i] * weights[2]
                + grids[3][i] * weights[3]
            })
            .fold(0.0f32, f32::max);
        let norm_factor = if max_blended > 0.0 { 1.0 / max_blended } else { 1.0 };
        
        let uniforms = BlendUniforms {
            weights,
            color: [color.0, color.1, color.2],
            norm_factor,
            grid_width: width as u32,
            grid_height: height as u32,
            apply_srgb: 0,  // Linear for benchmarks
            _padding: 0,
        };
        
        // Benchmark GPU with readback (old way)
        let staging_size = (size * std::mem::size_of::<u32>()) as u64;
        let staging_buffer = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Warmup
        for _ in 0..warmup {
            blend_pipeline.upload_grids(&gpu_ctx, &[&grids[0], &grids[1], &grids[2], &grids[3]]);
            blend_pipeline.upload_walls(&gpu_ctx, &walls);
            blend_pipeline.upload_uniforms(&gpu_ctx, &uniforms);
            blend_pipeline.dispatch_blend(&gpu_ctx);
            
            let mut encoder = gpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(blend_pipeline.output_buffer(), 0, &staging_buffer, 0, staging_size);
            gpu_ctx.queue.submit(std::iter::once(encoder.finish()));
            
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu_ctx.device.poll(wgpu::Maintain::Wait);
            { let _ = buffer_slice.get_mapped_range(); }
            staging_buffer.unmap();
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            blend_pipeline.upload_grids(&gpu_ctx, &[&grids[0], &grids[1], &grids[2], &grids[3]]);
            blend_pipeline.upload_walls(&gpu_ctx, &walls);
            blend_pipeline.upload_uniforms(&gpu_ctx, &uniforms);
            blend_pipeline.dispatch_blend(&gpu_ctx);
            
            // Copy to staging
            let mut encoder = gpu_ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            encoder.copy_buffer_to_buffer(blend_pipeline.output_buffer(), 0, &staging_buffer, 0, staging_size);
            gpu_ctx.queue.submit(std::iter::once(encoder.finish()));
            
            // Map and read
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
            gpu_ctx.device.poll(wgpu::Maintain::Wait);
            {
                let _data = buffer_slice.get_mapped_range();
                std::hint::black_box(&_data);
            }
            staging_buffer.unmap();
        }
        let gpu_readback_elapsed = start.elapsed();
        let gpu_readback_avg_us = gpu_readback_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        
        // ==================== GPU Texture (no readback - new optimized way) ====================
        let texture_pipeline = BlendToTexturePipeline::new(&gpu_ctx, width as u32, height as u32);
        
        // Warmup
        for _ in 0..warmup {
            texture_pipeline.upload_grids(&gpu_ctx, &[&grids[0], &grids[1], &grids[2], &grids[3]]);
            texture_pipeline.upload_walls(&gpu_ctx, &walls);
            texture_pipeline.upload_uniforms(&gpu_ctx, &uniforms);
            texture_pipeline.dispatch_blend(&gpu_ctx);
            gpu_ctx.device.poll(wgpu::Maintain::Wait);
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            texture_pipeline.upload_grids(&gpu_ctx, &[&grids[0], &grids[1], &grids[2], &grids[3]]);
            texture_pipeline.upload_walls(&gpu_ctx, &walls);
            texture_pipeline.upload_uniforms(&gpu_ctx, &uniforms);
            texture_pipeline.dispatch_blend(&gpu_ctx);
        }
        gpu_ctx.device.poll(wgpu::Maintain::Wait);
        let gpu_texture_elapsed = start.elapsed();
        let gpu_texture_avg_us = gpu_texture_elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        
        // Results
        println!("  CPU blend+color:          {:>8.1} µs", cpu_avg_us);
        println!("  GPU w/ readback (old):    {:>8.1} µs", gpu_readback_avg_us);
        println!("  GPU to texture (new):     {:>8.1} µs", gpu_texture_avg_us);
        println!();
        
        let speedup_readback = cpu_avg_us / gpu_readback_avg_us;
        let speedup_texture = cpu_avg_us / gpu_texture_avg_us;
        
        if speedup_readback > 1.0 {
            println!("  GPU (old) is {:.2}x faster than CPU", speedup_readback);
        } else {
            println!("  CPU is {:.2}x faster than GPU (old)", 1.0 / speedup_readback);
        }
        
        if speedup_texture > 1.0 {
            println!("  GPU (new) is {:.2}x faster than CPU", speedup_texture);
        } else {
            println!("  CPU is {:.2}x faster than GPU (new)", 1.0 / speedup_texture);
        }
        
        if gpu_readback_avg_us > 0.0 {
            let readback_savings = (gpu_readback_avg_us - gpu_texture_avg_us) / gpu_readback_avg_us * 100.0;
            println!("  Texture saves {:.1}% vs readback", readback_savings);
        }
        
        println!();
    }
    
    println!("Note: 'GPU to texture' writes directly to GPU texture,");
    println!("      avoiding CPU readback. This is used for display.");
}