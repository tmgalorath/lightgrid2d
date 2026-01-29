mod attenuation;
mod color;
mod interactive;
mod render;

#[cfg(test)]
mod tests;

// Re-export public API
pub use attenuation::{AttenuationAlgorithm, Sweeping, flatten_grid, attenuation_to_string};
pub use color::{RGBA, ColoredLight, apply_light_color, blend_lights, rgba_grid_to_string};
pub use render::{save_ppm, save_ppm_with_walls, normalize_grid, normalize_grid_osb, normalize_grid_perceptual, NormalizationMode};
pub use interactive::{InteractiveViewer, ViewerConfig};

fn main() {
    // Check for command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() > 1 && args[1] == "--interactive" {
        run_interactive();
    } else if args.len() > 1 && args[1] == "--benchmark" {
        run_benchmark();
    } else {
        println!("Lighting Test");
        println!("Run with --interactive for minifb viewer");
        println!("Run with --benchmark to test performance");
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
        
        // Create decay grids
        let decay_nested: Vec<Vec<f32>> = vec![vec![decay_value; height]; width];
        let decay_flat: Vec<f32> = flatten_grid(&decay_nested);
        
        let light_x = width / 2;
        let light_y = height / 2;
        
        // Benchmark nested Vec API (for compatibility)
        let sweeping = Sweeping::new();
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sweeping.calculate(&decay_nested, (light_x, light_y));
        }
        let elapsed_nested = start.elapsed();
        let avg_nested_ms = elapsed_nested.as_secs_f64() * 1000.0 / iterations as f64;
        
        // Benchmark flat memory API
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = sweeping.calculate_flat(&decay_flat, width, height, light_x, light_y);
        }
        let elapsed_flat = start.elapsed();
        let avg_flat_ms = elapsed_flat.as_secs_f64() * 1000.0 / iterations as f64;
        
        // Calculate speedup
        let speedup = avg_nested_ms / avg_flat_ms;
        
        println!("  Nested Vec API:  {:.3} ms/iter", avg_nested_ms);
        println!("  Flat memory API: {:.3} ms/iter", avg_flat_ms);
        println!("  Speedup: {:.2}x", speedup);
        println!();
    }
    
    // Benchmark 4-grid bilinear scenario (the real use case)
    println!("=== 4-Grid Bilinear Scenario ===");
    println!("(Simulates subpixel light movement)\n");
    
    let (width, height) = (100, 100);
    let decay_nested: Vec<Vec<f32>> = vec![vec![decay_value; height]; width];
    let decay_flat: Vec<f32> = flatten_grid(&decay_nested);
    
    let positions = [(49, 49), (50, 49), (49, 50), (50, 50)];
    
    // Sequential with nested Vec
    let sweeping = Sweeping::new();
    let start = Instant::now();
    for _ in 0..iterations {
        let _grids: Vec<_> = positions.iter().map(|&(x, y)| {
            sweeping.calculate(&decay_nested, (x, y))
        }).collect();
    }
    let elapsed_sequential = start.elapsed();
    let avg_sequential_ms = elapsed_sequential.as_secs_f64() * 1000.0 / iterations as f64;
    
    // Parallel with rayon + flat memory
    let start = Instant::now();
    for _ in 0..iterations {
        let _grids: Vec<_> = positions.par_iter().map(|&(x, y)| {
            let sweeping = Sweeping::new();
            sweeping.calculate_flat(&decay_flat, width, height, x, y)
        }).collect();
    }
    let elapsed_parallel = start.elapsed();
    let avg_parallel_ms = elapsed_parallel.as_secs_f64() * 1000.0 / iterations as f64;
    
    let speedup = avg_sequential_ms / avg_parallel_ms;
    
    println!("Grid size: {}x{}, 4 grids", width, height);
    println!("-----------------------");
    println!("  Sequential (nested Vec): {:.3} ms/iter", avg_sequential_ms);
    println!("  Parallel (rayon + flat): {:.3} ms/iter", avg_parallel_ms);
    println!("  Speedup: {:.2}x", speedup);
    println!();
    
    // FPS estimate
    let fps_sequential = 1000.0 / avg_sequential_ms;
    let fps_parallel = 1000.0 / avg_parallel_ms;
    println!("Estimated max FPS (lighting only):");
    println!("  Sequential: {:.1} FPS", fps_sequential);
    println!("  Parallel:   {:.1} FPS", fps_parallel);
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