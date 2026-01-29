# lightgrid2d

A fast 2D light propagation library in Rust, using a sweeping algorithm for calculating light attenuation through grids.

![Rust](https://img.shields.io/badge/Rust-2024%20Edition-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **Sweeping Algorithm** – Bidirectional sweep-based light propagation with O(n) complexity
- **Colored Lights** – Full RGBA support with intensity and multi-light blending
- **Multiple Normalization Modes** – Standard, brightness-limited (OpenStarbound-style), and perceptual luminance
- **Subpixel Blending** – Smooth light movement with bilinear interpolation
- **Interactive Viewer** – Real-time visualization with mouse-controlled light sources
- **Parallelized** – Uses [rayon](https://github.com/rayon-rs/rayon) for parallel forward/reverse sweeps
- **Flat Memory Layout** – Cache-friendly `Vec<f32>` for optimal performance

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
lighting_test = { path = "." }
```

## Quick Start

```rust
use lighting_test::{Sweeping, flatten_grid};

// Create a decay grid (0.0 = transparent, 1.0 = opaque)
let width = 50;
let height = 50;
let decay_flat: Vec<f32> = vec![0.1; width * height];

// Calculate light attenuation from a point source
let sweeping = Sweeping::new();
let attenuation = sweeping.calculate_flat(&decay_flat, width, height, 25, 25);
// attenuation is Vec<f32> in row-major order: index = y * width + x

// Access a specific cell
let light_at_10_20 = attenuation[20 * width + 10];
```

## Usage

### Running the Interactive Viewer

```bash
cargo run --release -- --interactive
```

**Controls:**
| Key | Action |
|-----|--------|
| Mouse | Move light source |
| Left Click | Toggle wall |
| Right Click | Clear all walls |
| `1` / `2` / `3` | Normalization: Standard / OSB / Perceptual |
| `R` / `G` / `B` / `Y` / `W` | Color: Red / Green / Blue / Yellow / White |
| `+` / `-` | Adjust decay rate |
| `T` | Toggle subpixel blending |
| `C` | Clear walls |
| `ESC` | Exit |

### Running Benchmarks

```bash
cargo run --release -- --benchmark
```

## Architecture

The library is organized in layers:

1. **Attenuation** (`src/attenuation/`) – Pure geometry/physics calculation of light propagation
2. **Color** (`src/color.rs`) – Applies colors to attenuation grids, blends multiple lights
3. **Render** (`src/render.rs`) – Normalization and output (PPM export, display buffers)
4. **Interactive** (`src/interactive/`) – Real-time minifb-based viewer

## Algorithm

The sweeping algorithm propagates light through the grid using four directional passes:

1. **Forward Pass:** Top-left → Bottom-right, then Bottom-right → Top-left, Down, Up
2. **Reverse Pass:** Up, Down, Bottom-right → Top-left, Top-left → Bottom-right
3. **Merge:** `max(forward, reverse)` for symmetric light distribution

This approach is significantly faster than traditional flood-fill or raycast methods for 2D grids.

## Dependencies

- [rayon](https://crates.io/crates/rayon) – Parallel iteration
- [minifb](https://crates.io/crates/minifb) – Windowing for interactive viewer
- [pixels](https://crates.io/crates/pixels) + [winit](https://crates.io/crates/winit) – GPU-accelerated rendering (optional)

## License

MIT License – see [LICENSE](LICENSE) for details.
