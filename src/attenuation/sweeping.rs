//! Sweeping neighbor algorithm for light attenuation.
//!
//! This algorithm uses four directional sweeps to propagate light through the grid,
//! taking the maximum value from already-processed neighbors at each cell.
//!
//! Key optimizations:
//! - Flat Vec<f32> for better cache locality
//! - Parallel forward/reverse passes using rayon
//! - Hand-unrolled sweep loops for maximum performance

/// Sweeping neighbor attenuation algorithm.
///
/// Uses bidirectional sweeping for symmetric light propagation:
/// - Forward pass: TL→BR, BR→TL, Down, Up
/// - Reverse pass: Up, Down, BR→TL, TL→BR
/// - Merge: max(forward, reverse) for each cell
#[derive(Debug, Clone)]
pub struct Sweeping {
    /// Multiplier for diagonal distance (default: √2 ≈ 1.414)
    pub diagonal_decay_mult: f32,
}

impl Default for Sweeping {
    fn default() -> Self {
        Sweeping {
            diagonal_decay_mult: std::f32::consts::SQRT_2,
        }
    }
}

impl Sweeping {
    /// Create a new Sweeping algorithm with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new Sweeping algorithm with custom diagonal multiplier.
    pub fn with_diagonal_mult(diagonal_decay_mult: f32) -> Self {
        Sweeping { diagonal_decay_mult }
    }

    /// Calculate attenuation with flat memory layout.
    /// Returns a flat Vec<f32> with row-major order: index = y * width + x
    pub fn calculate_flat(
        &self,
        decay_flat: &[f32],
        width: usize,
        height: usize,
        light_x: usize,
        light_y: usize,
    ) -> Vec<f32> {
        let diag = self.diagonal_decay_mult;
        let light_idx = light_y * width + light_x;

        // Run forward and reverse passes in parallel using rayon
        let (mut forward, reverse) = rayon::join(
            || {
                let mut grid = vec![0.0f32; width * height];
                grid[light_idx] = 1.0;
                run_forward_sweeps(decay_flat, &mut grid, width, height, diag);
                grid
            },
            || {
                let mut grid = vec![0.0f32; width * height];
                grid[light_idx] = 1.0;
                run_reverse_sweeps(decay_flat, &mut grid, width, height, diag);
                grid
            },
        );

        // Merge with max
        for i in 0..forward.len() {
            forward[i] = forward[i].max(reverse[i]);
        }

        forward
    }
}

// ============================================================================
// Propagation helper
// ============================================================================

/// Calculate light propagation from neighbor to current cell
#[inline]
fn propagate(att: &[f32], decay: &[f32], ni: usize, mult: f32) -> f32 {
    att[ni] * (1.0 - decay[ni] * mult)
}

// ============================================================================
// Four sweep patterns (hand-unrolled for performance)
// ============================================================================

/// Sweep from top-left to bottom-right, checking left/up/up-left neighbors
#[inline]
fn sweep_tl_to_br(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut max_prop = att[idx];
            if x > 0 {
                max_prop = max_prop.max(propagate(att, decay, idx - 1, 1.0));
            }
            if y > 0 {
                max_prop = max_prop.max(propagate(att, decay, idx - w, 1.0));
            }
            if x > 0 && y > 0 {
                max_prop = max_prop.max(propagate(att, decay, idx - w - 1, diag));
            }
            att[idx] = max_prop;
        }
    }
}

/// Sweep from bottom-right to top-left, checking right/down/down-right neighbors
#[inline]
fn sweep_br_to_tl(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = y * w + x;
            let mut max_prop = att[idx];
            if x + 1 < w {
                max_prop = max_prop.max(propagate(att, decay, idx + 1, 1.0));
            }
            if y + 1 < h {
                max_prop = max_prop.max(propagate(att, decay, idx + w, 1.0));
            }
            if x + 1 < w && y + 1 < h {
                max_prop = max_prop.max(propagate(att, decay, idx + w + 1, diag));
            }
            att[idx] = max_prop;
        }
    }
}

/// Sweep top-down, checking left/up and both upper diagonals
#[inline]
fn sweep_down(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let mut max_prop = att[idx];
            if x > 0 {
                max_prop = max_prop.max(propagate(att, decay, idx - 1, 1.0));
            }
            if y > 0 {
                max_prop = max_prop.max(propagate(att, decay, idx - w, 1.0));
                if x > 0 {
                    max_prop = max_prop.max(propagate(att, decay, idx - w - 1, diag));
                }
                if x + 1 < w {
                    max_prop = max_prop.max(propagate(att, decay, idx - w + 1, diag));
                }
            }
            att[idx] = max_prop;
        }
    }
}

/// Sweep bottom-up, checking right/down and both lower diagonals
#[inline]
fn sweep_up(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let idx = y * w + x;
            let mut max_prop = att[idx];
            if x + 1 < w {
                max_prop = max_prop.max(propagate(att, decay, idx + 1, 1.0));
            }
            if y + 1 < h {
                max_prop = max_prop.max(propagate(att, decay, idx + w, 1.0));
                if x > 0 {
                    max_prop = max_prop.max(propagate(att, decay, idx + w - 1, diag));
                }
                if x + 1 < w {
                    max_prop = max_prop.max(propagate(att, decay, idx + w + 1, diag));
                }
            }
            att[idx] = max_prop;
        }
    }
}

// ============================================================================
// Forward and reverse pass orchestration
// ============================================================================

/// Forward sweeps: TL→BR, BR→TL, Down, Up
#[inline]
fn run_forward_sweeps(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    sweep_tl_to_br(decay, att, w, h, diag);
    sweep_br_to_tl(decay, att, w, h, diag);
    sweep_down(decay, att, w, h, diag);
    sweep_up(decay, att, w, h, diag);
}

/// Reverse sweeps: Up, Down, BR→TL, TL→BR (opposite order)
#[inline]
fn run_reverse_sweeps(decay: &[f32], att: &mut [f32], w: usize, h: usize, diag: f32) {
    sweep_up(decay, att, w, h, diag);
    sweep_down(decay, att, w, h, diag);
    sweep_br_to_tl(decay, att, w, h, diag);
    sweep_tl_to_br(decay, att, w, h, diag);
}

// ============================================================================
// Grid conversion utilities
// ============================================================================

/// Convert Vec<Vec<f32>> to flat Vec<f32> (row-major: y * width + x)
pub fn flatten_grid(grid: &Vec<Vec<f32>>) -> Vec<f32> {
    let width = grid.len();
    let height = grid[0].len();
    let mut flat = vec![0.0f32; width * height];
    for x in 0..width {
        for y in 0..height {
            flat[y * width + x] = grid[x][y];
        }
    }
    flat
}
