//! Tests for the lighting system

use crate::{Sweeping, flatten_grid, ColoredLight, apply_light_color, blend_lights, rgba_grid_to_string};

/// Convert flat Vec<f32> back to Vec<Vec<f32>> (for test convenience)
fn unflatten_grid(flat: &[f32], width: usize, height: usize) -> Vec<Vec<f32>> {
    let mut grid = vec![vec![0.0f32; height]; width];
    for x in 0..width {
        for y in 0..height {
            grid[x][y] = flat[y * width + x];
        }
    }
    grid
}

// Helper to create default sweeping algorithm
fn calculate_light_attenuation(decay_grid: &Vec<Vec<f32>>, light_pos: (usize, usize)) -> Vec<Vec<f32>> {
    let width = decay_grid.len();
    let height = decay_grid[0].len();
    let decay_flat = flatten_grid(decay_grid);
    let result_flat = Sweeping::new().calculate_flat(&decay_flat, width, height, light_pos.0, light_pos.1);
    unflatten_grid(&result_flat, width, height)
}

#[test]
fn test_main() {
    crate::main();
}

#[test]
fn test_attenuation_basic() {
    let decay_grid = vec![vec![0.1f32; 5]; 5];
    let attenuation = calculate_light_attenuation(&decay_grid, (2, 2));

    // Light source should be at 100%
    assert!((attenuation[2][2] - 1.0).abs() < 0.001);
    // Adjacent cells should be ~90% (1.0 - 0.1 decay)
    assert!(attenuation[2][1] > 0.85 && attenuation[2][1] < 0.95);
}

#[test]
fn test_attenuation_strong_decay() {
    let mut decay_grid = vec![vec![0.1f32; 5]; 5];

    // Add strong decay around light position
    decay_grid[1][2] = 0.6;
    decay_grid[3][2] = 0.6;
    decay_grid[2][1] = 0.6;
    decay_grid[2][3] = 0.6;
    decay_grid[1][1] = 0.6;

    let attenuation = calculate_light_attenuation(&decay_grid, (2, 2));

    // Light source should still be 100%
    assert!((attenuation[2][2] - 1.0).abs() < 0.001);
}

#[test]
fn test_attenuation_2x2_barriers() {
    let mut decay_grid = vec![vec![0.1f32; 9]; 9];

    // Top barrier
    decay_grid[3][1] = 0.9;
    decay_grid[4][1] = 0.9;
    decay_grid[3][2] = 0.9;
    decay_grid[4][2] = 0.9;

    // Bottom barrier
    decay_grid[3][6] = 0.9;
    decay_grid[4][6] = 0.9;
    decay_grid[3][7] = 0.9;
    decay_grid[4][7] = 0.9;

    // Left barrier
    decay_grid[1][3] = 0.9;
    decay_grid[2][3] = 0.9;
    decay_grid[1][4] = 0.9;
    decay_grid[2][4] = 0.9;

    // Right barrier
    decay_grid[6][3] = 0.9;
    decay_grid[7][3] = 0.9;
    decay_grid[6][4] = 0.9;
    decay_grid[7][4] = 0.9;

    let attenuation = calculate_light_attenuation(&decay_grid, (4, 4));

    // Light source at 100%
    assert!((attenuation[4][4] - 1.0).abs() < 0.001);
}

#[test]
fn test_asymmetric_diagonal_barrier() {
    // Layout (7x7 grid, light at center):
    //      0   1   2   3   4   5   6
    //  0 | A |   |   |   |   |   |   |  <- corner A
    //  1 |   | █ |   |   |   |   |   |  <- barrier at (1,1)
    //  3 |   |   |   | ☀ |   |   |   |  <- light at (3,3)
    //  5 |   |   |   |   |   | █ |   |  <- barrier at (5,5)
    //  6 |   |   |   |   |   |   | B |  <- corner B
    let mut decay_grid = vec![vec![0.1f32; 7]; 7];

    decay_grid[1][1] = 0.9;
    decay_grid[5][5] = 0.9;

    let attenuation = calculate_light_attenuation(&decay_grid, (3, 3));

    let corner_a = attenuation[0][0];
    let corner_b = attenuation[6][6];
    println!("Corner A (0,0): {:.4}", corner_a);
    println!("Corner B (6,6): {:.4}", corner_b);
    println!("Difference: {:.6}", (corner_a - corner_b).abs());

    assert!(
        (corner_a - corner_b).abs() < 0.001,
        "Asymmetry detected! A={:.4}, B={:.4}",
        corner_a,
        corner_b
    );
}

#[test]
fn test_cave_pocket() {
    let mut decay_grid = vec![vec![0.1f32; 10]; 10];

    // L-shaped wall
    decay_grid[6][1] = 0.9;
    decay_grid[6][2] = 0.9;
    decay_grid[6][3] = 0.9;
    decay_grid[6][4] = 0.9;
    decay_grid[6][5] = 0.9;
    decay_grid[6][6] = 0.9;

    decay_grid[3][6] = 0.9;
    decay_grid[4][6] = 0.9;
    decay_grid[5][6] = 0.9;

    decay_grid[7][1] = 0.9;
    decay_grid[7][2] = 0.9;
    decay_grid[8][2] = 0.9;
    decay_grid[7][7] = 0.9;

    let attenuation = calculate_light_attenuation(&decay_grid, (5, 4));

    let near_source = attenuation[4][4];
    let in_pocket = attenuation[8][8];
    println!("Near source (4,4): {:.4}", near_source);
    println!("In pocket (8,8): {:.4}", in_pocket);

    assert!(in_pocket > 0.0, "Pocket should receive some light");
    assert!(
        in_pocket < near_source,
        "Pocket should be dimmer than near source"
    );
}

#[test]
fn test_cup_symmetry() {
    // Cup-shaped enclosure (open at top) with centered light
    // Should produce symmetric left-right attenuation
    let mut decay_grid = vec![vec![0.1f32; 15]; 15];

    // Left wall at x=4, right wall at x=10 (interior x: 5-9, center at 7)
    // Bottom wall at y=10 (interior y: 0-9, center at... well, open top)
    for y in 3..11 {
        decay_grid[4][y] = 1.0; // Left wall
        decay_grid[10][y] = 1.0; // Right wall
    }
    for x in 4..11 {
        decay_grid[x][10] = 1.0; // Bottom wall
    }

    // Light at center (7, 6)
    let light_pos = (7, 6);
    let attenuation = calculate_light_attenuation(&decay_grid, light_pos);

    // Print the grid for debugging
    println!("Cup symmetry test - attenuation grid:");
    for y in 0..15 {
        for x in 0..15 {
            print!("{:5.2} ", attenuation[x][y]);
        }
        println!();
    }

    // Check left-right symmetry around x=7
    let mut max_diff = 0.0f32;
    let mut worst_pos = (0, 0);
    for y in 0..15 {
        for dx in 1..=7 {
            let left_x = 7usize.saturating_sub(dx);
            let right_x = (7 + dx).min(14);
            let left = attenuation[left_x][y];
            let right = attenuation[right_x][y];
            let diff = (left - right).abs();
            if diff > max_diff {
                max_diff = diff;
                worst_pos = (dx, y);
            }
        }
    }

    println!(
        "Max asymmetry: {:.4} at dx={}, y={}",
        max_diff, worst_pos.0, worst_pos.1
    );
    println!(
        "Left (x={}): {:.4}, Right (x={}): {:.4}",
        7 - worst_pos.0,
        attenuation[7 - worst_pos.0][worst_pos.1],
        7 + worst_pos.0,
        attenuation[7 + worst_pos.0][worst_pos.1]
    );

    assert!(
        max_diff < 0.01,
        "Cup should be symmetric! Max diff: {:.4} at dx={}, y={}",
        max_diff,
        worst_pos.0,
        worst_pos.1
    );
}

#[test]
fn test_colored_light() {
    let decay_grid = vec![vec![0.1f32; 5]; 5];
    let attenuation = calculate_light_attenuation(&decay_grid, (2, 2));

    // Orange torch light
    let torch = ColoredLight {
        color: (1.0, 0.6, 0.2),
        intensity: 10.0,
        position: (2, 2),
    };

    let color_grid = apply_light_color(&attenuation, &torch);

    // At light source: full intensity
    let source_color = &color_grid[2][2];
    assert!((source_color.r - 10.0).abs() < 0.001);
    assert!((source_color.g - 6.0).abs() < 0.001);
    assert!((source_color.b - 2.0).abs() < 0.001);

    println!("Colored light grid:\n{}", rgba_grid_to_string(&color_grid));
}

#[test]
fn test_multiple_lights_blend() {
    let decay_grid = vec![vec![0.1f32; 7]; 7];

    // Red light on left
    let red_attenuation = calculate_light_attenuation(&decay_grid, (1, 3));
    let red_light = ColoredLight {
        color: (1.0, 0.0, 0.0),
        intensity: 5.0,
        position: (1, 3),
    };
    let red_contribution = apply_light_color(&red_attenuation, &red_light);

    // Blue light on right
    let blue_attenuation = calculate_light_attenuation(&decay_grid, (5, 3));
    let blue_light = ColoredLight {
        color: (0.0, 0.0, 1.0),
        intensity: 5.0,
        position: (5, 3),
    };
    let blue_contribution = apply_light_color(&blue_attenuation, &blue_light);

    // Blend them
    let blended = blend_lights(&[red_contribution, blue_contribution]);

    // Center should have both red and blue (purple-ish)
    let center = &blended[3][3];
    println!(
        "Center color: R={:.2}, G={:.2}, B={:.2}",
        center.r, center.g, center.b
    );
    assert!(center.r > 0.0, "Should have red component");
    assert!(center.b > 0.0, "Should have blue component");

    // Left side should be more red
    let left = &blended[1][3];
    assert!(left.r > left.b, "Left should be more red than blue");

    // Right side should be more blue
    let right = &blended[5][3];
    assert!(right.b > right.r, "Right should be more blue than red");

    println!("Blended light grid:\n{}", rgba_grid_to_string(&blended));
}
