// Compute shader for bilinear blending of 4 attenuation grids
// and applying color + normalization in a single GPU pass

struct Uniforms {
    weights: vec4<f32>,      // w00, w10, w01, w11 (bilinear weights)
    color: vec3<f32>,        // RGB light color
    norm_factor: f32,        // Normalization multiplier
    grid_width: u32,
    grid_height: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> grid0: array<f32>;
@group(0) @binding(1) var<storage, read> grid1: array<f32>;
@group(0) @binding(2) var<storage, read> grid2: array<f32>;
@group(0) @binding(3) var<storage, read> grid3: array<f32>;
@group(0) @binding(4) var<storage, read> walls: array<u32>;  // Wall flags (packed bits)
@group(0) @binding(5) var<storage, read_write> output: array<u32>;  // RGBA pixels
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn blend_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Bounds check
    if (x >= uniforms.grid_width || y >= uniforms.grid_height) {
        return;
    }
    
    let idx = y * uniforms.grid_width + x;
    
    // Bilinear blend of 4 attenuation grids
    let att = grid0[idx] * uniforms.weights.x
            + grid1[idx] * uniforms.weights.y
            + grid2[idx] * uniforms.weights.z
            + grid3[idx] * uniforms.weights.w;
    
    // Apply color and normalization
    var r = clamp(uniforms.color.x * att * uniforms.norm_factor, 0.0, 1.0);
    var g = clamp(uniforms.color.y * att * uniforms.norm_factor, 0.0, 1.0);
    var b = clamp(uniforms.color.z * att * uniforms.norm_factor, 0.0, 1.0);
    
    // Check if this cell is a wall (add tint)
    let wall_idx = idx / 32u;
    let wall_bit = idx % 32u;
    let is_wall = (walls[wall_idx] >> wall_bit) & 1u;
    
    if (is_wall == 1u) {
        // Add minimum brightness for walls
        r = max(r, 30.0 / 255.0);
        g = max(g, 30.0 / 255.0);
        b = max(b, 30.0 / 255.0);
    }
    
    // Pack as RGBA8 (0xAABBGGRR format for little-endian)
    let r_byte = u32(r * 255.0);
    let g_byte = u32(g * 255.0);
    let b_byte = u32(b * 255.0);
    
    // Output format: 0xAARRGGBB (matches what display expects)
    output[idx] = (255u << 24u) | (r_byte << 16u) | (g_byte << 8u) | b_byte;
}

// Single grid version (no bilinear blending)
@compute @workgroup_size(8, 8)
fn single_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.grid_width || y >= uniforms.grid_height) {
        return;
    }
    
    let idx = y * uniforms.grid_width + x;
    
    // Use only grid0 (weights.x should be 1.0, others 0.0)
    let att = grid0[idx];
    
    var r = clamp(uniforms.color.x * att * uniforms.norm_factor, 0.0, 1.0);
    var g = clamp(uniforms.color.y * att * uniforms.norm_factor, 0.0, 1.0);
    var b = clamp(uniforms.color.z * att * uniforms.norm_factor, 0.0, 1.0);
    
    let wall_idx = idx / 32u;
    let wall_bit = idx % 32u;
    let is_wall = (walls[wall_idx] >> wall_bit) & 1u;
    
    if (is_wall == 1u) {
        r = max(r, 30.0 / 255.0);
        g = max(g, 30.0 / 255.0);
        b = max(b, 30.0 / 255.0);
    }
    
    let r_byte = u32(r * 255.0);
    let g_byte = u32(g * 255.0);
    let b_byte = u32(b * 255.0);
    
    output[idx] = (255u << 24u) | (r_byte << 16u) | (g_byte << 8u) | b_byte;
}
