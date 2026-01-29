// Compute shader for bilinear blending of 4 attenuation grids
// Writes directly to a storage texture (no CPU readback needed)
// Supports both LINEAR and "sRGB-style" output modes

struct Uniforms {
    weights: vec4<f32>,      // w00, w10, w01, w11 (bilinear weights)
    color: vec3<f32>,        // RGB light color
    norm_factor: f32,        // Normalization multiplier
    grid_width: u32,
    grid_height: u32,
    apply_srgb: u32,         // 1 = apply sRGB-style gamma (tighter falloff), 0 = linear
    _padding: u32,
}

// Apply sRGB-style gamma curve (power 2.2)
// This simulates what happened when we used Rgba8UnormSrgb texture format:
// - Mid-tones get crushed darker
// - Light falloff appears tighter/shorter
// - More "Terraria-like" classic torch look
fn apply_srgb_gamma(linear: f32) -> f32 {
    return pow(max(linear, 0.0), 2.2);
}

@group(0) @binding(0) var<storage, read> grid0: array<f32>;
@group(0) @binding(1) var<storage, read> grid1: array<f32>;
@group(0) @binding(2) var<storage, read> grid2: array<f32>;
@group(0) @binding(3) var<storage, read> grid3: array<f32>;
@group(0) @binding(4) var<storage, read> walls: array<u32>;  // Wall flags (packed bits)
@group(0) @binding(5) var output_texture: texture_storage_2d<rgba8unorm, write>;
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
    
    // Apply color and normalization (linear space)
    var r = clamp(uniforms.color.x * att * uniforms.norm_factor, 0.0, 1.0);
    var g = clamp(uniforms.color.y * att * uniforms.norm_factor, 0.0, 1.0);
    var b = clamp(uniforms.color.z * att * uniforms.norm_factor, 0.0, 1.0);
    
    // Wall tint is now applied AFTER blur (see wall_overlay.wgsl)
    
    // Apply sRGB-style gamma if enabled (makes light falloff appear tighter)
    if (uniforms.apply_srgb == 1u) {
        r = apply_srgb_gamma(r);
        g = apply_srgb_gamma(g);
        b = apply_srgb_gamma(b);
    }
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(r, g, b, 1.0));
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
    
    // Use only grid0
    let att = grid0[idx];
    
    var r = clamp(uniforms.color.x * att * uniforms.norm_factor, 0.0, 1.0);
    var g = clamp(uniforms.color.y * att * uniforms.norm_factor, 0.0, 1.0);
    var b = clamp(uniforms.color.z * att * uniforms.norm_factor, 0.0, 1.0);
    
    // Wall tint is now applied AFTER blur (see wall_overlay.wgsl)
    
    // Apply sRGB-style gamma if enabled
    if (uniforms.apply_srgb == 1u) {
        r = apply_srgb_gamma(r);
        g = apply_srgb_gamma(g);
        b = apply_srgb_gamma(b);
    }
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(r, g, b, 1.0));
}
