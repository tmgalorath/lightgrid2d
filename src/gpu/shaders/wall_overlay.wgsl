// Wall overlay shader - applies wall tint AFTER blur
// This keeps walls crisp and visible while lighting is blurred

struct WallUniforms {
    grid_width: u32,
    grid_height: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, read_write>;
@group(0) @binding(1) var<storage, read> walls: array<u32>;  // Wall flags (packed bits)
@group(0) @binding(2) var<uniform> uniforms: WallUniforms;

@compute @workgroup_size(8, 8)
fn wall_overlay_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.grid_width || y >= uniforms.grid_height) {
        return;
    }
    
    let idx = y * uniforms.grid_width + x;
    
    // Check if this cell is a wall
    let wall_idx = idx / 32u;
    let wall_bit = idx % 32u;
    let is_wall = (walls[wall_idx] >> wall_bit) & 1u;
    
    if (is_wall == 1u) {
        // Read current pixel
        let current = textureLoad(output_texture, vec2<i32>(i32(x), i32(y)));
        
        // Apply wall tint - ensure minimum brightness
        let wall_min = 30.0 / 255.0;
        let r = max(current.r, wall_min);
        let g = max(current.g, wall_min);
        let b = max(current.b, wall_min);
        
        textureStore(output_texture, vec2<i32>(i32(x), i32(y)), vec4<f32>(r, g, b, 1.0));
    }
}
