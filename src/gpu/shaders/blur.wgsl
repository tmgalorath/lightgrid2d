// Dual-Kawase Blur Shader
// Based on ARM's SIGGRAPH 2015 presentation
// https://community.arm.com/cfs-file/__key/communityserver-blogs-components-weblogfiles/00-00-00-20-66/siggraph2015_2D00_mmg_2D00_marius_2D00_notes.pdf
//
// Two passes:
// 1. Downsample: 5 taps in X pattern, write to half-res texture
// 2. Upsample: 8 taps in circle pattern, write back to full-res

struct BlurUniforms {
    texture_size: vec2<u32>,      // Size of the texture being processed
    pixel_size: vec2<f32>,        // 1.0 / texture_size (for sampling offsets)
}

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: BlurUniforms;
@group(0) @binding(3) var tex_sampler: sampler;

// Downsample pass: read from full-res, write to half-res
// Samples 5 points in an X pattern with center weighted 4x
@compute @workgroup_size(8, 8)
fn downsample_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    // Output is half resolution
    let out_width = uniforms.texture_size.x / 2u;
    let out_height = uniforms.texture_size.y / 2u;
    
    if (x >= out_width || y >= out_height) {
        return;
    }
    
    // Sample coordinates in the input texture (normalized 0-1)
    // Map output pixel to center of 2x2 input region
    let uv = vec2<f32>(
        (f32(x) * 2.0 + 1.0) / f32(uniforms.texture_size.x),
        (f32(y) * 2.0 + 1.0) / f32(uniforms.texture_size.y)
    );
    
    let offset = uniforms.pixel_size;
    
    // 5-tap X pattern: center (4x weight) + 4 corners
    var sum = textureSampleLevel(input_texture, tex_sampler, uv, 0.0) * 4.0;
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-offset.x, -offset.y), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( offset.x, -offset.y), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-offset.x,  offset.y), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( offset.x,  offset.y), 0.0);
    
    let result = sum * (1.0 / 8.0);
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}

// Upsample pass: read from half-res, write to full-res
// Samples 8 points in a circle pattern
@compute @workgroup_size(8, 8)
fn upsample_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= uniforms.texture_size.x || y >= uniforms.texture_size.y) {
        return;
    }
    
    // Sample coordinates in the half-res input texture
    let uv = vec2<f32>(
        (f32(x) + 0.5) / f32(uniforms.texture_size.x),
        (f32(y) + 0.5) / f32(uniforms.texture_size.y)
    );
    
    // Offset is relative to the half-res texture (so 2x the full-res pixel size)
    let offset = uniforms.pixel_size * 2.0;
    
    // 8-tap circle pattern with varying weights
    // Cardinal directions (weight 2) + diagonals (weight 1)
    var sum = vec4<f32>(0.0);
    
    // Cardinals (weight 2 each)
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-offset.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( offset.x * 2.0, 0.0), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(0.0, -offset.y * 2.0), 0.0);
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(0.0,  offset.y * 2.0), 0.0);
    
    // Diagonals (weight 2 each, at 1x offset)
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-offset.x,  offset.y), 0.0) * 2.0;
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( offset.x,  offset.y), 0.0) * 2.0;
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>(-offset.x, -offset.y), 0.0) * 2.0;
    sum += textureSampleLevel(input_texture, tex_sampler, uv + vec2<f32>( offset.x, -offset.y), 0.0) * 2.0;
    
    let result = sum * (1.0 / 12.0);
    
    textureStore(output_texture, vec2<i32>(i32(x), i32(y)), result);
}
