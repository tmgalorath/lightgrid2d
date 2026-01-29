// Display shader - renders a texture to a full-screen quad

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
};

// Vertex shader - generates a full-screen triangle (3 vertices cover the screen)
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate full-screen triangle coordinates
    // vertex 0: (-1, -1), vertex 1: (3, -1), vertex 2: (-1, 3)
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    
    // Texture coordinates (0,0) to (1,1), flipped Y for correct orientation
    out.tex_coord = vec2<f32>(
        (x + 1.0) * 0.5,
        (1.0 - y) * 0.5
    );
    
    return out;
}

// Texture and sampler bindings
@group(0) @binding(0) var t_diffuse: texture_2d<f32>;
@group(0) @binding(1) var s_diffuse: sampler;

// Fragment shader - samples the texture
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coord);
}
