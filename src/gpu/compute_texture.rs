//! Optimized compute pipeline that writes directly to a texture
//! Avoids GPU→CPU→GPU round-trip for maximum performance

use super::context::GpuContext;
use super::compute::BlendUniforms;

/// Compute pipeline that writes directly to a texture (no readback)
pub struct BlendToTexturePipeline {
    blend_pipeline: wgpu::ComputePipeline,
    single_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    
    // Input buffers
    grid_buffers: [wgpu::Buffer; 4],
    wall_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    
    // Output texture (shared with display pipeline)
    output_texture: wgpu::Texture,
    output_texture_view: wgpu::TextureView,
    
    // Bind group (includes texture)
    bind_group: wgpu::BindGroup,
    
    // Grid dimensions
    grid_size: (u32, u32),
}

impl BlendToTexturePipeline {
    pub fn new(ctx: &GpuContext, grid_width: u32, grid_height: u32) -> Self {
        let device = &ctx.device;
        let grid_size = (grid_width, grid_height);
        let num_cells = (grid_width * grid_height) as usize;
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend-to-Texture Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blend_texture.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blend-to-Texture Bind Group Layout"),
            entries: &[
                // grid0
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // grid1
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // grid2
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // grid3
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // walls (packed bits)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // output texture (storage texture, write-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blend-to-Texture Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipelines
        let blend_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blend-to-Texture Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("blend_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let single_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Single-to-Texture Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("single_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create buffers
        let grid_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        let wall_buffer_size = ((num_cells + 31) / 32 * std::mem::size_of::<u32>()) as u64;
        
        let create_grid_buffer = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: grid_buffer_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        
        let grid_buffers = [
            create_grid_buffer("Grid 0 Buffer"),
            create_grid_buffer("Grid 1 Buffer"),
            create_grid_buffer("Grid 2 Buffer"),
            create_grid_buffer("Grid 3 Buffer"),
        ];
        
        let wall_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Buffer"),
            size: wall_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blend Uniform Buffer"),
            size: std::mem::size_of::<BlendUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create output texture (used for both compute write and display read)
        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width: grid_width,
                height: grid_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Rgba8Unorm for storage texture (compute write)
            // Also usable as TEXTURE_BINDING for fragment shader read
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let output_texture_view = output_texture.create_view(&Default::default());
        
        // Create bind group
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &grid_buffers,
            &wall_buffer,
            &output_texture_view,
            &uniform_buffer,
        );
        
        Self {
            blend_pipeline,
            single_pipeline,
            bind_group_layout,
            grid_buffers,
            wall_buffer,
            uniform_buffer,
            output_texture,
            output_texture_view,
            bind_group,
            grid_size,
        }
    }
    
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        grid_buffers: &[wgpu::Buffer; 4],
        wall_buffer: &wgpu::Buffer,
        output_texture_view: &wgpu::TextureView,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend-to-Texture Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: grid_buffers[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_buffers[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grid_buffers[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: grid_buffers[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wall_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(output_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        })
    }
    
    /// Upload attenuation grids to GPU
    pub fn upload_grids(&self, ctx: &GpuContext, grids: &[&[f32]; 4]) {
        for (i, grid) in grids.iter().enumerate() {
            ctx.queue.write_buffer(&self.grid_buffers[i], 0, bytemuck::cast_slice(grid));
        }
    }
    
    /// Upload a single grid (to slot 0)
    pub fn upload_single_grid(&self, ctx: &GpuContext, grid: &[f32]) {
        ctx.queue.write_buffer(&self.grid_buffers[0], 0, bytemuck::cast_slice(grid));
    }
    
    /// Upload wall data (packed as bits)
    pub fn upload_walls(&self, ctx: &GpuContext, walls: &[bool]) {
        let num_words = (walls.len() + 31) / 32;
        let mut packed = vec![0u32; num_words];
        
        for (i, &is_wall) in walls.iter().enumerate() {
            if is_wall {
                packed[i / 32] |= 1 << (i % 32);
            }
        }
        
        ctx.queue.write_buffer(&self.wall_buffer, 0, bytemuck::cast_slice(&packed));
    }
    
    /// Upload uniforms
    pub fn upload_uniforms(&self, ctx: &GpuContext, uniforms: &BlendUniforms) {
        ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));
    }
    
    /// Run the blend compute shader (4 grids → texture)
    pub fn dispatch_blend(&self, ctx: &GpuContext) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Blend-to-Texture Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Blend-to-Texture Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.blend_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            let workgroups_x = (self.grid_size.0 + 7) / 8;
            let workgroups_y = (self.grid_size.1 + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Run the single-grid compute shader (1 grid → texture)
    pub fn dispatch_single(&self, ctx: &GpuContext) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Single-to-Texture Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Single-to-Texture Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.single_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            let workgroups_x = (self.grid_size.0 + 7) / 8;
            let workgroups_y = (self.grid_size.1 + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Get the output texture view (for display pipeline to sample from)
    pub fn output_texture_view(&self) -> &wgpu::TextureView {
        &self.output_texture_view
    }
    
    /// Get grid dimensions
    pub fn grid_size(&self) -> (u32, u32) {
        self.grid_size
    }
}
