//! Compute pipeline for GPU-accelerated blending and color application

use super::context::GpuContext;

/// Uniform data sent to the blend compute shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlendUniforms {
    pub weights: [f32; 4],     // w00, w10, w01, w11
    pub color: [f32; 3],       // RGB
    pub norm_factor: f32,
    pub grid_width: u32,
    pub grid_height: u32,
    pub apply_srgb: u32,       // 1 = apply sRGB gamma encoding, 0 = linear
    pub _padding: u32,
}

/// Compute pipeline for blending 4 attenuation grids on GPU
pub struct BlendPipeline {
    blend_pipeline: wgpu::ComputePipeline,
    single_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    
    // Buffers (resized as needed)
    grid_buffers: [wgpu::Buffer; 4],
    wall_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    
    // Current bind group
    bind_group: wgpu::BindGroup,
    
    // Grid dimensions
    grid_size: (u32, u32),
}

impl BlendPipeline {
    pub fn new(ctx: &GpuContext, grid_width: u32, grid_height: u32) -> Self {
        let device = &ctx.device;
        let grid_size = (grid_width, grid_height);
        let num_cells = (grid_width * grid_height) as usize;
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blend Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blend.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blend Bind Group Layout"),
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
                // output (RGBA pixels)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
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
            label: Some("Blend Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create compute pipelines
        let blend_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Blend Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("blend_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let single_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Single Grid Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("single_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create buffers
        let grid_buffer_size = (num_cells * std::mem::size_of::<f32>()) as u64;
        let wall_buffer_size = ((num_cells + 31) / 32 * std::mem::size_of::<u32>()) as u64;
        let output_buffer_size = (num_cells * std::mem::size_of::<u32>()) as u64;
        
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
        
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Blend Uniform Buffer"),
            size: std::mem::size_of::<BlendUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create initial bind group
        let bind_group = Self::create_bind_group(
            device,
            &bind_group_layout,
            &grid_buffers,
            &wall_buffer,
            &output_buffer,
            &uniform_buffer,
        );
        
        Self {
            blend_pipeline,
            single_pipeline,
            bind_group_layout,
            grid_buffers,
            wall_buffer,
            output_buffer,
            uniform_buffer,
            bind_group,
            grid_size,
        }
    }
    
    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        grid_buffers: &[wgpu::Buffer; 4],
        wall_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blend Bind Group"),
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
                    resource: output_buffer.as_entire_binding(),
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
    
    /// Upload a single grid (copies to all 4 slots)
    pub fn upload_single_grid(&self, ctx: &GpuContext, grid: &[f32]) {
        ctx.queue.write_buffer(&self.grid_buffers[0], 0, bytemuck::cast_slice(grid));
    }
    
    /// Upload wall data (packed as bits)
    pub fn upload_walls(&self, ctx: &GpuContext, walls: &[bool]) {
        // Pack bools into u32 bits
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
    
    /// Run the blend compute shader (4 grids → 1 output)
    pub fn dispatch_blend(&self, ctx: &GpuContext) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Blend Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Blend Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.blend_pipeline);
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
            
            // Dispatch workgroups (8x8 threads each)
            let workgroups_x = (self.grid_size.0 + 7) / 8;
            let workgroups_y = (self.grid_size.1 + 7) / 8;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Run the single-grid compute shader (1 grid → 1 output)
    pub fn dispatch_single(&self, ctx: &GpuContext) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Single Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Single Compute Pass"),
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
    
    /// Get the output buffer for reading results
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }
    
    /// Get grid dimensions
    pub fn grid_size(&self) -> (u32, u32) {
        self.grid_size
    }
}
