//! Dual-Kawase blur pipeline for soft lighting effects
//! 
//! Uses a two-pass approach:
//! 1. Downsample to half resolution with 5-tap X pattern
//! 2. Upsample back to full resolution with 8-tap circle pattern
//!
//! Multiple iterations can be chained for stronger blur.

use super::context::GpuContext;

/// Uniform data for blur shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlurUniforms {
    pub texture_size: [u32; 2],
    pub pixel_size: [f32; 2],
}

/// Dual-Kawase blur pipeline
pub struct BlurPipeline {
    downsample_pipeline: wgpu::ComputePipeline,
    upsample_pipeline: wgpu::ComputePipeline,
    
    // Bind group layouts
    downsample_bind_group_layout: wgpu::BindGroupLayout,
    upsample_bind_group_layout: wgpu::BindGroupLayout,
    
    // Half-resolution intermediate texture
    half_texture: wgpu::Texture,
    half_texture_view: wgpu::TextureView,
    
    // Uniform buffers
    downsample_uniform_buffer: wgpu::Buffer,
    upsample_uniform_buffer: wgpu::Buffer,
    
    // Samplers
    sampler: wgpu::Sampler,
    
    // Bind groups (created dynamically based on input/output textures)
    downsample_bind_group: Option<wgpu::BindGroup>,
    upsample_bind_group: Option<wgpu::BindGroup>,
    
    // Size tracking
    full_size: (u32, u32),
}

impl BlurPipeline {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let device = &ctx.device;
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blur.wgsl").into()),
        });
        
        // Create sampler for texture sampling (linear for smooth interpolation)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blur Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        // Downsample bind group layout:
        // - input: sampled texture (full res)
        // - output: storage texture (half res)
        // - uniforms
        // - sampler
        let downsample_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Downsample Bind Group Layout"),
            entries: &[
                // input_texture (sampled)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // output_texture (storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Upsample uses the same layout
        let upsample_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Upsample Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        
        // Create pipelines
        let downsample_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Downsample Pipeline Layout"),
            bind_group_layouts: &[&downsample_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Downsample Pipeline"),
            layout: Some(&downsample_pipeline_layout),
            module: &shader,
            entry_point: Some("downsample_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        let upsample_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Upsample Pipeline Layout"),
            bind_group_layouts: &[&upsample_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Upsample Pipeline"),
            layout: Some(&upsample_pipeline_layout),
            module: &shader,
            entry_point: Some("upsample_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Create half-resolution intermediate texture
        let half_width = width / 2;
        let half_height = height / 2;
        
        let half_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Blur Half-Res Texture"),
            size: wgpu::Extent3d {
                width: half_width.max(1),
                height: half_height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let half_texture_view = half_texture.create_view(&Default::default());
        
        // Create uniform buffers
        let downsample_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Downsample Uniform Buffer"),
            size: std::mem::size_of::<BlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let upsample_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Upsample Uniform Buffer"),
            size: std::mem::size_of::<BlurUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            downsample_pipeline,
            upsample_pipeline,
            downsample_bind_group_layout,
            upsample_bind_group_layout,
            half_texture,
            half_texture_view,
            downsample_uniform_buffer,
            upsample_uniform_buffer,
            sampler,
            downsample_bind_group: None,
            upsample_bind_group: None,
            full_size: (width, height),
        }
    }
    
    /// Set up bind groups for blurring a specific texture in-place
    /// The texture must support both TEXTURE_BINDING and STORAGE_BINDING
    pub fn setup_for_texture(&mut self, ctx: &GpuContext, texture_view: &wgpu::TextureView) {
        let device = &ctx.device;
        
        let (width, height) = self.full_size;
        let half_width = width / 2;
        let half_height = height / 2;
        
        // Upload uniforms for downsample (full -> half)
        let downsample_uniforms = BlurUniforms {
            texture_size: [width, height],
            pixel_size: [1.0 / width as f32, 1.0 / height as f32],
        };
        ctx.queue.write_buffer(
            &self.downsample_uniform_buffer,
            0,
            bytemuck::cast_slice(&[downsample_uniforms]),
        );
        
        // Upload uniforms for upsample (half -> full)
        let upsample_uniforms = BlurUniforms {
            texture_size: [width, height],
            pixel_size: [1.0 / width as f32, 1.0 / height as f32],
        };
        ctx.queue.write_buffer(
            &self.upsample_uniform_buffer,
            0,
            bytemuck::cast_slice(&[upsample_uniforms]),
        );
        
        // Downsample: read from input texture, write to half-res texture
        self.downsample_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Downsample Bind Group"),
            layout: &self.downsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.half_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.downsample_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
        
        // Upsample: read from half-res texture, write back to input texture
        self.upsample_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Upsample Bind Group"),
            layout: &self.upsample_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.half_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.upsample_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
    }
    
    /// Run the blur (downsample + upsample)
    /// Call setup_for_texture first!
    pub fn dispatch(&self, ctx: &GpuContext) {
        let downsample_bg = match &self.downsample_bind_group {
            Some(bg) => bg,
            None => return,
        };
        let upsample_bg = match &self.upsample_bind_group {
            Some(bg) => bg,
            None => return,
        };
        
        let (width, height) = self.full_size;
        let half_width = width / 2;
        let half_height = height / 2;
        
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Blur Encoder"),
        });
        
        // Pass 1: Downsample (full -> half)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Downsample Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.downsample_pipeline);
            pass.set_bind_group(0, downsample_bg, &[]);
            pass.dispatch_workgroups(
                (half_width + 7) / 8,
                (half_height + 7) / 8,
                1,
            );
        }
        
        // Pass 2: Upsample (half -> full)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Upsample Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.upsample_pipeline);
            pass.set_bind_group(0, upsample_bg, &[]);
            pass.dispatch_workgroups(
                (width + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
        
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
    
    /// Run multiple blur iterations for stronger effect
    pub fn dispatch_iterations(&self, ctx: &GpuContext, iterations: u32) {
        for _ in 0..iterations {
            self.dispatch(ctx);
        }
    }
}

/// Uniform data for wall overlay shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WallOverlayUniforms {
    pub grid_width: u32,
    pub grid_height: u32,
    pub _padding: [u32; 2],
}

/// Pipeline to apply wall tint after blur
/// This keeps walls crisp and visible while lighting is blurred
pub struct WallOverlayPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    wall_buffer: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    grid_size: (u32, u32),
}

impl WallOverlayPipeline {
    pub fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let device = &ctx.device;
        let num_cells = (width * height) as usize;
        
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Wall Overlay Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/wall_overlay.wgsl").into()),
        });
        
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Wall Overlay Bind Group Layout"),
            entries: &[
                // output_texture (read_write storage texture)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadWrite,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // walls buffer
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
                // uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
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
        
        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Wall Overlay Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Wall Overlay Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("wall_overlay_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        
        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Overlay Uniform Buffer"),
            size: std::mem::size_of::<WallOverlayUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Wall buffer
        let wall_buffer_size = ((num_cells + 31) / 32 * std::mem::size_of::<u32>()) as u64;
        let wall_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Wall Overlay Wall Buffer"),
            size: wall_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            wall_buffer,
            bind_group: None,
            grid_size: (width, height),
        }
    }
    
    /// Set up bind group for the target texture
    pub fn setup_for_texture(&mut self, ctx: &GpuContext, texture_view: &wgpu::TextureView) {
        let device = &ctx.device;
        let (width, height) = self.grid_size;
        
        // Upload uniforms
        let uniforms = WallOverlayUniforms {
            grid_width: width,
            grid_height: height,
            _padding: [0, 0],
        };
        ctx.queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
        
        // Create bind group
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wall Overlay Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.wall_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        }));
    }
    
    /// Upload wall data
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
    
    /// Apply wall overlay to the texture
    pub fn dispatch(&self, ctx: &GpuContext) {
        let bind_group = match &self.bind_group {
            Some(bg) => bg,
            None => return,
        };
        
        let (width, height) = self.grid_size;
        
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Wall Overlay Encoder"),
        });
        
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Wall Overlay Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(
                (width + 7) / 8,
                (height + 7) / 8,
                1,
            );
        }
        
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
}
