use std::sync::Arc;

use voxtree::{Node, Voxtree};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferToImageInfo, PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::{
        allocator::{StandardDescriptorSetAlloc, StandardDescriptorSetAllocator},
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex as InputVertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, PresentMode, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    DeviceSize, Validated, VulkanError, VulkanLibrary,
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{pixels_pipeline::PixelsPipeline, raytracing_pipeline::RaytracingPipeline};

pub struct State {
    window: Arc<Window>,

    device: Arc<Device>,
    queue: Arc<Queue>,

    swapchain: Arc<Swapchain>,
    recreate_swapchain: bool,
    render_pass: Arc<RenderPass>,

    viewport: Viewport,

    pixels_pipeline: PixelsPipeline,
    raytracing_pipeline: RaytracingPipeline,

    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    node_buffer: Subbuffer<[[u32; 8]]>,
    voxel_buffer: Subbuffer<[f32]>,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    // descriptor_set: Arc<PersistentDescriptorSet<StandardDescriptorSetAlloc>>,
    tree: Voxtree<f32>,
}

impl State {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        println!("app initialization...\n");

        let library = VulkanLibrary::new().unwrap();
        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("[𐄂] failed to create instance");
        println!("[✓] instance created");

        let tree: Voxtree<f32> = Voxtree::builder()
            .with_max_depth(8)
            .with_root(Node::Branch(Box::new([
                Node::Leaf(Some(0.0)),
                Node::Leaf(None),
                Node::Leaf(None),
                Node::Leaf(None),
                Node::Leaf(None),
                Node::Leaf(None),
                Node::Leaf(None),
                Node::Leaf(None),
            ])))
            .build();
        println!("[✓] voxtree created");

        let window = Arc::new(
            WindowBuilder::new()
                .with_title("app")
                .with_inner_size(PhysicalSize::new(1024, 1024))
                .build(event_loop)
                .expect("[𐄂] failed to create window"),
        );
        println!("[✓] window created");
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            khr_vulkan_memory_model: true,
            ..Default::default()
        };
        let features = Features {
            vulkan_memory_model: true,
            ..Features::empty()
        };
        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter(|p| p.supported_features().contains(&features))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        println!(
            "[✓] using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .expect("[𐄂] failed to create device");
        println!("[✓] device created");

        let queue = queues.next().expect("[𐄂] failed to create queue");
        println!("[✓] queue created");
        //
        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .unwrap(),
                    present_mode: PresentMode::Mailbox,
                    ..Default::default()
                },
            )
            .expect("[𐄂] failed to create swapchain")
        };
        println!("[✓] swapchain created");
        //
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        //
        // // let vertices = [
        // //     Vertex {
        // //         position: [-0.5, -0.5],
        // //     },
        // //     Vertex {
        // //         position: [-0.5, 0.5],
        // //     },
        // //     Vertex {
        // //         position: [0.5, -0.5],
        // //     },
        // //     Vertex {
        // //         position: [0.5, 0.5],
        // //     },
        // // ];
        // // let vertex_buffer = Buffer::from_iter(
        // //     memory_allocator.clone(),
        // //     BufferCreateInfo {
        // //         usage: BufferUsage::VERTEX_BUFFER,
        // //         ..Default::default()
        // //     },
        // //     AllocationCreateInfo {
        // //         memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
        // //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        // //         ..Default::default()
        // //     },
        // //     vertices,
        // // )
        // // .expect("[𐄂] failed to create vertex buffer");
        // // println!("[✓] vertex buffer created");
        //
        let packed_tree = tree.pack();
        //
        let node_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            packed_tree.nodes,
        )
        .expect("[𐄂] failed to create node buffer");
        println!("[✓] node buffer created");

        let voxel_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            packed_tree.voxels,
        )
        .expect("[𐄂] failed to create voxel buffer");
        println!("[✓] voxel buffer created");
        //
        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .expect("[𐄂] failed to create render pass");
        println!("[✓] render pass created");
        //
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let mut uploads = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        //
        // let texture = {
        //     let png_bytes = include_bytes!("image_img.png").as_slice();
        //     let decoder = png::Decoder::new(png_bytes);
        //     let mut reader = decoder.read_info().unwrap();
        //     let info = reader.info();
        //     let extent = [info.width, info.height, 1];
        //
        //     let upload_buffer = Buffer::new_slice(
        //         memory_allocator.clone(),
        //         BufferCreateInfo {
        //             usage: BufferUsage::TRANSFER_SRC,
        //             ..Default::default()
        //         },
        //         AllocationCreateInfo {
        //             memory_type_filter: MemoryTypeFilter::PREFER_HOST
        //                 | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
        //             ..Default::default()
        //         },
        //         (info.width * info.height * 4) as DeviceSize,
        //     )
        //     .unwrap();
        //
        //     reader
        //         .next_frame(&mut upload_buffer.write().unwrap())
        //         .unwrap();
        //
        //     let image = Image::new(
        //         memory_allocator,
        //         ImageCreateInfo {
        //             image_type: ImageType::Dim2d,
        //             format: Format::R8G8B8A8_SRGB,
        //             extent,
        //             usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
        //             ..Default::default()
        //         },
        //         AllocationCreateInfo::default(),
        //     )
        //     .unwrap();
        //
        //     uploads
        //         .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        //             upload_buffer,
        //             image.clone(),
        //         ))
        //         .unwrap();
        //
        //     ImageView::new_default(image).unwrap()
        // };
        //
        // let sampler = Sampler::new(
        //     device.clone(),
        //     SamplerCreateInfo {
        //         mag_filter: Filter::Linear,
        //         min_filter: Filter::Linear,
        //         address_mode: [SamplerAddressMode::Repeat; 3],
        //         ..Default::default()
        //     },
        // )
        // .unwrap();
        //
        // let pipeline = {
        //     let vs = crate::vs::load(device.clone())
        //         .unwrap()
        //         .single_entry_point()
        //         .unwrap();
        //     let fs = crate::fs::load(device.clone())
        //         .unwrap()
        //         .single_entry_point()
        //         .unwrap();
        //     let vertex_input_state = Vertex::per_vertex()
        //         .definition(&vs.info().input_interface)
        //         .unwrap();
        //     let stages = [
        //         PipelineShaderStageCreateInfo::new(vs),
        //         PipelineShaderStageCreateInfo::new(fs),
        //     ];
        //     let layout = PipelineLayout::new(
        //         device.clone(),
        //         PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        //             .into_pipeline_layout_create_info(device.clone())
        //             .unwrap(),
        //     )
        //     .unwrap();
        //     let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        //
        //     GraphicsPipeline::new(
        //         device.clone(),
        //         None,
        //         GraphicsPipelineCreateInfo {
        //             stages: stages.into_iter().collect(),
        //             input_assembly_state: Some(InputAssemblyState {
        //                 topology: PrimitiveTopology::TriangleStrip,
        //                 ..Default::default()
        //             }),
        //             vertex_input_state: Some(vertex_input_state),
        //             viewport_state: Some(ViewportState::default()),
        //             rasterization_state: Some(RasterizationState::default()),
        //             multisample_state: Some(MultisampleState::default()),
        //             color_blend_state: Some(ColorBlendState::with_attachment_states(
        //                 subpass.num_color_attachments(),
        //                 ColorBlendAttachmentState {
        //                     blend: Some(AttachmentBlend::alpha()),
        //                     ..Default::default()
        //                 },
        //             )),
        //             dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        //             subpass: Some(subpass.into()),
        //             ..GraphicsPipelineCreateInfo::layout(layout)
        //         },
        //     )
        //     .expect("[𐄂] failed to create graphics pipeline")
        // };
        // println!("[✓] graphics pipeline created");
        //
        // let layout = pipeline.layout().set_layouts().get(0).unwrap();
        // let set = PersistentDescriptorSet::new(
        //     &descriptor_set_allocator,
        //     layout.clone(),
        //     [
        //         WriteDescriptorSet::sampler(0, sampler),
        //         WriteDescriptorSet::image_view(1, texture),
        //     ],
        //     [],
        // )
        // .unwrap();
        //
        let mut viewport = Viewport {
            offset: [0.0, 0.0],
            extent: [0.0, 0.0],
            depth_range: 0.0..=1.0,
        };
        let framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut viewport);
        //
        let previous_frame_end = Some(
            uploads
                .build()
                .unwrap()
                .execute(queue.clone())
                .unwrap()
                .boxed(),
        );
        //
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let pixels_pipeline = PixelsPipeline::new(
            queue,
            subpass,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        );

        let raytracing_pipeline = RaytracingPipeline::new(
            queue,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        );

        println!("\n...app initalization");

        Self {
            window,

            device,
            queue,

            swapchain,
            recreate_swapchain: false,
            render_pass,

            pixels_pipeline,
            raytracing_pipeline,

            viewport,
            // pipeline,
            command_buffer_allocator,

            // vertex_buffer,
            node_buffer,
            voxel_buffer,

            previous_frame_end,
            framebuffers,

            // set,
            tree,
        }
    }

    pub fn render(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();

        if image_extent.contains(&0) {
            return;
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            let (new_swapchain, new_images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            self.swapchain = new_swapchain;
            self.framebuffers = window_size_dependent_setup(
                &new_images,
                self.render_pass.clone(),
                &mut self.viewport,
            );
            self.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0; 4].into())],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[0])
                },
                SubpassBeginInfo {
                    contents: SubpassContents::SecondaryCommandBuffers,
                    ..Default::default()
                },
            )
            .unwrap();

        let image_view = 
            ImageView::new_default(image.clone()).unwrap();

        let cb = self.pixels_pipeline.render(image_extent, )

        // let mut builder = AutoCommandBufferBuilder::primary(
        //     &self.command_buffer_allocator,
        //     self.queue.queue_family_index(),
        //     CommandBufferUsage::OneTimeSubmit,
        // )
        // .unwrap();
        // builder
        //     .begin_render_pass(
        //         RenderPassBeginInfo {
        //             clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
        //             ..RenderPassBeginInfo::framebuffer(
        //                 self.framebuffers[image_index as usize].clone(),
        //             )
        //         },
        //         Default::default(),
        //     )
        //     .unwrap()
        //     .set_viewport(0, [self.viewport.clone()].into_iter().collect())
        //     .unwrap()
        //     .bind_pipeline_graphics(self.pipeline.clone())
        //     .unwrap()
        //     .bind_descriptor_sets(
        //         PipelineBindPoint::Graphics,
        //         self.pipeline.layout().clone(),
        //         0,
        //         self.set.clone(),
        //     )
        //     .unwrap()
        //     .bind_vertex_buffers(0, self.vertex_buffer.clone())
        //     .unwrap()
        //     .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
        //     .unwrap()
        //     .end_render_pass(Default::default())
        //     .unwrap();
        // let command_buffer = builder.build().unwrap();
        //
        // let future = self
        //     .previous_frame_end
        //     .take()
        //     .unwrap()
        //     .join(acquire_future)
        //     .then_execute(self.queue.clone(), command_buffer)
        //     .unwrap()
        //     .then_swapchain_present(
        //         self.queue.clone(),
        //         SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_index),
        //     )
        //     .then_signal_fence_and_flush();
        //
        // match future.map_err(Validated::unwrap) {
        //     Ok(future) => {
        //         self.previous_frame_end = Some(future.boxed());
        //     }
        //     Err(VulkanError::OutOfDate) => {
        //         self.recreate_swapchain = true;
        //         self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        //     }
        //     Err(e) => {
        //         println!("failed to flush future: {e}");
        //         self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
        //     }
        // }
    }

    pub fn handle_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        match event {
            Event::WindowEvent { window_id, event } if window_id == self.window.id() => match event
            {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(_) => self.recreate_swapchain = true,
                _ => (),
            },
            Event::RedrawEventsCleared => self.render(),
            _ => (),
        }
    }
}

/// This function is called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let extent = images[0].extent();
    viewport.extent = [extent[0] as f32, extent[1] as f32];

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
