use std::sync::Arc;

use voxtree::{Node, Voxtree};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{
            CommandBufferAllocator, StandardCommandBufferAllocator,
            StandardCommandBufferAllocatorCreateInfo,
        },
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
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
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::{VulkanoWindowRenderer, DEFAULT_IMAGE_FORMAT},
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use crate::{pixels_pipeline::PixelsPipeline, raytracing_pipeline::RaytracingPipeline};

pub struct State {
    // window: Arc<Window>,
    primary_window_renderer: &'static mut VulkanoWindowRenderer,

    // image: Arc<Image>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    render_pass: Arc<RenderPass>,

    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pixels_pipeline: PixelsPipeline,
    raytracing_pipeline: RaytracingPipeline,

    node_buffer: Subbuffer<[[u32; 8]]>,
    voxel_buffer: Subbuffer<[f32]>,

    tree: Voxtree<f32>,
}

impl State {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        println!("app initialization...\n");

        let library = VulkanLibrary::new().unwrap();
        let context = VulkanoContext::new(VulkanoConfig::default());
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
        let windows = VulkanoWindows::default();
        let window_id = windows.create_window(
            event_loop,
            &context,
            &WindowDescriptor {
                title: "app".to_string(),
                present_mode: PresentMode::Mailbox,
                ..Default::default()
            },
            |_| {},
        );
        println!("[✓] window created");

        let primary_window_renderer = windows
            .get_primary_renderer_mut()
            .expect("[𐄂] failed to create primary window renderer");
        println!("[✓] primary window renderer created");

        primary_window_renderer.add_additional_image_view(
            0,
            DEFAULT_IMAGE_FORMAT,
            ImageUsage::SAMPLED | ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
        );

        let gfx_queue = context.graphics_queue();
        // let device = queue.device();

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

        // let window = Arc::new(
        //     WindowBuilder::new()
        //         .with_title("app")
        //         .with_inner_size(PhysicalSize::new(1024, 1024))
        //         .build(event_loop)
        //         .expect("[𐄂] failed to create window"),
        // );
        // println!("[✓] window created");
        // let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        // let device_extensions = DeviceExtensions {
        //     khr_swapchain: true,
        //     khr_vulkan_memory_model: true,
        //     ..Default::default()
        // };
        // let features = Features {
        //     vulkan_memory_model: true,
        //     ..Features::empty()
        // };
        // let (physical_device, queue_family_index) = instance
        //     .enumerate_physical_devices()
        //     .unwrap()
        //     .filter(|p| p.supported_extensions().contains(&device_extensions))
        //     .filter(|p| p.supported_features().contains(&features))
        //     .filter_map(|p| {
        //         p.queue_family_properties()
        //             .iter()
        //             .enumerate()
        //             .position(|(i, q)| {
        //                 q.queue_flags.intersects(QueueFlags::GRAPHICS)
        //                     && p.surface_support(i as u32, &surface).unwrap_or(false)
        //             })
        //             .map(|i| (p, i as u32))
        //     })
        //     .min_by_key(|(p, _)| match p.properties().device_type {
        //         PhysicalDeviceType::DiscreteGpu => 0,
        //         PhysicalDeviceType::IntegratedGpu => 1,
        //         PhysicalDeviceType::VirtualGpu => 2,
        //         PhysicalDeviceType::Cpu => 3,
        //         PhysicalDeviceType::Other => 4,
        //         _ => 5,
        //     })
        //     .unwrap();
        //
        // println!(
        //     "[✓] using device: {} (type: {:?})",
        //     physical_device.properties().device_name,
        //     physical_device.properties().device_type,
        // );
        //
        // let (device, mut queues) = Device::new(
        //     physical_device,
        //     DeviceCreateInfo {
        //         enabled_extensions: device_extensions,
        //         enabled_features: features,
        //         queue_create_infos: vec![QueueCreateInfo {
        //             queue_family_index,
        //             ..Default::default()
        //         }],
        //         ..Default::default()
        //     },
        // )
        // .expect("[𐄂] failed to create device");
        // println!("[✓] device created");

        // let queue = queues.next().expect("[𐄂] failed to create queue");
        // println!("[✓] queue created");
        //
        // let (swapchain, images) = {
        //     let surface_capabilities = device
        //         .physical_device()
        //         .surface_capabilities(&surface, Default::default())
        //         .unwrap();
        //     let image_format = device
        //         .physical_device()
        //         .surface_formats(&surface, Default::default())
        //         .unwrap()[0]
        //         .0;
        //
        //     Swapchain::new(
        //         device.clone(),
        //         surface,
        //         SwapchainCreateInfo {
        //             min_image_count: surface_capabilities.min_image_count.max(2),
        //             image_format,
        //             image_extent: window.inner_size().into(),
        //             image_usage: ImageUsage::COLOR_ATTACHMENT,
        //             composite_alpha: surface_capabilities
        //                 .supported_composite_alpha
        //                 .into_iter()
        //                 .next()
        //                 .unwrap(),
        //             present_mode: PresentMode::Mailbox,
        //             ..Default::default()
        //         },
        //     )
        //     .expect("[𐄂] failed to create swapchain")
        // };
        //
        // println!("[✓] swapchain created");

        let device = gfx_queue.device();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let packed_tree = tree.pack();

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

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: primary_window_renderer.swapchain_format(),
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

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let pixels_pipeline = PixelsPipeline::new(
            gfx_queue.clone(),
            subpass,
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        );

        let raytracing_pipeline = RaytracingPipeline::new(
            gfx_queue.clone(),
            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        );

        println!("\n...app initalization");

        Self {
            // window,
            primary_window_renderer,

            // image: images[0],
            device: device.clone(),
            queue: gfx_queue.clone(),

            render_pass,

            command_buffer_allocator,

            pixels_pipeline,
            raytracing_pipeline,

            node_buffer,
            voxel_buffer,

            tree,
        }
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) {
        let before_pipeline_future = match self.primary_window_renderer.acquire() {
            Err(e) => {
                println!("{e}");
                return;
            }
            Ok(future) => future,
        };

        let image = self.primary_window_renderer.get_additional_image_view(0);

        let after_compute = self.raytracing_pipeline.compute(image);

        let after_renderpass_future = {
            let target = self.primary_window_renderer.swapchain_image_view();
            let image_dimensions: [u32; 2] = target.image().extent()[0..2].try_into().unwrap();

            let framebuffer = Framebuffer::new(
                self.render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![target],
                    ..Default::default()
                },
            )
            .unwrap();

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
                        ..RenderPassBeginInfo::framebuffer(framebuffer)
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::SecondaryCommandBuffers,
                        ..Default::default()
                    },
                )
                .unwrap();

            let cb = self.pixels_pipeline.render(image_dimensions, image);

            command_buffer_builder.execute_commands(cb).unwrap();

            command_buffer_builder
                .end_render_pass(Default::default())
                .unwrap();

            let command_buffer = command_buffer_builder.build().unwrap();

            after_compute
                .then_execute(self.queue.clone(), command_buffer)
                .unwrap()
                .boxed()
        };

        self.primary_window_renderer
            .present(after_renderpass_future, true);
    }

    pub fn handle_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        match event {
            Event::WindowEvent { window_id, event }
                if window_id == self.primary_window_renderer.window().id() =>
            {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    // WindowEvent::Resized(_) => self.recreate_swapchain = true,
                    _ => (),
                }
            }
            Event::RedrawEventsCleared => self.render(),
            _ => (),
        }
    }
}
