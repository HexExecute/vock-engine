use std::sync::Arc;

use shared::ShaderConstants;
use voxtree::{Node, Voxtree};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassBeginInfo,
        SubpassContents,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, DeviceExtensions, Features, Queue},
    image::ImageUsage,
    instance::{InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{PresentMode, Surface},
    sync::GpuFuture,
};
use vulkano_util::{
    context::{VulkanoConfig, VulkanoContext},
    renderer::DEFAULT_IMAGE_FORMAT,
    window::{VulkanoWindows, WindowDescriptor},
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};

use crate::{pixels_pipeline::PixelsPipeline, raytracing_pipeline::RaytracingPipeline};

pub struct State {
    // window: Arc<Window>,
    // primary_window_renderer: &mut VulkanoWindowRenderer,
    windows: VulkanoWindows,

    // image: Arc<Image>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    render_pass: Arc<RenderPass>,

    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pixels_pipeline: PixelsPipeline,
    raytracing_pipeline: RaytracingPipeline,

    node_buffer: Subbuffer<[[u32; 8]]>,
    voxel_buffer: Subbuffer<[u32]>,

    push_constants: ShaderConstants,

    tree: Voxtree<u32>,
}

impl State {
    pub fn new(event_loop: &EventLoop<()>) -> Self {
        println!("app initialization...\n");

        let required_extensions = Surface::required_extensions(&event_loop);
        let context = VulkanoContext::new(VulkanoConfig {
            device_extensions: DeviceExtensions {
                khr_swapchain: true,
                khr_vulkan_memory_model: true,
                ..Default::default()
            },
            device_features: Features {
                vulkan_memory_model: true,
                ..Features::empty()
            },
            instance_create_info: InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
            ..Default::default()
        });

        println!("[✓] instance created");

        let mut windows = VulkanoWindows::default();

        windows.create_window(
            event_loop,
            &context,
            &WindowDescriptor {
                title: "app".to_string(),
                present_mode: PresentMode::Mailbox,
                width: 512.0,
                height: 512.0,
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

        let tree: Voxtree<u32> = Voxtree::builder()
            .with_max_depth(8)
            .with_root(Node::Branch(Box::new([
                Node::Leaf(Some(0)),
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
            command_buffer_allocator.clone(),
            descriptor_set_allocator.clone(),
        );

        let raytracing_pipeline = RaytracingPipeline::new(
            gfx_queue.clone(),
            command_buffer_allocator.clone(),
            descriptor_set_allocator,
        );

        println!("\n...app initalization");

        Self {
            windows,

            device: device.clone(),
            queue: gfx_queue.clone(),

            render_pass,

            command_buffer_allocator,

            pixels_pipeline,
            raytracing_pipeline,

            node_buffer,
            voxel_buffer,

            push_constants: ShaderConstants {
                tree_root: packed_tree.root,
                tree_scale: packed_tree.scale,
            },

            tree,
        }
    }

    pub fn update(&mut self) {}

    pub fn render(&mut self) {
        let primary_window_renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("[𐄂] failed to create primary window renderer");

        let before_pipeline_future = match primary_window_renderer.acquire() {
            Err(e) => {
                println!("{e}");
                return;
            }
            Ok(future) => future,
        };

        let image = primary_window_renderer.get_additional_image_view(0);

        let after_compute = self
            .raytracing_pipeline
            .compute(
                image.clone(),
                self.node_buffer.clone(),
                self.voxel_buffer.clone(),
                self.push_constants,
            )
            .join(before_pipeline_future);

        let after_renderpass_future = {
            let target = primary_window_renderer.swapchain_image_view();
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

        primary_window_renderer.present(after_renderpass_future, true);
    }

    pub fn handle_event(&mut self, event: Event<()>, control_flow: &mut ControlFlow) {
        let primary_window_renderer = self
            .windows
            .get_primary_renderer_mut()
            .expect("[𐄂] failed to create primary window renderer");

        match event {
            Event::WindowEvent { window_id, event }
                if window_id == primary_window_renderer.window().id() =>
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
