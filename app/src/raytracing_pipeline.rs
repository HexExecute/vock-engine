use shared::ShaderConstants;
use std::sync::Arc;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    image::view::ImageView,
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
};

pub struct RaytracingPipeline {
    queue: Arc<Queue>,
    pipeline: Arc<ComputePipeline>,

    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl RaytracingPipeline {
    pub fn new(
        queue: Arc<Queue>,

        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Self {
        let pipeline = {
            let device = queue.device();
            let cs = crate::cs::load(device.clone())
                .unwrap()
                .single_entry_point()
                .unwrap();
            let stage = PipelineShaderStageCreateInfo::new(cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        Self {
            queue,
            pipeline,

            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn compute(
        &self,
        image_view: Arc<ImageView>,

        node_buffer: Subbuffer<[[u32; 8]]>,
        voxel_buffer: Subbuffer<[u32]>,

        push_constants: ShaderConstants,
    ) -> Box<dyn GpuFuture> {
        let image_extent = image_view.image().extent();
        let pipeline_layout = self.pipeline.layout();
        let descriptor_set_layout = &pipeline_layout.set_layouts()[0];
        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, image_view),
                WriteDescriptorSet::buffer(1, node_buffer),
                WriteDescriptorSet::buffer(2, voxel_buffer),
            ],
            [],
        )
        .unwrap();
        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.as_ref(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_constants)
            .unwrap()
            .dispatch([image_extent[0] / 8, image_extent[1] / 8, 1])
            .unwrap();

        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(self.queue.clone()).unwrap();

        finished.then_signal_fence_and_flush().unwrap().boxed()
    }
}
