use spirv_std::{
    glam::{vec4, UVec2, UVec3},
    spirv, Image,
};

#[spirv(compute(threads(8, 8, 1)))]
fn main_cs(
    #[spirv(launch_id)] id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),
) {
    unsafe {
        image.write(UVec2 { x: id.x, y: id.y }, vec4(1.0, 0.0, 0.0, 1.0));
    }
}
