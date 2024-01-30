use spirv_std::{
    glam::{vec2, vec4, UVec2, UVec3},
    spirv, Image,
};

#[spirv(compute(threads(8, 8, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),
) {
    // let image_size: UVec2 = image.width();
    // let uv = vec2(id.as_vec3().x, id.as_vec3().y) / image_size.as_vec2();

    unsafe {
        image.write(UVec2 { x: id.x, y: id.y }, vec4(1.0, 0.0, 0.0, 1.0));
    }
}
