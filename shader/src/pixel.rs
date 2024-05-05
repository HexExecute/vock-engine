use spirv_std::glam::{vec2, vec4, Vec2, Vec4};
use spirv_std::{spirv, Image, Sampler};

#[spirv(vertex)]
pub fn main_vs(
    position: Vec2,
    tex_coord: &mut Vec2,
    #[spirv(position, invariant)] out_pos: &mut Vec4,
) {
    *out_pos = vec4(position.x, position.y, 0.0, 1.0);
    *tex_coord = (position + vec2(1.0, 1.0)) / 2.0;
}

#[spirv(fragment)]
pub fn main_fs(
    tex_coord: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 1)] image: &Image!(2D, type=f32, sampled),
    frag_color: &mut Vec4,
) {
    *frag_color = image.sample(*sampler, tex_coord);
}
