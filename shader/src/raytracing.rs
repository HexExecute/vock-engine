const SPHERE_POSITION: Vec3 = vec3(0.0, 0.0, 4.0);

use shared::ShaderConstants;
use spirv_std::{
    glam::{ivec3, vec2, vec3, vec4, BVec3, UVec2, UVec3, Vec2, Vec3},
    num_traits::float::Float,
    spirv, Image,
};
use voxtree::PackedVoxtree;

struct Ray {
    position: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(uv: Vec2, camera: Camera) -> Self {
        let direction = Self::direction(uv, &camera);

        Self {
            position: camera.position,
            direction,
        }
    }

    fn direction(uv: Vec2, camera: &Camera) -> Vec3 {
        vec3(uv.x * camera.fov * 2.0, uv.y, 1.0).normalize()
    }

    pub fn trace(&mut self, tree: &PackedVoxtree<u32>) -> Vec3 {
        // let l = SPHERE_POSITION - self.position;
        // let tca = l.dot(self.direction);
        // let d2 = l.dot(l) - tca * tca;
        // if d2 > 1.0 {
        //     return Vec3::ZERO;
        // }
        // let thc = (1.0 - d2).sqrt();
        // let t0 = tca - thc;
        //
        // let hit = self.position + self.direction * t0;
        // let normal = (hit - SPHERE_POSITION).normalize();
        //
        // normal * 0.5 + 0.5

        let mut map_pos = self.position.as_ivec3();
        let delta_dist = (self.direction.length() / self.direction).abs();
        let ray_step = self.direction.signum().as_ivec3();
        let mut side_dist = (self.direction.signum() * (map_pos.as_vec3() - self.position)
            + self.direction.signum() * 0.5
            + 0.5)
            * delta_dist;
        let mut mask = vec3(0.0, 0.0, 0.0);

        for _ in 0..64 {
            if tree.voxels[tree.fetch(map_pos.x as u32, map_pos.y as u32, map_pos.z as u32, 8)] == 0
            {
                break;
            }
            mask = Vec3 {
                x: if side_dist.x <= side_dist.y.min(side_dist.z) {
                    1.0
                } else {
                    0.0
                },
                y: if side_dist.y <= side_dist.z.min(side_dist.x) {
                    1.0
                } else {
                    0.0
                },
                z: if side_dist.z <= side_dist.x.min(side_dist.y) {
                    1.0
                } else {
                    0.0
                },
            };

            side_dist += mask * delta_dist;
            map_pos += mask.as_ivec3() * ray_step;
        }

        let mut color = vec3(0.0, 0.0, 0.0);
        if mask.x == 1.0 {
            color = vec3(0.5, 0.5, 0.5);
        }
        if mask.y == 1.0 {
            color = vec3(1.0, 1.0, 1.0);
        }
        if mask.z == 1.0 {
            color = vec3(0.75, 0.75, 0.75);
        }
        color
    }

    fn step(&mut self) {}
}

struct Camera {
    position: Vec3,
    fov: f32,
}

#[spirv(compute(threads(16, 16, 1)))]
pub fn main_cs(
    #[spirv(global_invocation_id)] invocation_id: UVec3,
    #[spirv(descriptor_set = 0, binding = 0)] image: &Image!(2D, format = rgba8, sampled = false),

    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] nodes: &[[u32; 8]],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] voxels: &[u32],

    #[spirv(push_constant)] constants: &ShaderConstants,
) {
    let image_size: Vec2 = image.query_size::<UVec2>().as_vec2();
    let uv = vec2(
        invocation_id.x as f32 / image_size.x - 0.5,
        invocation_id.y as f32 / image_size.y - 0.5,
    );

    let camera = Camera {
        position: vec3(0.0, 0.0, 0.0),
        fov: 0.5, // 1.0 = 1pi rad = 180 deg
    };
    let mut ray = Ray::new(uv, camera);

    let tree = PackedVoxtree {
        nodes,
        voxels,
        root: constants.tree_root,
        scale: constants.tree_scale,
    };

    tree.fetch(0, 0, 0, 8);

    let color = ray.trace(&tree);

    unsafe {
        image.write(
            UVec2 {
                x: invocation_id.x,
                y: invocation_id.y,
            },
            vec4(color.x, color.y, color.z, 1.0), // vec4(0.0, 0.0, 0.0, 1.0),
        );
    }
}
