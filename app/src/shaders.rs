// #[cfg(not(feature = "use-glsl-shader"))]
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        root_path_env: "SHADER_OUT_DIR",
        bytes: "pixel-main_vs.spv",
    }
}

// #[cfg(not(feature = "use-glsl-shader"))]
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        root_path_env: "SHADER_OUT_DIR",
        bytes: "pixel-main_fs.spv",
    }
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        root_path_env: "SHADER_OUT_DIR",
        bytes: "raytracing-main_cs.spv"
    }
}
