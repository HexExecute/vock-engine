#![no_std]

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Pod, Zeroable, Clone, Copy)]
pub struct ShaderConstants {
    pub tree_root: u32,
    pub tree_scale: u32,
}
