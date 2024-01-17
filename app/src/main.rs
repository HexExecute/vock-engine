mod app;
mod pixels_pipeline;
mod raytracing_pipeline;
mod shaders;

use app::State;
pub use shaders::{cs, fs, vs};
use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new();

    let mut state = State::new(&event_loop);

    event_loop.run(move |event, _, control_flow| state.handle_event(event, control_flow));
}
