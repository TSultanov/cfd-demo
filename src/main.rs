#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use cfd_playground::App;
use eframe::NativeOptions;

fn main() -> eframe::Result {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    let native_options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1270.0, 740.0])
            .with_min_inner_size([300.0, 220.0]),
        vsync: true,
        ..Default::default()
    };
    eframe::run_native(
        "CFD Playground",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
