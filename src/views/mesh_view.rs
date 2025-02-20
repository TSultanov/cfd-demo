pub struct MeshView {

}

impl Default for MeshView {
    fn default() -> Self {
        Self {}
    }
}

impl eframe::App for MeshView {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Mesh View");
            ui.label("This is a mesh view.");
        });
    }
}