use crate::model::{
    Cylinder, Grid, InletProfile, Model, PressureSolver, SimSnapshot, SimulationControlHandle,
    SimulationParams, VelocityScheme,
};
use eframe::egui;

/// Updated App struct with simulation running in a background thread.
pub struct App {
    // --- New fields for logging ---
    log: Vec<String>, // stores all log messages
    // --- New field for visualization mode ---
    vis_mode: VisualizationMode,

    grid: Grid,
    simulation_handle: Option<SimulationControlHandle>,
    simulation_params: SimulationParams,
    last_snapshot: Option<SimSnapshot>,
}

// --- New helper function to create the default grid ---
fn default_grid() -> Grid {
    let nx = 400;
    let ny = 132;
    let lx = 30.0;
    let ly = 10.0;
    let dx = lx / nx as f32;
    let dy = ly / ny as f32;
    Grid {
        nx,
        ny,
        lx,
        ly,
        dx,
        dy,
        obstacle: Some(Cylinder {
            center_x: lx / 4.0,
            center_y: ly / 2.0,
            radius: 0.75,
        }),
    }
}

impl Default for App {
    fn default() -> Self {
        Self {
            // --- New initializations ---
            log: Vec::new(), // stores all log messages
            // --- New field for visualization mode ---
            vis_mode: VisualizationMode::Pressure,

            simulation_handle: None,
            grid: default_grid(),
            simulation_params: Default::default(),
            last_snapshot: None,
        }
    }
}

impl App {
    // New constructor that does NOT spawn a background thread.
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        Self::default()
    }

    fn reset_simulation(&mut self) {
        match &self.simulation_handle {
            Some(handle) => {
                handle.stop();
            }
            None => {}
        }

        self.log.clear();
        self.simulation_handle = None;
        self.last_snapshot = None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn run_simulation(&mut self) {
        let simulation = Model::new(self.grid.clone(), &self.simulation_params);
        self.simulation_handle = Some(simulation.run());
    }

    #[cfg(target_arch = "wasm32")]
    fn run_simulation(&mut self) {
        use std::sync::*;
        let (command_sender, command_receiver) = mpsc::channel();
        let (snapshot_sender, snapshot_receiver) = mpsc::channel();
        let (residuals_sender, residuals_receiver) = mpsc::channel();

        self.simulation_handle = Some(SimulationControlHandle {
            residuals_receiver,
            command_sender,
            snapshot_receiver,
        });

        let simulation = Model::new(self.grid.clone(), &self.simulation_params);
        wasm_thread::spawn(move || {
            simulation.run(command_receiver, snapshot_sender, residuals_sender);
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request the simulation state update from the simulation thread.
        if let Some(handle) = &self.simulation_handle {
            let last_available_snapshot = handle.get_last_available_snapshot();
            match last_available_snapshot {
                Some(snapshot) => {
                    self.simulation_params.dt = snapshot.dt;
                    self.last_snapshot = Some(snapshot);
                }
                None => {}
            }
        }

        // --- LEFT CONTROL PANEL: Simulation Controls ---
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                // --- START BUTTON ---
                if let Some(handle) = &self.simulation_handle {
                    if ui.button("Update").clicked() {
                        handle.set_params(self.simulation_params.clone());
                    }
                } else {
                    if ui.button("Start").clicked() {
                        self.run_simulation();
                    }
                }

                // --- PAUSE / RESUME BUTTON ---
                if let Some(handle) = &self.simulation_handle {
                    if let Some(snapshot) = &self.last_snapshot {
                        if snapshot.paused {
                            if ui.button("Resume").clicked() {
                                handle.resume();
                            }
                        } else {
                            if ui.button("Pause").clicked() {
                                handle.pause();
                            }
                        }
                    } else {
                        ui.add_enabled(false, egui::Button::new("Pause"));
                    }
                } else {
                    ui.add_enabled(false, egui::Button::new("Pause"));
                }

                if ui.button("Reset").clicked() {
                    self.reset_simulation();
                }

                ui.label("Simulation Parameters");
                {
                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.dt, 0.0..=1.0)
                            .text("Time Step")
                            .custom_formatter(|v, _range| format!("{:.6}", v)),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.simulation_params.viscosity, 0.0..=0.1)
                            .text("Viscosity")
                            .custom_formatter(|v, _range| format!("{:.6}", v)),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.simulation_params.target_inlet_velocity,
                            0.0..=5.0,
                        )
                        .text("Target Inlet Velocity")
                        .custom_formatter(|v, _range| format!("{:.6}", v)),
                    );
                    egui::ComboBox::from_label("Velocity Scheme")
                        .selected_text(format!("{:?}", self.simulation_params.velocity_scheme))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.simulation_params.velocity_scheme,
                                VelocityScheme::FirstOrder,
                                "FirstOrder Upwind",
                            );
                            ui.selectable_value(
                                &mut self.simulation_params.velocity_scheme,
                                VelocityScheme::SecondOrder,
                                "SecondOrder Upwind",
                            );
                        });
                    egui::ComboBox::from_label("Inlet Profile")
                        .selected_text(format!("{:?}", self.simulation_params.inlet_profile))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.simulation_params.inlet_profile,
                                InletProfile::Uniform,
                                "Uniform",
                            );
                            ui.selectable_value(
                                &mut self.simulation_params.inlet_profile,
                                InletProfile::Parabolic,
                                "Parabolic",
                            );
                        });
                    egui::ComboBox::from_label("Pressure Solver")
                        .selected_text(format!("{:?}", self.simulation_params.pressure_solver))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.simulation_params.pressure_solver,
                                PressureSolver::Jacobi,
                                "Jacobi",
                            );
                        });
                }
            });
        });

        // --- CENTRAL PANEL: Visualization ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Mode Selector: Buttons above the visualization ---
            ui.horizontal(|ui| {
                if ui
                    .selectable_label(self.vis_mode == VisualizationMode::Pressure, "Pressure")
                    .clicked()
                {
                    self.vis_mode = VisualizationMode::Pressure;
                }
                if ui
                    .selectable_label(self.vis_mode == VisualizationMode::Velocity, "Velocity")
                    .clicked()
                {
                    self.vis_mode = VisualizationMode::Velocity;
                }
                if ui
                    .selectable_label(self.vis_mode == VisualizationMode::Vorticity, "Vorticity")
                    .clicked()
                {
                    self.vis_mode = VisualizationMode::Vorticity;
                }
            });

            let nx = self.grid.nx;
            let ny = self.grid.ny;
            if nx == 0 || ny == 0 {
                return;
            }

            if let Some(snapshot) = &self.last_snapshot {
                let image = match self.vis_mode {
                    VisualizationMode::Pressure => {
                        let pressure = &snapshot.p;
                        let (mut min_val, mut max_val) = (f32::INFINITY, f32::NEG_INFINITY);
                        for &p in pressure.iter() {
                            if p < min_val {
                                min_val = p;
                            }
                            if p > max_val {
                                max_val = p;
                            }
                        }
                        if (max_val - min_val).abs() < 1e-6 {
                            max_val = min_val + 1.0;
                        }
                        let mut pixels = Vec::with_capacity(nx * ny);
                        for j in 0..ny {
                            for i in 0..nx {
                                let index = i + j * nx;
                                let val = pressure[index];
                                let norm = (val - min_val) / (max_val - min_val);
                                let r = (norm * 255.0) as u8;
                                let b = ((1.0 - norm) * 255.0) as u8;
                                pixels.push(egui::Color32::from_rgb(r, 0, b));
                            }
                        }
                        // Overlay the obstacle
                        if let Some(cyl) = &self.grid.obstacle {
                            for j in 0..ny {
                                for i in 0..nx {
                                    let x = (i as f32 + 0.5) * self.grid.dx;
                                    let y = (j as f32 + 0.5) * self.grid.dy;
                                    if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2))
                                        .sqrt()
                                        <= cyl.radius
                                    {
                                        pixels[i + j * nx] = egui::Color32::from_rgb(128, 128, 128);
                                    }
                                }
                            }
                        }
                        egui::ColorImage {
                            size: [nx, ny],
                            pixels,
                        }
                    }
                    VisualizationMode::Velocity => {
                        let u = &snapshot.u;
                        let v = &snapshot.v;
                        let nx_plus_one = self.grid.nx + 1;
                        let mut mags = Vec::with_capacity(nx * ny);
                        let mut min_val = f32::INFINITY;
                        let mut max_val = f32::NEG_INFINITY;
                        for j in 0..ny {
                            for i in 0..nx {
                                let u_left = u[i + j * nx_plus_one];
                                let u_right = u[i + 1 + j * nx_plus_one];
                                let u_cell = 0.5 * (u_left + u_right);
                                let v_bottom = v[i + j * self.grid.nx];
                                let v_top = v[i + (j + 1) * self.grid.nx];
                                let v_cell = 0.5 * (v_bottom + v_top);
                                let mag = (u_cell * u_cell + v_cell * v_cell).sqrt();
                                if mag < min_val {
                                    min_val = mag;
                                }
                                if mag > max_val {
                                    max_val = mag;
                                }
                                mags.push(mag);
                            }
                        }
                        if (max_val - min_val).abs() < 1e-6 {
                            max_val = min_val + 1.0;
                        }
                        let mut pixels: Vec<egui::Color32> = mags
                            .iter()
                            .map(|&mag| {
                                let norm = (mag - min_val) / (max_val - min_val);
                                let r = (norm * 255.0) as u8;
                                let b = ((1.0 - norm) * 255.0) as u8;
                                egui::Color32::from_rgb(r, 0, b)
                            })
                            .collect();
                        // Overlay the obstacle
                        if let Some(cyl) = &self.grid.obstacle {
                            for j in 0..ny {
                                for i in 0..nx {
                                    let x = (i as f32 + 0.5) * self.grid.dx;
                                    let y = (j as f32 + 0.5) * self.grid.dy;
                                    if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2))
                                        .sqrt()
                                        <= cyl.radius
                                    {
                                        pixels[i + j * nx] = egui::Color32::from_rgb(128, 128, 128);
                                    }
                                }
                            }
                        }
                        egui::ColorImage {
                            size: [nx, ny],
                            pixels,
                        }
                    }
                    VisualizationMode::Vorticity => {
                        let u = &snapshot.u;
                        let v = &snapshot.v;
                        let nx_plus_one = self.grid.nx + 1;
                        let mut vort = vec![0.0; nx * ny];
                        // Compute vorticity for interior cells using central differences.
                        for j in 1..(ny - 1) {
                            for i in 1..(nx - 1) {
                                let u_bottom =
                                    0.5 * (u[i + j * nx_plus_one] + u[i + 1 + j * nx_plus_one]);
                                let u_top = 0.5
                                    * (u[i + (j + 1) * nx_plus_one]
                                        + u[i + 1 + (j + 1) * nx_plus_one]);
                                let du_dy = (u_top - u_bottom) / self.grid.dy;
                                let v_left =
                                    0.5 * (v[i + j * self.grid.nx] + v[i + (j + 1) * self.grid.nx]);
                                let v_right = 0.5
                                    * (v[i + 1 + j * self.grid.nx]
                                        + v[i + 1 + (j + 1) * self.grid.nx]);
                                let dv_dx = (v_right - v_left) / self.grid.dx;
                                vort[i + j * nx] = dv_dx - du_dy;
                            }
                        }
                        let mut min_val = f32::INFINITY;
                        let mut max_val = f32::NEG_INFINITY;
                        for &w in vort.iter() {
                            if w < min_val {
                                min_val = w;
                            }
                            if w > max_val {
                                max_val = w;
                            }
                        }
                        if (max_val - min_val).abs() < 1e-6 {
                            max_val = min_val + 1.0;
                        }
                        let mut pixels: Vec<egui::Color32> = vort
                            .into_iter()
                            .map(|w| {
                                let norm = (w - min_val) / (max_val - min_val);
                                let r = (norm * 255.0) as u8;
                                let b = ((1.0 - norm) * 255.0) as u8;
                                egui::Color32::from_rgb(r, 0, b)
                            })
                            .collect();
                        // Overlay the obstacle
                        if let Some(cyl) = &self.grid.obstacle {
                            for j in 0..ny {
                                for i in 0..nx {
                                    let x = (i as f32 + 0.5) * self.grid.dx;
                                    let y = (j as f32 + 0.5) * self.grid.dy;
                                    if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2))
                                        .sqrt()
                                        <= cyl.radius
                                    {
                                        pixels[i + j * nx] = egui::Color32::from_rgb(128, 128, 128);
                                    }
                                }
                            }
                        }
                        egui::ColorImage {
                            size: [nx, ny],
                            pixels,
                        }
                    }
                };

                // Load the texture from the image. Using "Nearest" filtering preserves cell sharpness.
                let texture = ctx.load_texture(
                    "simulation",
                    image,
                    egui::TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Nearest,
                        mipmap_mode: Some(egui::TextureFilter::Nearest),
                        wrap_mode: egui::TextureWrapMode::ClampToEdge,
                    },
                );

                // Display the texture while maintaining the aspect ratio of the simulation domain.
                let available_size = ui.available_rect_before_wrap().size();
                let domain_aspect = self.grid.lx / self.grid.ly;
                let available_aspect = available_size.x / available_size.y;
                let (img_width, img_height) = if available_aspect > domain_aspect {
                    (available_size.y * domain_aspect, available_size.y)
                } else {
                    (available_size.x, available_size.x / domain_aspect)
                };
                let img_size = egui::Vec2::new(img_width, img_height);
                ui.image((texture.id(), img_size));
            }
        });

        // --- Update log history if a new simulation step occurred ---
        {
            if let Some(handle) = &self.simulation_handle {
                let new_log_messages = handle.get_new_log_messages();

                for message in new_log_messages {
                    let new_message = format!(
                        "Step: {}, Time: {:.3} s, dt: {:.3e} s, Pressure Residual: {:.3e}, U Residual: {:.3e}, V Residual: {:.3e}, Step computed in {:?} ({} substeps)",
                        message.simulation_step,
                        message.simulation_time,
                        message.dt,
                        message.p,
                        message.u,
                        message.v,
                        message.step_time,
                        message.piso_substeps,
                    );
                    self.log.push(new_message);
                }
            }
        }

        // --- BOTTOM PANEL: Log Display ---
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.separator();
            egui::ScrollArea::vertical()
                .min_scrolled_height(200.0)
                .show(ui, |ui| {
                    for message in &self.log {
                        ui.label(message);
                    }
                    // Autoscroll to the bottom if new message was added.
                    ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                });
        });

        match &self.simulation_handle {
            Some(handle) => {
                handle.request_snapshot();
                ctx.request_repaint();
            }
            None => {}
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum VisualizationMode {
    Pressure,
    Velocity,
    Vorticity,
}
