use eframe::egui;
use crate::model::{Grid, Cylinder, Model, VelocityScheme, InletProfile, PressureSolver};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Updated App struct with simulation running in a background thread.
pub struct App {
    simulation_running: Arc<AtomicBool>,
    simulation_snapshot: Option<SimSnapshot>,
    simulation_req_tx: Option<std::sync::mpsc::Sender<()>>,
    simulation_res_rx: Option<std::sync::mpsc::Receiver<SimSnapshot>>,
    simulation_params: Arc<Mutex<SimulationParams>>,
    // --- New fields for logging ---
    log: Vec<String>,              // stores all log messages
    last_logged_step: usize,       // last simulation step for which a message was logged
    should_autoscroll: bool,       // trigger to autoscroll the log view
    // --- New field for visualization mode ---
    vis_mode: VisualizationMode,
    // --- New field for managing the simulation thread ---
    simulation_handle: Option<thread::JoinHandle<()>>,
}

/// New helper structure to hold the simulation state.
pub struct SimulationState {
    pub model: Model,
    pub simulation_time: f32,
}

/// A snapshot structure to copy the data needed for visualization and logging.
#[derive(Clone)]
pub struct SimSnapshot {
    pub grid: Grid,
    pub pressure: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub simulation_step: usize,
    pub dt: f32,
    pub last_pressure_residual: f32,
    pub last_u_residual: f32,
    pub last_v_residual: f32,
    pub simulation_time: f32,
}

/// New helper structure to hold simulation parameters.
pub struct SimulationParams {
    pub dt: f32,
    pub viscosity: f32,
    pub target_inlet_velocity: f32,
    pub velocity_scheme: VelocityScheme,
    pub inlet_profile: InletProfile,
    pub pressure_solver: PressureSolver,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt: 0.5,
            viscosity: 0.000001,
            target_inlet_velocity: 1.0,
            velocity_scheme: VelocityScheme::FirstOrder,
            inlet_profile: InletProfile::Uniform,
            pressure_solver: PressureSolver::Jacobi,
        }
    }
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
        let grid = default_grid();
        Self {
            simulation_running: Arc::new(AtomicBool::new(false)),
            simulation_snapshot: None,
            simulation_req_tx: None,
            simulation_res_rx: None,
            simulation_params: Arc::new(Mutex::new(SimulationParams::default())),
            // --- New initializations ---
            log: Vec::new(),              // stores all log messages
            last_logged_step: 0,       // last simulation step for which a message was logged
            should_autoscroll: false,       // trigger to autoscroll the log view
            // --- New field for visualization mode ---
            vis_mode: VisualizationMode::Pressure,
            simulation_handle: None,
        }
    }
}

impl App {
    // New constructor that does NOT spawn a background thread.
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        Self::default()
    }

    /// Resets the simulation by reinitializing the model, simulation time, and resetting logs.
    fn reset_simulation(&mut self) {
        // Stop the simulation thread.
        self.simulation_running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.simulation_handle.take() {
            handle.join().unwrap();
        }

        // Clear the snapshot and channels.
        self.simulation_snapshot = None;
        self.simulation_req_tx = None;
        self.simulation_res_rx = None;

        // Clear the log and reset tracking.
        self.log.clear();
        self.last_logged_step = 0;
        self.should_autoscroll = false;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request the simulation state update from the simulation thread.
        if let Some(req_tx) = &self.simulation_req_tx {
            let _ = req_tx.send(());
        }
        // Drain any simulation state responses waiting.
        if let Some(rx) = self.simulation_res_rx.as_mut() {
            while let Ok(snapshot) = rx.try_recv() {
                self.simulation_snapshot = Some(snapshot);
            }
        }

        // --- LEFT CONTROL PANEL: Simulation Controls ---
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                // --- START BUTTON ---
                if ui.button("Start").clicked() {
                    if self.simulation_handle.is_none() {
                        self.simulation_running.store(true, Ordering::Relaxed);
                        // Create channels for requests and responses.
                        let (req_tx, req_rx) = std::sync::mpsc::channel::<()>();
                        let (res_tx, res_rx) = std::sync::mpsc::channel::<SimSnapshot>();
                        self.simulation_req_tx = Some(req_tx);
                        self.simulation_res_rx = Some(res_rx);
                        let simulation_params = Arc::clone(&self.simulation_params);
                        let simulation_running = Arc::clone(&self.simulation_running);
                        self.simulation_handle = Some(thread::spawn(move || {
                            // Create a local simulation state.
                            let mut sim_state = SimulationState {
                                model: Model::new(default_grid()),
                                simulation_time: 0.0,
                            };
                            while simulation_running.load(Ordering::Relaxed) {
                                let start_time = Instant::now();
                                // Copy simulation parameters while holding only one lock.
                                let (dt, viscosity, target_inlet_velocity, velocity_scheme, inlet_profile, pressure_solver) = {
                                    let params = simulation_params.lock().unwrap();
                                    (
                                        params.dt,
                                        params.viscosity,
                                        params.target_inlet_velocity,
                                        params.velocity_scheme,
                                        params.inlet_profile,
                                        params.pressure_solver,
                                    )
                                };
                                // When a UI request is pending, send a copy of the simulation snapshot.
                                while let Ok(()) = req_rx.try_recv() {
                                    let _ = res_tx.send(sim_state.snapshot());
                                }
                                {
                                    // Update simulation state using the copied parameters.
                                    sim_state.model.set_dt(dt);
                                    sim_state.model.set_viscosity(viscosity);
                                    sim_state.model.set_target_inlet_velocity(target_inlet_velocity);
                                    sim_state.model.set_velocity_scheme(velocity_scheme);
                                    sim_state.model.set_inlet_profile(inlet_profile);
                                    sim_state.model.set_pressure_solver(pressure_solver);
                                    sim_state.model.update();
                                    sim_state.simulation_time += dt;
                                }
                                println!("Time taken: {:?}", start_time.elapsed());
                                std::thread::yield_now();
                            }
                        }));
                    }
                }

                // --- PAUSE BUTTON ---
                if ui.button("Pause").clicked() {
                    self.simulation_running.store(false, Ordering::Relaxed);
                    if let Some(handle) = self.simulation_handle.take() {
                        handle.join().unwrap();
                    }
                }

                if ui.button("Reset").clicked() {
                    self.reset_simulation();
                }
                ui.label("Simulation Parameters");
                {
                    let mut params = self.simulation_params.lock().unwrap();
                    ui.add(egui::Slider::new(&mut params.dt, 0.0..=1.0)
                        .text("Time Step")
                        .custom_formatter(|v, _range| format!("{:.6}", v)));
                    ui.add(egui::Slider::new(&mut params.viscosity, 0.0..=0.1)
                        .text("Viscosity")
                        .custom_formatter(|v, _range| format!("{:.6}", v)));
                    ui.add(egui::Slider::new(&mut params.target_inlet_velocity, 0.0..=5.0)
                        .text("Target Inlet Velocity")
                        .custom_formatter(|v, _range| format!("{:.6}", v)));
                    egui::ComboBox::from_label("Velocity Scheme")
                        .selected_text(format!("{:?}", params.velocity_scheme))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut params.velocity_scheme, VelocityScheme::FirstOrder, "FirstOrder Upwind");
                            ui.selectable_value(&mut params.velocity_scheme, VelocityScheme::SecondOrder, "SecondOrder Upwind");
                            ui.selectable_value(&mut params.velocity_scheme, VelocityScheme::Quick, "QUICK Scheme");
                        });
                    egui::ComboBox::from_label("Inlet Profile")
                        .selected_text(format!("{:?}", params.inlet_profile))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut params.inlet_profile, InletProfile::Uniform, "Uniform");
                            ui.selectable_value(&mut params.inlet_profile, InletProfile::Parabolic, "Parabolic");
                        });
                    egui::ComboBox::from_label("Pressure Solver")
                        .selected_text(format!("{:?}", params.pressure_solver))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut params.pressure_solver, PressureSolver::Jacobi, "Jacobi");
                            ui.selectable_value(&mut params.pressure_solver, PressureSolver::SOR, "SOR");
                        });
                }
            });
        });

        // --- CENTRAL PANEL: Visualization ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Mode Selector: Buttons above the visualization ---
            ui.horizontal(|ui| {
                if ui.selectable_label(self.vis_mode == VisualizationMode::Pressure, "Pressure").clicked() {
                    self.vis_mode = VisualizationMode::Pressure;
                }
                if ui.selectable_label(self.vis_mode == VisualizationMode::Velocity, "Velocity").clicked() {
                    self.vis_mode = VisualizationMode::Velocity;
                }
                if ui.selectable_label(self.vis_mode == VisualizationMode::Vorticity, "Vorticity").clicked() {
                    self.vis_mode = VisualizationMode::Vorticity;
                }
            });

            let snapshot = match &self.simulation_snapshot {
                Some(s) => s,
                None => return,
            };
            let grid = &snapshot.grid;
            let nx = grid.nx;
            let ny = grid.ny;
            if nx == 0 || ny == 0 {
                return;
            }

            let image = match self.vis_mode {
                VisualizationMode::Pressure => {
                    let pressure = &snapshot.pressure;
                    let (mut min_val, mut max_val) = (f32::INFINITY, f32::NEG_INFINITY);
                    for &p in pressure.iter() {
                        if p < min_val { min_val = p; }
                        if p > max_val { max_val = p; }
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
                            let b = (((1.0 - norm) * 255.0)) as u8;
                            pixels.push(egui::Color32::from_rgb(r, 0, b));
                        }
                    }
                    // Overlay the obstacle
                    if let Some(cyl) = &grid.obstacle {
                        for j in 0..ny {
                            for i in 0..nx {
                                let x = (i as f32 + 0.5) * grid.dx;
                                let y = (j as f32 + 0.5) * grid.dy;
                                if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2)).sqrt() <= cyl.radius {
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
                    let nx_plus_one = grid.nx + 1;
                    let mut mags = Vec::with_capacity(nx * ny);
                    let mut min_val = f32::INFINITY;
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..ny {
                        for i in 0..nx {
                            let u_left = u[i + j * nx_plus_one];
                            let u_right = u[i + 1 + j * nx_plus_one];
                            let u_cell = 0.5 * (u_left + u_right);
                            let v_bottom = v[i + j * grid.nx];
                            let v_top = v[i + (j + 1) * grid.nx];
                            let v_cell = 0.5 * (v_bottom + v_top);
                            let mag = (u_cell * u_cell + v_cell * v_cell).sqrt();
                            if mag < min_val { min_val = mag; }
                            if mag > max_val { max_val = mag; }
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
                            let b = (((1.0 - norm) * 255.0)) as u8;
                            egui::Color32::from_rgb(r, 0, b)
                        })
                        .collect();
                    // Overlay the obstacle
                    if let Some(cyl) = &grid.obstacle {
                        for j in 0..ny {
                            for i in 0..nx {
                                let x = (i as f32 + 0.5) * grid.dx;
                                let y = (j as f32 + 0.5) * grid.dy;
                                if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2)).sqrt() <= cyl.radius {
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
                    let nx_plus_one = grid.nx + 1;
                    let mut vort = vec![0.0; nx * ny];
                    // Compute vorticity for interior cells using central differences.
                    for j in 1..(ny - 1) {
                        for i in 1..(nx - 1) {
                            let u_bottom = 0.5 * (u[i + j * nx_plus_one] + u[i + 1 + j * nx_plus_one]);
                            let u_top = 0.5 * (u[i + (j + 1) * nx_plus_one] + u[i + 1 + (j + 1) * nx_plus_one]);
                            let du_dy = (u_top - u_bottom) / grid.dy;
                            let v_left = 0.5 * (v[i + j * grid.nx] + v[i + (j + 1) * grid.nx]);
                            let v_right = 0.5 * (v[i + 1 + j * grid.nx] + v[i + 1 + (j + 1) * grid.nx]);
                            let dv_dx = (v_right - v_left) / grid.dx;
                            vort[i + j * nx] = dv_dx - du_dy;
                        }
                    }
                    let mut min_val = f32::INFINITY;
                    let mut max_val = f32::NEG_INFINITY;
                    for &w in vort.iter() {
                        if w < min_val { min_val = w; }
                        if w > max_val { max_val = w; }
                    }
                    if (max_val - min_val).abs() < 1e-6 {
                        max_val = min_val + 1.0;
                    }
                    let mut pixels: Vec<egui::Color32> = vort
                        .into_iter()
                        .map(|w| {
                            let norm = (w - min_val) / (max_val - min_val);
                            let r = (norm * 255.0) as u8;
                            let b = (((1.0 - norm) * 255.0)) as u8;
                            egui::Color32::from_rgb(r, 0, b)
                        })
                        .collect();
                    // Overlay the obstacle
                    if let Some(cyl) = &grid.obstacle {
                        for j in 0..ny {
                            for i in 0..nx {
                                let x = (i as f32 + 0.5) * grid.dx;
                                let y = (j as f32 + 0.5) * grid.dy;
                                if ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2)).sqrt() <= cyl.radius {
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
            let domain_aspect = grid.lx / grid.ly;
            let available_aspect = available_size.x / available_size.y;
            let (img_width, img_height) = if available_aspect > domain_aspect {
                (available_size.y * domain_aspect, available_size.y)
            } else {
                (available_size.x, available_size.x / domain_aspect)
            };
            let img_size = egui::Vec2::new(img_width, img_height);
            ui.image((texture.id(), img_size));
        });

        // --- New: Update log history if a new simulation step occurred ---
        {
            if let Some(snapshot) = &self.simulation_snapshot {
                let current_step = snapshot.simulation_step;
                if current_step > self.last_logged_step {
                    let new_message = format!(
                        "Step: {}, Time: {:.3} s, dt: {:.3e} s, Pressure Residual: {:.3e}, U Residual: {:.3e}, V Residual: {:.3e}",
                        current_step,
                        snapshot.simulation_time,
                        snapshot.dt,
                        snapshot.last_pressure_residual,
                        snapshot.last_u_residual,
                        snapshot.last_v_residual,
                    );
                    self.log.push(new_message);
                    self.last_logged_step = current_step;
                    self.should_autoscroll = true;
                }
            }
        }

        // --- BOTTOM PANEL: Log Display with Autoscroll ---
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.separator();
            egui::ScrollArea::vertical()
                .min_scrolled_height(200.0)
                .show(ui, |ui| {
                    for message in &self.log {
                        ui.label(message);
                    }
                    // Autoscroll to the bottom if new message was added.
                    if self.should_autoscroll {
                        ui.scroll_to_cursor(Some(egui::Align::BOTTOM));
                    }
                });
        });
        // Reset autoscroll flag after rendering the log.
        self.should_autoscroll = false;

        // Propagate automatic dt updates from the model to the simulation parameters,
        // so that the UI slider and the log display the current dt.
        {
            let snapshot = match &self.simulation_snapshot {
                Some(s) => s,
                None => return,
            };
            let new_dt = snapshot.dt;

            let mut params = self.simulation_params.lock().unwrap();
            params.dt = new_dt;
        }

        // Request a repaint if the simulation is running so that the UI updates continuously.
        if self.simulation_running.load(Ordering::Relaxed) {
            ctx.request_repaint();
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum VisualizationMode {
    Pressure,
    Velocity,
    Vorticity,
}

impl SimulationState {
    fn snapshot(&self) -> SimSnapshot {
        SimSnapshot {
            grid: self.model.grid.clone(), // now works after implementing Clone for Grid in model.rs
            pressure: self.model.get_pressure().to_vec(),  // convert &[f32] to Vec<f32>
            u: self.model.get_u().to_vec(),                // convert &[f32] to Vec<f32>
            v: self.model.get_v().to_vec(),                // convert &[f32] to Vec<f32>
            simulation_step: self.model.simulation_step,
            dt: self.model.dt,
            last_pressure_residual: self.model.get_last_pressure_residual(),
            last_u_residual: self.model.get_last_u_residual(),
            last_v_residual: self.model.get_last_v_residual(),
            simulation_time: self.simulation_time,
        }
    }
}

