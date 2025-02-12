use eframe::egui;
use crate::model::{Grid, Cylinder, Model, VelocityScheme, InletProfile, PressureSolver};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Updated App struct with simulation running in a background thread.
pub struct App {
    simulation_running: Arc<AtomicBool>,
    simulation_state: Arc<Mutex<SimulationState>>,
    simulation_params: Arc<Mutex<SimulationParams>>,
    // --- New fields for logging ---
    log: Vec<String>,              // stores all log messages
    last_logged_step: usize,       // last simulation step for which a message was logged
    should_autoscroll: bool,       // trigger to autoscroll the log view
    // --- New field for visualization mode ---
    vis_mode: VisualizationMode,
}

/// New helper structure to hold the simulation state.
pub struct SimulationState {
    pub model: Model,
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

impl Default for App {
    fn default() -> Self {
        // Create a default grid matching the HTML reference:
        let nx = 400;
        let ny = 132;
        let lx = 30.0;
        let ly = 10.0;
        let dx = lx / nx as f32;
        let dy = ly / ny as f32;
        let grid = Grid {
            nx,
            ny,
            lx,
            ly,
            dx,
            dy,
            obstacle: Some(Cylinder {
                center_x: lx / 4.0,       // place cylinder at one-fourth of the channel length
                center_y: ly / 2.0,       // centered vertically
                radius: 0.75,
            }),
        };
        Self {
            simulation_running: Arc::new(AtomicBool::new(false)),
            simulation_state: Arc::new(Mutex::new(SimulationState {
                model: Model::new(grid),
                simulation_time: 0.0,
            })),
            simulation_params: Arc::new(Mutex::new(SimulationParams::default())),
            // --- New initializations ---
            log: Vec::new(),              // stores all log messages
            last_logged_step: 0,       // last simulation step for which a message was logged
            should_autoscroll: false,       // trigger to autoscroll the log view
            // --- New field for visualization mode ---
            vis_mode: VisualizationMode::Pressure,
        }
    }
}

impl App {
    // New constructor that also spawns the simulation background thread.
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        let app = Self::default();

        // Spawn background simulation thread.
        let simulation_state = Arc::clone(&app.simulation_state);
        let simulation_params = Arc::clone(&app.simulation_params);
        let simulation_running = Arc::clone(&app.simulation_running);

        thread::spawn(move || loop {
            if simulation_running.load(Ordering::Relaxed) {
                // Lock simulation parameters and state.
                let params = simulation_params.lock().unwrap();
                let mut state = simulation_state.lock().unwrap();
                state.model.set_dt(params.dt);
                state.model.set_viscosity(params.viscosity);
                state
                    .model
                    .set_target_inlet_velocity(params.target_inlet_velocity);
                state.model.set_velocity_scheme(params.velocity_scheme);
                state.model.set_inlet_profile(params.inlet_profile);
                state.model.set_pressure_solver(params.pressure_solver);
                state.model.update();
                state.simulation_time += params.dt;
                // Locks are released here.
            }
            thread::sleep(Duration::from_millis(16));
        });

        app
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- LEFT CONTROL PANEL: Simulation Controls ---
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                if ui.button("Start").clicked() {
                    self.simulation_running.store(true, Ordering::Relaxed);
                }
                if ui.button("Pause").clicked() {
                    self.simulation_running.store(false, Ordering::Relaxed);
                }
                if ui.button("Reset").clicked() {
                    // Reinitialize the simulation model and simulation time:
                    let nx = 400;
                    let ny = 132;
                    let lx = 30.0;
                    let ly = 10.0;
                    let dx = lx / nx as f32;
                    let dy = ly / ny as f32;
                    let grid = Grid {
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
                    };
                    let mut state = self.simulation_state.lock().unwrap();
                    state.model = Model::new(grid);
                    state.simulation_time = 0.0;
                    self.simulation_running.store(false, Ordering::Relaxed);

                    // --- New: Clear the log and reset the simulation step tracker ---
                    self.log.clear();
                    self.last_logged_step = 0;
                    self.should_autoscroll = false;
                }
                ui.label("Simulation Parameters");
                {
                    let mut params = self.simulation_params.lock().unwrap();
                    ui.add(egui::Slider::new(&mut params.dt, 0.001..=1.0).text("Time Step"));
                    ui.add(egui::Slider::new(&mut params.viscosity, 1e-6..=0.1).text("Viscosity"));
                    ui.add(egui::Slider::new(&mut params.target_inlet_velocity, 0.0..=5.0).text("Target Inlet Velocity"));
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
                            ui.selectable_value(&mut params.pressure_solver, PressureSolver::Multigrid, "Multigrid");
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

            let state = self.simulation_state.lock().unwrap();
            let grid = &state.model.grid;
            let nx = grid.nx;
            let ny = grid.ny;
            if nx == 0 || ny == 0 {
                return;
            }

            let image = match self.vis_mode {
                VisualizationMode::Pressure => {
                    // Pressure visualization.
                    let pressure = state.model.get_pressure();
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
                    egui::ColorImage {
                        size: [nx, ny],
                        pixels,
                    }
                }
                VisualizationMode::Velocity => {
                    // Velocity magnitude visualization.
                    let u = state.model.get_u();
                    let v = state.model.get_v();
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
                    let pixels: Vec<egui::Color32> = mags
                        .iter()
                        .map(|&mag| {
                            let norm = (mag - min_val) / (max_val - min_val);
                            let r = (norm * 255.0) as u8;
                            let b = (((1.0 - norm) * 255.0)) as u8;
                            egui::Color32::from_rgb(r, 0, b)
                        })
                        .collect();
                    egui::ColorImage {
                        size: [nx, ny],
                        pixels,
                    }
                }
                VisualizationMode::Vorticity => {
                    // Vorticity visualization.
                    let u = state.model.get_u();
                    let v = state.model.get_v();
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
                    let pixels: Vec<egui::Color32> = vort
                        .into_iter()
                        .map(|w| {
                            let norm = (w - min_val) / (max_val - min_val);
                            let r = (norm * 255.0) as u8;
                            let b = (((1.0 - norm) * 255.0)) as u8;
                            egui::Color32::from_rgb(r, 0, b)
                        })
                        .collect();
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
            let state = self.simulation_state.lock().unwrap();
            let current_step = state.model.simulation_step;
            if current_step > self.last_logged_step {
                let new_message = format!(
                    "Step: {}, Time: {:.3} s, dt: {:.3} s, Residual: {:.3e}",
                    state.model.simulation_step,
                    state.simulation_time,
                    self.simulation_params.lock().unwrap().dt,
                    state.model.get_last_pressure_residual(),
                );
                self.log.push(new_message);
                self.last_logged_step = current_step;
                self.should_autoscroll = true;
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

