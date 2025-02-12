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
            let state = self.simulation_state.lock().unwrap();
            let grid = &state.model.grid;
            let nx = grid.nx;
            let ny = grid.ny;
            if nx == 0 || ny == 0 {
                return;
            }

            // Get the simulation pressure field.
            let pressure = state.model.get_pressure();

            // Compute min and max pressure for color mapping.
            let (mut min_p, mut max_p) = (f32::INFINITY, f32::NEG_INFINITY);
            for &p_val in pressure.iter() {
                if p_val < min_p { min_p = p_val; }
                if p_val > max_p { max_p = p_val; }
            }
            if (max_p - min_p).abs() < 1e-6 {
                max_p = min_p + 1.0;
            }

            // Create a color image from the pressure field, mapping one cell to one pixel.
            let mut pixels = Vec::with_capacity(nx * ny);
            for j in 0..ny {
                for i in 0..nx {
                    let index = i + j * nx;
                    let val = pressure[index];
                    let norm = (val - min_p) / (max_p - min_p);
                    let r = (norm * 255.0) as u8;
                    let b = (((1.0 - norm) * 255.0)) as u8;
                    pixels.push(egui::Color32::from_rgb(r, 0, b));
                }
            }

            let image = egui::ColorImage {
                size: [nx, ny],
                pixels,
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

            // Display the texture as an image that stretches to the available space.
            let available = ui.available_rect_before_wrap();
            ui.image((texture.id(), available.size()));
        });

        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            ui.separator();
            let state = self.simulation_state.lock().unwrap();
            let simulation_time = state.simulation_time;
            let log_text = format!(
                "Step: {}, Time: {:.3} s, dt: {:.3} s, Residual: {:.3e}",
                state.model.simulation_step,
                simulation_time,
                self.simulation_params.lock().unwrap().dt,
                state.model.get_last_pressure_residual(),
            );
            egui::ScrollArea::vertical()
                .min_scrolled_height(200.0)
                .show(ui, |ui| {
                    ui.label(log_text);
                });
        });

        // Request a repaint if the simulation is running so that the UI updates continuously.
        if self.simulation_running.load(Ordering::Relaxed) {
            ctx.request_repaint();
        }
    }
}

