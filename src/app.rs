use eframe::egui;
use crate::model::{Grid, Cylinder, Model, VelocityScheme, InletProfile, PressureSolver};

/// Updated App struct holding simulation state and parameters.
pub struct App {
    simulation_running: bool,
    model: Model,
    // Simulation parameters (which can be controlled via the UI)
    dt: f32,
    viscosity: f32,
    target_inlet_velocity: f32,
    velocity_scheme: VelocityScheme,
    inlet_profile: InletProfile,
    pressure_solver: PressureSolver,
    simulation_time: f32, // Accumulated simulation time
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
            simulation_running: false,
            model: Model::new(grid),
            dt: 0.5,
            viscosity: 0.000001,
            target_inlet_velocity: 1.0,
            velocity_scheme: VelocityScheme::FirstOrder,
            inlet_profile: InletProfile::Uniform,
            pressure_solver: PressureSolver::Jacobi,
            simulation_time: 0.0,
        }
    }
}

impl App {
    // New constructor calling Default::default()
    pub fn new(_cc: &eframe::CreationContext) -> Self {
        Self::default()
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- LEFT CONTROL PANEL: Simulation Controls ---
        egui::SidePanel::left("control_panel").show(ctx, |ui| {
            ui.vertical(|ui| {
                if ui.button("Start").clicked() {
                    self.simulation_running = true;
                }
                if ui.button("Pause").clicked() {
                    self.simulation_running = false;
                }
                if ui.button("Reset").clicked() {
                    // Reinitialize the simulation model with the same grid parameters:
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
                    self.model = Model::new(grid);
                    self.simulation_running = false;
                    self.simulation_time = 0.0;
                }
                ui.label("Simulation Parameters");
                ui.add(egui::Slider::new(&mut self.dt, 0.001..=1.0).text("Time Step"));
                ui.add(egui::Slider::new(&mut self.viscosity, 1e-6..=0.1).text("Viscosity"));
                ui.add(egui::Slider::new(&mut self.target_inlet_velocity, 0.0..=5.0).text("Target Inlet Velocity"));
                // ComboBox for Velocity Scheme.
                egui::ComboBox::from_label("Velocity Scheme")
                    .selected_text(format!("{:?}", self.velocity_scheme))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.velocity_scheme, VelocityScheme::FirstOrder, "FirstOrder Upwind");
                        ui.selectable_value(&mut self.velocity_scheme, VelocityScheme::SecondOrder, "SecondOrder Upwind");
                        ui.selectable_value(&mut self.velocity_scheme, VelocityScheme::Quick, "QUICK Scheme");
                    });
                // ComboBox for Inlet Profile.
                egui::ComboBox::from_label("Inlet Profile")
                    .selected_text(format!("{:?}", self.inlet_profile))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.inlet_profile, InletProfile::Uniform, "Uniform");
                        ui.selectable_value(&mut self.inlet_profile, InletProfile::Parabolic, "Parabolic");
                    });
                // ComboBox for Pressure Solver.
                egui::ComboBox::from_label("Pressure Solver")
                    .selected_text(format!("{:?}", self.pressure_solver))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.pressure_solver, PressureSolver::Jacobi, "Jacobi");
                        ui.selectable_value(&mut self.pressure_solver, PressureSolver::SOR, "SOR");
                        ui.selectable_value(&mut self.pressure_solver, PressureSolver::Multigrid, "Multigrid");
                    });
            });
        });

        // --- CENTRAL PANEL: Simulation update and visualization on the right ---
        egui::CentralPanel::default().show(ctx, |ui| {
            // Update simulation if running:
            if self.simulation_running {
                self.model.set_dt(self.dt);
                self.model.set_viscosity(self.viscosity);
                self.model.set_target_inlet_velocity(self.target_inlet_velocity);
                self.model.set_velocity_scheme(self.velocity_scheme);
                self.model.set_inlet_profile(self.inlet_profile);
                self.model.set_pressure_solver(self.pressure_solver);
                self.model.update();
                // Accumulate simulation time using the current dt.
                self.simulation_time += self.dt;
            }

            let grid = &self.model.grid;
            let nx = grid.nx;
            let ny = grid.ny;
            if nx == 0 || ny == 0 {
                return;
            }

            // Get the simulation pressure field.
            let pressure = self.model.get_pressure();

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
            // Use the accumulated simulation time.
            let simulation_time = self.simulation_time;
            let log_text = format!(
                "Step: {}, Time: {:.3} s, dt: {:.3} s, Residual: {:.3e}",
                self.model.simulation_step,
                simulation_time,
                self.dt,
                self.model.get_last_pressure_residual(),
            );
            egui::ScrollArea::vertical()
                .min_scrolled_height(200.0)
                .show(ui, |ui| {
                    ui.label(log_text);
                });
        });
    }
}

