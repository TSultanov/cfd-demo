use nalgebra::DMatrix;

/// A simple grid definition holding the number of pressure cells,
/// as well as the physical dimensions and cell sizes.
pub struct Grid {
    pub nx: usize, // number of pressure cells in x-direction
    pub ny: usize, // number of pressure cells in y-direction
    pub lx: f32,
    pub ly: f32,
    pub dx: f32,
    pub dy: f32,
    /// Optionally, we set a circular obstacle.
    pub obstacle: Option<Cylinder>,
}

/// A circular obstacle defined by its center and radius.
pub struct Cylinder {
    pub center_x: f32,
    pub center_y: f32,
    pub radius: f32,
}

/// Which convection scheme is used for the velocity discretization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityScheme {
    FirstOrder,
    SecondOrder,
    Quick,
}

/// Which pressure solver to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureSolver {
    Jacobi,
    SOR,
    Multigrid,
}

/// Inlet velocity profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InletProfile {
    Uniform,
    Parabolic,
}

/// The simulation "model". Here we store all simulation parameters and
/// fields. The staggered grid is defined as follows:
/// • Pressure: size = nx * ny
/// • u: size = (nx+1) * ny (defined on vertical faces)
/// • v: size = nx * (ny+1)  (defined on horizontal faces)
pub struct Model {
    pub grid: Grid,

    // Simulation parameters
    pub dt: f32,
    pub nu: f32,
    pub substep_count: usize,
    pub simulation_step: usize,
    pub ramp_up_steps: usize,
    pub current_inlet_velocity: f32,
    pub target_inlet_velocity: f32,
    pub velocity_scheme: VelocityScheme,
    pub pressure_solver: PressureSolver,
    pub inlet_profile: InletProfile,

    // Primary fields
    /// Horizontal velocities on vertical cell faces: size = (nx+1)*ny.
    pub u: Vec<f32>,
    /// Vertical velocities on horizontal cell faces: size = nx*(ny+1).
    pub v: Vec<f32>,
    /// Pressure field: size = nx*ny.
    pub p: Vec<f32>,

    // Previous fields (for extrapolation and residual computation)
    pub u_prev: Vec<f32>,
    pub v_prev: Vec<f32>,
    pub u_old: Vec<f32>,
    pub v_old: Vec<f32>,

    // Predictor (star) fields
    pub u_star: Vec<f32>,
    pub v_star: Vec<f32>,

    /// Right-hand side for pressure correction (divergence residual)
    pub rhs: Vec<f32>,
    /// Pressure correction field and temporary buffer for Jacobi iteration.
    pub p_prime: Vec<f32>,
    pub p_prime_new: Vec<f32>,

    /// Precomputed obstacle mask on the pressure grid (nx*ny).
    pub obstacle: Vec<bool>,

    /// The last pressure residual (for time–step scaling).
    pub last_pressure_residual: f32,
}

impl Model {
    /// Create a new simulation model given a grid.
    /// (The field vectors are allocated based on grid dimensions.)
    pub fn new(grid: Grid) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;

        // Store local copies before moving grid into the struct.
        let dx = grid.dx;
        let dy = grid.dy;

        let size_u = (nx + 1) * ny;
        let size_v = nx * (ny + 1);
        let size_p = nx * ny;

        let u = vec![0.0; size_u];
        let v = vec![0.0; size_v];
        let p = vec![0.0; size_p];

        // Compute the obstacle mask from grid before moving it.
        let obstacle_mask = if let Some(ref cyl) = grid.obstacle {
            let mut mask = vec![false; size_p];
            for j in 0..ny {
                for i in 0..nx {
                    let idx = i + j * nx;
                    // Center of a pressure cell using local dx, dy:
                    let x = (i as f32 + 0.5) * dx;
                    let y = (j as f32 + 0.5) * dy;
                    mask[idx] = ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2))
                        .sqrt()
                        <= cyl.radius;
                }
            }
            mask
        } else {
            vec![false; size_p]
        };

        Self {
            grid,
            dt: 0.5,
            nu: 0.000001,
            substep_count: 5,
            simulation_step: 0,
            ramp_up_steps: 1000,
            current_inlet_velocity: 0.0,
            target_inlet_velocity: 1.0,
            velocity_scheme: VelocityScheme::FirstOrder,
            pressure_solver: PressureSolver::Jacobi,
            inlet_profile: InletProfile::Uniform,
            u: u.clone(),
            v: v.clone(),
            p,
            u_prev: u.clone(),
            v_prev: v.clone(),
            u_old: u.clone(),
            v_old: v.clone(),
            u_star: u.clone(),
            v_star: v.clone(),
            rhs: vec![0.0; size_p],
            p_prime: vec![0.0; size_p],
            p_prime_new: vec![0.0; size_p],
            obstacle: obstacle_mask,
            last_pressure_residual: 0.0,
        }
    }

    /// Perform a simulation update (one "time step").
    /// Implements the extrapolation, multiple substeps (the PISO algorithm)
    /// and automatic dt adjustment.
    pub fn update(&mut self) {
        // Extrapolate previous velocities if not at the very first step.
        if self.simulation_step > 0 {
            for i in 0..self.u.len() {
                self.u[i] = 2.0 * self.u[i] - self.u_prev[i];
            }
            for i in 0..self.v.len() {
                self.v[i] = 2.0 * self.v[i] - self.v_prev[i];
            }
        }

        // Save current fields for residual computation.
        self.u_old.copy_from_slice(&self.u);
        self.v_old.copy_from_slice(&self.v);

        // Gradually ramp up inlet velocity.
        if self.simulation_step < self.ramp_up_steps {
            self.current_inlet_velocity =
                (self.simulation_step as f32 / self.ramp_up_steps as f32) * self.target_inlet_velocity;
        } else {
            self.current_inlet_velocity = self.target_inlet_velocity;
        }
        let dt_sub = self.dt / self.substep_count as f32;
        let mut max_pressure_residual = 0.0;

        // Perform the required number of substeps (PISO substeps)
        for _ in 0..self.substep_count {
            self.piso_step(dt_sub);
            let p_residual = self.get_last_pressure_residual();
            if p_residual > max_pressure_residual {
                max_pressure_residual = p_residual;
            }
        }

        // Compute residual differences between the updated velocity fields and their old values.
        let max_residual_u: f32 = self
            .u
            .iter()
            .zip(self.u_old.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f32, |acc, delta| acc.max(delta));
        let max_residual_v: f32 = self
            .v
            .iter()
            .zip(self.v_old.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f32, |acc, delta| acc.max(delta));

        self.simulation_step += 1;

        // Adjust the number of substeps if the error norm is too high.
        let error_norm: f32 = max_residual_u.max(max_residual_v).max(max_pressure_residual);
        let tolerance = 1e-3;
        if error_norm > tolerance {
            let factor = error_norm / tolerance;
            self.substep_count =
                ((self.substep_count as f32) * factor).ceil().min(20.0) as usize;
        } else if error_norm < tolerance / 10.0 && self.substep_count > 1 {
            self.substep_count = (self.substep_count as f32 / 2.0).floor() as usize;
            if self.substep_count < 1 {
                self.substep_count = 1;
            }
        }

        // Automatic dt control (using a simple CFL condition).
        let previous_dt = self.dt;
        let dt_cfl = self.compute_automatic_time_step();

        // Optionally scale dt based on pressure residual.
        let new_dt = if max_pressure_residual > 1e-3 {
            dt_cfl * (1e-3 / (max_pressure_residual + 1e-10))
        } else {
            dt_cfl
        };

        // Limit increase of dt to keep changes smooth.
        let max_increase_factor = 1.1;
        self.dt = if new_dt > previous_dt {
            new_dt.min(previous_dt * max_increase_factor)
        } else {
            new_dt
        };

        // Update previous fields for the next time step
        self.u_prev.copy_from_slice(&self.u);
        self.v_prev.copy_from_slice(&self.v);
    }

    /// The core PISO substep.
    /// This method:
    /// 1. Computes a velocity predictor (u_star, v_star).
    /// 2. Computes the pressure correction (using Jacobi, SOR or multigrid).
    /// 3. Corrects the velocity fields.
    /// 4. Applies boundary conditions.
    fn piso_step(&mut self, dt_sub: f32) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let nu = self.nu;

        // Copy velocities into predictor fields
        self.u_star.copy_from_slice(&self.u);
        self.v_star.copy_from_slice(&self.v);

        // ---------------- Predictor for u ----------------
        // Loop over internal u faces.
        for j in 1..(ny - 1) {
            for i in 1..nx {
                let idx = i + j * (nx + 1);
                let x = i as f32 * dx;
                let y = (j as f32 + 0.5) * dy;
                if self.is_point_in_obstacle(x, y) {
                    self.u_star[idx] = 0.0;
                    continue;
                }
                match self.velocity_scheme {
                    VelocityScheme::FirstOrder => {
                        // East face
                        let idx_e = idx + 1;
                        let u_avg_e = 0.5 * (self.u[idx] + self.u[idx_e]);
                        let u_face_e = if u_avg_e >= 0.0 {
                            self.u[idx]
                        } else {
                            self.u[idx_e]
                        };
                        let f_e = u_face_e * u_face_e;
                        // West face
                        let idx_w = idx - 1;
                        let u_avg_w = 0.5 * (self.u[idx_w] + self.u[idx]);
                        let u_face_w = if u_avg_w >= 0.0 {
                            self.u[idx_w]
                        } else {
                            self.u[idx]
                        };
                        let f_w = u_face_w * u_face_w;
                        // North face (need to index u in the (nx+1) grid and v appropriately)
                        let idx_n = i + (j + 1) * (nx + 1);
                        let u_north = self.u[idx_n];
                        let idx_v_nw = if i > 0 { (i - 1) + (j + 1) * nx } else { 0 };
                        let idx_v_n = i + (j + 1) * nx;
                        let v_n = 0.5 * (self.v[idx_v_nw] + self.v[idx_v_n]);
                        let u_face_n = if v_n >= 0.0 {
                            self.u[idx]
                        } else {
                            u_north
                        };
                        let f_n = v_n * u_face_n;
                        // South face
                        let idx_s = i + (j - 1) * (nx + 1);
                        let u_south = self.u[idx_s];
                        let idx_v_s = if i > 0 { (i - 1) + j * nx } else { 0 };
                        let idx_v_current = i + j * nx;
                        let v_s = 0.5 * (self.v[idx_v_s] + self.v[idx_v_current]);
                        let u_face_s = if v_s >= 0.0 {
                            u_south
                        } else {
                            self.u[idx]
                        };
                        let f_s = v_s * u_face_s;
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.u[idx_e] - 2.0 * self.u[idx] + self.u[idx_w]) / (dx * dx)
                            + (self.u[i + (j + 1) * (nx + 1)] - 2.0 * self.u[idx]
                                + self.u[i + (j - 1) * (nx + 1)])
                                / (dy * dy);
                        self.u_star[idx] = self.u[idx] + dt_sub * (-convective + nu * laplace);
                    }
                    VelocityScheme::SecondOrder => {
                        // Second order upwinding discretization for u.
                        let idx_e = idx + 1;
                        let u_face_e = if self.u[idx] >= 0.0 {
                            if i > 1 {
                                1.5 * self.u[idx] - 0.5 * self.u[idx - 1]
                            } else {
                                self.u[idx]
                            }
                        } else if (idx_e + 1) < self.u.len() && i < nx - 1 {
                            1.5 * self.u[idx_e] - 0.5 * self.u[idx_e + 1]
                        } else {
                            self.u[idx_e]
                        };
                        let f_e = u_face_e * u_face_e;
                        let idx_w = idx - 1;
                        let u_face_w = if self.u[idx_w] >= 0.0 {
                            if i > 2 {
                                1.5 * self.u[idx_w] - 0.5 * self.u[idx_w - 1]
                            } else {
                                self.u[idx_w]
                            }
                        } else {
                            if i < nx {
                                1.5 * self.u[idx] - 0.5 * self.u[idx + 1]
                            } else {
                                self.u[idx]
                            }
                        };
                        let f_w = u_face_w * u_face_w;
                        let idx_n = i + (j + 1) * (nx + 1);
                        let idx_v_nw = if i > 0 { (i - 1) + (j + 1) * nx } else { 0 };
                        let idx_v_n = i + (j + 1) * nx;
                        let v_n = 0.5 * (self.v[idx_v_nw] + self.v[idx_v_n]);
                        let u_face_n = if v_n >= 0.0 {
                            if j > 1 {
                                1.5 * self.u[idx] - 0.5 * self.u[i + (j - 1) * (nx + 1)]
                            } else {
                                self.u[idx]
                            }
                        } else if (i + (j + 2) * (nx + 1)) < self.u.len() && j < ny - 1 {
                            1.5 * self.u[idx_n] - 0.5 * self.u[i + (j + 2) * (nx + 1)]
                        } else {
                            self.u[idx_n]
                        };
                        let f_n = v_n * u_face_n;
                        let idx_s = i + (j - 1) * (nx + 1);
                        let u_face_s = if v_n >= 0.0 {
                            if j > 1 {
                                1.5 * self.u[idx_s] - 0.5 * self.u[i + (j - 2) * (nx + 1)]
                            } else {
                                self.u[idx_s]
                            }
                        } else if j < ny {
                            1.5 * self.u[idx] - 0.5 * self.u[i + (j + 1) * (nx + 1)]
                        } else {
                            self.u[idx]
                        };
                        let f_s = {
                            let idx_v_s = if i > 0 { (i - 1) + j * nx } else { 0 };
                            let idx_v_current = i + j * nx;
                            let v_s = 0.5 * (self.v[idx_v_s] + self.v[idx_v_current]);
                            v_s * u_face_s
                        };
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.u[idx_e] - 2.0 * self.u[idx] + self.u[idx_w]) / (dx * dx)
                            + (self.u[i + (j + 1) * (nx + 1)] - 2.0 * self.u[idx]
                                + self.u[i + (j - 1) * (nx + 1)])
                                / (dy * dy);
                        self.u_star[idx] = self.u[idx] + dt_sub * (-convective + nu * laplace);
                    }
                    VelocityScheme::Quick => {
                        // QUICK discretization for u.
                        let u_face_e = if self.u[idx] >= 0.0 {
                            if i >= 2 {
                                (-self.u[idx - 1] + 6.0 * self.u[idx] + 3.0 * self.u[idx + 1])
                                    / 8.0
                            } else {
                                1.5 * self.u[idx] - 0.5 * self.u[idx - 1]
                            }
                        } else if i <= nx - 2 {
                            (3.0 * self.u[idx] + 6.0 * self.u[idx + 1] - self.u[idx + 2]) / 8.0
                        } else {
                            self.u[idx + 1]
                        };
                        let f_e = u_face_e * u_face_e;
                        let u_face_w = if self.u[idx - 1] >= 0.0 {
                            if i >= 3 {
                                (-self.u[idx - 2] + 6.0 * self.u[idx - 1] + 3.0 * self.u[idx])
                                    / 8.0
                            } else {
                                1.5 * self.u[idx - 1] - 0.5 * self.u[idx]
                            }
                        } else {
                            (3.0 * self.u[idx - 1] + 6.0 * self.u[idx] - self.u[idx + 1]) / 8.0
                        };
                        let f_w = u_face_w * u_face_w;
                        let idx_n = i + (j + 1) * (nx + 1);
                        let u_n = self.u[idx_n];
                        let idx_v_nw = if i > 0 { (i - 1) + (j + 1) * nx } else { 0 };
                        let idx_v_n = i + (j + 1) * nx;
                        let v_n = 0.5 * (self.v[idx_v_nw] + self.v[idx_v_n]);
                        let u_face_n = if v_n >= 0.0 {
                            if j >= 2 {
                                (-self.u[i + (j - 1) * (nx + 1)] + 6.0 * self.u[idx] + 3.0 * u_n)
                                    / 8.0
                            } else {
                                1.5 * self.u[idx] - 0.5 * self.u[i + (j - 1) * (nx + 1)]
                            }
                        } else if j < ny - 2 {
                            (3.0 * self.u[idx] + 6.0 * u_n - self.u[i + (j + 2) * (nx + 1)]) / 8.0
                        } else {
                            u_n
                        };
                        let f_n = v_n * u_face_n;
                        let idx_s = i + (j - 1) * (nx + 1);
                        let u_face_s = if v_n >= 0.0 {
                            if j >= 2 {
                                (-self.u[i + (j - 2) * (nx + 1)] + 6.0 * self.u[idx_s] + 3.0 * self.u[idx])
                                    / 8.0
                            } else {
                                1.5 * self.u[idx_s] - 0.5 * self.u[idx]
                            }
                        } else if j < ny - 1 {
                            (3.0 * self.u[idx_s] + 6.0 * self.u[idx] - self.u[i + (j + 1) * (nx + 1)])
                                / 8.0
                        } else {
                            self.u[idx]
                        };
                        let f_s = {
                            let idx_v_s = if i > 0 { (i - 1) + j * nx } else { 0 };
                            let idx_v_current = i + j * nx;
                            let v_s = 0.5 * (self.v[idx_v_s] + self.v[idx_v_current]);
                            v_s * u_face_s
                        };
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.u[idx + 1] - 2.0 * self.u[idx] + self.u[idx - 1]) / (dx * dx)
                            + (self.u[i + (j + 1) * (nx + 1)] - 2.0 * self.u[idx]
                                + self.u[i + (j - 1) * (nx + 1)])
                                / (dy * dy);
                        self.u_star[idx] = self.u[idx] + dt_sub * (-convective + nu * laplace);
                    }
                }
            }
        }

        // ---------------- Predictor for v ----------------
        // Loop over internal v faces.
        for j in 1..ny {
            for i in 1..(nx - 1) {
                let idx = i + j * nx;
                let x = (i as f32 + 0.5) * dx;
                let y = j as f32 * dy;
                if self.is_point_in_obstacle(x, y) {
                    self.v_star[idx] = 0.0;
                    continue;
                }
                match self.velocity_scheme {
                    VelocityScheme::FirstOrder => {
                        // East face flux
                        let idx_e = idx + 1;
                        let u_e = self.u[(i + 1) + j * (nx + 1)];
                        let v_face_e = if u_e >= 0.0 {
                            self.v[idx]
                        } else {
                            self.v[idx_e]
                        };
                        let f_e = u_e * v_face_e;
                        // West face flux
                        let idx_w = idx - 1;
                        let u_w = self.u[i + j * (nx + 1)];
                        let v_face_w = if u_w >= 0.0 {
                            self.v[idx_w]
                        } else {
                            self.v[idx]
                        };
                        let f_w = u_w * v_face_w;
                        // North face flux
                        let idx_n = i + (j + 1) * nx;
                        let v_avg_n = 0.5 * (self.v[idx] + self.v[idx_n]);
                        let v_face_n = if v_avg_n >= 0.0 {
                            self.v[idx]
                        } else {
                            self.v[idx_n]
                        };
                        let f_n = v_face_n * v_face_n;
                        // South face flux
                        let idx_s = i + (j - 1) * nx;
                        let v_avg_s = 0.5 * (self.v[idx_s] + self.v[idx]);
                        let v_face_s = if v_avg_s >= 0.0 {
                            self.v[idx_s]
                        } else {
                            self.v[idx]
                        };
                        let f_s = v_face_s * v_face_s;
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.v[(i + 1) + j * nx] - 2.0 * self.v[idx] + self.v[(i - 1) + j * nx])
                            / (dx * dx)
                            + (self.v[i + (j + 1) * nx] - 2.0 * self.v[idx] + self.v[i + (j - 1) * nx])
                                / (dy * dy);
                        self.v_star[idx] = self.v[idx] + dt_sub * (-convective + nu * laplace);
                    }
                    VelocityScheme::SecondOrder => {
                        // Second order for v (similar in spirit)
                        let idx_e = idx + 1;
                        let u_e = self.u[(i + 1) + j * (nx + 1)];
                        let v_face_e = if u_e >= 0.0 {
                            if i > 0 {
                                1.5 * self.v[idx] - 0.5 * self.v[idx - 1]
                            } else {
                                self.v[idx]
                            }
                        } else if (idx_e + 1) < self.v.len() && i < nx - 2 {
                            1.5 * self.v[idx + 1] - 0.5 * self.v[idx + 2]
                        } else {
                            self.v[idx + 1]
                        };
                        let f_e = u_e * v_face_e;
                        let idx_w = idx - 1;
                        let u_w = self.u[i + j * (nx + 1)];
                        let v_face_w = if u_w >= 0.0 {
                            if i > 1 {
                                1.5 * self.v[idx_w] - 0.5 * self.v[idx_w - 1]
                            } else {
                                self.v[idx_w]
                            }
                        } else if i < nx - 1 {
                            1.5 * self.v[idx] - 0.5 * self.v[idx + 1]
                        } else {
                            self.v[idx]
                        };
                        let f_w = u_w * v_face_w;
                        let idx_n = i + (j + 1) * nx;
                        let v_face_n = if {
                            let v_avg = 0.5 * (self.v[idx] + self.v[idx_n]);
                            v_avg
                        } >= 0.0 {
                            if j > 1 {
                                1.5 * self.v[idx] - 0.5 * self.v[i + (j - 1) * nx]
                            } else {
                                self.v[idx]
                            }
                        } else if (i + (j + 2) * nx) < self.v.len() && j < ny - 1 {
                            1.5 * self.v[idx_n] - 0.5 * self.v[i + (j + 2) * nx]
                        } else {
                            self.v[idx_n]
                        };
                        let v_avg_n = 0.5 * (self.v[idx] + self.v[idx_n]);
                        let f_n = v_avg_n * v_face_n;
                        let idx_s = i + (j - 1) * nx;
                        let v_face_s = if {
                            let v_avg = 0.5 * (self.v[idx_s] + self.v[idx]);
                            v_avg
                        } >= 0.0 {
                            if j > 1 {
                                1.5 * self.v[idx_s] - 0.5 * self.v[i + (j - 2) * nx]
                            } else {
                                self.v[idx_s]
                            }
                        } else if j < ny {
                            1.5 * self.v[idx] - 0.5 * self.v[i + (j + 1) * nx]
                        } else {
                            self.v[idx]
                        };
                        let v_avg_s = 0.5 * (self.v[idx_s] + self.v[idx]);
                        let f_s = v_avg_s * v_face_s;
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.v[(i + 1) + j * nx] - 2.0 * self.v[idx] + self.v[(i - 1) + j * nx])
                            / (dx * dx)
                            + (self.v[i + (j + 1) * nx] - 2.0 * self.v[idx] + self.v[i + (j - 1) * nx])
                                / (dy * dy);
                        self.v_star[idx] = self.v[idx] + dt_sub * (-convective + nu * laplace);
                    }
                    VelocityScheme::Quick => {
                        // QUICK discretization for v.
                        let idx_e = idx + 1;
                        let u_e = self.u[(i + 1) + j * (nx + 1)];
                        let v_face_e = if u_e >= 0.0 {
                            if i >= 2 {
                                (-self.v[idx - 1] + 6.0 * self.v[idx] + 3.0 * self.v[idx_e]) / 8.0
                            } else {
                                1.5 * self.v[idx] - 0.5 * self.v[idx - 1]
                            }
                        } else if i < nx - 2 {
                            (3.0 * self.v[idx] + 6.0 * self.v[idx_e] - self.v[idx_e + 1]) / 8.0
                        } else {
                            self.v[idx_e]
                        };
                        let f_e = u_e * v_face_e;
                        let idx_w = idx - 1;
                        let u_w = self.u[i + j * (nx + 1)];
                        let v_face_w = if u_w >= 0.0 {
                            if i >= 3 {
                                (-self.v[idx - 2] + 6.0 * self.v[idx_w] + 3.0 * self.v[idx])
                                    / 8.0
                            } else {
                                1.5 * self.v[idx_w] - 0.5 * self.v[idx]
                            }
                        } else {
                            (3.0 * self.v[idx_w] + 6.0 * self.v[idx] - self.v[idx_e]) / 8.0
                        };
                        let f_w = u_w * v_face_w;
                        let idx_n = i + (j + 1) * nx;
                        let v_avg_n = 0.5 * (self.v[idx] + self.v[idx_n]);
                        let v_face_n = if v_avg_n >= 0.0 {
                            if j >= 2 {
                                (-self.v[i + (j - 1) * nx] + 6.0 * self.v[idx] + 3.0 * self.v[idx_n])
                                    / 8.0
                            } else {
                                1.5 * self.v[idx] - 0.5 * self.v[i + (j - 1) * nx]
                            }
                        } else if j < ny - 1 {
                            (3.0 * self.v[idx] + 6.0 * self.v[idx_n] - self.v[i + (j + 2) * nx])
                                / 8.0
                        } else {
                            self.v[idx_n]
                        };
                        let f_n = v_avg_n * v_face_n;
                        let idx_s = i + (j - 1) * nx;
                        let v_avg_s = 0.5 * (self.v[idx_s] + self.v[idx]);
                        let v_face_s = if v_avg_s >= 0.0 {
                            if j >= 2 {
                                (-self.v[i + (j - 2) * nx] + 6.0 * self.v[idx_s] + 3.0 * self.v[idx])
                                    / 8.0
                            } else {
                                1.5 * self.v[idx_s] - 0.5 * self.v[idx]
                            }
                        } else if j < ny - 1 {
                            (3.0 * self.v[idx_s] + 6.0 * self.v[idx] - self.v[i + (j + 1) * nx])
                                / 8.0
                        } else {
                            self.v[idx]
                        };
                        let f_s = v_avg_s * v_face_s;
                        let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                        let laplace = (self.v[(i + 1) + j * nx] - 2.0 * self.v[idx] + self.v[(i - 1) + j * nx])
                            / (dx * dx)
                            + (self.v[i + (j + 1) * nx] - 2.0 * self.v[idx] + self.v[i + (j - 1) * nx])
                                / (dy * dy);
                        self.v_star[idx] = self.v[idx] + dt_sub * (-convective + nu * laplace);
                    }
                }
            }
        }

        // ---------------- Pressure Correction (MAC form) ----------------
        // Compute the divergence (rhs) on pressure cells.
        for j in 0..ny {
            for i in 0..nx {
                let idx = i + j * nx;
                let u_right = self.u_star[i + 1 + j * (nx + 1)];
                let u_left = self.u_star[i + j * (nx + 1)];
                let v_top = self.v_star[i + (j + 1) * nx];
                let v_bottom = self.v_star[i + j * nx];
                self.rhs[idx] = ((u_right - u_left) / dx + (v_top - v_bottom) / dy) / dt_sub;
            }
        }

        // ---------------- Pressure Correction Solver ----------------
        let denom = 2.0 / (dx * dx) + 2.0 / (dy * dy);
        match self.pressure_solver {
            PressureSolver::SOR => {
                // For brevity, a SOR solver is implemented similarly here.
                self.p_prime.fill(0.0);
                let sor_omega = 1.7;
                let pressure_tolerance = 1e-4;
                let iterations = 50;
                for _ in 0..iterations {
                    let mut max_error = 0.0;
                    for j in 1..(ny - 1) {
                        for i in 1..(nx - 1) {
                            let idx = i + j * nx;
                            let p_old = self.p_prime[idx];
                            let p_update = ((self.p_prime[idx + 1] + self.p_prime[idx - 1]) / (dx * dx)
                                + (self.p_prime[i + (j + 1) * nx] + self.p_prime[i + (j - 1) * nx])
                                    / (dy * dy)
                                - self.rhs[idx])
                                / denom;
                            self.p_prime[idx] =
                                (1.0 - sor_omega) * p_old + sor_omega * p_update;
                            let error = (self.p_prime[idx] - p_old).abs();
                            if error > max_error {
                                max_error = error;
                            }
                        }
                    }
                    for i in 0..nx {
                        self.p_prime[i] = self.p_prime[i + nx]; // bottom
                        self.p_prime[i + (ny - 1) * nx] = self.p_prime[i + (ny - 2) * nx]; // top
                    }
                    for j in 0..ny {
                        self.p_prime[j * nx] = self.p_prime[1 + j * nx];
                        self.p_prime[(nx - 1) + j * nx] = 0.0;
                    }
                    if max_error < pressure_tolerance {
                        break;
                    }
                }
                self.last_pressure_residual = 0.0; // (optionally, compute the maximum residual)
            }
            PressureSolver::Multigrid => {
                // Multigrid solver can be implemented here.
                // For now we simply fall back to Jacobi.
                self.jacobi_pressure(denom, dx, dy, nx, ny);
            }
            PressureSolver::Jacobi => {
                self.jacobi_pressure(denom, dx, dy, nx, ny);
            }
        }

        // ---------------- Corrector Step ----------------
        // Correct u
        for j in 0..ny {
            for i in 1..nx {
                let idx = i + j * (nx + 1);
                let p_right = self.p_prime[i + j * nx];
                let p_left = self.p_prime[i.saturating_sub(1) + j * nx];
                self.u[idx] = self.u_star[idx] - dt_sub * (p_right - p_left) / dx;
            }
        }
        // Correct v
        for j in 1..ny {
            for i in 0..nx {
                let idx = i + j * nx;
                let p_top = self.p_prime[i + j * nx];
                let p_bottom = self.p_prime[i + (j.saturating_sub(1)) * nx];
                self.v[idx] = self.v_star[idx] - dt_sub * (p_top - p_bottom) / dy;
            }
        }
        // Accumulate the pressure correction
        for i in 0..self.p.len() {
            self.p[i] += self.p_prime[i];
        }

        // Enforce boundary conditions
        self.apply_boundary_conditions();
    }

    /// A helper for the Jacobi pressure correction solver.
    fn jacobi_pressure(&mut self, denom: f32, dx: f32, dy: f32, nx: usize, ny: usize) {
        self.p_prime.fill(0.0);
        let jacobi_omega = 0.7;
        let pressure_tolerance = 1e-6;
        let iterations = 50;
        for _iter in 0..iterations {
            self.p_prime_new.fill(0.0);
            for j in 1..(ny - 1) {
                for i in 1..(nx - 1) {
                    let idx = i + j * nx;
                    let p_update = ((self.p_prime[idx + 1] + self.p_prime[idx - 1]) / (dx * dx)
                        + (self.p_prime[i + (j + 1) * nx] + self.p_prime[i + (j - 1) * nx])
                            / (dy * dy)
                        - self.rhs[idx])
                        / denom;
                    self.p_prime_new[idx] =
                        jacobi_omega * p_update + (1.0 - jacobi_omega) * self.p_prime[idx];
                }
            }
            let mut max_error = 0.0;
            for j in 1..(ny - 1) {
                for i in 1..(nx - 1) {
                    let idx = i + j * nx;
                    let error = (self.p_prime_new[idx] - self.p_prime[idx]).abs();
                    if error > max_error {
                        max_error = error;
                    }
                    self.p_prime[idx] = self.p_prime_new[idx];
                }
            }
            for i in 0..nx {
                self.p_prime[i] = self.p_prime[i + nx]; // bottom
                self.p_prime[i + (ny - 1) * nx] = self.p_prime[i + (ny - 2) * nx]; // top
            }
            for j in 0..ny {
                self.p_prime[j * nx] = self.p_prime[1 + j * nx]; // left
                self.p_prime[(nx - 1) + j * nx] = 0.0; // outlet
            }
            if max_error < pressure_tolerance {
                break;
            }
        }
        self.last_pressure_residual = 0.0; // Optionally, store max_error here.
    }

    /// Enforce boundary conditions on the velocity fields.
    fn apply_boundary_conditions(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let dx = self.grid.dx;
        let dy = self.grid.dy;

        // For u: Inlet boundary: set left boundary faces according to the chosen profile.
        for j in 0..ny {
            let idx = 0 + j * (nx + 1);
            let y = (j as f32 + 0.5) * dy;
            let inlet_val = match self.inlet_profile {
                InletProfile::Uniform => self.current_inlet_velocity,
                InletProfile::Parabolic => {
                    let center = self.grid.ly / 2.0;
                    let radius = self.grid.ly / 2.0;
                    let val = self.current_inlet_velocity * (1.0 - ((y - center) / radius).powi(2));
                    if val < 0.0 { 0.0 } else { val }
                }
            };
            self.u[idx] = inlet_val;
        }
        // Outlet (right boundary): set u[nx, j] = u[nx-1, j]
        for j in 0..ny {
            let idx_out = nx + j * (nx + 1);
            let idx_in = (nx - 1) + j * (nx + 1);
            self.u[idx_out] = self.u[idx_in];
        }
        // No–slip on top and bottom walls for u.
        for i in 0..(nx + 1) {
            self.u[i + 0 * (nx + 1)] = 0.0;
            self.u[i + (ny - 1) * (nx + 1)] = 0.0;
        }
        // For v: enforce no–slip on top and bottom boundaries.
        for i in 0..nx {
            self.v[i + 0 * nx] = 0.0;
            // Note: v is defined on (ny+1) horizontal faces; set top (j = ny) to zero.
            self.v[i + ny * nx] = 0.0;
        }
        // Enforce zero velocity in the obstacle region (by checking cell center positions).
        for j in 0..ny {
            for i in 0..(nx + 1) {
                let x = i as f32 * dx;
                let y = (j as f32 + 0.5) * dy;
                if self.is_point_in_obstacle(x, y) {
                    self.u[i + j * (nx + 1)] = 0.0;
                }
            }
        }
        for j in 0..(ny + 1) {
            for i in 0..nx {
                let x = (i as f32 + 0.5) * dx;
                let y = j as f32 * dy;
                if self.is_point_in_obstacle(x, y) {
                    self.v[i + j * nx] = 0.0;
                }
            }
        }
    }

    /// Helper: check whether point (x,y) is inside the obstacle.
    fn is_point_in_obstacle(&self, x: f32, y: f32) -> bool {
        if let Some(cyl) = &self.grid.obstacle {
            ((x - cyl.center_x).powi(2) + (y - cyl.center_y).powi(2)).sqrt() <= cyl.radius
        } else {
            false
        }
    }

    /// Automatically compute a time step based on a CFL condition.
    fn compute_automatic_time_step(&self) -> f32 {
        let max_u = self.u.iter().map(|&v| v.abs()).fold(0.0, f32::max);
        let max_v = self.v.iter().map(|&v| v.abs()).fold(0.0, f32::max);
        let max_vel = max_u.max(max_v);
        if max_vel == 0.0 {
            self.dt
        } else {
            let cfl = 0.5;
            let dt_cfl = cfl * self.grid.dx.min(self.grid.dy) / max_vel;
            dt_cfl.min(self.dt)
        }
    }

    // ----- Methods for setting simulation parameters -----
    pub fn set_dt(&mut self, dt: f32) {
        self.dt = dt;
    }
    pub fn set_viscosity(&mut self, nu: f32) {
        self.nu = nu;
    }
    pub fn set_target_inlet_velocity(&mut self, velocity: f32) {
        self.target_inlet_velocity = velocity;
    }
    pub fn set_velocity_scheme(&mut self, scheme: VelocityScheme) {
        self.velocity_scheme = scheme;
    }
    pub fn set_pressure_solver(&mut self, solver: PressureSolver) {
        self.pressure_solver = solver;
    }
    pub fn set_inlet_profile(&mut self, profile: InletProfile) {
        self.inlet_profile = profile;
    }

    // ----- Methods for retrieving the current state -----
    /// Returns a reference to the pressure field.
    pub fn get_pressure(&self) -> &[f32] {
        &self.p
    }
    /// Returns a reference to the horizontal velocity field (u).
    pub fn get_u(&self) -> &[f32] {
        &self.u
    }
    /// Returns a reference to the vertical velocity field (v).
    pub fn get_v(&self) -> &[f32] {
        &self.v
    }

    /// Returns the last computed pressure residual.
    pub fn get_last_pressure_residual(&self) -> f32 {
        self.last_pressure_residual
    }

    /// Optionally, convert the pressure field into a nalgebra DMatrix.
    pub fn pressure_as_matrix(&self) -> DMatrix<f32> {
        DMatrix::from_vec(self.grid.ny, self.grid.nx, self.p.clone())
    }
}

