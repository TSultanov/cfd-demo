use std::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use std::simd::num::SimdFloat;
use std::simd::{Mask, Simd};

use std::{
    sync::mpsc::{self, TryRecvError},
    thread,
    time::{Duration, Instant},
};

const LANES: usize = 8;

#[derive(Clone)]
pub struct SimulationParams {
    pub dt: f32,
    pub viscosity: f32,
    pub target_inlet_velocity: f32,
    pub velocity_scheme: VelocityScheme,
    pub inlet_profile: InletProfile,
    pub pressure_solver: PressureSolver,
}

pub struct Residuals {
    pub simulation_step: usize,
    pub simulation_time: f32,
    pub dt: f32,
    pub p: f32,
    pub u: f32,
    pub v: f32,
}

/// A snapshot structure to copy the data needed for visualization and logging.
#[derive(Clone)]
pub struct SimSnapshot {
    pub p: Vec<f32>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub dt: f32,
    pub paused: bool,
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

pub enum Command {
    Stop,
    GetSnapshot,
    SetParams(SimulationParams),
    Pause,
    Resume,
}

pub struct SimulationControlHandle {
    command_sender: mpsc::Sender<Command>,
    snapshot_receiver: mpsc::Receiver<SimSnapshot>,
    residuals_receiver: mpsc::Receiver<Residuals>,
}

impl SimulationControlHandle {
    pub fn stop(&self) {
        self.command_sender.send(Command::Stop).unwrap();
    }

    pub fn get_last_available_snapshot(&self) -> Option<SimSnapshot> {
        let mut last_snapshot = None;
        loop {
            match self.snapshot_receiver.try_recv() {
                Ok(snapshot) => last_snapshot = Some(snapshot),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        last_snapshot
    }

    pub fn get_new_log_messages(&self) -> Vec<Residuals> {
        let mut last_residuals = vec![];
        loop {
            match self.residuals_receiver.try_recv() {
                Ok(residuals) => last_residuals.push(residuals),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        last_residuals
    }

    pub fn request_snapshot(&self) {
        self.command_sender.send(Command::GetSnapshot).unwrap();
    }

    pub fn set_params(&self, params: SimulationParams) {
        self.command_sender
            .send(Command::SetParams(params))
            .unwrap();
    }

    pub fn pause(&self) {
        self.command_sender.send(Command::Pause).unwrap();
    }

    pub fn resume(&self) {
        self.command_sender.send(Command::Resume).unwrap();
    }
}

/// A simple grid definition holding the number of pressure cells,
/// as well as the physical dimensions and cell sizes.
#[derive(Clone)]
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
#[derive(Clone)]
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

    obstacle_mask_u: Vec<u8>,
    obstacle_mask_v: Vec<u8>,
    obstacle_coords: Vec<(usize, usize)>,

    // Previous fields (for extrapolation and residual computation)
    pub u_old: Vec<f32>,
    pub v_old: Vec<f32>,

    // Predictor (star) fields
    u_v_n: Vec<f32>,
    u_v_s: Vec<f32>,
    u_u_e: Vec<f32>,
    u_u_w: Vec<f32>,
    u_u_n: Vec<f32>,
    u_u_s: Vec<f32>,
    pub u_star: Vec<f32>,

    v_u_e: Vec<f32>,
    v_u_w: Vec<f32>,
    v_v_n: Vec<f32>,
    v_v_s: Vec<f32>,
    v_v_e: Vec<f32>,
    v_v_w: Vec<f32>,
    pub v_star: Vec<f32>,

    /// Right-hand side for pressure correction (divergence residual)
    pub rhs: Vec<f32>,
    /// Pressure correction field and temporary buffer for Jacobi iteration.
    pub p_prime: Vec<f32>,
    pub p_prime_new: Vec<f32>,

    /// The last pressure residual (for time–step scaling).
    pub last_pressure_residual: f32,
    pub last_u_residual: f32,
    pub last_v_residual: f32,
    pub simulation_time: f32,
}

impl Model {
    /// Create a new simulation model given a grid.
    /// (The field vectors are allocated based on grid dimensions.)
    pub fn new(grid: Grid, params: &SimulationParams) -> Self {
        let nx = grid.nx;
        let ny = grid.ny;

        let size_u = (nx + 1) * ny;
        let size_v = nx * (ny + 1);
        let size_p = nx * ny;

        let u = vec![0.0; size_u];
        let v = vec![0.0; size_v];
        let p = vec![0.0; size_p];

        // Create and populate the obstacle mask.
        let mut obstacle_mask_u = vec![0; size_u];
        let mut obstacle_mask_v = vec![0; size_v];
        let mut obstacle_coords = vec![];
        if let Some(obstacle) = &grid.obstacle {
            for j in 0..ny {
                for i in 0..nx {
                    let x = (i as f32 + 0.5) * grid.dx;
                    let y = (j as f32 + 0.5) * grid.dy;
                    let dx = x - obstacle.center_x;
                    let dy = y - obstacle.center_y;
                    let distance = (dx * dx + dy * dy).sqrt();
                    if distance < obstacle.radius {
                        // Inside the obstacle.
                        if i > 0 {
                            obstacle_mask_u[i + j * (nx + 1)] = 1;
                        }
                        if i < nx {
                            obstacle_mask_u[(i + 1) + j * (nx + 1)] = 1;
                        }
                        if j > 0 {
                            obstacle_mask_v[i + j * nx] = 1;
                        }
                        if j < ny {
                            obstacle_mask_v[i + (j + 1) * nx] = 1;
                        }
                        obstacle_coords.push((i, j));
                    }
                }
            }
        }

        Self {
            grid,
            dt: params.dt,
            nu: params.viscosity,
            substep_count: 5,
            simulation_step: 0,
            ramp_up_steps: 100,
            current_inlet_velocity: 0.0,
            target_inlet_velocity: params.target_inlet_velocity,
            velocity_scheme: params.velocity_scheme,
            pressure_solver: params.pressure_solver,
            inlet_profile: params.inlet_profile,
            u: u.clone(),
            v: v.clone(),
            p,

            obstacle_mask_u,
            obstacle_mask_v,
            obstacle_coords,
            // u_prev: u.clone(),
            // v_prev: v.clone(),
            u_old: u.clone(),
            v_old: v.clone(),

            u_v_n: u.clone(),
            u_v_s: u.clone(),
            u_u_e: u.clone(),
            u_u_w: u.clone(),
            u_u_n: u.clone(),
            u_u_s: u.clone(),
            u_star: u.clone(),

            v_u_e: v.clone(),
            v_u_w: v.clone(),
            v_v_n: v.clone(),
            v_v_s: v.clone(),
            v_v_e: v.clone(),
            v_v_w: v.clone(),
            v_star: v.clone(),
            rhs: vec![0.0; size_p],
            p_prime: vec![0.0; size_p],
            p_prime_new: vec![0.0; size_p],
            last_pressure_residual: 0.0,
            last_u_residual: 0.0,
            last_v_residual: 0.0,
            simulation_time: 0.0,
        }
    }

    /// Perform a simulation update (one "time step").
    /// Implements the extrapolation, multiple substeps (the PISO algorithm)
    /// and automatic dt adjustment.
    pub fn update(&mut self) {
        // Save current fields for residual computation.
        self.u_old.copy_from_slice(&self.u);
        self.v_old.copy_from_slice(&self.v);

        // Gradually ramp up inlet velocity.
        if self.simulation_step < self.ramp_up_steps {
            self.current_inlet_velocity = (self.simulation_step as f32 / self.ramp_up_steps as f32)
                * self.target_inlet_velocity;
        } else {
            self.current_inlet_velocity = self.target_inlet_velocity;
        }
        let dt_sub = self.dt / self.substep_count as f32;
        let mut max_pressure_residual = 0.0;

        // Perform the required number of substeps (PISO substeps)
        let start = Instant::now();
        for _ in 0..self.substep_count {
            let start_time = Instant::now();
            println!("");
            self.piso_step(dt_sub);
            let p_residual = self.last_pressure_residual;
            if p_residual > max_pressure_residual {
                max_pressure_residual = p_residual;
            }
            let piso_time = start_time.elapsed();
            println!("piso time: {:?}", piso_time);
        }
        println!("piso total time: {:?}", start.elapsed());

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

        // Store U and V residuals
        self.last_u_residual = max_residual_u;
        self.last_v_residual = max_residual_v;

        self.simulation_step += 1;

        // Adjust the number of substeps if the error norm is too high.
        let error_norm: f32 = max_residual_u
            .max(max_residual_v)
            .max(max_pressure_residual);
        let tolerance = 1e-3;
        if error_norm > tolerance {
            let factor = error_norm / tolerance;
            self.substep_count = ((self.substep_count as f32) * factor).ceil().min(20.0) as usize;
        } else if error_norm < tolerance / 10.0 && self.substep_count > 1 {
            self.substep_count = (self.substep_count as f32 / 2.0).floor() as usize;
            if self.substep_count < 1 {
                self.substep_count = 1;
            }
        }

        self.simulation_time += self.dt;

        // Automatic dt control (using a simple CFL condition).
        let previous_dt = self.dt;
        let new_dt = self.compute_automatic_time_step();

        // Limit increase of dt to keep changes smooth.
        let max_increase_factor = 1.1;
        self.dt = if new_dt > previous_dt {
            new_dt.min(previous_dt * max_increase_factor)
        } else {
            new_dt
        };

        // Update previous fields for the next time step
        // self.u_prev.copy_from_slice(&self.u);
        // self.v_prev.copy_from_slice(&self.v);
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
        let start = Instant::now();
        match self.velocity_scheme {
            VelocityScheme::FirstOrder => {
                for j in 1..(ny - 1) {
                    for i in (1..nx).step_by(LANES) {
                        let idx = i + j * (nx + 1);
                        let idx_end = (i + LANES) + j * (nx + 1);

                        self.get_v_north(i, j)
                            .copy_to_slice(&mut self.u_v_n[idx..idx_end]);
                        self.get_v_south(i, j)
                            .copy_to_slice(&mut self.u_v_s[idx..idx_end]);
                        self.u_face_n_first_order(i, j)
                            .copy_to_slice(&mut self.u_u_n[idx..idx_end]);
                        self.u_face_s_first_order(i, j)
                            .copy_to_slice(&mut self.u_u_s[idx..idx_end]);
                        self.u_face_e_first_order(i, j)
                            .copy_to_slice(&mut self.u_u_e[idx..idx_end]);
                        self.u_face_w_first_order(i, j)
                            .copy_to_slice(&mut self.u_u_w[idx..idx_end]);
                    }
                }
            }
            VelocityScheme::SecondOrder => {
                for j in 1..(ny - 1) {
                    for i in (1..nx).step_by(LANES) {
                        let idx = i + j * (nx + 1);
                        let idx_end = (i + LANES) + j * (nx + 1);

                        self.get_v_north(i, j)
                            .copy_to_slice(&mut self.u_v_n[idx..idx_end]);
                        self.get_v_south(i, j)
                            .copy_to_slice(&mut self.u_v_s[idx..idx_end]);
                        self.u_face_n_second_order(i, j)
                            .copy_to_slice(&mut self.u_u_n[idx..idx_end]);
                        self.u_face_s_second_order(i, j)
                            .copy_to_slice(&mut self.u_u_s[idx..idx_end]);
                        self.u_face_e_second_order(i, j)
                            .copy_to_slice(&mut self.u_u_e[idx..idx_end]);
                        self.u_face_w_second_order(i, j)
                            .copy_to_slice(&mut self.u_u_w[idx..idx_end]);
                    }
                }
            }
            VelocityScheme::Quick => {
                for j in 1..(ny - 1) {
                    for i in (1..nx).step_by(LANES) {
                        let idx = i + j * (nx + 1);
                        let idx_end = (i + LANES) + j * (nx + 1);

                        self.get_v_north(i, j)
                            .copy_to_slice(&mut self.u_v_n[idx..idx_end]);
                        self.get_v_south(i, j)
                            .copy_to_slice(&mut self.u_v_s[idx..idx_end]);
                        self.u_face_n_quick(i, j)
                            .copy_to_slice(&mut self.u_u_n[idx..idx_end]);
                        self.u_face_s_quick(i, j)
                            .copy_to_slice(&mut self.u_u_s[idx..idx_end]);

                        for k in 0..LANES {
                            let idx = (i + k) + j * (nx + 1);
                            self.u_u_e[idx] = self.u_face_e_quick(i + k, j);
                            self.u_u_w[idx] = self.u_face_w_quick(i + k, j);
                        }
                    }
                }
            }
        }

        let dx_v = Simd::splat(dx);
        let dy_v = Simd::splat(dy);

        for j in 1..(ny - 1) {
            for i in (1..nx).step_by(LANES) {
                let idx = i + j * (nx + 1);
                let idx_end = (i + LANES) + j * (nx + 1);

                let obstacle_mask_u = Simd::from_slice(&self.obstacle_mask_u[idx..idx_end]);
                let obstacle_mask_u: Mask<i32, LANES> =
                    obstacle_mask_u.simd_eq(Simd::splat(1)).into();

                let convective = {
                    let u_e: Simd<f32, LANES> = Simd::from_slice(&self.u_u_e[idx..idx_end]);
                    let u_w: Simd<f32, LANES> = Simd::from_slice(&self.u_u_w[idx..idx_end]);
                    let v_n: Simd<f32, LANES> = Simd::from_slice(&self.u_v_n[idx..idx_end]);
                    let v_s: Simd<f32, LANES> = Simd::from_slice(&self.u_v_s[idx..idx_end]);
                    let u_n: Simd<f32, LANES> = Simd::from_slice(&self.u_u_n[idx..idx_end]);
                    let u_s: Simd<f32, LANES> = Simd::from_slice(&self.u_u_s[idx..idx_end]);

                    let f_e = u_e * u_e;
                    let f_w = u_w * u_w;

                    let f_n = v_n * u_n;
                    let f_s = v_s * u_s;
                    (f_e - f_w) / dx_v + (f_n - f_s) / dy_v
                };

                let u: Simd<f32, LANES> = Simd::from_slice(&self.u[idx..idx_end]);
                let laplace = {
                    let idx_e = (i + 1) + j * (nx + 1);
                    let idx_w = (i - 1) + j * (nx + 1);
                    let idx_s = i + (j - 1) * (nx + 1);
                    let idx_n = i + (j + 1) * (nx + 1);

                    let u_e: Simd<f32, LANES> = Simd::from_slice(&self.u[idx_e..idx_e + LANES]);
                    let u_w: Simd<f32, LANES> = Simd::from_slice(&self.u[idx_w..idx_w + LANES]);
                    let u_n: Simd<f32, LANES> = Simd::from_slice(&self.u[idx_n..idx_n + LANES]);
                    let u_s: Simd<f32, LANES> = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);

                    (u_e - Simd::splat(2.0) * u + u_w) / (dx_v * dx_v)
                        + (u_n - Simd::splat(2.0) * u + u_s) / (dy_v * dy_v)
                };

                let u_star = u + Simd::splat(dt_sub) * (-convective + Simd::splat(nu) * laplace);
                let u_star = obstacle_mask_u.select(Simd::splat(0.0), u_star);
                u_star.copy_to_slice(&mut self.u_star[idx..idx_end]);
            }
        }
        println!("u_star time: {:?}", start.elapsed());

        // ---------------- Predictor for v ----------------
        // Loop over internal v faces.
        let start = Instant::now();
        match self.velocity_scheme {
            VelocityScheme::FirstOrder => {
                for j in 1..ny {
                    for i in (1..(nx - 1)).step_by(LANES) {
                        let idx = i + j * nx;
                        let idx_end = (i + LANES) + j * nx;

                        if i + LANES > nx - 1 {
                            for k in 0..(nx - i) {
                                let idx = i + k + j * nx;
                                self.v_u_e[idx] = self.u[(i + k + 1) + j * (nx + 1)];
                                self.v_u_w[idx] = self.u[i + k + j * (nx + 1)];

                                self.v_v_n[idx] = self.v_face_n_first_order_scalar(i + k, j);
                                self.v_v_s[idx] = self.v_face_s_first_order_scalar(i + k, j);
                                self.v_v_e[idx] = self.v_face_e_first_order_scalar(i + k, j);
                                self.v_v_w[idx] = self.v_face_w_first_order_scalar(i + k, j);
                            }
                            continue;
                        }

                        self.v_u_e[idx..idx_end].copy_from_slice(
                            &self.u[(i + 1) + j * (nx + 1)..(i + LANES + 1) + j * (nx + 1)],
                        );
                        self.v_u_w[idx..idx_end]
                            .copy_from_slice(&self.u[i + j * (nx + 1)..i + LANES + j * (nx + 1)]);

                        self.v_face_n_first_order(i, j)
                            .copy_to_slice(&mut self.v_v_n[idx..idx_end]);
                        self.v_face_s_first_order(i, j)
                            .copy_to_slice(&mut self.v_v_s[idx..idx_end]);
                        self.v_face_e_first_order(i, j)
                            .copy_to_slice(&mut self.v_v_e[idx..idx_end]);
                        self.v_face_w_first_order(i, j)
                            .copy_to_slice(&mut self.v_v_w[idx..idx_end]);
                    }
                }
            }
            VelocityScheme::SecondOrder => {
                for j in 1..ny {
                    for i in (1..(nx - 1)).step_by(LANES) {
                        let idx = i + j * nx;
                        let idx_end = (i + LANES) + j * nx;

                        if i + LANES > nx - 1 {
                            for k in 0..(nx - i) {
                                let idx = i + k + j * nx;
                                self.v_u_e[idx] = self.u[(i + k + 1) + j * (nx + 1)];
                                self.v_u_w[idx] = self.u[i + k + j * (nx + 1)];

                                self.v_v_n[idx] = self.v_face_n_second_order(i + k, j);
                                self.v_v_s[idx] = self.v_face_s_second_order(i + k, j);
                                self.v_v_e[idx] = self.v_face_e_second_order(i + k, j);
                                self.v_v_w[idx] = self.v_face_w_second_order(i + k, j);
                            }
                            continue;
                        }

                        self.v_u_e[idx..idx_end].copy_from_slice(
                            &self.u[(i + 1) + j * (nx + 1)..(i + LANES + 1) + j * (nx + 1)],
                        );
                        self.v_u_w[idx..idx_end]
                            .copy_from_slice(&self.u[i + j * (nx + 1)..i + LANES + j * (nx + 1)]);

                        for k in 0..LANES {
                            let idx = (i + k) + j * nx;
                            self.v_v_n[idx] = self.v_face_n_second_order(i + k, j);
                            self.v_v_s[idx] = self.v_face_s_second_order(i + k, j);
                            self.v_v_e[idx] = self.v_face_e_second_order(i + k, j);
                            self.v_v_w[idx] = self.v_face_w_second_order(i + k, j);
                        }
                    }
                }
            }
            VelocityScheme::Quick => {
                for j in 1..ny {
                    for i in 1..(nx - 1) {
                        let idx = i + j * nx;
                        self.v_u_e[idx] = self.u[(i + 1) + j * (nx + 1)];
                        self.v_u_w[idx] = self.u[i + j * (nx + 1)];

                        self.v_v_n[idx] = self.v_face_n_quick(i, j);
                        self.v_v_s[idx] = self.v_face_s_quick(i, j);
                        self.v_v_e[idx] = self.v_face_e_quick(i, j);
                        self.v_v_w[idx] = self.v_face_w_quick(i, j);
                    }
                }
            }
        }

        for j in 1..ny {
            for i in (1..(nx - 1)).step_by(LANES) {
                let idx = i + j * nx;

                if i + LANES > nx - 1 {
                    for k in 0..(nx - i) {
                        let idx = i + k + j * nx;
                        if self.obstacle_mask_v[idx] == 1 {
                            self.v_star[idx] = 0.0;
                            continue;
                        }
                        // For v the east/west fluxes also involve the u–field.

                        let convective = {
                            let u_e = self.v_u_e[idx];
                            let u_w = self.v_u_w[idx];
                            let v_n = self.v_v_n[idx];
                            let v_s = self.v_v_s[idx];
                            let v_e = self.v_v_e[idx];
                            let v_w = self.v_v_w[idx];

                            let f_e = u_e * v_e;
                            let f_w = u_w * v_w;
                            let f_n = v_n * v_n;
                            let f_s = v_s * v_s;
                            (f_e - f_w) / dx + (f_n - f_s) / dy
                        };

                        let v = self.v[idx];

                        let laplace = {
                            let idx_e = i + 1 + j * nx;
                            let idx_w = i - 1 + j * nx;
                            let idx_n = i + (j + 1) * nx;
                            let idx_s = i + (j - 1) * nx;

                            let v_e = self.v[idx_e];
                            let v_w = self.v[idx_w];
                            let v_n = self.v[idx_n];
                            let v_s = self.v[idx_s];

                            (v_e - 2.0 * v + v_w) / (dx * dx) + (v_n - 2.0 * v + v_s) / (dy * dy)
                        };

                        self.v_star[idx] = self.v[idx] + dt_sub * (-convective + nu * laplace);
                    }

                    continue;
                }

                let obstacle_mask_v = Simd::from_slice(&self.obstacle_mask_v[idx..idx + LANES]);
                let obstacle_mask_v: Mask<i32, LANES> =
                    obstacle_mask_v.simd_eq(Simd::splat(1)).into();

                let convective = {
                    let u_e: Simd<f32, LANES> = Simd::from_slice(&self.v_u_e[idx..idx + LANES]);
                    let u_w: Simd<f32, LANES> = Simd::from_slice(&self.v_u_w[idx..idx + LANES]);
                    let v_n: Simd<f32, LANES> = Simd::from_slice(&self.v_v_n[idx..idx + LANES]);
                    let v_s: Simd<f32, LANES> = Simd::from_slice(&self.v_v_s[idx..idx + LANES]);
                    let v_e: Simd<f32, LANES> = Simd::from_slice(&self.v_v_e[idx..idx + LANES]);
                    let v_w: Simd<f32, LANES> = Simd::from_slice(&self.v_v_w[idx..idx + LANES]);

                    let f_e = u_e * v_e;
                    let f_w = u_w * v_w;
                    let f_n = v_n * v_n;
                    let f_s = v_s * v_s;
                    (f_e - f_w) / dx_v + (f_n - f_s) / dy_v
                };

                let v: Simd<f32, LANES> = Simd::from_slice(&self.v[idx..idx + LANES]);

                let laplace = {
                    let idx_e = i + 1 + j * nx;
                    let idx_w = i - 1 + j * nx;
                    let idx_n = i + (j + 1) * nx;
                    let idx_s = i + (j - 1) * nx;

                    let v_e: Simd<f32, LANES> = Simd::from_slice(&self.v[idx_e..idx_e + LANES]);
                    let v_w: Simd<f32, LANES> = Simd::from_slice(&self.v[idx_w..idx_w + LANES]);
                    let v_n: Simd<f32, LANES> = Simd::from_slice(&self.v[idx_n..idx_n + LANES]);
                    let v_s: Simd<f32, LANES> = Simd::from_slice(&self.v[idx_s..idx_s + LANES]);

                    (v_e - Simd::splat(2.0) * v + v_w) / (dx_v * dx_v)
                        + (v_n - Simd::splat(2.0) * v + v_s) / (dy_v * dy_v)
                };

                let v_star = v + Simd::splat(dt_sub) * (-convective + Simd::splat(nu) * laplace);
                let v_star = obstacle_mask_v.select(Simd::splat(0.0), v_star);
                v_star.copy_to_slice(&mut self.v_star[idx..idx + LANES]);
            }
        }
        println!("v_star time: {:?}", start.elapsed());

        // ---------------- Pressure Correction (MAC form) ----------------
        // Compute the divergence (rhs) on pressure cells.
        let start_time = Instant::now();
        for j in 0..ny {
            for i in (0..nx).step_by(LANES) {
                if i + LANES > nx {
                    for k in 0..(nx - i) {
                        let idx = i + k + j * nx;
                        let idx_e = (i + k + 1) + j * (nx + 1);
                        let idx_w = i + k + j * (nx + 1);
                        let idx_n = i + k + (j + 1) * nx;
                        let idx_s = i + k + j * nx;

                        let u_e = self.u_star[idx_e];
                        let u_w = self.u_star[idx_w];
                        let v_n = self.v_star[idx_n];
                        let v_s = self.v_star[idx_s];
                        self.rhs[idx] = ((u_e - u_w) / dx + (v_n - v_s) / dy) / dt_sub;
                    }
                    continue;
                }

                let idx = i + j * nx;
                let idx_e = (i + 1) + j * (nx + 1);
                let idx_w = i + j * (nx + 1);
                let idx_n = i + (j + 1) * nx;
                let idx_s = i + j * nx;

                // Load the velocities.
                let u_e = Simd::from_slice(&self.u_star[idx_e..idx_e + LANES]);
                let u_w = Simd::from_slice(&self.u_star[idx_w..idx_w + LANES]);
                let v_n = Simd::from_slice(&self.v_star[idx_n..idx_n + LANES]);
                let v_s = Simd::from_slice(&self.v_star[idx_s..idx_s + LANES]);

                // Compute the divergence.
                let rhs = ((u_e - u_w) / dx_v + (v_n - v_s) / dy_v) / Simd::splat(dt_sub);
                rhs.copy_to_slice(&mut self.rhs[idx..idx + LANES]);
            }
        }
        println!("rhs time: {:?}", start_time.elapsed());

        let start_time = Instant::now();

        // ---------------- Pressure Correction Solver ----------------
        let denom = 2.0 / (dx * dx) + 2.0 / (dy * dy);
        let denom_v = Simd::splat(denom);
        match self.pressure_solver {
            PressureSolver::SOR => {
                // SOR solver implementation.
                self.p_prime.fill(0.0);
                let sor_omega = 1.7;
                let sor_omega_v = Simd::splat(sor_omega);
                let one_v = Simd::splat(1.0);
                let pressure_tolerance = 1e-4;
                let iterations = 50;
                let mut max_error: f32 = 0.0;
                for _ in 0..iterations {
                    max_error = 0.0;
                    for j in 1..(ny - 1) {
                        for i in (1..(nx - 1)).step_by(LANES) {
                            if i + LANES > nx - 1 {
                                for k in 0..(nx - i) {
                                    let idx = i + k + j * nx;
                                    let idx_e = (i + k + 1) + j * nx;
                                    let idx_w = i + k + j * nx;
                                    let idx_s = i + k + (j - 1) * nx;
                                    let idx_n = i + k + (j + 1) * nx;

                                    let p_old = self.p_prime[idx];

                                    let p_prime_e = self.p_prime[idx_e];
                                    let p_prime_w = self.p_prime[idx_w];
                                    let p_prime_n = self.p_prime[idx_n];
                                    let p_prime_s = self.p_prime[idx_s];
                                    let rhs = self.rhs[idx];

                                    let p_update = ((p_prime_e + p_prime_w)
                                        / (dx * dx)
                                        + (p_prime_n + p_prime_s) / (dy * dy)
                                        - rhs)
                                        / denom;

                                    self.p_prime[idx] =
                                        (1.0 - sor_omega) * p_old + sor_omega * p_update;

                                    let error = (self.p_prime[idx] - p_old).abs();
                                    if error > max_error {
                                        max_error = error;
                                    }
                                }
                                continue;
                            }

                            let idx = i + j * nx;
                            let idx_e = (i + 1) + j * nx;
                            let idx_w = (i - 1) + j * nx;
                            let idx_s = i + (j - 1) * nx;
                            let idx_n = i + (j + 1) * nx;

                            // Load neighbor values into SIMD vectors.
                            let p_old = Simd::from_slice(&self.p_prime[idx..idx + LANES]);

                            let p_prime_e = Simd::from_slice(&self.p_prime[idx_e..idx_e + LANES]);
                            let p_prime_w = Simd::from_slice(&self.p_prime[idx_w..idx_w + LANES]);
                            let p_prime_n = Simd::from_slice(&self.p_prime[idx_n..idx_n + LANES]);
                            let p_prime_s = Simd::from_slice(&self.p_prime[idx_s..idx_s + LANES]);
                            let rhs = Simd::from_slice(&self.rhs[idx..idx + LANES]);

                            let p_update = ((p_prime_e + p_prime_w)
                                / dx_v
                                + (p_prime_n + p_prime_s) / dy_v
                                - rhs)
                                / denom_v;
                            
                            let new_val = sor_omega_v * p_update + (one_v - sor_omega_v) * p_old;
                            new_val.copy_to_slice(&mut self.p_prime[idx..idx + LANES]);

                            let error = (new_val - p_old).abs();
                            max_error = max_error.max(error.reduce_max());
                        }
                    }
                    for i in 0..nx {
                        self.p_prime[i] = self.p_prime[i + nx]; // bottom
                        self.p_prime[i + (ny - 1) * nx] = self.p_prime[i + (ny - 2) * nx];
                        // top
                    }
                    for j in 0..ny {
                        self.p_prime[j * nx] = self.p_prime[1 + j * nx];
                        self.p_prime[(nx - 1) + j * nx] = 0.0;
                    }
                    if max_error < pressure_tolerance {
                        break;
                    }
                }
                self.last_pressure_residual = max_error;
            }
            PressureSolver::Jacobi => {
                let residual = self.jacobi_pressure(dx, dy, nx, ny);
                self.last_pressure_residual = residual;
            }
        }
        let corrector_time = start_time.elapsed();
        println!("corrector time: {:?}", corrector_time);

        // ---------------- Corrector Step ----------------
        let start = Instant::now();
        // Correct u
        for j in 0..ny {
            for i in (1..nx).step_by(LANES) {
                if i + LANES > nx {
                    for k in 0..(nx - i) {
                        let idx = i + k + j * (nx + 1);
                        let p_right = self.p_prime[i + k + j * nx];
                        let p_left = self.p_prime[i.saturating_sub(1) + k + j * nx];
                        self.u[idx] = self.u_star[idx] - dt_sub * (p_right - p_left) / dx;
                    }
                    continue;
                }

                let idx_u = i + j * (nx + 1);
                let idx_p = i + j * nx;
                let left_offset = i.saturating_sub(1) + j * nx;

                let p_right = Simd::<f32, LANES>::from_slice(&self.p_prime[idx_p..idx_p + LANES]);
                let p_left = Simd::<f32, LANES>::from_slice(&self.p_prime[left_offset..left_offset + LANES]);
                let dt_sub_v = Simd::splat(dt_sub);
                let dx_v = Simd::splat(dx);
                let correction = dt_sub_v * ((p_right - p_left) / dx_v);

                let u_star = Simd::<f32, LANES>::from_slice(&self.u_star[idx_u..idx_u + LANES]);
                let result = u_star - correction;
                result.copy_to_slice(&mut self.u[idx_u..idx_u + LANES]);
            }
        }
        // Correct v
        for j in 1..ny {
            for i in (0..nx).step_by(LANES) {
                if i + LANES > nx {
                    for k in 0..(nx - i) {
                        let idx = i + k + j * nx;
                        let p_top = self.p_prime[i + k + j * nx];
                        let p_bottom = self.p_prime[i + k + (j - 1) * nx];
                        self.v[idx] = self.v_star[idx] - dt_sub * (p_top - p_bottom) / dy;
                    }
                    continue;
                }

                let idx = i + j * nx;
                let p_top = Simd::<f32, LANES>::from_slice(&self.p_prime[idx..idx + LANES]);
                let bottom_idx = i + (j.saturating_sub(1)) * nx;
                let p_bottom = Simd::<f32, LANES>::from_slice(&self.p_prime[bottom_idx..bottom_idx + LANES]);
                let dt_sub_v = Simd::splat(dt_sub);
                let dy_v = Simd::splat(dy);
                let v_star = Simd::<f32, LANES>::from_slice(&self.v_star[idx..idx + LANES]);
                let correction = dt_sub_v * ((p_top - p_bottom) / dy_v);
                let v_new = v_star - correction;
                v_new.copy_to_slice(&mut self.v[idx..idx + LANES]);
            }
        }
        println!("velocity corrector time: {:?}", start.elapsed());
        // Accumulate the pressure correction
        let start = Instant::now();
        for i in (0..self.p.len()).step_by(LANES) {
            if i + LANES > self.p.len() {
                for k in 0..(self.p.len() - i) {
                    self.p[i + k] += self.p_prime[i + k];
                }
                continue;
            }

            let p_prime: Simd<f32, LANES> = Simd::from_slice(&self.p_prime[i..i + LANES]);
            let p = Simd::from_slice(&self.p[i..i + LANES]);
            let p_new = p + p_prime;
            p_new.copy_to_slice(&mut self.p[i..i + LANES]);
        }
        println!("pressure correcttion accumulation time: {:?}", start.elapsed());

        // Enforce boundary conditions
        let start = Instant::now();
        self.apply_boundary_conditions();
        println!("boundary time: {:?}", start.elapsed());
    }

    /// A helper for the Jacobi pressure correction solver.
    #[inline(never)]
    fn jacobi_pressure(&mut self, dx: f32, dy: f32, nx: usize, ny: usize) -> f32 {
        let jacobi_omega = 0.7;
        let pressure_tolerance = 1e-6;
        let iterations = 50;
        let mut max_error = 0.0;
        // Precompute constants.
        let dx_sq = dx * dx;
        let dx_sq_v = Simd::splat(dx_sq);
        let dy_sq = dy * dy;
        let dy_sq_v = Simd::splat(dy_sq);
        let jacobi_omega_v = Simd::splat(jacobi_omega);
        let jacobi_omega_o_m_v = Simd::splat(1.0 - jacobi_omega);
        let denom = 2.0 / (dx * dx) + 2.0 / (dy * dy);
        let denom_v = Simd::splat(denom);
        for _iter in 0..iterations {
            max_error = 0.0;
            for j in 1..(ny - 1) {
                // Process indices in SIMD chunks.
                for i in (1..(nx - 1)).step_by(LANES) {
                    let stride = j * nx + i;

                    if i + LANES > nx - 1 {
                        // Run scalar code for the last chunk.
                        for k in 0..(nx - i) {
                            let idx = stride + k;
                            let right = self.p_prime[idx + 1];
                            let left = self.p_prime[idx - 1];
                            let top = self.p_prime[idx + nx];
                            let bot = self.p_prime[idx - nx];
                            let center = self.p_prime[idx];
                            let rhs = self.rhs[idx];
                            let horizontal = (right + left) / dx_sq;
                            let vertical = (top + bot) / dy_sq;
                            let p_update = (horizontal + vertical - rhs) / denom;
                            let new_val = jacobi_omega * p_update + (1.0 - jacobi_omega) * center;
                            self.p_prime_new[idx] = new_val;
                        }
                        continue;
                    }

                    // Load neighbor values into SIMD vectors.
                    let right: Simd<f32, LANES> =
                        Simd::from_slice(&self.p_prime[(stride + 1)..(stride + 1 + LANES)]);
                    let left = Simd::from_slice(&self.p_prime[(stride - 1)..(stride - 1 + LANES)]);
                    let top = Simd::from_slice(
                        &self.p_prime[(i + (j + 1) * nx)..(i + (j + 1) * nx + LANES)],
                    );
                    let bot = Simd::from_slice(
                        &self.p_prime[(i + (j - 1) * nx)..(i + (j - 1) * nx + LANES)],
                    );
                    let center = Simd::from_slice(&self.p_prime[stride..(stride + LANES)]);
                    let rhs = Simd::from_slice(&self.rhs[stride..(stride + LANES)]);

                    // Compute the update using SIMD operations.
                    let horizontal = (right + left) / dx_sq_v;
                    let vertical = (top + bot) / dy_sq_v;
                    let p_update = (horizontal + vertical - rhs) / denom_v;

                    // Apply relaxation parameter.
                    let new_val = jacobi_omega_v * p_update + jacobi_omega_o_m_v * center;

                    let error = (new_val - center).abs().reduce_max();
                    if error > max_error {
                        max_error = error;
                    }

                    // Store the updated values back.
                    new_val.copy_to_slice(&mut self.p_prime_new[stride..(stride + LANES)]);
                }
            }

            std::mem::swap(&mut self.p_prime, &mut self.p_prime_new);

            for i in 0..nx {
                self.p_prime[i] = self.p_prime[i + nx]; // bottom
                self.p_prime[i + (ny - 1) * nx] = self.p_prime[i + (ny - 2) * nx];
                // top
            }
            for j in 0..ny {
                self.p_prime[j * nx] = self.p_prime[1 + j * nx]; // left
                self.p_prime[(nx - 1) + j * nx] = 0.0; // outlet
            }
            if max_error < pressure_tolerance {
                break;
            }
        }
        self.last_pressure_residual = max_error;
        max_error
    }

    /// Enforce boundary conditions on the velocity fields.
    fn apply_boundary_conditions(&mut self) {
        let nx = self.grid.nx;
        let ny = self.grid.ny;
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
                    if val < 0.0 {
                        0.0
                    } else {
                        val
                    }
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
        for (i, j) in &self.obstacle_coords {
            let idx_u = i + j * (nx + 1);
            let idx_v = i + j * nx;
            self.u[idx_u] = 0.0;
            self.v[idx_v] = 0.0;
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

    // --- Helper methods for u velocity discretization ---
    #[inline(always)]
    fn u_face_e_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx_left = i + j * (nx + 1);
        let idx_right = (i + 1) + j * (nx + 1);

        // Load LANES values for left and right faces.
        let u_left = Simd::from_slice(&self.u[idx_left..idx_left + LANES]);
        let u_right = Simd::from_slice(&self.u[idx_right..idx_right + LANES]);

        // Compute average using SIMD operations.
        let u_avg = (u_left + u_right) * Simd::splat(0.5);

        // Create mask: if average is >= 0.0, choose u_left, else choose u_right.
        let mask = u_avg.simd_ge(Simd::splat(0.0));
        mask.select(u_left, u_right)
    }

    #[inline(always)]
    fn u_face_e_second_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_e = (i + 1) + j * (nx + 1);
        let idx_ee = (i + 2) + j * (nx + 1);
        let idx_w = (i - 1) + j * (nx + 1);

        // Load current and east u values as SIMD vectors.
        let u_current = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_west = Simd::from_slice(&self.u[idx_w..idx_w + LANES]);
        let u_east = Simd::from_slice(&self.u[idx_e..idx_e + LANES]);
        let u_east_east = Simd::from_slice(&self.u[idx_ee..idx_ee + LANES]);

        // Preselect 1.5 * u[idx] - 0.5 * self.u[idx_w] if i > 1, else u[idx].
        let mask_i: Mask<i32, LANES> = Simd::from_slice(&(i..i + LANES).collect::<Vec<_>>())
            .simd_gt(Simd::splat(1))
            .into();
        let u_a = mask_i.select(
            Simd::splat(1.5) * u_current - Simd::splat(0.5) * u_west,
            u_current,
        );

        // Preselect 1.5 * u[idx_e] - 0.5 * u[idx_ee] if i < nx - 1, else self.u[idx_e]
        let mask_i: Mask<i32, LANES> = Simd::from_slice(&(i..i + LANES).collect::<Vec<_>>())
            .simd_lt(Simd::splat(nx - 1))
            .into();
        let u_b = mask_i.select(
            Simd::splat(1.5) * u_east - Simd::splat(0.5) * u_east_east,
            u_east,
        );

        // If u_currect >= 0.0, then u_a, else u_b.
        let mask = u_a.simd_ge(Simd::splat(0.0));
        mask.select(u_a, u_b)
    }

    #[inline(always)]
    fn u_face_e_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        if self.u[idx] >= 0.0 {
            if i >= 2 {
                (-self.u[idx - 1] + 6.0 * self.u[idx] + 3.0 * self.u[idx + 1]) / 8.0
            } else {
                1.5 * self.u[idx] - 0.5 * self.u[idx - 1]
            }
        } else if i <= nx - 2 {
            (3.0 * self.u[idx] + 6.0 * self.u[idx + 1] - self.u[idx + 2]) / 8.0
        } else {
            self.u[idx + 1]
        }
    }

    #[inline(always)]
    fn u_face_w_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;

        let idx = i + j * (nx + 1);
        let idx_w = (i - 1) + j * (nx + 1);

        let u_current = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_w = Simd::from_slice(&self.u[idx_w..idx_w + LANES]);
        let u_avg = (u_w + u_current) * Simd::splat(0.5);

        let mask = u_avg.simd_ge(Simd::splat(0.0));
        mask.select(u_w, u_current)
    }

    #[inline(always)]
    fn u_face_w_second_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_w = (i - 1) + j * (nx + 1);
        let idx_ww = (i - 2) + j * (nx + 1);
        let idx_e = (i + 1) + j * (nx + 1);

        let u_current = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_w = Simd::from_slice(&self.u[idx_w..idx_w + LANES]);
        let u_ww = Simd::from_slice(&self.u[idx_ww..idx_ww + LANES]);
        let u_e = Simd::from_slice(&self.u[idx_e..idx_e + LANES]);

        // For west branch: if u_w >= 0.0, use second order if valid (i > 2), else fallback to u_w.
        let mask_w_valid: Mask<i32, LANES> = Simd::from_slice(&(i..i + LANES).collect::<Vec<_>>())
            .simd_gt(Simd::splat(2))
            .into();
        let candidate_left =
            mask_w_valid.select(Simd::splat(1.5) * u_w - Simd::splat(0.5) * u_ww, u_w);

        // For east branch: if u_w < 0.0, use second order if valid (i < nx), else fallback to u_current.
        let mask_e_valid: Mask<i32, LANES> = Simd::from_slice(&(i..i + LANES).collect::<Vec<_>>())
            .simd_lt(Simd::splat(nx))
            .into();
        let candidate_right = mask_e_valid.select(
            Simd::splat(1.5) * u_current - Simd::splat(0.5) * u_e,
            u_current,
        );

        let mask = u_w.simd_ge(Simd::splat(0.0));
        mask.select(candidate_left, candidate_right)
    }

    #[inline(always)]
    fn u_face_w_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        if self.u[idx - 1] >= 0.0 {
            if i >= 3 {
                (-self.u[idx - 2] + 6.0 * self.u[idx - 1] + 3.0 * self.u[idx]) / 8.0
            } else {
                1.5 * self.u[idx - 1] - 0.5 * self.u[idx]
            }
        } else {
            (3.0 * self.u[idx - 1] + 6.0 * self.u[idx] - self.u[idx + 1]) / 8.0
        }
    }

    #[inline(always)]
    fn u_face_n_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_n = i + (j + 1) * (nx + 1);

        // Load current and north u values as SIMD vectors.
        let u_curr = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_next = Simd::from_slice(&self.u[idx_n..idx_n + LANES]);

        // Get v at north face as SIMD vector and create a mask.
        let v_n = self.get_v_north(i, j);
        let mask = v_n.simd_ge(Simd::splat(0.0));

        // Select u_curr if mask true, else u_next.
        mask.select(u_curr, u_next)
    }

    #[inline(always)]
    fn u_face_n_second_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_n = i + (j + 1) * (nx + 1);

        // Load the current and next rows into SIMD vectors.
        let current = Simd::from_slice(&self.u[idx..idx + LANES]);
        let north = Simd::from_slice(&self.u[idx_n..idx_n + LANES]);

        // When v_n >= 0.0:
        //   If j > 1, use a second–order upwind (using the value from previous row),
        //   otherwise, fall back to current.
        let res_positive = if j > 1 {
            // Load south row SIMD values.
            let idx_s = i + (j - 1) * (nx + 1);
            let current_s = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);
            current * Simd::splat(1.5) - current_s * Simd::splat(0.5)
        } else {
            current
        };

        // When v_n < 0.0:
        //   If there is a valid (j+2) row, use a second–order upwind (using the value from row j+2),
        //   otherwise, fall back to next row.
        let res_negative = if (idx_n + nx + 1) < self.u.len() && j < self.grid.ny - 1 {
            let idx_north_north = i + (j + 2) * (nx + 1);
            let north_north = Simd::from_slice(&self.u[idx_north_north..idx_north_north + LANES]);
            north * Simd::splat(1.5) - north_north * Simd::splat(0.5)
        } else {
            north
        };

        // Use SIMD mask based on v_n to select between the two alternatives.
        let v_n = self.get_v_north(i, j);
        let mask = v_n.simd_ge(Simd::splat(0.0));
        mask.select(res_positive, res_negative)
    }

    #[inline(always)]
    fn u_face_n_quick(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_n = i + (j + 1) * (nx + 1);

        // Load current and north row velocities as SIMD vectors.
        let u = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_n = Simd::from_slice(&self.u[idx_n..idx_n + LANES]);

        // Load south row (j - 1) velocities.
        let idx_s = i + (j - 1) * (nx + 1);
        let u_s = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);

        // Load row j + 2 velocities.
        let idx_nn = i + (j + 2) * (nx + 1);
        let u_nn = Simd::from_slice(&self.u[idx_nn..idx_nn + LANES]);

        // Get the north face v values as a SIMD vector.
        let v_n = self.get_v_north(i, j);

        // For positive v_n, choose second-order upwind if j >= 2, else first-order.
        let res_positive_second =
            (-u_s + Simd::splat(6.0) * u + Simd::splat(3.0) * u_n) / Simd::splat(8.0);
        let res_positive_first = Simd::splat(1.5) * u - Simd::splat(0.5) * u_s;
        let candidate_positive = if j >= 2 {
            res_positive_second
        } else {
            res_positive_first
        };

        // For negative v_n, use second-order upwind if possible.
        let res_negative =
            (Simd::splat(3.0) * u + Simd::splat(6.0) * u_n - u_nn) / Simd::splat(8.0);
        let candidate_negative = if j < self.grid.ny - 2 {
            res_negative
        } else {
            u_n
        };

        // Use a SIMD mask based on v_n >= 0.0 to select the appropriate candidate.
        let mask = v_n.simd_ge(Simd::splat(0.0));
        mask.select(candidate_positive, candidate_negative)
    }

    #[inline(always)]
    fn u_face_s_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_s = i + (j - 1) * (nx + 1);

        // Load u values for the south and current row
        let u_south = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);
        let u_curr = Simd::from_slice(&self.u[idx..idx + LANES]);

        // Get v_south values using existing helper which returns Simd<f32, LANES>
        let v_s = self.get_v_south(i, j);

        // Create a mask: if v_s >= 0.0 select u_south, else select u_curr.
        let mask = v_s.simd_ge(Simd::splat(0.0));
        mask.select(u_south, u_curr)
    }

    #[inline(always)]
    fn u_face_s_second_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_s = i + (j - 1) * (nx + 1);

        // Load current and south row into SIMD vectors.
        let current = Simd::from_slice(&self.u[idx..idx + LANES]);
        let south = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);

        // When v_s >= 0.0:
        //   If j > 1, use second–order upwind (using the value from row j-2),
        //   otherwise, fall back to south row.
        let res_positive = if j > 1 {
            let idx_s_south = i + (j - 2) * (nx + 1);
            let south_south = Simd::from_slice(&self.u[idx_s_south..idx_s_south + LANES]);
            south * Simd::splat(1.5) - south_south * Simd::splat(0.5)
        } else {
            south
        };

        // When v_s < 0.0:
        //   If j < (grid.ny - 1), use second–order upwind (using the value from row j+1),
        //   otherwise, fall back to current.
        let res_negative = if j < self.grid.ny - 1 {
            let idx_north = i + (j + 1) * (nx + 1);
            let north = Simd::from_slice(&self.u[idx_north..idx_north + LANES]);
            current * Simd::splat(1.5) - north * Simd::splat(0.5)
        } else {
            current
        };

        // Use SIMD mask based on v_s to select between the two alternatives.
        let v_s = self.get_v_south(i, j);
        let mask = v_s.simd_ge(Simd::splat(0.0));
        mask.select(res_positive, res_negative)
    }

    #[inline(always)]
    fn u_face_s_quick(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let base_stride = nx + 1;
        let idx = i + j * base_stride;
        let idx_s = i + (j - 1) * base_stride;
        let u = Simd::from_slice(&self.u[idx..idx + LANES]);
        let u_s = Simd::from_slice(&self.u[idx_s..idx_s + LANES]);

        let candidate_positive = if j >= 2 {
            let idx_ss = i + (j - 2) * base_stride;
            let u_ss = Simd::from_slice(&self.u[idx_ss..idx_ss + LANES]);
            (-u_ss + Simd::splat(6.0) * u_s + Simd::splat(3.0) * u) / Simd::splat(8.0)
        } else {
            Simd::splat(1.5) * u_s - Simd::splat(0.5) * u
        };

        let candidate_negative = if j < self.grid.ny - 1 {
            let idx_n = i + (j + 1) * base_stride;
            let u_n = Simd::from_slice(&self.u[idx_n..idx_n + LANES]);
            (Simd::splat(3.0) * u_s + Simd::splat(6.0) * u - u_n) / Simd::splat(8.0)
        } else {
            u
        };

        let v_s = self.get_v_south(i, j);
        let mask = v_s.simd_ge(Simd::splat(0.0));
        mask.select(candidate_positive, candidate_negative)
    }

    #[inline(always)]
    fn get_v_north(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let start = i + (j + 1) * nx;
        // SAFETY: Caller must ensure that start+LANES does not exceed self.v.len()
        Simd::from_slice(&self.v[start..start + LANES])
    }

    #[inline(always)]
    fn get_v_south(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let start = i + j * nx;
        // SAFETY: Caller must ensure that start+LANES does not exceed self.v.len()
        Simd::from_slice(&self.v[start..start + LANES])
    }

    // --- Helper methods for v velocity discretization ---
    #[inline(always)]
    fn v_face_e_first_order_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_e = self.u[(i + 1) + j * (nx + 1)];
        if u_e >= 0.0 {
            self.v[idx]
        } else {
            self.v[idx + 1]
        }
    }

    #[inline(always)]
    fn v_face_e_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        // Load LANES consecutive values for u_e, v_current and v_next.
        let u_e = Simd::from_slice(&self.u[(i + 1) + j * (nx + 1)..(i + 1) + j * (nx + 1) + LANES]);
        let v_curr = Simd::from_slice(&self.v[idx..idx + LANES]);
        let v_next = Simd::from_slice(&self.v[idx + 1..idx + 1 + LANES]);
        // Create mask: if u_e >= 0.0, select v_curr, else v_next.
        let mask = u_e.simd_ge(Simd::splat(0.0));
        mask.select(v_curr, v_next)
    }

    #[inline(always)]
    fn v_face_e_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_e = self.u[(i + 1) + j * (nx + 1)];
        if u_e >= 0.0 {
            if i > 0 {
                1.5 * self.v[idx] - 0.5 * self.v[idx - 1]
            } else {
                self.v[idx]
            }
        } else if (idx + 2) < self.v.len() && i < nx - 2 {
            1.5 * self.v[idx + 1] - 0.5 * self.v[idx + 2]
        } else {
            self.v[idx + 1]
        }
    }

    #[inline(always)]
    fn v_face_e_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_e = self.u[(i + 1) + j * (nx + 1)];
        if u_e >= 0.0 {
            if i >= 2 {
                (-self.v[idx - 1] + 6.0 * self.v[idx] + 3.0 * self.v[idx + 1]) / 8.0
            } else {
                1.5 * self.v[idx] - 0.5 * self.v[idx - 1]
            }
        } else if i < nx - 2 {
            (3.0 * self.v[idx] + 6.0 * self.v[idx + 1] - self.v[idx + 2]) / 8.0
        } else {
            self.v[idx + 1]
        }
    }

    #[inline(always)]
    fn v_face_w_first_order_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_w = self.u[i + j * (nx + 1)];
        if u_w >= 0.0 {
            self.v[idx - 1]
        } else {
            self.v[idx]
        }
    }

    #[inline(always)]
    fn v_face_w_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let base_u = i + j * (nx + 1);
        let base_v = i + j * nx;
        // Load u values for LANES lanes starting at the given index.
        let u_w = Simd::from_slice(&self.u[base_u..base_u + LANES]);
        // Create a mask where u_w >= 0.0.
        let mask = u_w.simd_ge(Simd::splat(0.0));
        // Load corresponding v values:
        // For lanes where u_w is >= 0.0, load from v at index (base_v - 1);
        // otherwise load from v at index base_v.
        let v_left = Simd::from_slice(&self.v[(base_v - 1)..(base_v - 1 + LANES)]);
        let v_center = Simd::from_slice(&self.v[base_v..base_v + LANES]);
        mask.select(v_left, v_center)
    }

    #[inline(always)]
    fn v_face_w_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_w = self.u[i + j * (nx + 1)];
        if u_w >= 0.0 {
            if i > 1 {
                1.5 * self.v[idx - 1] - 0.5 * self.v[idx - 2]
            } else {
                self.v[idx - 1]
            }
        } else if i < nx - 1 {
            1.5 * self.v[idx] - 0.5 * self.v[idx + 1]
        } else {
            self.v[idx]
        }
    }

    #[inline(always)]
    fn v_face_w_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let u_w = self.u[i + j * (nx + 1)];
        if u_w >= 0.0 {
            if i >= 3 {
                (-self.v[idx - 2] + 6.0 * self.v[idx - 1] + 3.0 * self.v[idx]) / 8.0
            } else {
                1.5 * self.v[idx - 1] - 0.5 * self.v[idx]
            }
        } else {
            (3.0 * self.v[idx - 1] + 6.0 * self.v[idx] - self.v[idx + 1]) / 8.0
        }
    }

    #[inline(always)]
    fn v_face_n_first_order_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_n = i + (j + 1) * nx;
        let v_avg = 0.5 * (self.v[idx] + self.v[idx_n]);
        if v_avg >= 0.0 {
            self.v[idx]
        } else {
            self.v[idx_n]
        }
    }

    #[inline(always)]
    fn v_face_n_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_n = i + (j + 1) * nx;
        let v_current = Simd::from_slice(&self.v[idx..idx + LANES]);
        let v_north = Simd::from_slice(&self.v[idx_n..idx_n + LANES]);
        let v_avg = (v_current + v_north) * Simd::splat(0.5);
        let mask = v_avg.simd_ge(Simd::splat(0.0));
        mask.select(v_current, v_north)
    }

    #[inline(always)]
    fn v_face_n_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_n = i + (j + 1) * nx;
        let v_avg = 0.5 * (self.v[idx] + self.v[idx_n]);
        if v_avg >= 0.0 {
            if j > 1 {
                1.5 * self.v[idx] - 0.5 * self.v[i + (j - 1) * nx]
            } else {
                self.v[idx]
            }
        } else if (i + (j + 2) * nx) < self.v.len() && j < self.grid.ny - 1 {
            1.5 * self.v[idx_n] - 0.5 * self.v[i + (j + 2) * nx]
        } else {
            self.v[idx_n]
        }
    }

    #[inline(always)]
    fn v_face_n_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_n = i + (j + 1) * nx;
        let v_avg = 0.5 * (self.v[idx] + self.v[idx_n]);
        if v_avg >= 0.0 {
            if j >= 2 {
                (-self.v[i + (j - 1) * nx] + 6.0 * self.v[idx] + 3.0 * self.v[idx_n]) / 8.0
            } else {
                1.5 * self.v[idx] - 0.5 * self.v[i + (j - 1) * nx]
            }
        } else if j < self.grid.ny - 1 {
            (3.0 * self.v[idx] + 6.0 * self.v[idx_n] - self.v[i + (j + 2) * nx]) / 8.0
        } else {
            self.v[idx_n]
        }
    }

    #[inline(always)]
    fn v_face_s_first_order_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_s = i + (j - 1) * nx;
        let v_avg = 0.5 * (self.v[idx_s] + self.v[idx]);
        if v_avg >= 0.0 {
            self.v[idx_s]
        } else {
            self.v[idx]
        }
    }

    #[inline(always)]
    fn v_face_s_first_order(&self, i: usize, j: usize) -> Simd<f32, LANES> {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_s = i + (j - 1) * nx;
        let v_current = Simd::from_slice(&self.v[idx..idx + LANES]);
        let v_south = Simd::from_slice(&self.v[idx_s..idx_s + LANES]);
        let v_avg = (v_current + v_south) * Simd::splat(0.5);
        let mask = v_avg.simd_ge(Simd::splat(0.0));
        mask.select(v_south, v_current)
    }

    #[inline(always)]
    fn v_face_s_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_s = i + (j - 1) * nx;
        let v_avg = 0.5 * (self.v[idx_s] + self.v[idx]);
        if v_avg >= 0.0 {
            if j > 1 {
                1.5 * self.v[idx_s] - 0.5 * self.v[i + (j - 2) * nx]
            } else {
                self.v[idx_s]
            }
        } else if j < self.grid.ny {
            1.5 * self.v[idx] - 0.5 * self.v[i + (j + 1) * nx]
        } else {
            self.v[idx]
        }
    }

    #[inline(always)]
    fn v_face_s_quick(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * nx;
        let idx_s = i + (j - 1) * nx;
        let v_avg = 0.5 * (self.v[idx_s] + self.v[idx]);
        if v_avg >= 0.0 {
            if j >= 2 {
                (-self.v[i + (j - 2) * nx] + 6.0 * self.v[idx_s] + 3.0 * self.v[idx]) / 8.0
            } else {
                1.5 * self.v[idx_s] - 0.5 * self.v[idx]
            }
        } else if j < self.grid.ny - 1 {
            (3.0 * self.v[idx_s] + 6.0 * self.v[idx] - self.v[i + (j + 1) * nx]) / 8.0
        } else {
            self.v[idx]
        }
    }

    pub fn set_parameters(&mut self, params: &SimulationParams) {
        self.nu = params.viscosity;
        self.dt = params.dt;
        self.target_inlet_velocity = params.target_inlet_velocity;
        self.velocity_scheme = params.velocity_scheme;
        self.pressure_solver = params.pressure_solver;
        self.inlet_profile = params.inlet_profile;
    }

    pub fn get_snapshot(&self) -> SimSnapshot {
        SimSnapshot {
            u: self.u.clone(),
            v: self.v.clone(),
            p: self.p.clone(),
            dt: self.dt,
            paused: false,
        }
    }

    pub fn get_residuals(&self) -> Residuals {
        Residuals {
            simulation_step: self.simulation_step,
            simulation_time: self.simulation_time,
            dt: self.dt,
            u: self.last_u_residual,
            v: self.last_v_residual,
            p: self.last_pressure_residual,
        }
    }

    pub fn run(mut self) -> SimulationControlHandle {
        let (command_sender, command_receiver) = mpsc::channel();
        let (snapshot_sender, snapshot_receiver) = mpsc::channel();
        let (residuals_sender, residuals_receiver) = mpsc::channel();

        thread::spawn(move || {
            let mut paused = false;
            loop {
                let start = Instant::now();
                let commands_in_queue = command_receiver.try_iter();

                let mut snapshot_sent = false;
                for command in commands_in_queue {
                    match command {
                        Command::Stop => break,
                        Command::SetParams(params) => {
                            self.set_parameters(&params);
                        }
                        Command::GetSnapshot => {
                            if !snapshot_sent {
                                let mut snapshot = self.get_snapshot();
                                snapshot.paused = paused;
                                snapshot_sender.send(snapshot).unwrap();
                                snapshot_sent = true;
                            }
                        }
                        Command::Pause => {
                            paused = true;
                        }
                        Command::Resume => {
                            paused = false;
                        }
                    }
                }

                if !paused {
                    self.update();
                    residuals_sender.send(self.get_residuals()).unwrap();
                    println!("====== Step time: {:?} ======", start.elapsed());
                } else {
                    thread::sleep(Duration::from_millis(16));
                }
            }
        });

        SimulationControlHandle {
            residuals_receiver,
            command_sender,
            snapshot_receiver,
        }
    }
}
