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
    pub step_time: Duration,
    pub piso_substeps: usize,
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
            dt: 0.005,
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
}

/// Which pressure solver to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureSolver {
    Jacobi,
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
    pub u_star: Vec<f32>,
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
    pub last_step_time: Duration,
    pub last_piso_substeps_count: usize,
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
            substep_count: 1,
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

            u_star: u.clone(),
            v_star: v.clone(),
            rhs: vec![0.0; size_p],
            p_prime: vec![0.0; size_p],
            p_prime_new: vec![0.0; size_p],
            last_pressure_residual: 0.0,
            last_u_residual: 0.0,
            last_v_residual: 0.0,
            simulation_time: 0.0,
            last_step_time: Duration::from_secs(0),
            last_piso_substeps_count: 0,
        }
    }

    /// Perform a simulation update (one "time step").
    /// Implements the extrapolation, multiple substeps (the PISO algorithm)
    /// and automatic dt adjustment.
    pub fn update(&mut self) {
        let step_start = Instant::now();
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

        // Perform the required number of substeps (PISO substeps)
        self.last_piso_substeps_count = self.substep_count;
        let start = Instant::now();
        for _ in 0..self.substep_count {
            let start_time = Instant::now();
            println!("");
            self.piso_step(dt_sub);
            let p_residual = self.last_pressure_residual;
            let piso_time = start_time.elapsed();
            println!("piso time: {:?}, p_residual: {:.3e}", piso_time, p_residual);
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
        // let error_norm: f32 = self.last_pressure_residual;
        // let tolerance = 1e-3;
        // if error_norm > tolerance {
        //     let factor = error_norm / tolerance;
        //     self.substep_count = ((self.substep_count as f32) * factor).ceil().min(20.0) as usize;
        // } else if error_norm < tolerance / 2.0 && self.substep_count > 1 {
        //     self.substep_count = (self.substep_count as f32 / 2.0).floor() as usize;
        //     if self.substep_count < 1 {
        //         self.substep_count = 1;
        //     }
        // }

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
        self.last_step_time = step_start.elapsed();
    }

    #[inline(always)]
    fn compute_ustar(
        &mut self,
        dt_sub: f32,
        i: usize,
        j: usize,
        v_n: Simd<f32, LANES>,
        v_s: Simd<f32, LANES>,
        u_n: Simd<f32, LANES>,
        u_s: Simd<f32, LANES>,
        u_e: Simd<f32, LANES>,
        u_w: Simd<f32, LANES>,
    ) {
        let nx = self.grid.nx;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dx_v = Simd::splat(dx);
        let dy_v = Simd::splat(dy);
        let nu = self.nu;
        let nu_v = Simd::splat(nu);

        let idx = i + j * (nx + 1);
        let idx_end = (i + LANES) + j * (nx + 1);

        let obstacle_mask_u = Simd::from_slice(&self.obstacle_mask_u[idx..idx_end]);
        let obstacle_mask_u: Mask<i32, LANES> = obstacle_mask_u.simd_eq(Simd::splat(1)).into();

        let convective = {
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

        let u_star = u + Simd::splat(dt_sub) * (-convective + nu_v * laplace);
        let u_star = obstacle_mask_u.select(Simd::splat(0.0), u_star);
        u_star.copy_to_slice(&mut self.u_star[idx..idx_end]);
    }

    #[inline(always)]
    fn compute_vstar(
        &mut self,
        dt_sub: f32,
        i: usize,
        j: usize,
        u_e: Simd<f32, LANES>,
        u_w: Simd<f32, LANES>,
        v_n: Simd<f32, LANES>,
        v_s: Simd<f32, LANES>,
        v_e: Simd<f32, LANES>,
        v_w: Simd<f32, LANES>,
    ) {
        let nx = self.grid.nx;
        let dx = self.grid.dx;
        let dy = self.grid.dy;

        let idx = i + j * nx;
        if i + LANES > nx - 1 {
            let v_u_e = u_e.to_array();
            let v_u_w = u_w.to_array();
            let v_v_n = v_n.to_array();
            let v_v_s = v_s.to_array();
            let v_v_e = v_e.to_array();
            let v_v_w = v_w.to_array();

            for k in 0..(nx - i) {
                let scalar_idx = i + k + j * nx;

                if self.obstacle_mask_v[scalar_idx] == 1 {
                    self.v_star[scalar_idx] = 0.0;
                    continue;
                }
                let u_e = v_u_e[k];
                let u_w = v_u_w[k];
                let v_n = v_v_n[k];
                let v_s = v_v_s[k];
                let v_e = v_v_e[k];
                let v_w = v_v_w[k];
                let f_e = u_e * v_e;
                let f_w = u_w * v_w;
                let f_n = v_n * v_n;
                let f_s = v_s * v_s;
                let convective = (f_e - f_w) / dx + (f_n - f_s) / dy;
                let v_val = self.v[scalar_idx];
                let idx_e = (i + k + 1) + j * nx;
                let idx_w = (i + k).saturating_sub(1) + j * nx;
                let idx_n = i + k + (j + 1) * nx;
                let idx_s = i + k + (j - 1) * nx;
                let v_e_val = self.v[idx_e];
                let v_w_val = self.v[idx_w];
                let v_n_val = self.v[idx_n];
                let v_s_val = self.v[idx_s];
                let laplace = (v_e_val - 2.0 * v_val + v_w_val) / (dx * dx)
                    + (v_n_val - 2.0 * v_val + v_s_val) / (dy * dy);
                self.v_star[scalar_idx] = v_val + dt_sub * (-convective + self.nu * laplace);
            }
            return;
        }

        let obstacle_mask_v = Simd::from_slice(&self.obstacle_mask_v[idx..idx + LANES]);
        let obstacle_mask_v: Mask<i32, LANES> = obstacle_mask_v.simd_eq(Simd::splat(1)).into();

        let f_e = u_e * v_e;
        let f_w = u_w * v_w;
        let f_n = v_n * v_n;
        let f_s = v_s * v_s;
        let convective = (f_e - f_w) / Simd::splat(dx) + (f_n - f_s) / Simd::splat(dy);

        let v_vec = Simd::from_slice(&self.v[idx..idx + LANES]);
        let idx_e = i + 1 + j * nx;
        let idx_w = i - 1 + j * nx;
        let idx_n = i + (j + 1) * nx;
        let idx_s = i + (j - 1) * nx;
        let v_e_vec = Simd::from_slice(&self.v[idx_e..idx_e + LANES]);
        let v_w_vec = Simd::from_slice(&self.v[idx_w..idx_w + LANES]);
        let v_n_vec = Simd::from_slice(&self.v[idx_n..idx_n + LANES]);
        let v_s_vec = Simd::from_slice(&self.v[idx_s..idx_s + LANES]);
        let laplace = (v_e_vec - Simd::splat(2.0) * v_vec + v_w_vec) / (Simd::splat(dx * dx))
            + (v_n_vec - Simd::splat(2.0) * v_vec + v_s_vec) / (Simd::splat(dy * dy));
        let result = v_vec + Simd::splat(dt_sub) * (-convective + Simd::splat(self.nu) * laplace);
        let result = obstacle_mask_v.select(Simd::splat(0.0), result);
        result.copy_to_slice(&mut self.v_star[idx..idx + LANES]);
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

        // ---------------- Predictor for u ----------------
        // Loop over internal u faces.
        let start = Instant::now();
        match self.velocity_scheme {
            VelocityScheme::FirstOrder => {
                for j in 1..(ny - 1) {
                    for i in (1..nx).step_by(LANES) {
                        let u_v_n = self.get_v_north(i, j);
                        let u_v_s = self.get_v_south(i, j);
                        let u_u_n = self.u_face_n_first_order(i, j);
                        let u_u_s = self.u_face_s_first_order(i, j);
                        let u_u_e = self.u_face_e_first_order(i, j);
                        let u_u_w = self.u_face_w_first_order(i, j);

                        self.compute_ustar(dt_sub, i, j, u_v_n, u_v_s, u_u_n, u_u_s, u_u_e, u_u_w);
                    }
                }
            }
            VelocityScheme::SecondOrder => {
                for j in 1..(ny - 1) {
                    for i in (1..nx).step_by(LANES) {
                        let u_v_n = self.get_v_north(i, j);
                        let u_v_s = self.get_v_south(i, j);

                        let mut u_u_n = [0.0; LANES];
                        let mut u_u_s = [0.0; LANES];
                        let mut u_u_e = [0.0; LANES];
                        let mut u_u_w = [0.0; LANES];

                        for k in 0..LANES {
                            u_u_n[k] = self.u_face_n_second_order(i + k, j);
                            u_u_s[k] = self.u_face_s_second_order(i + k, j);
                            u_u_e[k] = self.u_face_e_second_order(i + k, j);
                            u_u_w[k] = self.u_face_w_second_order(i + k, j);
                        }

                        let u_u_n = Simd::from_slice(&u_u_n);
                        let u_u_s = Simd::from_slice(&u_u_s);
                        let u_u_e = Simd::from_slice(&u_u_e);
                        let u_u_w = Simd::from_slice(&u_u_w);

                        self.compute_ustar(dt_sub, i, j, u_v_n, u_v_s, u_u_n, u_u_s, u_u_e, u_u_w);
                    }
                }
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
                        // Handle the right boundary as scalar.
                        if i + LANES > nx - 1 {
                            let mut v_u_e = [0.0; LANES];
                            let mut v_u_w = [0.0; LANES];
                            let mut v_v_n = [0.0; LANES];
                            let mut v_v_s = [0.0; LANES];
                            let mut v_v_e = [0.0; LANES];
                            let mut v_v_w = [0.0; LANES];

                            for k in 0..(nx - i) {
                                v_u_e[k] = self.u[(i + k + 1) + j * (nx + 1)];
                                v_u_w[k] = self.u[(i + k) + j * (nx + 1)];
                                v_v_n[k] = self.v_face_n_first_order_scalar(i + k, j);
                                v_v_s[k] = self.v_face_s_first_order_scalar(i + k, j);
                                v_v_e[k] = self.v_face_e_first_order_scalar(i + k, j);
                                v_v_w[k] = self.v_face_w_first_order_scalar(i + k, j);
                            }

                            let v_u_e = Simd::from_slice(&v_u_e);
                            let v_u_w = Simd::from_slice(&v_u_w);
                            let v_v_n = Simd::from_slice(&v_v_n);
                            let v_v_s = Simd::from_slice(&v_v_s);
                            let v_v_e = Simd::from_slice(&v_v_e);
                            let v_v_w = Simd::from_slice(&v_v_w);

                            self.compute_vstar(
                                dt_sub, i, j, v_u_e, v_u_w, v_v_n, v_v_s, v_v_e, v_v_w,
                            );

                            continue;
                        }

                        let v_u_e = Simd::from_slice(
                            &self.u[(i + 1) + j * (nx + 1)..(i + LANES + 1) + j * (nx + 1)],
                        );
                        let v_u_w =
                            Simd::from_slice(&self.u[i + j * (nx + 1)..i + LANES + j * (nx + 1)]);

                        let v_v_n = self.v_face_n_first_order(i, j);
                        let v_v_s = self.v_face_s_first_order(i, j);
                        let v_v_e = self.v_face_e_first_order(i, j);
                        let v_v_w = self.v_face_w_first_order(i, j);

                        self.compute_vstar(dt_sub, i, j, v_u_e, v_u_w, v_v_n, v_v_s, v_v_e, v_v_w);
                    }
                }
            }
            VelocityScheme::SecondOrder => {
                for j in 1..ny {
                    for i in (1..(nx - 1)).step_by(LANES) {
                        let mut v_u_e = [0.0; LANES];
                        let mut v_u_w = [0.0; LANES];
                        let mut v_v_n = [0.0; LANES];
                        let mut v_v_s = [0.0; LANES];
                        let mut v_v_e = [0.0; LANES];
                        let mut v_v_w = [0.0; LANES];

                        for k in 0..LANES {
                            if i + k >= nx - 1 {
                                break;
                            }
                            v_u_e[k] = self.u[(i + k + 1) + j * (nx + 1)];
                            v_u_w[k] = self.u[(i + k) + j * (nx + 1)];
                            v_v_n[k] = self.v_face_n_second_order(i + k, j);
                            v_v_s[k] = self.v_face_s_second_order(i + k, j);
                            v_v_e[k] = self.v_face_e_second_order(i + k, j);
                            v_v_w[k] = self.v_face_w_second_order(i + k, j);
                        }

                        let v_u_e = Simd::from_slice(&v_u_e);
                        let v_u_w = Simd::from_slice(&v_u_w);
                        let v_v_n = Simd::from_slice(&v_v_n);
                        let v_v_s = Simd::from_slice(&v_v_s);
                        let v_v_e = Simd::from_slice(&v_v_e);
                        let v_v_w = Simd::from_slice(&v_v_w);

                        self.compute_vstar(dt_sub, i, j, v_u_e, v_u_w, v_v_n, v_v_s, v_v_e, v_v_w);
                    }
                }
            }
        }
        println!("v_face time: {:?}", start.elapsed());

        // ---------------- Pressure Correction (MAC form) ----------------
        // Compute the divergence (rhs) on pressure cells.
        let start_time = Instant::now();
        self.recompute_divergence(dt_sub, dx, dy, nx, ny);
        println!("rhs time: {:?}", start_time.elapsed());

        let start_time = Instant::now();

        // ---------------- Pressure Correction Solver ----------------
        match self.pressure_solver {
            PressureSolver::Jacobi => {
                let residual = self.jacobi_pressure(dx, dy, nx, ny);
                self.last_pressure_residual = residual;
            }
        }
        let corrector_time = start_time.elapsed();
        println!("corrector time: {:?}", corrector_time);

        // ---------------- Corrector Step ----------------
        let start = Instant::now();
        self.apply_corrector(dt_sub, dx, dy, nx, ny);
        println!("apply corrector time: {:?}", start.elapsed());

        for _iter in 0..20 {
            let start = Instant::now();
            self.u_star.copy_from_slice(&self.u);
            self.v_star.copy_from_slice(&self.v);
            println!("copy time: {:?}", start.elapsed());

            // Recompute divergence from the updated velocities.
            let start = Instant::now();
            self.recompute_divergence(dt_sub, dx, dy, nx, ny);
            println!("rhs time: {:?}", start.elapsed());
            // Re-solve for an additional pressure correction.
            let start = Instant::now();
            match self.pressure_solver {
                PressureSolver::Jacobi => {
                    let residual = self.jacobi_pressure(dx, dy, nx, ny);
                    self.last_pressure_residual = residual;
                }
            }
            let corrector_time = start.elapsed();
            println!("corrector time: {:?}", corrector_time);
            // Second corrector update.
            let start = Instant::now();
            self.apply_corrector(dt_sub, dx, dy, nx, ny);
            println!("apply corrector time: {:?}", start.elapsed());

            if self.last_pressure_residual < 1e-4 {
                break;
            }
        }

        // Enforce boundary conditions
        let start = Instant::now();
        self.apply_boundary_conditions();
        println!("boundary time: {:?}", start.elapsed());
    }

    /// A helper for the Jacobi pressure correction solver.
    #[inline(never)]
    fn jacobi_pressure(&mut self, dx: f32, dy: f32, nx: usize, ny: usize) -> f32 {
        let jacobi_omega = 0.75;
        let pressure_tolerance = 1e-4;
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
                println!("Jacobi solver converged after {} iterations.", _iter);
                break;
            }
        }
        println!("Jacobi solver max error: {}", max_error);
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
            let cfl = 0.2;
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
    fn u_face_e_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_e = (i + 1) + j * (nx + 1);
        if self.u[idx] >= 0.0 {
            if i > 1 {
                1.5 * self.u[idx] - 0.5 * self.u[idx - 1]
            } else {
                self.u[idx]
            }
        } else if (idx_e + 1) < self.u.len() && i < nx - 1 {
            1.5 * self.u[idx_e] - 0.5 * self.u[idx_e + 1]
        } else {
            self.u[idx_e]
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
    fn u_face_w_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_w = (i - 1) + j * (nx + 1);
        let idx_ww = (i - 2) + j * (nx + 1);
        let idx_e = (i + 1) + j * (nx + 1);
        if self.u[idx_w] >= 0.0 {
            if i > 2 {
                1.5 * self.u[idx_w] - 0.5 * self.u[idx_ww]
            } else {
                self.u[idx_w]
            }
        } else {
            if i < nx {
                1.5 * self.u[idx] - 0.5 * self.u[idx_e]
            } else {
                self.u[idx]
            }
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
    fn get_v_north_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx_v_nw = if i > 0 { (i - 1) + (j + 1) * nx } else { 0 };
        let idx_v_n = i + (j + 1) * nx;
        0.5 * (self.v[idx_v_nw] + self.v[idx_v_n])
    }

    #[inline(always)]
    fn u_face_n_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_n = i + (j + 1) * (nx + 1);
        let v_n = self.get_v_north_scalar(i, j);
        if v_n >= 0.0 {
            if j > 1 {
                1.5 * self.u[idx] - 0.5 * self.u[i + (j - 1) * (nx + 1)]
            } else {
                self.u[idx]
            }
        } else if (i + (j + 2) * (nx + 1)) < self.u.len() && j < self.grid.ny - 1 {
            1.5 * self.u[idx_n] - 0.5 * self.u[i + (j + 2) * (nx + 1)]
        } else {
            self.u[idx_n]
        }
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
    fn get_v_south_scalar(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx_v_s = if i > 0 { (i - 1) + j * nx } else { 0 };
        let idx_v = i + j * nx;
        0.5 * (self.v[idx_v_s] + self.v[idx_v])
    }

    #[inline(always)]
    fn u_face_s_second_order(&self, i: usize, j: usize) -> f32 {
        let nx = self.grid.nx;
        let idx = i + j * (nx + 1);
        let idx_s = i + (j - 1) * (nx + 1);
        let v_s = self.get_v_south_scalar(i, j);
        if v_s >= 0.0 {
            if j > 1 {
                1.5 * self.u[idx_s] - 0.5 * self.u[i + (j - 2) * (nx + 1)]
            } else {
                self.u[idx_s]
            }
        } else if j < self.grid.ny {
            1.5 * self.u[idx] - 0.5 * self.u[i + (j + 1) * (nx + 1)]
        } else {
            self.u[idx]
        }
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
            step_time: self.last_step_time,
            piso_substeps: self.last_piso_substeps_count,
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

    fn apply_corrector(&mut self, dt_sub: f32, dx: f32, dy: f32, nx: usize, ny: usize) {
        // Correct u
        for j in 0..ny {
            for i in (1..nx).step_by(LANES) {
                if i + LANES > nx {
                    for k in 0..(nx - i) {
                        let idx = i + k + j * (nx + 1);
                        let p_right = self.p_prime[i + k + j * nx];
                        let p_left = self.p_prime[i.saturating_sub(1) + k + j * nx];
                        self.u[idx] = self.u_star[idx] - dt_sub * (p_right - p_left) / dx;
                        // This is wrong. dt_sub should be multiplied by the correction term.
                    }
                    continue;
                }

                let idx_u = i + j * (nx + 1);
                let idx_p = i + j * nx;
                let left_offset = i.saturating_sub(1) + j * nx;

                let p_right = Simd::<f32, LANES>::from_slice(&self.p_prime[idx_p..idx_p + LANES]);
                let p_left =
                    Simd::<f32, LANES>::from_slice(&self.p_prime[left_offset..left_offset + LANES]);
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
                let p_bottom =
                    Simd::<f32, LANES>::from_slice(&self.p_prime[bottom_idx..bottom_idx + LANES]);
                let dt_sub_v = Simd::splat(dt_sub);
                let dy_v = Simd::splat(dy);
                let v_star = Simd::<f32, LANES>::from_slice(&self.v_star[idx..idx + LANES]);
                let correction = dt_sub_v * ((p_top - p_bottom) / dy_v);
                let v_new = v_star - correction;
                v_new.copy_to_slice(&mut self.v[idx..idx + LANES]);
            }
        }
        // Accumulate the pressure correction.
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
    }

    fn recompute_divergence(&mut self, dt_sub: f32, dx: f32, dy: f32, nx: usize, ny: usize) {
        let dx_v: Simd<f32, LANES> = Simd::splat(dx);
        let dy_v = Simd::splat(dy);
        let dt_sub_v = Simd::splat(dt_sub);
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
                let u_e = Simd::from_slice(&self.u_star[idx_e..idx_e + LANES]);
                let u_w = Simd::from_slice(&self.u_star[idx_w..idx_w + LANES]);
                let v_n = Simd::from_slice(&self.v_star[idx_n..idx_n + LANES]);
                let v_s = Simd::from_slice(&self.v_star[idx_s..idx_s + LANES]);
                let rhs = ((u_e - u_w) / dx_v + (v_n - v_s) / dy_v) / dt_sub_v;
                rhs.copy_to_slice(&mut self.rhs[idx..idx + LANES]);
            }
        }
    }
}
