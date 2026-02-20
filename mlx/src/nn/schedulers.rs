//! Learning rate schedulers for controlling optimizer learning rates during training.
//!
//! Schedulers adjust the learning rate over the course of training to improve
//! convergence. They are designed to work with any optimizer that exposes a
//! mutable `lr` field.
//!
//! # Usage
//!
//! ```no_run
//! use mlx::nn::{Adam, Optimizer, StepLR, LRScheduler};
//!
//! let mut optimizer = Adam::new(1e-3, &params).unwrap();
//! let mut scheduler = StepLR::new(1e-3, 30, 0.1);
//!
//! for epoch in 0..100 {
//!     // ... training step ...
//!     let new_lr = scheduler.step();
//!     optimizer.lr = new_lr;
//! }
//! ```

use std::f32::consts::PI;

// =========================================================================
// Scheduler Trait
// =========================================================================

/// Common interface for all learning rate schedulers.
///
/// Call `step()` once per epoch (or per iteration, depending on your schedule)
/// to advance the scheduler and get the updated learning rate.
pub trait LRScheduler {
 
    fn step(&mut self) -> f32;

    fn get_lr(&self) -> f32;

    fn reset(&mut self);
}

// =========================================================================
// StepLR
// =========================================================================

pub struct StepLR {
    base_lr: f32,
    step_size: u32,
    gamma: f32,
    current_epoch: u32,
    current_lr: f32,
}

impl StepLR {
    pub fn new(base_lr: f32, step_size: u32, gamma: f32) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            current_epoch: 0,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self) -> f32 {
        self.current_epoch += 1;
        let power = self.current_epoch / self.step_size;
        self.current_lr = self.base_lr * self.gamma.powi(power as i32);
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.base_lr;
    }
}

// =========================================================================
// ExponentialLR
// =========================================================================
pub struct ExponentialLR {
    base_lr: f32,
    gamma: f32,
    current_epoch: u32,
    current_lr: f32,
}

impl ExponentialLR {
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        Self {
            base_lr,
            gamma,
            current_epoch: 0,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn step(&mut self) -> f32 {
        self.current_epoch += 1;
        self.current_lr = self.base_lr * self.gamma.powi(self.current_epoch as i32);
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.base_lr;
    }
}

// =========================================================================
// CosineAnnealingLR
// =========================================================================

pub struct CosineAnnealingLR {
    base_lr: f32,
    t_max: u32,
    eta_min: f32,
    current_epoch: u32,
    current_lr: f32,
}

impl CosineAnnealingLR {
    pub fn new(base_lr: f32, t_max: u32, eta_min: f32) -> Self {
        Self {
            base_lr,
            t_max,
            eta_min,
            current_epoch: 0,
            current_lr: base_lr,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self) -> f32 {
        self.current_epoch += 1;
        let t = (self.current_epoch % self.t_max) as f32;
        let t_max = self.t_max as f32;
        self.current_lr = self.eta_min
            + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * t / t_max).cos());
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_lr = self.base_lr;
    }
}

// =========================================================================
// ReduceLROnPlateau
// =========================================================================

/// Reduces the learning rate when a metric has stopped improving.
///
/// Unlike the other schedulers, this one requires you to pass in a metric
/// value (e.g. validation loss) via `step_with_metric()`.
///
/// If the metric does not improve for `patience` consecutive calls,
/// the learning rate is multiplied by `factor`.
pub struct ReduceLROnPlateau {
    current_lr: f32,
    factor: f32,
    patience: u32,
    min_lr: f32,
    threshold: f32,
    mode: PlateauMode,
    best: f32,
    num_bad_epochs: u32,
}

/// Whether we're tracking a metric to minimize (e.g. loss) or maximize (e.g. accuracy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    Min,
    Max,
}

impl ReduceLROnPlateau {
    pub fn new(
        initial_lr: f32,
        factor: f32,
        patience: u32,
        min_lr: f32,
        threshold: f32,
        mode: PlateauMode,
    ) -> Self {
        let best = match mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };
        Self {
            current_lr: initial_lr,
            factor,
            patience,
            min_lr,
            threshold,
            mode,
            best,
            num_bad_epochs: 0,
        }
    }


    pub fn for_loss(initial_lr: f32) -> Self {
        Self::new(initial_lr, 0.1, 10, 1e-6, 1e-4, PlateauMode::Min)
    }

    pub fn step_with_metric(&mut self, metric: f32) -> f32 {
        let improved = match self.mode {
            PlateauMode::Min => metric < self.best - self.threshold,
            PlateauMode::Max => metric > self.best + self.threshold,
        };

        if improved {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.num_bad_epochs > self.patience {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
            self.num_bad_epochs = 0;
        }

        self.current_lr
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn step(&mut self) -> f32 {
        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.num_bad_epochs = 0;
        self.best = match self.mode {
            PlateauMode::Min => f32::INFINITY,
            PlateauMode::Max => f32::NEG_INFINITY,
        };
    }
}

// =========================================================================
// OneCycleLR
// =========================================================================

/// Implements the 1cycle policy (Smith & Topin, 2018).

pub struct OneCycleLR {
    initial_lr: f32,
    max_lr: f32,
    final_lr: f32,
    total_steps: u32,
    pct_start: f32,
    current_step: u32,
    current_lr: f32,
}

impl OneCycleLR {
    pub fn new(
        max_lr: f32,
        total_steps: u32,
        pct_start: f32,
        div_factor: f32,
        final_div_factor: f32,
    ) -> Self {
        let initial_lr = max_lr / div_factor;
        let final_lr = initial_lr / final_div_factor;
        Self {
            initial_lr,
            max_lr,
            final_lr,
            total_steps,
            pct_start,
            current_step: 0,
            current_lr: initial_lr,
        }
    }

    pub fn default(max_lr: f32, total_steps: u32) -> Self {
        Self::new(max_lr, total_steps, 0.3, 25.0, 1e4)
    }
}

impl LRScheduler for OneCycleLR {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        let step = self.current_step.min(self.total_steps) as f32;
        let total = self.total_steps as f32;
        let warmup_steps = total * self.pct_start;

        self.current_lr = if step <= warmup_steps {
            // Phase 1: linear warmup from initial_lr to max_lr
            let pct = step / warmup_steps;
            self.initial_lr + (self.max_lr - self.initial_lr) * pct
        } else {
            // Phase 2: cosine annealing from max_lr to final_lr
            let pct = (step - warmup_steps) / (total - warmup_steps);
            self.final_lr
                + 0.5 * (self.max_lr - self.final_lr) * (1.0 + (PI * pct).cos())
        };

        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = self.initial_lr;
    }
}

// =========================================================================
// Linear Warmup + Decay
// =========================================================================


pub struct WarmupSchedule {
    base_lr: f32,
    final_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
    current_step: u32,
    current_lr: f32,
}

impl WarmupSchedule {
    pub fn new(base_lr: f32, warmup_steps: u32, total_steps: u32, final_lr: f32) -> Self {
        Self {
            base_lr,
            final_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
            current_lr: 0.0,
        }
    }

    
    pub fn cosine(base_lr: f32, warmup_steps: u32, total_steps: u32) -> WarmupCosineSchedule {
        WarmupCosineSchedule {
            base_lr,
            warmup_steps,
            total_steps,
            eta_min: 0.0,
            current_step: 0,
            current_lr: 0.0,
        }
    }
}

impl LRScheduler for WarmupSchedule {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        let step = self.current_step as f32;

        self.current_lr = if self.current_step <= self.warmup_steps {
            // Linear warmup: 0 -> base_lr
            self.base_lr * step / self.warmup_steps as f32
        } else {
            // Linear decay: base_lr -> final_lr
            let decay_steps = (self.total_steps - self.warmup_steps) as f32;
            let decay_progress = (step - self.warmup_steps as f32) / decay_steps;
            let decay_progress = decay_progress.min(1.0);
            self.base_lr + (self.final_lr - self.base_lr) * decay_progress
        };

        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = 0.0;
    }
}

// =========================================================================
// Warmup + Cosine Decay (Transformer Standard)
// =========================================================================

pub struct WarmupCosineSchedule {
    base_lr: f32,
    warmup_steps: u32,
    total_steps: u32,
    eta_min: f32,
    current_step: u32,
    current_lr: f32,
}

impl WarmupCosineSchedule {
    pub fn new(base_lr: f32, warmup_steps: u32, total_steps: u32, eta_min: f32) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            eta_min,
            current_step: 0,
            current_lr: 0.0,
        }
    }
}

impl LRScheduler for WarmupCosineSchedule {
    fn step(&mut self) -> f32 {
        self.current_step += 1;
        let step = self.current_step as f32;

        self.current_lr = if self.current_step <= self.warmup_steps {
            self.base_lr * step / self.warmup_steps as f32
        } else {
            let decay_steps = (self.total_steps - self.warmup_steps) as f32;
            let progress = (step - self.warmup_steps as f32) / decay_steps;
            let progress = progress.min(1.0);
            self.eta_min
                + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * progress).cos())
        };

        self.current_lr
    }

    fn get_lr(&self) -> f32 {
        self.current_lr
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.current_lr = 0.0;
    }
}

