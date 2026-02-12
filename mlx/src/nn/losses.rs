//! Loss functions for training neural networks.

use crate::{Array, Result};

/// MSELoss

pub fn mse_loss(predictions: &Array, targets: &Array) -> Result<Array> {
    let diff = predictions.subtract(targets)?;
    let squared = diff.multiply(&diff)?;
    let axes: Vec<i32> = (0..predictions.ndim() as i32).collect();
    squared.mean_axes(&axes, false)
}

/// Cross Entropy Loss
pub fn cross_entropy(logits: &Array, targets: &Array) -> Result<Array> {
    let max_logits = logits.max_axis(-1, true)?; 
    let centered = logits.subtract(&max_logits)?;
    
    let exp_logits = centered.exp()?;
    let sum_exp = exp_logits.sum_axis(-1, true)?; 
    let log_z = sum_exp.log()?.add(&max_logits)?;

    let nll = log_z.subtract(logits)?;
    let weighted_nll = nll.multiply(targets)?;

    let per_sample_loss = weighted_nll.sum_axis(-1, false)?;
    // [Batch] -> Scalar
    per_sample_loss.mean_axis(0, false)
}