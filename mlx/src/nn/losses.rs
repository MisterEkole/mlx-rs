//! Loss functions for training neural networks.

use crate::{Array, Result};
use crate::nn::layers::activations::{sigmoid};

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



/// Binary Cross Entropy Loss
pub fn binary_cross_entropy(logits: &Array, targets: &Array) -> Result<Array> {
    // formula: -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    let probs = sigmoid(logits)?;
    
    let one = Array::full(&[], 1.0, probs.dtype())?;
    
    let log_probs = probs.log()?;
    let one_minus_probs = one.subtract(&probs)?;
    let log_one_minus_probs = one_minus_probs.log()?;
    
    let term1 = targets.multiply(&log_probs)?;
    let one_minus_targets = one.subtract(targets)?;
    let term2 = one_minus_targets.multiply(&log_one_minus_probs)?;
    
    let combined = term1.add(&term2)?;
    let negative_loss = combined.mean_axis(0, false)?;
    
    negative_loss.multiply_scalar(-1.0)
}

/// L1 Loss (Mean Absolute Error)
pub fn l1_loss(predictions: &Array, targets: &Array) -> Result<Array> {
    let diff = predictions.subtract(targets)?;
    let abs_diff = diff.abs()?; // Assumes .abs() is in array.rs
    let axes: Vec<i32> = (0..predictions.ndim() as i32).collect();
    abs_diff.mean_axes(&axes, false)
}

/// Smooth L1 Loss (Huber Loss)
pub fn smooth_l1_loss(predictions: &Array, targets: &Array, beta: f32) -> Result<Array> {
    let diff = predictions.subtract(targets)?;
    let abs_diff = diff.abs()?;
    
    // if abs(diff) < beta: 0.5 * diff^2 / beta
    // else: abs(diff) - 0.5 * beta
    let squared_part = diff.multiply(&diff)?.multiply_scalar(0.5 / beta)?;
    let linear_part = abs_diff.subtract_scalar(0.5 * beta)?;
    
    let beta_arr = Array::full(&[], beta, abs_diff.dtype())?;
    let mask = abs_diff.less_than(&beta_arr)?;
    
    // Assumes .where_op() is in array.rs
    let loss = mask.where_op(&squared_part, &linear_part)?;
    let axes: Vec<i32> = (0..predictions.ndim() as i32).collect();
    loss.mean_axes(&axes, false)
}

/// KL Divergence Loss
pub fn kl_div(log_probs: &Array, targets: &Array) -> Result<Array> {
    // formula: targets * (log(targets) - log_probs)
    let log_targets = targets.log()?;
    let diff = log_targets.subtract(log_probs)?;
    let kl = targets.multiply(&diff)?;
    
    let axes: Vec<i32> = (0..kl.ndim() as i32).collect();
    kl.sum_axes(&axes, false)
}

/// Cosine Embedding Loss
pub fn cosine_embedding_loss(x1: &Array, x2: &Array, y: &Array, margin: f32) -> Result<Array> {
    // cos = (x1 . x2) / (||x1|| * ||x2||)
    let dot = x1.multiply(x2)?.sum_axis(-1, false)?;
    let norm1 = x1.multiply(x1)?.sum_axis(-1, false)?.sqrt()?;
    let norm2 = x2.multiply(x2)?.sum_axis(-1, false)?.sqrt()?;
    
    let cos = dot.divide(&norm1.multiply(&norm2)?)?;
    
    // if y == 1: 1 - cos
    // if y == -1: max(0, cos - margin)
    let loss_y1 = Array::full(&[], 1.0, cos.dtype())?.subtract(&cos)?;
    
    let margin_arr = Array::full(&[], margin, cos.dtype())?;
    let zero_arr = Array::full(&[], 0.0, cos.dtype())?;
    let loss_y_neg1 = cos.subtract(&margin_arr)?.maximum(&zero_arr)?;
    
    let y_is_1 = y.equal(&Array::full(&[], 1.0, y.dtype())?)?;
    let per_sample_loss = y_is_1.where_op(&loss_y1, &loss_y_neg1)?;
    
    per_sample_loss.mean_axis(0, false)
}