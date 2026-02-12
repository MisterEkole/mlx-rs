//! Flattening layers to bridge spatial and linear layers.

use crate::{Array, Result};
use crate::nn::Module;

pub struct Flatten {
    pub start_axis: i32,
    pub end_axis: i32,
}

impl Flatten {
    /// Creates a new Flatten layer. 
    /// Standard behavior is to keep the batch dimension (axis 0) 
    /// and flatten everything else.
    pub fn new() -> Self {
        Self {
            start_axis: 1,
            end_axis: -1,
        }
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?;
        if shape.len() < 2 {
            return Ok(x.clone());
        }

        // We assume axis 0 is Batch [N]
        let batch_size = shape[0] as i32;
        
        // Calculate the product of all dimensions from start_axis to end_axis
        // For a standard flatten, this is H * W * C
        let flattened_dim: usize = shape.iter().skip(1).product();

        // Reshape to [Batch, Total_Features]
        x.reshape(&[batch_size, flattened_dim as i32])
    }

    fn parameters(&self) -> Vec<&Array> {
        // Flatten is a functional layer and has no learnable parameters
        vec![]
    }

    fn train(&mut self, _training: bool) {
        // No-op
    }
}