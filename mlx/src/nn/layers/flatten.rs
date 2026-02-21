//! Flattening layers to bridge spatial and linear layers.
use crate::{Array, Result};
use crate::nn::{Module, ModuleParams};
use crate::TreeFlatten;

pub struct Flatten {
    pub start_axis: i32,
    pub end_axis: i32,
}

impl Flatten {
    /// Standard behavior: keep axis 0 (Batch) and flatten the rest.
    pub fn new() -> Self {
        Self {
            start_axis: 1,
            end_axis: -1,
        }
    }
}
impl ModuleParams for Flatten {}
impl TreeFlatten for Flatten {
    fn flatten_state(&self) -> Vec<Array> {
        Vec::new()
    }

    fn unflatten_state(&mut self, _flat_arrays: &mut std::slice::Iter<'_, Array>) {
       
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?; // returns Vec<usize>
        if shape.len() < 2 {
            return Ok(x.clone());
        }

        // MLX-C reshape expects &[i32]. We must cast our usize values.
        let batch_size = shape[0] as i32;
        
        // Calculate the product of all dimensions after the batch axis
        let flattened_dim: usize = shape.iter().skip(1).product();

        // Pass an array of i32 to the MLX reshape method
        x.reshape(&[batch_size, flattened_dim as i32])
    }

}

