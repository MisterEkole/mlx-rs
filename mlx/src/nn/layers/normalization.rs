//! Normalization layers.

use crate::{Array, Result, Dtype};
use crate::nn::Module;
use mlx_derive::ModuleParams;
use crate::TreeFlatten;

/// Layer Normalization.

#[derive(ModuleParams)]
pub struct LayerNorm {
    pub weight: Array,
    pub bias: Array,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(dims: usize, eps: f32) -> Result<Self> {
        // Initialize weights to 1s and bias to 0s
        // Shape is [dims]
        let weight = Array::full(&[dims as i32], 1.0, Dtype::Float32)?;
        let bias = Array::full(&[dims as i32], 0.0, Dtype::Float32)?;
        
        Ok(LayerNorm {
            weight,
            bias,
            eps,
        })
    }
}
impl TreeFlatten for LayerNorm {
    fn flatten_state(&self) -> Vec<Array> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.weight = flat_arrays.next().unwrap().clone();
        self.bias = flat_arrays.next().unwrap().clone();
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        layer_norm(x, &self.weight, &self.bias, self.eps)
    }
    
}

pub fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let axis = -1;
    let mean = x.mean(axis, true)?;
    let var = x.var(axis, true)?;
    
    let den = var.add_scalar(eps)?.sqrt()?;
    let x_norm = x.subtract(&mean)?.divide(&den)?;
    
    x_norm.multiply(weight)?.add(bias)
}

/// RMS Normalization.

#[derive(ModuleParams)]
pub struct RMSNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dims: usize, eps: f32) -> Result<Self> {
        let weight = Array::full(&[dims as i32], 1.0, Dtype::Float32)?;
        Ok(RMSNorm { weight, eps })
    }
}

impl TreeFlatten for RMSNorm {
    fn flatten_state(&self) -> Vec<Array> {
        vec![self.weight.clone()]
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.weight = flat_arrays.next().unwrap().clone();
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let x_squared = x.multiply(x)?;
        let mean_squared = x_squared.mean(-1, true)?;
        let rms = mean_squared.add_scalar(self.eps)?.sqrt()?;
        
        let normalized = x.divide(&rms)?;
        normalized.multiply(&self.weight)
    }
    
}