//! Activation functions for neural networks.
//!
//! Provides both module-based and functional APIs.

use crate::{Array, Result};
use crate::nn::Module;

// ===== ReLU =====

/// Rectified Linear Unit activation.
///
/// Applies: f(x) = max(0, x)
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU
    }
}

impl Module for ReLU {
    fn forward(&self, x: &Array) -> Result<Array> {
        relu(x)
    }
    
    fn parameters(&self) -> Vec<&Array> {
        vec![]
    }
    
    fn train(&mut self, _training: bool) {}
}

/// Functional ReLU: max(0, x)
pub fn relu(x: &Array) -> Result<Array> {
    let zero= Array::full(&[],0.0,x.dtype())?;
    unsafe {
        let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = crate::sys::mlx_maximum(&mut res_handle, x.handle, zero.handle,Array::default_stream());
        
        x.check_status(status, res_handle)

      
    }
}

// ===== GELU =====

/// Gaussian Error Linear Unit.
///
/// Smoother than ReLU, often used in transformers.
pub struct GELU;

impl GELU {
    pub fn new() -> Self { GELU }
}

impl Module for GELU {
    fn forward(&self, x: &Array) -> Result<Array> {
        gelu(x)
    }
    
    fn parameters(&self) -> Vec<&Array> {
        vec![]
    }
    
    fn train(&mut self, _training: bool) {}
}

/// GELU approximation using Tanh:
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: &Array) -> Result<Array> {
    let constant_sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    
    let x_sq = x.multiply(x)?;
    let x_cube = x_sq.multiply(x)?;
    
    let inner_term = x_cube.multiply_scalar(0.044715)?;
    let combined_inner = x.add(&inner_term)?;
    
    // 3. Apply the tanh: tanh(sqrt(2/pi) * combined_inner)
    let tanh_input = combined_inner.multiply_scalar(constant_sqrt_2_pi)?;
    let tanh_output = tanh(&tanh_input)?; 
    
    let one = Array::full(&[], 1.0, x.dtype())?;
    let plus_one = tanh_output.add(&one)?;
    let half_x = x.multiply_scalar(0.5)?;
    
    half_x.multiply(&plus_one)
}

// ===== Softmax =====

pub struct Softmax {
    pub axis: i32,
}

impl Softmax {
    pub fn new(axis: i32) -> Self {
        Softmax { axis }
    }
}

impl Module for Softmax {
    fn forward(&self, x: &Array) -> Result<Array> {
        softmax(x, self.axis)
    }
    
    fn parameters(&self) -> Vec<&Array> {
        vec![]
    }

    fn train(&mut self, _training: bool) {
        
    }
}

/// Computes the softmax along the specified axis.
pub fn softmax(x: &Array, axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
    
        let status = crate::sys::mlx_softmax_axis(
            &mut res_handle, 
            x.handle, 
            axis, //axis
            false,  // precise
            Array::default_stream()
        );
        x.check_status(status, res_handle)
    }
}

// ===== Other Activations =====

pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, x: &Array) -> Result<Array> { sigmoid(x) }
    fn parameters(&self) -> Vec<&Array> { vec![] }
    fn train(&mut self, _: bool) {}
}

pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, x: &Array) -> Result<Array> { tanh(x) }
    fn parameters(&self) -> Vec<&Array> { vec![] }
    fn train(&mut self, _: bool) {}
}

pub fn sigmoid(x: &Array) -> Result<Array> {
    unsafe{
        let mut res_handle = crate::sys::mlx_array{ctx: std::ptr::null_mut()}; let status = crate::sys::mlx_sigmoid(&mut res_handle,x.handle,Array::default_stream()); x.check_status(status, res_handle) }
    
}


pub fn tanh(x: &Array) -> Result<Array> {
  unsafe{
    let mut res_handle = crate::sys::mlx_array{ctx:std::ptr::null_mut()};
    let status = crate::sys::mlx_tanh(&mut res_handle,x.handle,Array::default_stream()); x.check_status(status,res_handle)
  }
}



// ===== Leaky ReLU =====

pub fn leaky_relu(x: &Array, alpha:f32) -> Result<Array>{
    let condition = x.greater_than_scalar(0.0)?;
   let scaled_x = x.multiply_scalar(alpha)?;
    condition.where_op(x, &scaled_x)
}

// ===== ELU=====

pub fn elu(x: &Array, alpha: f32) -> Result<Array> {
    let condition = x.greater_than_scalar(0.0)?;
    let exp_x = x.exp()?;
    let scaled_exp_x = exp_x.multiply_scalar(alpha)?.subtract_scalar(alpha)?;
    condition.where_op(x, &scaled_exp_x)
}