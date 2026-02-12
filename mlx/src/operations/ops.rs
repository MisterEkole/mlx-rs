use crate::{sys, Array, Result};
use std::ptr;

impl Array {

    /// Primitve element-wise operations (these directly call the C-API)
    // --- Unary Ops ---
    pub fn exp(&self) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_exp(&mut res_handle, self.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn log(&self) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_log(&mut res_handle, self.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn sqrt(&self) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_sqrt(&mut res_handle, self.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    // --- Reduction Ops ---
    pub fn mean(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_mean_axis(&mut res_handle, self.handle,axis, keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn max(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_max_axis(&mut res_handle, self.handle, axis,keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    //Absolute value Ops
    pub fn abs(&self) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = sys::mlx_abs(&mut res_handle, self.handle, Self::default_stream());
        self.check_status(status, res_handle)
    }
}


    /// --- High Level Utilities (Safe Rust) ---///

    pub fn flatten(&self) -> Result<Array> {
    let shape = self.shape()?;
    let total_elements: usize = shape.iter().product(); 
    self.reshape(&[total_elements as i32])
    }

    pub fn expand_dims(&self, axis: i32) -> Result<Array> {
    let mut new_shape = self.shape()?;
    // Convert negative axis to positive (e.g., -1 is the last index)
    let len = new_shape.len() as i32;
    let pos = if axis < 0 {
        (len + axis + 1).max(0) as usize
    } else {
        axis as usize
    };
    new_shape.insert(pos, 1);
    // Convert to i32 for the reshape primitive
    let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
    self.reshape(&new_shape_i32)
    }

    pub fn square(&self) -> Result<Array> {
        self.multiply(self)
    }

    /// Variance: mean((x - mean(x))^2)
    pub fn var(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let mu = self.mean(axis, keepdims)?;
        let diff = self.subtract(&mu)?; 
        let squared_diff = diff.square()?;
        
        squared_diff.mean(axis, keepdims)
    }

    /// Standard Deviation: sqrt(variance)
    pub fn std(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let v = self.var(axis, keepdims)?;
        v.sqrt() 
    }

    /// Creates an array of zeros with the same shape and dtype as self
    pub fn zeros_like(&self) -> Result<Array> {
        let shape = self.shape()?;
        let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        Self::full(&shape_i32, 0.0, self.dtype())
    }

    /// Creates an array of ones with the same shape and dtype as self
    pub fn ones_like(&self) -> Result<Array> {
        let shape = self.shape()?;
        let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        Self::full(&shape_i32, 1.0, self.dtype())
    }


















}