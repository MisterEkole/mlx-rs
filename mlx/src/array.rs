// Array.rs - Core Array struct and operations

use std::fmt;
use crate::{sys, Error, Result, Dtype};
use std::ptr;
use std::os::raw::c_int;

// =========================================================================
// Core Struct Definition
// =========================================================================

/// The core Array struct wrapping the MLX C-API handle.
pub struct Array {
    pub(crate) handle: sys::mlx_array,
}

// =========================================================================
// Initialization & Internal Helpers
// =========================================================================

impl Array {
    /// Helper to create a default stream handle for MLX operations.
    pub(crate) fn default_stream() -> sys::mlx_stream {
        unsafe {
            let mut device = sys::mlx_device_ { ctx: std::ptr::null_mut() };
            sys::mlx_get_default_device(&mut device);

            let mut stream = sys::mlx_stream_ { ctx: std::ptr::null_mut() };
            let status = sys::mlx_get_default_stream(&mut stream, device);
            
            if status != 0 || stream.ctx.is_null() {
                panic!("Failed to get MLX default stream. Status: {}", status);
            }
            stream
        }
    }

    /// Helper to reduce boilerplate error checking for MLX C-API calls.
    pub fn check_status(&self, status: i32, handle: sys::mlx_array) -> Result<Array> {
        if status != 0 || handle.ctx.is_null() {
            Err(Error::OperationFailed("Failed to perform operation".into()))
        } else {
            Ok(Array { handle })
        }
    }
}

// =========================================================================
// Array Creation Methods
// =========================================================================

impl Array {
    /// Create an array from a slice of data and a shape.
    pub fn from_slice<T>(data: &[T], shape: &[usize], dtype: Dtype) -> Result<Self> {
        let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
        unsafe {
            let handle = sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape_i32.as_ptr(),
                shape_i32.len() as c_int,
                dtype.to_sys(),
            );
            if handle.ctx.is_null() { Err(Error::NullPointer) } else { Ok(Array { handle }) }
        }
    }

    /// Create an array filled with a single value.
    pub fn full(shape: &[i32], val: f32, dtype: crate::Dtype) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let val_array = Array::from_slice(&[val], &[], dtype)?;
            let status = sys::mlx_full(
                &mut res_handle,
                shape.as_ptr(),
                shape.len(),
                val_array.handle, 
                dtype.to_sys(),
                Self::default_stream(),
            );
            val_array.check_status(status, res_handle)
        }
    }

    /// Create a range of values.
    pub fn arange(start: f64, stop: f64, step: f64, dtype: Dtype) -> Result<Self> {
        let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        unsafe {
            let status = sys::mlx_arange(&mut handle, start, stop, step, dtype.into(), Self::default_stream());
            if status != 0 { return Err(Error::OperationFailed("mlx_arange failed".into())); }
            Ok(Array { handle })
        }
    }
    pub fn zeros(shape: &[i32], dtype: Dtype) -> Result<Array> {
        Self::full(shape, 0.0, dtype)
    }

    /// Random initialization (Uniform).
    pub fn random_uniform(shape: &[usize], low: f32, high: f32, dtype: Dtype, key: &Array) -> Result<Array> {
        let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let low_arr = Array::full(&[], low, dtype)?;
        let high_arr = Array::full(&[], high, dtype)?;
        unsafe {
            let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = crate::sys::mlx_random_uniform(
                &mut res_handle, low_arr.handle, high_arr.handle,
                shape_i32.as_ptr(), shape_i32.len(), dtype.into(),
                key.handle, Self::default_stream(),
            );
            low_arr.check_status(status, res_handle)
        }
    }

    pub fn key(seed: u64) -> Result<Self> {
        unsafe {
            let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = crate::sys::mlx_random_key(&mut res_handle, seed);
            if status != 0 || res_handle.ctx.is_null() { Err(Error::OperationFailed("Failed key gen".into())) } else { Ok(Array { handle: res_handle }) }
        }
    }

    pub fn split(&self) -> Result<(Array, Array)> {
        unsafe {
            let mut res1 = sys::mlx_array { ctx: ptr::null_mut() };
            let mut res2 = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_random_split(&mut res1, &mut res2, self.handle, Self::default_stream());
            if status != 0 { return Err(Error::OperationFailed("split failed".into())); }
            Ok((Array { handle: res1 }, Array { handle: res2 }))
        }
    }
}

// =========================================================================
// Mathematical Operations (Arithmetic & Logic)
// =========================================================================

impl Array {
    pub fn add(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_add(&mut res, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn subtract(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_subtract(&mut res, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn multiply(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_multiply(&mut res, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn divide(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_divide(&mut res, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn matmul(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_matmul(&mut res, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn equal(&self, other: &Array) -> Result<Self> {
        let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        unsafe {
            let status = sys::mlx_equal(&mut handle, self.handle, other.handle, Self::default_stream());
            if status != 0 { return Err(Error::OperationFailed("equal failed".into())); }
            Ok(Array { handle })
        }
    }

    // --- Scalar Helpers ---
    pub fn add_scalar(&self, val: f32) -> Result<Array> { self.add(&Self::full(&[], val, self.dtype())?) }
    pub fn subtract_scalar(&self, val: f32) -> Result<Array> { self.subtract(&Self::full(&[], val, self.dtype())?) }
    pub fn multiply_scalar(&self, val: f32) -> Result<Array> { self.multiply(&Self::full(&[], val, self.dtype())?) }
    pub fn divide_scalar(&self, val: f32) -> Result<Array> { self.divide(&Self::full(&[], val, self.dtype())?) }
}

// =========================================================================
// Shape & Transformation Operations
// =========================================================================

impl Array {
    pub fn shape(&self) -> Result<Vec<usize>> {
        unsafe {
            let ndim = sys::mlx_array_ndim(self.handle) as usize;
            let shape_ptr = sys::mlx_array_shape(self.handle);
            Ok(std::slice::from_raw_parts(shape_ptr, ndim).iter().map(|&x| x as usize).collect())
        }
    }

    pub fn ndim(&self) -> usize { unsafe { sys::mlx_array_ndim(self.handle) } }

    pub fn dtype(&self) -> crate::Dtype {
        unsafe { crate::Dtype::from_sys(sys::mlx_array_dtype(self.handle)) }
    }

    pub fn reshape(&self, new_shape: &[i32]) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_reshape(&mut res, self.handle, new_shape.as_ptr(), new_shape.len(), Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn transpose(&self, _axes: &[i32]) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_transpose(&mut res, self.handle, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_transpose_axes(&mut res, self.handle, axes.as_ptr(), axes.len(), Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_broadcast_to(&mut res, self.handle, shape.as_ptr(), shape.len(), Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn squeeze(&self, axes: Option<&[i32]>) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let status = match axes {
                None => sys::mlx_squeeze(&mut res, self.handle, Self::default_stream()),
                Some(ax) => sys::mlx_squeeze_axes(&mut res, self.handle, ax.as_ptr(), ax.len(), Self::default_stream()),
            };
            self.check_status(status, res)
        }
    }

    pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: ptr::null_mut() };
            let handles: Vec<_> = arrays.iter().map(|a| a.handle).collect();
            let vec_handle = sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len());
            let status = sys::mlx_concatenate_axis(&mut res, vec_handle, axis, Self::default_stream());
            sys::mlx_vector_array_free(vec_handle);
            if status != 0 { Err(Error::OperationFailed("concat failed".into())) } else { Ok(Array { handle: res }) }
        }
    }

    pub fn cast(&self, dtype: Dtype) -> Result<Self> {
        let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        unsafe {
            let status = sys::mlx_astype(&mut handle, self.handle, dtype.into(), Self::default_stream());
            if status != 0 { return Err(Error::OperationFailed("cast failed".into())); }
            Ok(Array { handle })
        }
    }
}

// =========================================================================
// Data Retrieval & Evaluation
// =========================================================================

impl Array {
    pub fn eval(&self) -> Result<()> { Self::eval_all(&[self.clone()]) }

    pub fn eval_all(arrays: &[Array]) -> Result<()> {
        if arrays.is_empty() { return Ok(()); }
        unsafe {
            let handles: Vec<_> = arrays.iter().map(|a| a.handle).collect();
            let vec_handle = sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len());
            sys::mlx_eval(vec_handle);
            sys::mlx_vector_array_free(vec_handle);
            Ok(())
        }
    }

    pub fn item<T: Copy + 'static>(&self) -> Result<T> {
        self.eval()?;
        unsafe {
            let data_ptr = sys::mlx_array_data_float32(self.handle);
            if data_ptr.is_null() { return Err(Error::OperationFailed("null data".into())); }
            Ok(*(data_ptr as *const T))
        }
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.eval()?;
        unsafe {
            let data_ptr = sys::mlx_array_data_float32(self.handle);
            let size = sys::mlx_array_size(self.handle);
            if data_ptr.is_null() { return Err(Error::OperationFailed("null data".into())); }
            Ok(std::slice::from_raw_parts(data_ptr, size as usize).to_vec())
        }
    }
}



//============================================================================
// Trigonometric Operations
//============================================================================

impl Array{
    pub fn cos(&self) -> Result<Array> {
        unsafe{
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_cos(&mut res_handle,self.handle,Self::default_stream());
            self.check_status(status,res_handle)
        }
        }

    pub fn sin(&self) -> Result<Array> {
        unsafe{
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_sin(&mut res_handle,self.handle,Self::default_stream());
            self.check_status(status,res_handle)
        }
        }
}

/// Negation Operation
impl Array{
    pub fn negative(&self) -> Result<Array> {
        unsafe{
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_negative(&mut res_handle,self.handle,Self::default_stream());
            self.check_status(status,res_handle)
        }
        }

}

/// Argmax/Argmin
impl Array{
    pub fn argmax_axis(&self,axis: i32,keepdims: bool) -> Result<Array>{
    unsafe{
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_argmax_axis(&mut res_handle,self.handle,axis,keepdims, Self::default_stream());
        self.check_status(status,res_handle)
    }
}
    pub fn argmin_axis(&self,axis: i32,keepdims: bool) -> Result<Array>{
    unsafe{
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_argmin_axis(&mut res_handle, self.handle, axis, keepdims, Self::default_stream());
        self.check_status(status, res_handle)
    }
 
    
}
}

// =========================================================================
// Trait Implementations (Memory & Formatting)
// =========================================================================

impl Drop for Array {
    fn drop(&mut self) {
        unsafe { if !self.handle.ctx.is_null() { sys::mlx_array_free(self.handle); } }
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        unsafe {
            let mut new_handle = sys::mlx_array { ctx: ptr::null_mut() };
            sys::mlx_array_set(&mut new_handle, self.handle);
            Array { handle: new_handle }
        }
    }
}

impl fmt::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unsafe {
            let mut str_handle = sys::mlx_string { ctx: ptr::null_mut() };
            sys::mlx_array_tostring(&mut str_handle, self.handle);
            let c_str_ptr = sys::mlx_string_data(str_handle);
            if c_str_ptr.is_null() {
                write!(f, "Array(<error>)")?;
            } else {
                write!(f, "{}", std::ffi::CStr::from_ptr(c_str_ptr).to_string_lossy())?;
            }
            if !str_handle.ctx.is_null() { sys::mlx_string_free(str_handle); }
            Ok(())
        }
    }
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_vec_f32() {
            Ok(data) => write!(f, "{:?}", data),
            Err(_) => write!(f, "[Error fetching data]"),
        }
    }
}