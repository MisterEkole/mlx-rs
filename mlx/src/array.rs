// Array.rs - Core Array struct and operations

use std::fmt;
use crate::{sys, Error, Result, Dtype};
use std::ptr;
use std::os::raw::c_int;

/// The core Array struct
pub struct Array {
    // We keep the inner handle public to the crate so other modules (like nn.rs) 
    // can access the C-pointer, but private to the outside world.
    pub(crate) handle: sys::mlx_array,
}

impl Array {
    /// Helper to create a default stream handle
pub(crate)fn default_stream() -> sys::mlx_stream {
        unsafe {
            // 1. Initialize the device struct with a null context
            let mut device = sys::mlx_device_ { ctx: std::ptr::null_mut() };
            
            // 2. Fill the device struct using the C-API
            sys::mlx_get_default_device(&mut device);

            // 3. Initialize the stream struct with a null context
            let mut stream = sys::mlx_stream_ { ctx: std::ptr::null_mut() };
            
            // 4. Fill the stream struct using the device we just got
            let status = sys::mlx_get_default_stream(&mut stream, device);
            
            if status != 0 || stream.ctx.is_null() {
                panic!("Failed to get MLX default stream. Status: {}", status);
            }

            stream
        }
    }

    /// Create an array from a slice of data
    pub fn from_slice<T>(data: &[T], shape: &[usize], dtype: Dtype) -> Result<Self> {
        let shape_i32: Vec<c_int> = shape.iter().map(|&x| x as c_int).collect();
        
        unsafe {
            let handle = sys::mlx_array_new_data(
                data.as_ptr() as *const _,
                shape_i32.as_ptr(),
                shape_i32.len() as c_int,
                dtype.to_sys(),
            );
            
            if handle.ctx.is_null() {
                Err(Error::NullPointer)
            } else {
                Ok(Array { handle })
            }
        }
    }
    pub fn random_uniform(
    shape: &[usize], 
    low: f32, 
    high: f32, 
    dtype: Dtype,
    key: &Array 
    ) -> Result<Array> {
    let shape_i32: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
    
    // Convert bounds to MLX scalar arrays
    let low_arr = Array::full(&[], low, dtype)?;
    let high_arr = Array::full(&[], high, dtype)?;

    unsafe {
        let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
        
        let status = crate::sys::mlx_random_uniform(
            &mut res_handle,
            low_arr.handle,     
            high_arr.handle,     
            shape_i32.as_ptr(),
            shape_i32.len(),
            dtype.into(), 
            key.handle,          // Pass the entropy key
            Self::default_stream(),
        );
        
        low_arr.check_status(status, res_handle)
    }
}
    /// Element-wise addition
    pub fn add(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_add(
                &mut res_handle, 
                self.handle, 
                other.handle, 
                Self::default_stream()
            );
            
            if status != 0 || res_handle.ctx.is_null() {
                Err(Error::OperationFailed)
            } else {
                Ok(Array { handle: res_handle })
            }
        }
    }


    /// Element-wise subtraction
    pub fn subtract(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_subtract(&mut res_handle, self.handle, other.handle, Self::default_stream());
            
            if status != 0 || res_handle.ctx.is_null() {
                Err(Error::OperationFailed)
            } else {
                Ok(Array { handle: res_handle })
            }
        }
    }

    /// Element-wise multiplication
    pub fn multiply(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_multiply(&mut res_handle, self.handle, other.handle, Self::default_stream());
            
            if status != 0 || res_handle.ctx.is_null() {
                Err(Error::OperationFailed)
            } else {
                Ok(Array { handle: res_handle })
            }
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_matmul(&mut res_handle, self.handle, other.handle, Self::default_stream());
            
            if status != 0 || res_handle.ctx.is_null() {
                Err(Error::OperationFailed)
            } else {
                Ok(Array { handle: res_handle })
            }
        }
    }

    /// Element-wise division
    pub fn divide(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_divide(&mut res_handle, self.handle, other.handle, Self::default_stream());
            
            if status != 0 || res_handle.ctx.is_null() {
                Err(Error::OperationFailed)
            } else {
                Ok(Array { handle: res_handle })
            }
        }
    }

    /// Multiplies the array by a scalar float.
    pub fn multiply_scalar(&self, value: f32) -> Result<Array> {
        // Create a 0-dimension (scalar) array
        let scalar = Self::full(&[], value, self.dtype())?;
        self.multiply(&scalar)
    }
    /// Adds a scalar float to the array.
    pub fn add_scalar(&self, value: f32) -> Result<Array> {
        let scalar = Self::full(&[], value, self.dtype())?;
        self.add(&scalar)
    }


    /// Generates a new base key from a seed.
    pub fn key(seed: u64) -> Result<Self> {
        unsafe {
            let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = crate::sys::mlx_random_key(&mut res_handle, seed);
            if status != 0 || res_handle.ctx.is_null() { Err(Error::OperationFailed) } else { Ok(Array { handle: res_handle }) }
        }
    }

    /// Splits a key into two new keys.
 
    pub fn split(&self) -> Result<(Array, Array)> {
    unsafe {
        let mut res1_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
        let mut res2_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };

        let status = crate::sys::mlx_random_split(
            &mut res1_handle, 
            &mut res2_handle, 
            self.handle, 
            Self::default_stream()
        );
        if status != 0 || res1_handle.ctx.is_null() || res2_handle.ctx.is_null() {
            return Err(Error::OperationFailed);
        }

        Ok((
            Array { handle: res1_handle }, 
            Array { handle: res2_handle }
        ))
    }
}


    

    /// Trigger evaluation
    pub fn eval(&self) -> Result<()> {
        unsafe {
            let vec = sys::mlx_vector_array_new();
            
            // Append THIS array to the vector
            let status = sys::mlx_vector_array_append_data(vec, &self.handle, 1);
            
            if status != 0 {
                sys::mlx_vector_array_free(vec);
                return Err(Error::OperationFailed);
            }

            sys::mlx_eval(vec);
            sys::mlx_vector_array_free(vec);
            Ok(())
        }
    }

    // shape method to get the shape of the array
    pub fn shape(&self) -> Result<Vec<usize>> {
        unsafe{
            let ndim = sys::mlx_array_ndim(self.handle) as usize;
            let shape_ptr= sys::mlx_array_shape(self.handle);
            std::slice::from_raw_parts(shape_ptr, ndim).iter().map(|&x| Ok(x as usize)).collect()
        }
    }

    pub fn ndim(&self) -> usize {
        unsafe { sys::mlx_array_ndim(self.handle) }
    }

    // reshape method to change the shape of the array
  pub fn reshape(&self, new_shape: &[i32]) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        
        // MLX C-API usually expects the shape as a pointer and its length
        let status = sys::mlx_reshape(
            &mut res_handle,
            self.handle,
            new_shape.as_ptr(),
            new_shape.len() as usize,
            Self::default_stream()
        );

        if status != 0 || res_handle.ctx.is_null() {
            Err(Error::OperationFailed)
        } else {
            Ok(Array { handle: res_handle })
        }
    }
}

    /// default transpose method (reverses dimensions)
    pub fn transpose(&self, __axes: &[i32]) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            // axes.len() as i32 matches the MLX C-API signature for dimension counts
            let status = sys::mlx_transpose(
                &mut res_handle, 
                self.handle, 
                
                Self::default_stream()
            );
            self.check_status(status, res_handle)
        }
    }

    /// Custom transpose (requires axes, usually mlx_transpose_axes in C)
    pub fn transpose_axes(&self, axes: &[i32]) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        
            let status = sys::mlx_transpose_axes(
                &mut res_handle, 
                self.handle, 
                axes.as_ptr(),
                axes.len() as usize,
                Self::default_stream()
            );
            self.check_status(status, res_handle)
        }
    }

    /// Broadcast array to a new shape
    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_broadcast_to(
                &mut res_handle, 
                self.handle, 
                shape.as_ptr(), 
                shape.len() as usize, 
                Self::default_stream()
            );
            self.check_status(status, res_handle)
        }
    }

    /// if axes None remove all size 1 dims, if axes provided remove only those
    pub fn squeeze(&self, axes: Option<&[i32]>) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            
            let status = match axes {
                // Case 1: No axes provided (Python's a.squeeze())
                None => sys::mlx_squeeze(
                    &mut res_handle, 
                    self.handle, 
                    Self::default_stream()
                ),
                // Case 2: Specific axes provided (Python's a.squeeze(axis=0))
                Some(ax) => sys::mlx_squeeze_axes(
                    &mut res_handle,
                    self.handle,
                    ax.as_ptr(),
                    ax.len() as usize,
                    Self::default_stream(),
                ),
            };

            self.check_status(status, res_handle)
        }
    }


    /// Concatenate multiple arrays along an axis
    pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        let handles: Vec<sys::mlx_array> = arrays.iter().map(|a| a.handle).collect();
        let vec_handle = sys::mlx_vector_array_new_data(
            handles.as_ptr(),
            handles.len() as usize, 
        );
        
        let status = sys::mlx_concatenate_axis(
            &mut res_handle,
            vec_handle,
            axis,
            Self::default_stream()
        );

        sys::mlx_vector_array_free(vec_handle);

        if status != 0 || res_handle.ctx.is_null() {
            Err(Error::OperationFailed)
        } else {
            Ok(Array { handle: res_handle })
        }
    }
}

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

        if status != 0 || res_handle.ctx.is_null() {
            Err(crate::Error::OperationFailed)
        } else {
            Ok(Array { handle: res_handle })
        }
    }
}


    // Helper to reduce boilerplate
    pub (crate)fn check_status(&self, status: i32, handle: sys::mlx_array) -> Result<Array> {
        if status != 0 || handle.ctx.is_null() {
            Err(Error::OperationFailed)
        } else {
            Ok(Array { handle })
        }
    }


    /// Retrieve a single scalar value
  pub fn item<T: Copy + 'static>(&self) -> Result<T> {
        self.eval()?;

        unsafe {
            // HYPOTHESIS A: The function returns the pointer directly
            // It takes 1 argument (the handle) and returns *mut f32
            let data_ptr = sys::mlx_array_data_float32(self.handle);

            if data_ptr.is_null() {
                return Err(Error::OperationFailed);
            }

            // Cast the float pointer to a T pointer
            // WARNING: This assumes the array actually contains floats!
            let typed_ptr = data_ptr as *const T;
            Ok(*typed_ptr)
        }
    }

    /// Retrieve all elements as a Rust Vector
  pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.eval()?;

        unsafe {
          
            let data_ptr = sys::mlx_array_data_float32(self.handle);
           
            let size = sys::mlx_array_size(self.handle); 
            if data_ptr.is_null() {
                return Err(Error::OperationFailed);
            }
            let slice = std::slice::from_raw_parts(data_ptr, size as usize);
            
            Ok(slice.to_vec())
        }
    }


    pub fn dtype(&self) -> crate::Dtype {
        unsafe {
            // This calls the C-API to get the underlying type
            let sys_dtype = sys::mlx_array_dtype(self.handle);
            crate::Dtype::from_sys(sys_dtype)
        }
    }
}

// Memory Management
impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.ctx.is_null() {
                sys::mlx_array_free(self.handle);
            }
        }
    }
}

// Copy Management
impl Clone for Array {
    fn clone(&self) -> Self {
        unsafe {
            let mut new_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_array_set(&mut new_handle, self.handle);
            
            if status != 0 {
                panic!("MLX Clone failed");
            }
            Array { handle: new_handle }
        }
    }
}

// Debug trait for easy printing
impl fmt ::Debug for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Array(handle: {:?})", self.handle)
    }

}

impl std::fmt::Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Remove the ::<f32> part here
        match self.to_vec_f32() { 
            Ok(data) => write!(f, "{:?}", data),
            Err(_) => write!(f, "[Error fetching data]"),
        }
    }
}

