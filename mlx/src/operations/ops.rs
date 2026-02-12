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
    

    /// Reduces the array by taking the mean along a single specified axis.
    pub fn mean_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_mean_axis(
                &mut res_handle,
                self.handle,
                axis,
                keepdims,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    /// Reduces the array by taking the mean along multiple specified axes.
    pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_mean_axes(
                &mut res_handle,
                self.handle,
                axes.as_ptr(),
                axes.len(), // The axes_num argument
                keepdims,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    /// Reduces the array by taking the maximum along a single specified axis.
    pub fn max_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_max_axis(
                &mut res_handle,
                self.handle,
                axis,
                keepdims,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_sum_axis(
                &mut res_handle,
                self.handle,
                axis,
                keepdims,
                Self::default_stream(),
            );
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


    pub fn slice(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_slice(
                &mut res_handle,
                self.handle,
                start.as_ptr(),
                start.len(),
                stop.as_ptr(),
                stop.len(),
                strides.as_ptr(),
                strides.len(),
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    /// Pulls a specific index from the first dimension and removes that dimension.
   
    pub fn index(&self, idx: usize) -> Result<Array> {
        let shape = self.shape()?;
        if shape.is_empty() {
            return Err(crate::Error::OperationFailed);
        }
        let mut start = vec![0; shape.len()];
        let mut stop: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let strides = vec![1; shape.len()];

        start[0] = idx as i32;
        stop[0] = (idx + 1) as i32;

        // Slice it: shape [D0, D1] becomes [1, D1]
        let sliced = self.slice(&start, &stop, &strides)?;
        
        // Squeeze it: shape [1, D1] becomes [D1]
        sliced.squeeze(Some(&[0]))
    }
    pub fn greater_than_scalar(&self, threshold: f32) -> Result<Array> {
        let t = Array::full(&[], threshold, self.dtype())?;
        unsafe {
            let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
            // Standard MLX comparison
            let status = crate::sys::mlx_greater(
                &mut res_handle,
                self.handle,
                t.handle,
                Self::default_stream()
            );
            if status != 0 { return Err(crate::Error::OperationFailed); } Ok(Array { handle: res_handle })
        }
    }




/// Standard Conv Layer Primitives

    pub fn conv1d(&self, w: &Array, s: [i32; 1], p: [i32; 1], d: [i32; 1], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv1d(
                &mut res, 
                self.handle, 
                w.handle, 
                s[0], p[0], d[0], // Unpack the array elements
                g, 
                Self::default_stream()
            );
            if status != 0 { return Err(crate::Error::OperationFailed); } 
            Ok(Array { handle: res })
        }
    }

    pub fn conv2d(&self, w: &Array, s: [i32; 2], p: [i32; 2], d: [i32; 2], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv2d(&mut res, self.handle, w.handle, s[0], s[1], p[0], p[1], d[0], d[1], g, Self::default_stream());
            if status != 0 { return Err(crate::Error::OperationFailed); } Ok(Array { handle: res })
        }
    }

    pub fn conv3d(&self, w: &Array, s: [i32; 3], p: [i32; 3], d: [i32; 3], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv3d(&mut res, self.handle, w.handle, s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2], g, Self::default_stream());
            if status != 0 { return Err(crate::Error::OperationFailed); } Ok(Array { handle: res })
        }
    }

    // Transposed Convolutions (Deconvolution)
    pub fn conv_transpose1d(
        &self, 
        w: &Array, 
        s: [i32; 1], 
        p: [i32; 1], 
        d: [i32; 1], 
        op: [i32; 1], 
        g: i32
                ) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose1d(
                &mut res, 
                self.handle, 
                w.handle, 
                s[0], p[0], d[0], op[0], // Unpack the array elements
                g, 
                Self::default_stream()
            );
            if status != 0 { return Err(crate::Error::OperationFailed); } 
            Ok(Array { handle: res })
        }
    }

    pub fn conv_transpose2d(
        &self, weight: &Array, stride: [i32; 2], padding: [i32; 2], 
        dilation: [i32; 2], out_padding: [i32; 2], groups: i32
    ) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose2d(
                &mut res, self.handle, weight.handle,
                stride[0], stride[1], padding[0], padding[1],
                dilation[0], dilation[1], out_padding[0], out_padding[1],
                groups, Self::default_stream()
            );
            if status != 0 { return Err(crate::Error::OperationFailed); } Ok(Array { handle: res})
        }
    }

    pub fn conv_transpose3d(
        &self, weight: &Array, stride: [i32; 3], padding: [i32; 3], 
        dilation: [i32; 3], out_padding: [i32; 3], groups: i32
    ) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose3d(
                &mut res, self.handle, weight.handle,
                stride[0], stride[1], stride[2], padding[0], padding[1], padding[2],
                dilation[0], dilation[1], dilation[2], out_padding[0], out_padding[1], out_padding[2],
                groups, Self::default_stream()
            );
           if status != 0 { return Err(crate::Error::OperationFailed); } Ok(Array { handle: res })
        }
    }




    /// Specialized max reduction for multiple axes (used by pooling)
    pub fn max_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            // Note: We use the generic mlx_max, NOT mlx_max_axis
            let status = sys::mlx_max_axes(
                &mut res_handle,
                self.handle,
                axes.as_ptr(),
                axes.len(),
                keepdims,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }


    /// Returns the strides of the array as i64.
    pub fn strides(&self) -> Result<Vec<i64>> {
        unsafe {
            let ptr = sys::mlx_array_strides(self.handle);
            let ndim = sys::mlx_array_ndim(self.handle);
            let mut v = Vec::with_capacity(ndim);
            for i in 0..ndim {
                v.push(*ptr.add(i) as i64);
            }
            Ok(v)
        }
    }


    // as strided primitve
    pub fn as_strided(
        &self,
        shape: &[i32],
        strides: &[i64],
        offset: usize,
    ) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_as_strided(
                &mut res_handle,
                self.handle,
                shape.as_ptr(),
                shape.len(),
                strides.as_ptr(),
                strides.len(),
                offset,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    //take primitves for mlx_take_axis
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_take_axis(
                &mut res_handle,
                self.handle,
                indices.handle,
                axis,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }
}








