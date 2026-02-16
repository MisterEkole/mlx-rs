use crate::{sys, Array, Result};
use std::ptr;

impl Array {
    // =========================================================================
    // Unary Operations (Element-wise, Single Input)
    // =========================================================================

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

    pub fn abs(&self) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_abs(&mut res_handle, self.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }


    pub fn sign(&self) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_sign(&mut res_handle, self.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    // =========================================================================
    // Binary & Conditional Operations (Element-wise, Comparison)
    // =========================================================================

    /// Select elements from 'on_true' or 'on_false' based on the condition (self)
    pub fn where_op(&self, on_true: &Array, on_false: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_where(
                &mut res_handle,
                self.handle,
                on_true.handle,
                on_false.handle,
                Self::default_stream(),
            );
            self.check_status(status, res_handle)
        }
    }

    pub fn less_than(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_less(&mut res_handle, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    /// Returns element-wise maximum of two arrays
    pub fn maximum(&self, other: &Array) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_maximum(&mut res_handle, self.handle, other.handle, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn greater_than_scalar(&self, threshold: f32) -> Result<Array> {
        let t = Array::full(&[], threshold, self.dtype())?;
        unsafe {
            let mut res_handle = crate::sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = crate::sys::mlx_greater(
                &mut res_handle,
                self.handle,
                t.handle,
                Self::default_stream()
            );
            self.check_status(status, res_handle)
        }
    }

    // =========================================================================
    // Reduction Operations (Collapsing Dimensions)
    // =========================================================================

    pub fn mean_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_mean_axis(&mut res_handle, self.handle, axis, keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn mean_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_mean_axes(&mut res_handle, self.handle, axes.as_ptr(), axes.len(), keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn max_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_max_axis(&mut res_handle, self.handle, axis, keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn max_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_max_axes(&mut res_handle, self.handle, axes.as_ptr(), axes.len(), keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn sum_axis(&self, axis: i32, keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_sum_axis(&mut res_handle, self.handle, axis, keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn sum_axes(&self, axes: &[i32], keepdims: bool) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_sum_axes(&mut res_handle, self.handle, axes.as_ptr(), axes.len(), keepdims, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    // Aliases for common reductions
    pub fn mean(&self, axis: i32, keepdims: bool) -> Result<Array> { self.mean_axis(axis, keepdims) }
    pub fn max(&self, axis: i32, keepdims: bool) -> Result<Array> { self.max_axis(axis, keepdims) }

    // =========================================================================
    // Shape, Slicing & Indexing Operations
    // =========================================================================

    pub fn slice(&self, start: &[i32], stop: &[i32], strides: &[i32]) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_slice(&mut res_handle, self.handle, start.as_ptr(), start.len(), stop.as_ptr(), stop.len(), strides.as_ptr(), strides.len(), Self::default_stream());
            self.check_status(status, res_handle)
        }
    }
    
    pub fn slice_axis(&self, axis: i32, start: i32, stop: i32) -> Result<Array> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let ax = if axis < 0 { 
            (ndim as i32 + axis) as usize 
        } else { 
            axis as usize 
        };
        let mut starts = vec![0; ndim];
        let mut stops: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let strides = vec![1; ndim];
        starts[ax] = start;
        stops[ax] = stop;
        self.slice(&starts, &stops, &strides)
    }

    pub fn slice_axes(&self, axes: &[i32], starts_in: &[i32], stops_in: &[i32]) -> Result<Array> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let mut starts = vec![0; ndim];
        let mut stops: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let strides = vec![1; ndim];

        // Apply the slices to all requested axes
        for i in 0..axes.len() {
            let ax = if axes[i] < 0 { 
                (ndim as i32 + axes[i]) as usize 
            } else { 
                axes[i] as usize 
            };
            
            starts[ax] = starts_in[i];
            stops[ax] = stops_in[i];
        }

        self.slice(&starts, &stops, &strides)
    }

    pub fn index(&self, idx: usize) -> Result<Array> {
        let shape = self.shape()?;
        if shape.is_empty() { return Err(crate::Error::OperationFailed("index failed: empty array".into())); }
        let mut start = vec![0; shape.len()];
        let mut stop: Vec<i32> = shape.iter().map(|&s| s as i32).collect();
        let strides = vec![1; shape.len()];
        start[0] = idx as i32;
        stop[0] = (idx + 1) as i32;
        let sliced = self.slice(&start, &stop, &strides)?;
        sliced.squeeze(Some(&[0]))
    }

    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_take_axis(&mut res_handle, self.handle, indices.handle, axis, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn as_strided(&self, shape: &[i32], strides: &[i64], offset: usize) -> Result<Array> {
        unsafe {
            let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_as_strided(&mut res_handle, self.handle, shape.as_ptr(), shape.len(), strides.as_ptr(), strides.len(), offset, Self::default_stream());
            self.check_status(status, res_handle)
        }
    }

    pub fn strides(&self) -> Result<Vec<i64>> {
        unsafe {
            let ptr = sys::mlx_array_strides(self.handle);
            let ndim = sys::mlx_array_ndim(self.handle);
            let mut v = Vec::with_capacity(ndim);
            for i in 0..ndim { v.push(*ptr.add(i) as i64); }
            Ok(v)
        }
    }

    // =========================================================================
    // Neural Network Primitives (Convolutions)
    // =========================================================================

    pub fn conv1d(&self, w: &Array, s: [i32; 1], p: [i32; 1], d: [i32; 1], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv1d(&mut res, self.handle, w.handle, s[0], p[0], d[0], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn conv2d(&self, w: &Array, s: [i32; 2], p: [i32; 2], d: [i32; 2], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv2d(&mut res, self.handle, w.handle, s[0], s[1], p[0], p[1], d[0], d[1], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn conv3d(&self, w: &Array, s: [i32; 3], p: [i32; 3], d: [i32; 3], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv3d(&mut res, self.handle, w.handle, s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn conv_transpose1d(&self, w: &Array, s: [i32; 1], p: [i32; 1], d: [i32; 1], op: [i32; 1], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose1d(&mut res, self.handle, w.handle, s[0], p[0], d[0], op[0], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn conv_transpose2d(&self, w: &Array, s: [i32; 2], p: [i32; 2], d: [i32; 2], op: [i32; 2], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose2d(&mut res, self.handle, w.handle, s[0], s[1], p[0], p[1], d[0], d[1], op[0], op[1], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    pub fn conv_transpose3d(&self, w: &Array, s: [i32; 3], p: [i32; 3], d: [i32; 3], op: [i32; 3], g: i32) -> Result<Array> {
        unsafe {
            let mut res = sys::mlx_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_conv_transpose3d(&mut res, self.handle, w.handle, s[0], s[1], s[2], p[0], p[1], p[2], d[0], d[1], d[2], op[0], op[1], op[2], g, Self::default_stream());
            self.check_status(status, res)
        }
    }

    // =========================================================================
    // High-Level Utilities (Safe Rust, Composite Ops)
    // =========================================================================

    pub fn flatten(&self) -> Result<Array> {
        let shape = self.shape()?;
        let total_elements: usize = shape.iter().product(); 
        self.reshape(&[total_elements as i32])
    }

    pub fn expand_dims(&self, axis: i32) -> Result<Array> {
        let mut new_shape = self.shape()?;
        let len = new_shape.len() as i32;
        let pos = if axis < 0 { (len + axis + 1).max(0) as usize } else { axis as usize };
        new_shape.insert(pos, 1);
        let new_shape_i32: Vec<i32> = new_shape.iter().map(|&x| x as i32).collect();
        self.reshape(&new_shape_i32)
    }

    pub fn square(&self) -> Result<Array> {
        self.multiply(self)
    }

    pub fn var(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let mu = self.mean(axis, keepdims)?;
        let diff = self.subtract(&mu)?; 
        let squared_diff = diff.square()?;
        squared_diff.mean(axis, keepdims)
    }

    pub fn std(&self, axis: i32, keepdims: bool) -> Result<Array> {
        let v = self.var(axis, keepdims)?;
        v.sqrt() 
    }

    pub fn zeros_like(&self) -> Result<Array> {
        let shape = self.shape()?;
        let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        Self::full(&shape_i32, 0.0, self.dtype())
    }

    pub fn ones_like(&self) -> Result<Array> {
        let shape = self.shape()?;
        let shape_i32: Vec<i32> = shape.iter().map(|&x| x as i32).collect();
        Self::full(&shape_i32, 1.0, self.dtype())
    }
}