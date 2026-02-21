
//! Pooling layers supporting overlapping windows via as_strided.

use crate::{Array, Result};
use crate::nn::{Module, ModuleParams};
use crate::TreeFlatten;


/// 2D Max Pooling layer.
pub struct MaxPool2d {
    pub kernel_size: [i32; 2],
    pub stride: [i32; 2],
}

impl MaxPool2d {
    pub fn new(kernel_size: [i32; 2], stride: [i32; 2]) -> Self {
        Self { kernel_size, stride }
    }
}
impl ModuleParams for MaxPool2d {}
impl TreeFlatten for MaxPool2d {
    fn flatten_state(&self) -> Vec<Array> {
        Vec::new()
    }

    fn unflatten_state(&mut self, _flat_arrays: &mut std::slice::Iter<'_, Array>) {
        
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?; // [N, H, W, C]
        let current_strides = x.strides()?; // [i64; 4]

        let (kh, kw) = (self.kernel_size[0] as usize, self.kernel_size[1] as usize);
        let (sh, sw) = (self.stride[0] as usize, self.stride[1] as usize);

        let n = shape[0];
        let h_out = (shape[1] - kh) / sh + 1;
        let w_out = (shape[2] - kw) / sw + 1;
        let c = shape[3];

        // 6D View: [Batch, H_out, W_out, Window_H, Window_W, Channels]
        let new_shape = [
            n as i32, h_out as i32, w_out as i32, 
            kh as i32, kw as i32, c as i32
        ];

        // Strides calculation:
        // Jumping 'sh' rows in the original H dimension means skipping (sh * H_stride) elements.
        let new_strides = [
            current_strides[0],             // Batch
            current_strides[1] * sh as i64, // Vertically slide window
            current_strides[2] * sw as i64, // Horizontally slide window
            current_strides[1],             // Inside window: move down
            current_strides[2],             // Inside window: move right
            current_strides[3],             // Channel
        ];

        let windows = x.as_strided(&new_shape, &new_strides, 0)?;

        // Reduce across the Kh and Kw dimensions (axes 3 and 4)
        windows.max_axes(&[3, 4], false)
    }
}

/// 2D Average Pooling layer.
pub struct AvgPool2d {
    pub kernel_size: [i32; 2],
    pub stride: [i32; 2],
}

impl AvgPool2d {
    pub fn new(kernel_size: [i32; 2], stride: [i32; 2]) -> Self {
        Self { kernel_size, stride }
    }
}
impl ModuleParams for AvgPool2d {}
impl TreeFlatten for AvgPool2d {
    fn flatten_state(&self) -> Vec<Array> {
        Vec::new()
    }

    fn unflatten_state(&mut self, _flat_arrays: &mut std::slice::Iter<'_, Array>) {
        
    }
}


impl Module for AvgPool2d {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?;
        let current_strides = x.strides()?;

        let (kh, kw) = (self.kernel_size[0] as usize, self.kernel_size[1] as usize);
        let (sh, sw) = (self.stride[0] as usize, self.stride[1] as usize);

        let h_out = (shape[1] - kh) / sh + 1;
        let w_out = (shape[2] - kw) / sw + 1;

        let new_shape = [
            shape[0] as i32, h_out as i32, w_out as i32, 
            kh as i32, kw as i32, shape[3] as i32
        ];

        let new_strides = [
            current_strides[0],
            current_strides[1] * sh as i64,
            current_strides[2] * sw as i64,
            current_strides[1],
            current_strides[2],
            current_strides[3],
        ];

        let windows = x.as_strided(&new_shape, &new_strides, 0)?;
        windows.mean_axes(&[3, 4], false)
    }


}