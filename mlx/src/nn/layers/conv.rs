//! Convolutional and Transposed Convolutional layers.

use crate::{Array, Result, Dtype};
use crate::nn::Module;
use mlx_derive::ModuleParams;
use crate::TreeFlatten;


// --- Helper for Weight Initialization ---
fn kaiming_uniform(shape: &[usize], fan_in: usize, key: &Array) -> Result<Array> {
    let bound = (2.0 / fan_in as f32).sqrt();
    Array::random_uniform(shape, -bound, bound, Dtype::Float32, key)
}

// --- Standard Convolutional Layers ---

macro_rules! define_conv {
    ($name:ident, $dim:expr, $op:ident) => {
        #[derive(ModuleParams)]
        pub struct $name {
            #[param]
            pub weight: Array,
            #[param(optional)]
            pub bias: Option<Array>,
            pub stride: [i32; $dim],
            pub padding: [i32; $dim],
            pub dilation: [i32; $dim],
            pub groups: i32,
        }

        impl $name {
            pub fn new(
                in_channels: usize,
                out_channels: usize,
                kernel_size: [usize; $dim],
                stride: [i32; $dim],
                padding: [i32; $dim],
                dilation: [i32; $dim],
                groups: i32,
                bias: bool,
                key: &Array,
            ) -> Result<Self> {
                let fan_in = in_channels * kernel_size.iter().product::<usize>();
                let (w_key, _b_key) = key.split()?;

                // MLX Weight Layout: [Out, Kernel..., In/Groups]
                let mut w_shape = vec![out_channels];
                w_shape.extend(kernel_size);
                w_shape.push(in_channels / groups as usize);

                let weight = kaiming_uniform(&w_shape, fan_in, &w_key)?;
                let bias = if bias {
                    Some(Array::full(&[out_channels as i32], 0.0, Dtype::Float32)?)
                } else {
                    None
                };

                Ok(Self { weight, bias, stride, padding, dilation, groups })
            }
        }
        impl TreeFlatten for $name {
            fn flatten_state(&self) -> Vec<Array> {
                let mut flat = vec![self.weight.clone()];
                if let Some(b) = &self.bias {
                    flat.push(b.clone());
                }
                flat
            }

            fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
                self.weight = flat_arrays.next().unwrap().clone();
                if self.bias.is_some() {
                    self.bias = Some(flat_arrays.next().unwrap().clone());
                }
            }
        }

        impl Module for $name {
            fn forward(&self, x: &Array) -> Result<Array> {
                // Perform the core convolution operation
                let mut out = x.$op(&self.weight, self.stride, self.padding, self.dilation, self.groups)?;
                
                // Add bias if it exists
                if let Some(ref b) = self.bias {
                    out = out.add(b)?;
                }
                Ok(out)
            }

            
        }
    };
}

define_conv!(Conv1d, 1, conv1d);
define_conv!(Conv2d, 2, conv2d);
define_conv!(Conv3d, 3, conv3d);

// --- Transposed Convolutional Layers ---



macro_rules! define_conv_transpose {
    ($name:ident, $dim:expr, $op:ident) => {
        #[derive(ModuleParams)]
        pub struct $name {
            #[param]
            pub weight: Array,
            #[param(optional)]
            pub bias: Option<Array>,
            pub stride: [i32; $dim],
            pub padding: [i32; $dim],
            pub dilation: [i32; $dim],
            pub output_padding: [i32; $dim],
            pub groups: i32,
        }

        impl $name {
            pub fn new(
                in_channels: usize,
                out_channels: usize,
                kernel_size: [usize; $dim],
                stride: [i32; $dim],
                padding: [i32; $dim],
                output_padding: [i32; $dim],
                dilation: [i32; $dim],
                groups: i32,
                bias: bool,
                key: &Array,
            ) -> Result<Self> {
                let fan_in = in_channels; 
                let (w_key, _b_key) = key.split()?;

                // Transposed Layout: [In, Kernel..., Out/Groups]
                let mut w_shape = vec![in_channels];
                w_shape.extend(kernel_size);
                w_shape.push(out_channels / groups as usize);

                let weight = kaiming_uniform(&w_shape, fan_in, &w_key)?;
                let bias = if bias {
                    Some(Array::full(&[out_channels as i32], 0.0, Dtype::Float32)?)
                } else {
                    None
                };

                Ok(Self { weight, bias, stride, padding, dilation, output_padding, groups })
            }
        }
        impl TreeFlatten for $name {
            fn flatten_state(&self) -> Vec<Array> {
                let mut flat = vec![self.weight.clone()];
                if let Some(b) = &self.bias {
                    flat.push(b.clone());
                }
                flat
            }

            fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
                self.weight = flat_arrays.next().unwrap().clone();
                if self.bias.is_some() {
                    self.bias = Some(flat_arrays.next().unwrap().clone());
                }
            }
        }

        impl Module for $name {
            fn forward(&self, x: &Array) -> Result<Array> {
                let mut out = x.$op(
                    &self.weight, 
                    self.stride, 
                    self.padding, 
                    self.dilation, 
                    self.output_padding, 
                    self.groups
                )?;
                if let Some(ref b) = self.bias {
                    out = out.add(b)?;
                }
                Ok(out)
            }

            
        }
    };
}

define_conv_transpose!(ConvTranspose1d, 1, conv_transpose1d);
define_conv_transpose!(ConvTranspose2d, 2, conv_transpose2d);
define_conv_transpose!(ConvTranspose3d, 3, conv_transpose3d);