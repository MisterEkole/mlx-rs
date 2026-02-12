//! Convolutional and Transposed Convolutional layers.

use crate::{Array, Result, Dtype};
use crate::nn::Module;

// --- Helper for Weight Initialization ---
fn kaiming_uniform(shape: &[usize], fan_in: usize, key: &Array) -> Result<Array> {
    let bound = (2.0 / fan_in as f32).sqrt();
    Array::random_uniform(shape, -bound, bound, Dtype::Float32, key)
}

// --- Standard Convolutional Layers ---

macro_rules! define_conv {
    ($name:ident, $dim:expr, $op:ident) => {
        pub struct $name {
            pub weight: Array,
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
                let (w_key, __b_key) = key.split()?;

                // Layout: [Out, Kernel..., In/Groups]
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

        impl Module for $name {
            fn forward(&self, x: &Array) -> Result<Array> {
                let mut out = x.$op(&self.weight, self.stride, self.padding, self.dilation, self.groups)?;
                if let Some(ref b) = self.bias {
                    out = out.add(b)?;
                }
                Ok(out)
            }

            fn parameters(&self) -> Vec<&Array> {
                let mut p = vec![&self.weight];
                if let Some(ref b) = self.bias { p.push(b); }
                p
            }
            
            fn train(&mut self, _: bool) {}
        }
    };
}

define_conv!(Conv1d, 1, conv1d);
define_conv!(Conv2d, 2, conv2d);
define_conv!(Conv3d, 3, conv3d);

// --- Transposed Convolutional Layers ---



macro_rules! define_conv_transpose {
    ($name:ident, $dim:expr, $op:ident) => {
        pub struct $name {
            pub weight: Array,
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
                let fan_in = in_channels; // Fan-in for transposed is based on input channels
                let (w_key, __b_key) = key.split()?;

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

            fn parameters(&self) -> Vec<&Array> {
                let mut p = vec![&self.weight];
                if let Some(ref b) = self.bias { p.push(b); }
                p
            }

            fn train(&mut self, _: bool) {}
        }
    };
}

define_conv_transpose!(ConvTranspose1d, 1, conv_transpose1d);
define_conv_transpose!(ConvTranspose2d, 2, conv_transpose2d);
define_conv_transpose!(ConvTranspose3d, 3, conv_transpose3d);