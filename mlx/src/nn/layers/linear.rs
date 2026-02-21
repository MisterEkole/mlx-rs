// mlx/src/nn/layers/linear.rs

// The derive macro auto generates: parameters(), parameters_mut(), update_parameters(), train() based on field annotations.


use crate::{Array, Result, Dtype};
use crate::nn::Module;
use mlx_derive::ModuleParams;
use crate::TreeFlatten;

/// A linear (fully connected) layer.
/// 
/// Applies a linear transformation to the incoming data: y = xA^T + b


impl TreeFlatten for Linear {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = vec![self.weight.clone()];
        if let Some(bias) = &self.bias {
            flat.push(bias.clone());
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
#[derive(ModuleParams)]


pub struct Linear {
    #[param]
    pub weight: Array,
    #[param(optional)]
    pub bias: Option<Array>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Creates a new Linear layer.
    /// 
    /// # Arguments
    /// * `in_features` - Size of each input sample.
    /// * `out_features` - Size of each output sample.
    /// * `bias` - If set to false, the layer will not learn an additive bias.
    /// * `key` - The PRNG key for weight initialization.
    pub fn new(in_features: usize, out_features: usize, bias: bool, key: &Array) -> Result<Self> {
        // PyTorch initialization: U(-sqrt(k), sqrt(k)) where k = 1/in_features
        let k = 1.0 / (in_features as f32);
        let bound = k.sqrt();
        
        // Split the key: 0 for weight, 1 for bias (if needed)
        let (w_key, b_key) = key.split()?;
    
        // Weight shape: [out_features, in_features]
        let weight = Array::random_uniform(
            &[out_features, in_features],
            -bound,
            bound,
            Dtype::Float32,
            &w_key,
        )?;
        
        let bias_array = if bias {
            Some(Array::random_uniform(
                &[out_features],
                -bound,
                bound,
                Dtype::Float32,
                &b_key,
            )?)
        } else {
            None
        };
        
        Ok(Self {
            weight,
            bias: bias_array,
            in_features,
            out_features,
        })
    }
    
    /// Creates a Linear layer from existing weights.
    pub fn from_weights(weight: Array, bias: Option<Array>) -> Result<Self> {
        let shape = weight.shape()?;
        if shape.len() != 2 {
            return Err(crate::Error::InvalidShape(
                format!("Weight must be 2D, got shape {:?}", shape)
            ));
        }
        
        Ok(Self {
            out_features: shape[0],
            in_features: shape[1],
            weight,
            bias,
        })
    }
}

impl Module for Linear {
    /// Computes y = x @ W^T + b
    fn forward(&self, x: &Array) -> Result<Array> {
        // In MLX, we typically store weights as [Out, In]
        // To compute the dot product, we transpose the weights to [In, Out]
        // result: [Batch, In] @ [In, Out] -> [Batch, Out]
        let weight_t= self.weight.transpose(&[])?;
        //println!("X shape: {:?}, Weight shape (T): {:?}", x.shape()?, self.weight.transpose(&[1, 0])?.shape()?);
        let mut out = x.matmul(&weight_t)?;
        
        if let Some(ref b) = self.bias {
            out = out.add(b)?;
        }
        
        Ok(out)
    }
 
    
}