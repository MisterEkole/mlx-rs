use crate::{Array, Result};
use crate::nn::{Module, ModuleParams};
use crate::nn::layers::linear::Linear;
use crate::nn::transformers::multi_head_attention::MultiHeadAttention;
use crate::nn::layers::normalization::LayerNorm;
use crate::nn::layers::activations::relu;
use crate::nn::layers::dropouts::Dropout;
use crate::tree::TreeFlatten; // <-- 1. Import TreeFlatten
use mlx_derive::ModuleParams;

#[derive(ModuleParams)]
pub struct TransformerEncoderLayer {
    #[module]
    pub self_attn: MultiHeadAttention,
    #[module]
    pub norm1: LayerNorm,
    #[module]
    pub norm2: LayerNorm,
    #[module]
    pub ff_linear1: Linear,
    #[module]
    pub ff_linear2: Linear,
    #[module]
    pub dropout: Dropout,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout_p: f32, key: &Array) -> Result<Self>{
        let (k_1, rest)= key.split()?;
        let (k_2, rest)= rest.split()?;
        let (k_3, k_4)= rest.split()?;

        Ok(Self{
            self_attn: MultiHeadAttention::new(d_model, n_heads, true, &k_1)?,
            norm1: LayerNorm::new(d_model, 1e-5)?,
            norm2: LayerNorm::new(d_model, 1e-5)?,
            ff_linear1: Linear::new(d_model, d_ff, true, &k_2)?,
            ff_linear2: Linear::new(d_ff, d_model, true, &k_3)?,
            dropout: Dropout::new(dropout_p, k_4)?,
        })
    }

    pub fn forward_with_mask(&self, x: &Array, mask: Option<&Array>) -> Result<Array>{
        // Self-attention block 
        let normed = self.norm1.forward(x)?;
        let attn_out = self.self_attn.forward_qkv(&normed, &normed, &normed, mask)?;
        let x = x.add(&self.dropout.forward(&attn_out)?)?;

        // Feed-forward block
        let normed = self.norm2.forward(&x)?;
        let ff_out = relu(&self.ff_linear1.forward(&normed)?)?;
        let ff_out = self.ff_linear2.forward(&ff_out)?;

        Ok(x.add(&self.dropout.forward(&ff_out)?)?)
    }
}

// 2. NEW: Implement TreeFlatten for TransformerEncoderLayer
impl TreeFlatten for TransformerEncoderLayer {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = Vec::new();
        flat.extend(self.self_attn.flatten_state());
        flat.extend(self.norm1.flatten_state());
        flat.extend(self.norm2.flatten_state());
        flat.extend(self.ff_linear1.flatten_state());
        flat.extend(self.ff_linear2.flatten_state());
        flat.extend(self.dropout.flatten_state());
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.self_attn.unflatten_state(flat_arrays);
        self.norm1.unflatten_state(flat_arrays);
        self.norm2.unflatten_state(flat_arrays);
        self.ff_linear1.unflatten_state(flat_arrays);
        self.ff_linear2.unflatten_state(flat_arrays);
        self.dropout.unflatten_state(flat_arrays);
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Array) -> Result<Array> {
        self.forward_with_mask(input, None)
    }
}

/// Stacking N encoder layers to form the full Transformer Encoder.

pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    pub fn new(num_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, dropout_p: f32, key: &Array) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let mut current_key = key.clone();

        for _ in 0..num_layers {
            let (layer_key, rest) = current_key.split()?;
            layers.push(TransformerEncoderLayer::new(d_model, n_heads, d_ff, dropout_p, &layer_key)?);
            current_key = rest;
        }

        Ok(Self { layers })
    }
}


impl TreeFlatten for TransformerEncoder {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = Vec::new();
        for layer in &self.layers {
            flat.extend(layer.flatten_state());
        }
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        for layer in &mut self.layers {
            layer.unflatten_state(flat_arrays);
        }
    }
}

// Manual ModuleParams for Vec<T> pattern, since #[module] doesn't work directly on Vec<T> where T: Module.
impl ModuleParams for TransformerEncoder {
    fn parameters(&self) -> Vec<&Array> {
        self.layers.iter().flat_map(|layer| layer.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        self.layers.iter_mut().flat_map(|layer| layer.parameters_mut()).collect()
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        let mut offset = 0;
        for layer in &mut self.layers {
            let n = layer.parameters().len();
            if offset + n <= new_params.len() {
                layer.update_parameters(&new_params[offset..offset + n]);
                offset += n;
            }
        }
    }

    fn train(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.train(training);
        }
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}