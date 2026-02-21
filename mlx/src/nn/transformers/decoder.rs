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
pub struct TransformerDecoderLayer {
    #[module]
    pub self_attn: MultiHeadAttention,
    #[module(optional)]
    pub cross_attn: Option<MultiHeadAttention>,
    #[module]
    pub norm1: LayerNorm,
    #[module(optional)]
    pub norm2: Option<LayerNorm>,
    #[module]
    pub norm3: LayerNorm,
    #[module]
    pub ff_linear1: Linear,
    #[module]
    pub ff_linear2: Linear,
    #[module]
    pub dropout: Dropout,
}

impl TransformerDecoderLayer {
    pub fn new(d_model: usize, n_heads: usize, d_ff: usize, dropout_p: f32, key: &Array, use_cross_attn: bool) -> Result<Self> {
        let (k_1, rest) = key.split()?;
        let (k_2, rest) = rest.split()?;
        let (k_3, rest) = rest.split()?;
        let (k_4, k_5) = rest.split()?;

        Ok(Self {
            self_attn: MultiHeadAttention::new(d_model, n_heads, true, &k_1)?,
            cross_attn: if use_cross_attn {
                Some(MultiHeadAttention::new(d_model, n_heads, true, &k_2)?)
            } else { None },
            norm1: LayerNorm::new(d_model, 1e-5)?,
            norm2: if use_cross_attn {
                Some(LayerNorm::new(d_model, 1e-5)?)
            } else { None },
            norm3: LayerNorm::new(d_model, 1e-5)?,
            ff_linear1: Linear::new(d_model, d_ff, true, &k_3)?,
            ff_linear2: Linear::new(d_ff, d_model, true, &k_4)?,
            dropout: Dropout::new(dropout_p, k_5)?,
        })
    }

    pub fn forward_full(
        &self,
        x: &Array,
        enc_output: Option<&Array>,
        self_attn_mask: Option<&Array>,
        cross_attn_mask: Option<&Array>,
    ) -> Result<Array> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.self_attn.forward_qkv(&normed, &normed, &normed, self_attn_mask)?;
        let mut out = x.add(&self.dropout.forward(&attn_out)?)?;

        if let (Some(cross_attn), Some(norm2), Some(enc_out)) = (&self.cross_attn, &self.norm2, enc_output) {
            let normed = norm2.forward(&out)?;
            let attn_out = cross_attn.forward_qkv(&normed, enc_out, enc_out, cross_attn_mask)?;
            out = out.add(&self.dropout.forward(&attn_out)?)?;
        }

        let normed = self.norm3.forward(&out)?;
        let ff_out = relu(&self.ff_linear1.forward(&normed)?)?;
        let ff_out = self.ff_linear2.forward(&ff_out)?;
        Ok(out.add(&self.dropout.forward(&ff_out)?)?)
    }
}

// 2. NEW: Implement TreeFlatten for TransformerDecoderLayer
impl TreeFlatten for TransformerDecoderLayer {
    fn flatten_state(&self) -> Vec<Array> {
        let mut flat = Vec::new();
        flat.extend(self.self_attn.flatten_state());
        
        if let Some(ca) = &self.cross_attn {
            flat.extend(ca.flatten_state());
        }
        
        flat.extend(self.norm1.flatten_state());
        
        if let Some(n2) = &self.norm2 {
            flat.extend(n2.flatten_state());
        }
        
        flat.extend(self.norm3.flatten_state());
        flat.extend(self.ff_linear1.flatten_state());
        flat.extend(self.ff_linear2.flatten_state());
        flat.extend(self.dropout.flatten_state());
        flat
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.self_attn.unflatten_state(flat_arrays);
        
        if let Some(ca) = &mut self.cross_attn {
            ca.unflatten_state(flat_arrays);
        }
        
        self.norm1.unflatten_state(flat_arrays);
        
        if let Some(n2) = &mut self.norm2 {
            n2.unflatten_state(flat_arrays);
        }
        
        self.norm3.unflatten_state(flat_arrays);
        self.ff_linear1.unflatten_state(flat_arrays);
        self.ff_linear2.unflatten_state(flat_arrays);
        self.dropout.unflatten_state(flat_arrays);
    }
}

impl Module for TransformerDecoderLayer {
    fn forward(&self, input: &Array) -> Result<Array> {
        self.forward_full(input, None, None, None)
    }
}

// --- Stack N Decoder Layers --- Manual ModuleParams for Vec<T>.

pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
}

impl TransformerDecoder {
    pub fn new(num_layers: usize, d_model: usize, n_heads: usize, d_ff: usize, dropout_p: f32, key: &Array, use_cross_attn: bool) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        let mut current_key = key.clone();

        for _ in 0..num_layers {
            let (k_layer, rest) = current_key.split()?;
            layers.push(TransformerDecoderLayer::new(d_model, n_heads, d_ff, dropout_p, &k_layer, use_cross_attn)?);
            current_key = rest;
        }

        Ok(Self { layers })
    }

    pub fn forward_full(
        &self,
        x: &Array,
        enc_output: Option<&Array>,
        self_attn_mask: Option<&Array>,
        cross_attn_mask: Option<&Array>,
    ) -> Result<Array> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward_full(&out, enc_output, self_attn_mask, cross_attn_mask)?;
        }
        Ok(out)
    }
}

// 3. NEW: Implement TreeFlatten for the full Decoder stack
impl TreeFlatten for TransformerDecoder {
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

impl ModuleParams for TransformerDecoder {
    fn parameters(&self) -> Vec<&Array> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        self.layers.iter_mut().flat_map(|l| l.parameters_mut()).collect()
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

impl Module for TransformerDecoder {
    fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_full(x, None, None, None)
    }
}