use crate::{Array, Result};
use crate::nn::layers::activations::Softmax;
use crate::nn::Module;

pub fn scaled_dot_product_attention(
    q: &Array,
    k: &Array,
    v: &Array,
    mask: Option<&Array>,
) -> Result<Array> {
    let shape = q.shape()?;
    let head_dim = *shape.last().unwrap_or(&1) as f32;
    let scale = head_dim.sqrt();

    let k_t = k.transpose_axes(&[0, 1, 3, 2])?;

    let mut scores = q.matmul(&k_t)?.divide_scalar(scale)?;
    if let Some(m) = mask {
        scores = scores.add(m)?;
    }


    let weights = Softmax::new(-1).forward(&scores)?;
    weights.matmul(v)
}