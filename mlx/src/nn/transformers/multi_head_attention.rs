use crate::{Array, Result};
use crate::nn::{Module};
use crate::nn::layers::linear::Linear;
use crate::nn::transformers::scaled_dot_product::scaled_dot_product_attention;
use mlx_derive::ModuleParams;



#[derive(ModuleParams)]
pub struct MultiHeadAttention{
    pub num_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,

    #[module]
    pub q_proj: Linear,
    #[module]
    pub k_proj: Linear,
    #[module]
    pub v_proj: Linear,
    #[module]
    pub out_proj: Linear,


}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, bias: bool, key: &Array) -> Result<Self> {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;

        let (k_1, rest) = key.split()?;
        let (k_2, rest) = rest.split()?;
        let (k_3, k_4) = rest.split()?;

        Ok(Self {
            num_heads,
            head_dim,
            embed_dim,
            q_proj: Linear::new(embed_dim, embed_dim, bias, &k_1)?,
            k_proj: Linear::new(embed_dim, embed_dim, bias, &k_2)?,
            v_proj: Linear::new(embed_dim, embed_dim, bias, &k_3)?,
            out_proj: Linear::new(embed_dim, embed_dim, bias, &k_4)?,
        })
    }

    pub fn forward_qkv(&self, query: &Array, key: &Array, value: &Array, mask: Option<&Array>) -> Result<Array> {
        let batch_size = query.shape()?[0] as i32;
        let q_seq_len = query.shape()?[1] as i32;
        let k_seq_len = key.shape()?[1] as i32;

        let n_heads = self.num_heads as i32;
        let h_dim = self.head_dim as i32;
        let e_dim = self.embed_dim as i32;

        let q = self.q_proj.forward(query)?
            .reshape(&[batch_size, q_seq_len, n_heads, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self.k_proj.forward(key)?
            .reshape(&[batch_size, k_seq_len, n_heads, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self.v_proj.forward(value)?
            .reshape(&[batch_size, k_seq_len, n_heads, h_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let attn_output = scaled_dot_product_attention(&q, &k, &v, mask)?;
        let attn_output = attn_output
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[batch_size, q_seq_len, e_dim])?;

        self.out_proj.forward(&attn_output)
    }
}


impl Module for MultiHeadAttention{
    fn forward(&self, x: &Array) -> Result<Array>{
        self.forward_qkv(x, x, x, None)
    }

}




        
