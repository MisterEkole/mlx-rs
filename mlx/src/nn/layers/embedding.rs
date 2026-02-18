//! Embedding layers for discrete inputs.

use mlx_derive::ModuleParams;

use crate::{Array, Result, Dtype};
use crate::nn::{Module};

#[derive(ModuleParams)]
pub struct Embedding {
    pub weight: Array,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub padding_idx: Option<usize>,
}

impl Embedding {
    pub fn new(
        num_embeddings: usize, 
        embedding_dim: usize, 
        padding_idx: Option<usize>,
        key: &Array
    ) -> Result<Self> {
        let bound = 0.0346;

        let weight = Array::random_uniform(
            &[num_embeddings, embedding_dim],
            -bound,
            bound,
            Dtype::Float32,
            key,
        )?;
        
        Ok(Embedding {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx,
        })
    }
}



impl Module for Embedding {
    fn forward(&self, x: &Array) -> Result<Array> {
        self.weight.take(x, 0)
    }
    
   
}