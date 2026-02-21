//! Embedding layers for discrete inputs.

use mlx_derive::ModuleParams;

use crate::{Array, Result, Dtype};
use crate::nn::{Module};
use crate::TreeFlatten;
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

impl TreeFlatten for Embedding {
    fn flatten_state(&self) -> Vec<Array> {
        // Only the learnable weights are tracked by the JIT
        vec![self.weight.clone()]
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        self.weight = flat_arrays.next().unwrap().clone();
    }
}

impl Module for Embedding {
    fn forward(&self, x: &Array) -> Result<Array> {
        self.weight.take(x, 0)
    }
    
   
}