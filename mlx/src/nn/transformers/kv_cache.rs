// =============================================================================
// mlx/src/nn/transformers/kv_cache.rs
//
// Key-Value cache for efficient autoregressive generation.
// =============================================================================

use crate::{Array, Result};

pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    /// Current sequence length in the cache
    offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }
    pub fn offset(&self) -> usize {
        self.offset
    }


    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

 
    pub fn update(
        &mut self,
        new_keys: &Array,
        new_values: &Array,
    ) -> Result<(Array, Array)> {
        let (k, v) = match (&self.keys, &self.values) {
            (Some(prev_k), Some(prev_v)) => {
                // Concatenate along sequence dimension (axis 2)
                let k = Array::concatenate(&[prev_k, new_keys], 2)?;
                let v = Array::concatenate(&[prev_v, new_values], 2)?;
                (k, v)
            }
            _ => {
            
                (new_keys.clone(), new_values.clone())
            }
        };

        self.keys = Some(k.clone());
        self.values = Some(v.clone());

    
        let new_seq_len = new_keys.shape()?[2] as usize;
        self.offset += new_seq_len;

        Ok((k, v))
    }

 
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
    }
}

/// A collection of KV caches, one per transformer layer.

pub struct KVCacheCollection {
    caches: Vec<KVCache>,
}

impl KVCacheCollection {
    pub fn new(n_layers: usize) -> Self {
        let caches = (0..n_layers).map(|_| KVCache::new()).collect();
        Self { caches }
    }

    pub fn get_mut(&mut self, layer_idx: usize) -> &mut KVCache {
        &mut self.caches[layer_idx]
    }

    pub fn get(&self, layer_idx: usize) -> &KVCache {
        &self.caches[layer_idx]
    }

    pub fn offset(&self) -> usize {
        self.caches.first().map_or(0, |c| c.offset())
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn len(&self) -> usize {
        self.caches.len()
    }
}