//! Dropout regularization layers.
use crate::{Array, Result, Dtype};
use crate::nn::{Module, ModuleParams};
use std::cell::RefCell;
use crate::TreeFlatten;


pub struct Dropout {
    pub p: f32,
    training: bool,
    // use RefCell because forward takes &self, but we need to update the key
    key: RefCell<Array>,
}

impl Dropout {
    pub fn new(p: f32, key: Array) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(crate::Error::InvalidShape(
                format!("Dropout probability must be in [0, 1], got {}", p)
            ));
        }
        
        Ok(Dropout {
            p,
            training: true,
            key: RefCell::new(key),
        })
    }
}
// --- NEW: TreeFlatten Implementation ---
// We MUST flatten the PRNG key so the JIT compiler updates the random state
// for every batch, instead of reusing the same dropout mask forever!
impl TreeFlatten for Dropout {
    fn flatten_state(&self) -> Vec<Array> {
        vec![self.key.borrow().clone()]
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        // We have &mut self here, so we can safely replace the inner key
        self.key.replace(flat_arrays.next().unwrap().clone());
    }
}

impl ModuleParams for Dropout {
    fn train(&mut self, training: bool) {
        self.training = training;
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Array) -> Result<Array> {
        if !self.training || self.p == 0.0 {
            return Ok(x.clone());
        }

        //Get and update the key
        let (current_key, next_key) = {
            let k = self.key.borrow();
            k.split()?
        };
        self.key.replace(next_key);

        // Generate random mask U(0, 1)
        let shape = x.shape()?;
        let mask_probs = Array::random_uniform(&shape, 0.0, 1.0, Dtype::Float32, &current_key)?;
        let scale = 1.0 / (1.0 - self.p);
        // apply mask
        let mask = mask_probs.greater_than_scalar(self.p)?; 
        x.multiply(&mask)?.multiply_scalar(scale)
    }
    
}