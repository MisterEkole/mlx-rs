/// Module trait and related utilities for building neural network layers and models.

use crate::{Array, Result};
pub trait Module {
    fn forward(&self, x: &Array) -> Result<Array>;

    fn parameters(&self) -> Vec<&Array> { Vec::new() }
    
    fn parameters_mut(&mut self) -> Vec<&mut Array> { Vec::new() }

    fn parameters_owned(&self) -> Vec<Array> {
        self.parameters().into_iter().cloned().collect()
    }

    // The Bridge: This must be overridden by structs that have parameters
    fn update_parameters(&mut self, _new_params: &[Array]) {
        // Default: do nothing (for layers like ReLU)
    }

    fn train(&mut self, _training: bool) {}
}