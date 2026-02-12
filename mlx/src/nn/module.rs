/// Module trait and related utilities for building neural network layers and models.

use crate::{Array, Result};

pub trait Module {
    fn forward(&self, x: &Array) -> Result<Array>;

    fn parameters(&self) -> Vec<&Array> {
        Vec::new()
    }

    fn train(&mut self, _training: bool) {
        // Default: do nothing
    }
}