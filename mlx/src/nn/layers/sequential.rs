//! Sequential container for stacking neural network layers.

use crate::{Array,Result};
use crate::nn::{Module,ModuleParams};

/// A container that wraps a sequence of modules and applies them one after another.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    /// Creates a new Sequential container from a vector of boxed modules.
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    /// Appends a new layer to the end of the sequence.
    /// The 'static bound ensures the layer doesn't contain temporary references.
    pub fn add<M: Module + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer));
    }
}

/// Manual ModuleParam -Vec<Box<dyn Module>> pattern

impl ModuleParams for Sequential {
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

impl Module for Sequential {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}
