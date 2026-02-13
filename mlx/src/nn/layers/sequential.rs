//! Sequential container for stacking neural network layers.

use crate::{Array, Result};
use crate::nn::Module;

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

impl Module for Sequential {
    /// Pass the input through each layer in the sequence.
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }

    /// Collects all parameters from all constituent layers.
    fn parameters(&self) -> Vec<&Array> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Provides mutable access to the weights for the Optimizer
  fn parameters_mut(&mut self) -> Vec<&mut Array> {
    self.layers
        .iter_mut()
        .flat_map(|layer| layer.parameters_mut())
        .collect()
    }

    /// Sets the training mode for all constituent layers.
    fn train(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.train(training);
        }
    }


    fn update_parameters(&mut self, new_params: &[Array]) {
        let mut i = 0;
        for layer in &mut self.layers {
            let n = layer.parameters().len();
            // Give the layer the slice of params it owns
            layer.update_parameters(&new_params[i..i+n]);
            i += n;
        }
    }
}