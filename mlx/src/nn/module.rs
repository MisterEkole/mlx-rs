
/// Module trait and related utilities for building neural network layers and models.

use crate::{Array, Result, TreeFlatten};


pub trait ModuleParams {
    fn parameters(&self) -> Vec<&Array> { Vec::new() }
    fn parameters_mut(&mut self) -> Vec<&mut Array> { Vec::new() }
    fn update_parameters(&mut self, _new_params: &[Array]) {}
    fn train(&mut self, _training: bool) {}

    fn parameters_owned(&self) -> Vec<Array> {
        self.parameters().into_iter().cloned().collect()
    }
}

pub trait Module: ModuleParams + TreeFlatten{
    fn forward(&self, x: &Array) -> Result<Array>;
}

// Blanket impls for Box<dyn Module> — makes Sequential work.
impl ModuleParams for Box<dyn Module> {
    fn parameters(&self) -> Vec<&Array> { (**self).parameters() }
    fn parameters_mut(&mut self) -> Vec<&mut Array> { (**self).parameters_mut() }
    fn update_parameters(&mut self, new_params: &[Array]) { (**self).update_parameters(new_params) }
    fn train(&mut self, training: bool) { (**self).train(training) }
}

impl Module for Box<dyn Module> {
    fn forward(&self, x: &Array) -> Result<Array> { (**self).forward(x) }
}

impl TreeFlatten for Box<dyn Module> {
    fn flatten_state(&self) -> Vec<Array> {
        (**self).flatten_state()
    }

    fn unflatten_state(&mut self, flat_arrays: &mut std::slice::Iter<'_, Array>) {
        (**self).unflatten_state(flat_arrays)
    }
}

// pub trait ModuleParams {
//     // This trait is a marker for structs that have parameters. It can be used for code generation or reflection.
//     fn parameters(&self) -> Vec<&Array>;
//     fn parameters_mut(&mut self) -> Vec<&mut Array>;
//     fn update_parameters(&mut self, new_params: &[Array]);
//     fn train(&mut self, training: bool);
// }
// pub trait Module {
//     fn forward(&self, x: &Array) -> Result<Array>;

//     fn parameters(&self) -> Vec<&Array> { Vec::new() }
    
//     fn parameters_mut(&mut self) -> Vec<&mut Array> { Vec::new() }

//     fn parameters_owned(&self) -> Vec<Array> {
//         self.parameters().into_iter().cloned().collect()
//     }

//     // The Bridge: This must be overridden by structs that have parameters
//     fn update_parameters(&mut self, _new_params: &[Array]) {
//         // Default: do nothing (for layers like ReLU)
//     }

//     fn train(&mut self, _training: bool) {}
// }