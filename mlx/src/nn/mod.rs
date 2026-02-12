// mlx/src/nn/mod.rs

pub trait Module {
    fn forward(&self, x: &Array) -> crate::Result<Array>;
    
    // In the future, you'll add methods to get parameters for the optimizer
    // fn parameters(&self) -> Vec<&Array>;
}