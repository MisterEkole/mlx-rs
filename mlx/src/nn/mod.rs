// mlx/src/nn/mod.rs
pub mod module;  // Points to module.rs
pub mod layers;  // Points to the layers/ folder
pub mod losses;
pub mod optimizers;
pub mod transformers; // Points to the transformers/ folder

// 2. Re-export for the user
// This allows: use mlx::nn::Module;
pub use module::{Module, ModuleParams};
pub use losses::*; 
pub use optimizers::*;


// This allows: use mlx::nn::Linear; (instead of mlx::nn::layers::linear::Linear)
pub use layers::*;
