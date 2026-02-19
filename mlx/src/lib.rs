// lib.rs

use std::fmt;

// 1. Declare the sub-modules (this tells Rust to look for array.rs and dtype.rs)
pub mod array;
pub mod dtype;
pub mod device;
pub mod operations; 
pub mod nn;
pub mod transforms;
pub mod quantization;
pub mod io;


// 2. Re-export items so users can just type `mlx::Array` instead of `mlx::array::Array`
pub use array::Array;
pub use dtype::Dtype;
pub use device::{Device, DeviceType};
pub use operations::*;
pub use transforms::*;
pub use quantization::*;
pub use io::*;

// 3. Import the C-bindings here once, so other files can use them via `crate::sys`
// (Assuming your bindings crate is named mlx_sys)
pub use mlx_sys as sys; 

// 4. Define Global Errors (used by all files)
#[derive(Debug)]
pub enum Error {
    NullPointer,
    OperationFailed(String),
    InvalidUtf8,
    MlxError(i32),
    InvalidShape(String)
    
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NullPointer => write!(f, "Null pointer returned from MLX"),
            Error::OperationFailed(msg) => write!(f, "MLX operation failed: {}", msg),
            Error::InvalidUtf8 => write!(f, "Invalid UTF-8 in MLX string"),
            Error::MlxError(code) => write!(f, "MLX error with code: {}", code),
            Error::InvalidShape(msg) => write!(f, "Invalid shape: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

// A convenient alias for Result
pub type Result<T> = std::result::Result<T, Error>;