// lib.rs

use std::fmt;

// 1. Declare the sub-modules (this tells Rust to look for array.rs and dtype.rs)
pub mod array;
pub mod dtype;
pub mod device;
pub mod operations; 


// 2. Re-export items so users can just type `mlx::Array` instead of `mlx::array::Array`
pub use array::Array;
pub use dtype::Dtype;
pub use device::{Device, DeviceType};
pub use operations::*; 

// 3. Import the C-bindings here once, so other files can use them via `crate::sys`
// (Assuming your bindings crate is named mlx_sys)
pub use mlx_sys as sys; 

// 4. Define Global Errors (used by all files)
#[derive(Debug)]
pub enum Error {
    NullPointer,
    OperationFailed,
    InvalidUtf8,
    // You can add more errors here as you grow
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NullPointer => write!(f, "Null pointer returned from MLX"),
            Error::OperationFailed => write!(f, "MLX operation failed"),
            Error::InvalidUtf8 => write!(f, "Invalid UTF-8 in MLX string"),
        }
    }
}

impl std::error::Error for Error {}

// A convenient alias for Result
pub type Result<T> = std::result::Result<T, Error>;