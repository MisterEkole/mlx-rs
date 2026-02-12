//! Low-level FFI bindings to mlx-c
//! 
//! This crate provides raw, unsafe Rust bindings to the MLX C API
//! generated automatically from the mlx-c headers.

// These allows are necessary because C naming conventions 
// (like mlx_array) trigger Rust's style warnings.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// This pulls in every single function, struct, and enum 
// from the mlx.h header that bindgen translated.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));