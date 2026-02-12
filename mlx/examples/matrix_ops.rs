//! Matrix operations example
//! 
//! This example demonstrates working with matrices in MLX.
//! Note: Full matrix multiplication and other operations will be
//! added as the API is expanded.

use mlx::{Array, Dtype, Result};

fn main() -> Result<()> {
    println!("MLX-RS Matrix Example\n");
    
    // Create a 2D matrix (2x3)
    let matrix_data = vec![
        1.0f32, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ];
    
    let matrix = Array::from_slice(
        &matrix_data,
        &[2, 3],  // shape: 2 rows, 3 columns
        Dtype::Float32,
    )?;
    
    println!("Matrix (2x3):");
    println!("{:?}", matrix);
    
    // Create a vector (3x1) - conceptually
    let vector_data = vec![1.0f32, 2.0, 3.0];
    let vector = Array::from_slice(
        &vector_data,
        &[3],
        Dtype::Float32,
    )?;
    
    println!("\nVector (3):");
    println!("{:?}", vector);
    
    // Element-wise operations work on matrices too
    let scaled = matrix.multiply(&matrix)?;
    scaled.eval();
    
    println!("\nElement-wise square:");
    println!("{:?}", scaled);
    
    println!("\n📝 Note: Matrix multiplication, broadcasting, and other");
    println!("   advanced operations will be added in future updates.");
    
    Ok(())
}
