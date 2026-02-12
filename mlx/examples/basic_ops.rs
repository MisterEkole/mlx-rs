//! Basic array operations example
//! 
//! This example demonstrates:
//! - Creating arrays from Rust slices
//! - Element-wise arithmetic operations
//! - Lazy evaluation in MLX

use mlx::{Array, Dtype, Result,Device, DeviceType};

fn main() -> Result<()> {
    println!("MLX-RS Basic Array Operations Example\n");
    println!("Initializing GPU device...");
    let gpu = Device::new(DeviceType::Gpu);

    gpu.set_default()?;
    println!("GPU set as default device.\n");
    
    // Create some arrays
    println!("Creating arrays...");
    let a = Array::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0], 
        &[4], 
        Dtype::Float32
    )?;
    let b = Array::from_slice(
        &[5.0f32, 6.0, 7.0, 8.0], 
        &[4], 
        Dtype::Float32
    )?;
    
    println!("Array a: {:?}", a);
    println!("Array b: {:?}", b);
    
    // Element-wise addition
    println!("\nComputing a + b...");
    let sum = a.add(&b)?;
    
    // Element-wise multiplication
    println!("Computing a * b...");
    let product = a.multiply(&b)?;

    // Element wise matmul
    println!("Computing a @ b...");
   let dot_product = a.matmul(&b)?;  
    
    // Note: At this point, operations are lazy - not yet computed!
    println!("\nOperations are lazy - not yet evaluated");
    
    // Force evaluation
    println!("Evaluating results...");
    sum.eval()?;
    //product.eval();
    
    // Display results
    println!("\nResults:");
    println!("a + b = {}", sum);
    println!("a * b = {:}", product);
    println!("a @ b = {:}", dot_product);




//     // 1. Create a flat array of 6 elements
//     let data = Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], Dtype::Float32)?;

// // 2. Reshape it into a 2x3 matrix
//     let matrix_a = data.reshape(&[2, 3])?;
// println!("Matrix A shape: {:?}", matrix_a.shape());

// // 3. Create a 3x1 matrix for multiplication
//     let matrix_b = Array::from_slice(&[1.0, 1.0, 1.0], &[3], Dtype::Float32)?.reshape(&[3, 1])?;

// // 4. Multiply on GPU!
//     let result = matrix_a.matmul(&matrix_b)?;
//     result.eval()?;
//     println!("Matmul Result:\n{}", result);
    
    Ok(())
}
