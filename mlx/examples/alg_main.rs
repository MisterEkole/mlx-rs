use mlx::{Array, Dtype, Result,Device, DeviceType};
use mlx::operations::linalg;

fn main() -> Result<()> {
    println!("Starting MLX Linear Algebra Tests...\n");
    let cpu_device = Device::new(DeviceType::Cpu);
   cpu_device.set_default()?;

  
    let a_data = [2.0f32, 1.0, 1.0, 2.0];
    let a = Array::from_slice(&a_data, &[2, 2], Dtype::Float32)?;
    a.eval()?;
    println!("Matrix A:\n{:?}\n", a);


    println!("--- Inverse ---");
    let a_inv = linalg::inv(&a)?;
    a_inv.eval()?;
    println!("{:?}\n", a_inv);


    println!("--- Norm (Frobenius) ---");
    let norm = linalg::norm(&a, 2.0, &[], false)?;
    norm.eval()?;
    println!("{:?}\n", norm);


    println!("--- QR Decomposition ---");
    let (q, r) = linalg::qr(&a)?;
    q.eval()?;
    r.eval()?;
    println!("Q:\n{:?}", q);
    println!("R:\n{:?}\n", r);

    println!("--- Cholesky Decomposition (Lower) ---");
    let chol = linalg::cholesky(&a, false)?;
    chol.eval()?;
    println!("{:?}\n", chol);

  
    println!("--- Eigen Decomposition ---");
    let (eig_vals, eig_vecs) = linalg::eigh(&a, false)?;
    eig_vals.eval()?;
    eig_vecs.eval()?;
    println!("Values:\n{:?}", eig_vals);
    println!("Vectors:\n{:?}\n", eig_vecs);

    println!("--- Solve Ax = b ---");
    let b_data = [1.0f32, 2.0];
    let b = Array::from_slice(&b_data, &[2], Dtype::Float32)?;
    let x = linalg::solve(&a, &b)?;
    x.eval()?;
    println!("x:\n{:?}\n", x);

  
    println!("--- SVD ---");
    let svd_res = linalg::svd(&a, true)?;
 
    for (i, arr) in svd_res.iter().enumerate() {
        arr.eval()?;
        println!("Component {}:\n{:?}", i, arr);
    }
    
    println!("\n✅ All Linear Algebra operations completed successfully!");
    Ok(())
}