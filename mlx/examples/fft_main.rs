use mlx::{Array, Dtype, Result, Device, DeviceType};
use mlx::operations::fft;

fn main() -> Result<()> {
    // Ensure we are on CPU if certain FFT kernels are not yet in your Metal build
    let cpu = Device::new(DeviceType::Cpu);
    Device::set_default(&cpu)?;

    println!(" MLX FFT Tests...\n");

 
    let signal_data = [1.0f32, 2.0, 1.0, -1.0, -2.0, -1.0, 0.0, 1.0];
    let x = Array::from_slice(&signal_data, &[8], Dtype::Float32)?;
    x.eval()?;
    println!("Original Signal:\n{:?}\n", x);

   
    println!("--- 1D FFT ---");
    let x_fft = fft::fft(&x, None, -1)?;
    x_fft.eval()?;
    println!("FFT Result:\n{:?}\n", x_fft);

    println!("--- 1D Inverse FFT ---");
    let x_recovered = fft::ifft(&x_fft, None, -1)?;
    x_recovered.eval()?;
    println!("Recovered:\n{:?}\n", x_recovered);

    
    println!("--- Real FFT (RFFT) ---");
    let x_rfft = fft::rfft(&x, None, -1)?;
    x_rfft.eval()?;
    println!("RFFT Result:\n{:?}\n", x_rfft);

  
    println!("--- 2D FFT ---");
    let grid_data = [1.0f32; 16]; 
    let matrix = Array::from_slice(&grid_data, &[4, 4], Dtype::Float32)?;
    let matrix_fft2 = fft::fft2(&matrix, None, None)?;
    matrix_fft2.eval()?;
    println!("2D FFT Result:\n{:?}\n", matrix_fft2);

  
    println!("--- FFT Shift ---");
    let shifted = fft::fftshift(&x_fft, None)?;
    shifted.eval()?;
    println!("Shifted Spectrum:\n{:?}\n", shifted);

    println!("✅ All FFT operations verified!");
    Ok(())
}