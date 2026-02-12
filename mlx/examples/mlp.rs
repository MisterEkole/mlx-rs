use mlx::{Array, Dtype, Result};
use mlx::nn::{Sequential, Linear, ReLU,cross_entropy};
use mlx::nn::Module;

fn main() -> Result<()> {
    let key = Array::key(42)?; 

    //  MLP: Input(10) -> Hidden(32) -> Output(5)
    let model = Sequential::new(vec![
        Box::new(Linear::new(10, 32, true, &key)?),
        Box::new(ReLU::new()),
        Box::new(Linear::new(32, 5, true, &key)?),
    ]);
// dummy input and target for testing
    let x = Array::random_uniform(&[4, 10], -1.0, 1.0, Dtype::Float32, &key)?;
    
    let targets = Array::full(&[4, 5], 0.2, Dtype::Float32)?; 

    println!("Input shape: {:?}", x.shape()?);

   
    let logits = model.forward(&x)?;
    println!("Output (Logits) shape: {:?}", logits.shape()?);

    let loss = cross_entropy(&logits, &targets)?;
    println!("Calculated Cross Entropy Loss: {:?}", loss);

    Ok(())
}