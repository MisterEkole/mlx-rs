/// mlx/examples/mlp.rs
use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Sequential, Linear, ReLU, cross_entropy, Adam, Optimizer, Module, ModuleParams};
use std::cell::RefCell;

fn main() -> Result<()> {

    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;
    println!("--- Using Device: GPU ---");
    let key = Array::key(42)?; 

    let model = RefCell::new(Sequential::new(vec![
        Box::new(Linear::new(10, 32, true, &key)?),
        Box::new(ReLU::new()),
        Box::new(Linear::new(32, 5, true, &key)?),
    ]));

    let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

    let x = Array::random_uniform(&[8, 10], -1.0, 1.0, Dtype::Float32, &key)?;
    let targets = Array::full(&[8, 5], 0.2, Dtype::Float32)?; 

    println!("--- Training Loop Started ---");

    for step in 0..100 {
        let current_params = model.borrow().parameters_owned();

        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            // Update weights: Scoped block to ensure the mutable borrow is dropped
            {
                let mut model_mut = model.borrow_mut();
                model_mut.update_parameters(p); 
            } 
            
            // Forward pass: Use a fresh immutable borrow
            let model_ref = model.borrow();
            let logits = model_ref.forward(&x)?;
            cross_entropy(&logits, &targets)
        }, &current_params)?;

        // Apply gradients
        optimizer.update(model.borrow_mut().parameters_mut(), grads)?;

        // Batch Evaluation (The Big Flush)
        let mut to_eval = model.borrow().parameters_owned();
        to_eval.extend(optimizer.get_state_handles()); 
        to_eval.push(loss.clone());

        Array::eval_all(&to_eval[..])?;

        let loss_val: f32 = loss.item()?; 
        println!("Step {}: Loss = {:.6}", step, loss_val);
    }

    Ok(())
}
// use mlx::{Array, Dtype, Result};
// use mlx::nn::{Sequential, Linear, ReLU,cross_entropy};
// use mlx::nn::Module;

// fn main() -> Result<()> {
//     let key = Array::key(42)?; 

//     //  MLP: Input(10) -> Hidden(32) -> Output(5)
//     let model = Sequential::new(vec![
//         Box::new(Linear::new(10, 32, true, &key)?),
//         Box::new(ReLU::new()),
//         Box::new(Linear::new(32, 5, true, &key)?),
//     ]);
// // dummy input and target for testing
//     let x = Array::random_uniform(&[4, 10], -1.0, 1.0, Dtype::Float32, &key)?;
    
//     let targets = Array::full(&[4, 5], 0.2, Dtype::Float32)?; 

//     println!("Input shape: {:?}", x.shape()?);

   
//     let logits = model.forward(&x)?;
//     println!("Output (Logits) shape: {:?}", logits.shape()?);

//     let loss = cross_entropy(&logits, &targets)?;
//     println!("Calculated Cross Entropy Loss: {:?}", loss);

//     Ok(())
// }



