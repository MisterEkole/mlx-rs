use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Sequential, Linear, ReLU, Conv2d, Flatten, cross_entropy, Adam, Optimizer, Module, ModuleParams};
use std::cell::RefCell;


fn main() -> Result<()> {

    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;// fyi unified nature of MLX means we can write code once and run on CPU or GPU without changes
    println!("--- Using Device: GPU ---");
    let key = Array::key(42)?; 

   
    let model = RefCell::new(Sequential::new(vec![
       
        Box::new(Conv2d::new(1, 16, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, &key)?),
        Box::new(ReLU::new()),
       
        Box::new(Conv2d::new(16, 32, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, &key)?),
        Box::new(ReLU::new()),
        
        Box::new(Flatten::new()),
        
    
        Box::new(Linear::new(1568, 10, true, &key)?),
    ]));

    let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

    let x = Array::random_uniform(&[8, 28, 28, 1], -1.0, 1.0, Dtype::Float32, &key)?;

    let raw_labels = Array::random_uniform(&[8], 0.0, 10.0, Dtype::Float32, &key)?
        .cast(Dtype::Int32)?;
    let range = Array::arange(0.0, 10.0, 1.0, Dtype::Float32)?;

    let targets = raw_labels.reshape(&[8, 1])?
        .equal(&range.reshape(&[1, 10])?)?
        .cast(Dtype::Float32)?;

    println!("--- CNN Training Started ---");

    for step in 0..10000 {
  
        let mut params = model.borrow().parameters_owned();

        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            let logits = {
                let mut model_mut = model.borrow_mut();
                model_mut.update_parameters(p); 
                model_mut.forward(&x)?
            };
            cross_entropy(&logits, &targets)
        }, &params)?;

        optimizer.update(params.iter_mut().collect(), grads)?;

        model.borrow_mut().update_parameters(&params);

        let mut to_eval = params.clone();
        to_eval.push(loss.clone());
        Array::eval_all(&to_eval[..])?;

        if step % 10 == 0 {
            let loss_val: f32 = loss.item()?;
            println!("Step {}: Loss = {:.6}", step, loss_val);
        }
    }

    println!("--- Training Complete ---");
    Ok(())
}