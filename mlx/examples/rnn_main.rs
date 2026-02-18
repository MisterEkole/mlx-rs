use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Linear, cross_entropy, Adam, Optimizer, Module, ModuleParams};
use mlx::nn::layers::recurrent::LSTM;
use std::cell::RefCell;

// =========================================================================
// 1. Define the Model Architecture
// =========================================================================
pub struct SequenceClassifier {
    lstm: LSTM,
    head: Linear,
}

impl SequenceClassifier {
    pub fn new(input_dim: usize, hidden_dim: usize, num_classes: usize, key: &Array) -> Result<Self> {
        Ok(Self {
            lstm: LSTM::new(input_dim, hidden_dim, true, key)?,
            head: Linear::new(hidden_dim, num_classes, true, key)?,
        })
    }
}

// Parameter management — delegates to sub-modules
impl ModuleParams for SequenceClassifier {
    fn parameters(&self) -> Vec<&Array> {
        [self.lstm.parameters(), self.head.parameters()].concat()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array> {
        let mut p = self.lstm.parameters_mut();
        p.extend(self.head.parameters_mut());
        p
    }

    fn update_parameters(&mut self, new_params: &[Array]) {
        let lstm_n = self.lstm.parameters().len();
        self.lstm.update_parameters(&new_params[0..lstm_n]);
        self.head.update_parameters(&new_params[lstm_n..]);
    }
}

// Module — only forward()
impl Module for SequenceClassifier {
    fn forward(&self, x: &Array) -> Result<Array> {
        let final_hidden_state = self.lstm.forward(x)?;
        self.head.forward(&final_hidden_state)
    }
}

// =========================================================================
// 2. Data Generator
// =========================================================================
fn next_batch(batch_size: usize, seq_len: usize, input_dim: usize, num_classes: usize, key: &Array) -> Result<(Array, Array)> {
    let x = Array::random_uniform(&[batch_size, seq_len, input_dim], -1.0f32, 1.0f32, Dtype::Float32, key)?;

    let raw_labels = Array::random_uniform(&[batch_size], 0.0f32, num_classes as f32, Dtype::Float32, key)?
        .cast(Dtype::Int32)?;

    let range = Array::arange(0.0f64, num_classes as f64, 1.0f64, Dtype::Float32)?;

    let targets = raw_labels.reshape(&[batch_size as i32, 1])?
        .equal(&range.reshape(&[1, num_classes as i32])?)?
        .cast(Dtype::Float32)?;

    Ok((x, targets))
}

// =========================================================================
// 3. Main Training Loop
// =========================================================================
fn main() -> Result<()> {
    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;
    println!("--- Using Device: GPU ---");

    let batch_size = 32;
    let seq_len = 20;
    let input_dim = 10;
    let hidden_dim = 64;
    let num_classes = 5;
    let learning_rate: f32 = 1e-3;
    let num_steps = 5000;

    let mut key = Array::key(42)?;

    let model = RefCell::new(SequenceClassifier::new(input_dim, hidden_dim, num_classes, &key)?);
    let mut optimizer = Adam::new(learning_rate, &model.borrow().parameters_owned())?;

    println!("--- Sequence Classifier Training Started ---");

    for step in 1..=num_steps {
        key = Array::key((42 + step) as u64)?;
        let (x, targets) = next_batch(batch_size, seq_len, input_dim, num_classes, &key)?;

        let mut params = model.borrow().parameters_owned();

        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            let mut model_mut = model.borrow_mut();
            model_mut.update_parameters(p);
            let logits = model_mut.forward(&x)?;
            cross_entropy(&logits, &targets)
        }, &params)?;

        optimizer.update(params.iter_mut().collect(), grads)?;

        model.borrow_mut().update_parameters(&params);

        let mut to_eval = params.clone();
        to_eval.push(loss.clone());
        Array::eval_all(&to_eval[..])?;

        if step % 100 == 0 || step == 1 {
            let loss_val: f32 = loss.item()?;
            println!("Step {:04} | Loss: {:.4}", step, loss_val);
        }
    }

    println!("--- Training Complete ---");
    Ok(())
}