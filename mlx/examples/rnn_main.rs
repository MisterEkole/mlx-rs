use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Linear, cross_entropy, Adam, Optimizer, Module, ModuleParams};
use mlx::nn::layers::recurrent::LSTM;
use mlx::TreeFlatten;
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

// Support for the JIT boundary
impl TreeFlatten for SequenceClassifier {
    fn flatten_state(&self) -> Vec<Array> {
        let mut state = self.lstm.flatten_state();
        state.extend(self.head.flatten_state());
        state
    }

    fn unflatten_state(&mut self, iter: &mut std::slice::Iter<Array>) {
        self.lstm.unflatten_state(iter);
        self.head.unflatten_state(iter);
    }
}

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

impl Module for SequenceClassifier {
    fn forward(&self, x: &Array) -> Result<Array> {
        // LSTM in mlx-rs typically returns the full sequence or final state
        let final_hidden_state = self.lstm.forward(x)?;
        self.head.forward(&final_hidden_state)
    }
}

// =========================================================================
// 2. Data Generator
// =========================================================================
fn next_batch(batch_size: usize, seq_len: usize, input_dim: usize, num_classes: usize, key: &Array) -> Result<(Array, Array)> {
    let x = Array::random_uniform(&[batch_size, seq_len, input_dim], -1.0f32, 1.0f32, Dtype::Float32, key)?;
    let raw_labels = Array::random_uniform(&[batch_size], 0.0f32, num_classes as f32, Dtype::Float32, key)?.cast(Dtype::Int32)?;
    let range = Array::arange(0.0f64, num_classes as f64, 1.0f64, Dtype::Float32)?;
    let targets = raw_labels.reshape(&[batch_size as i32, 1])?.equal(&range.reshape(&[1, num_classes as i32])?)?.cast(Dtype::Float32)?;
    Ok((x, targets))
}

// =========================================================================
// 3. Main Training Loop
// =========================================================================
fn main() -> Result<()> {
    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;

    let batch_size = 32;
    let seq_len = 20;
    let input_dim = 10;
    let hidden_dim = 64;
    let num_classes = 5;
    let learning_rate: f32 = 1e-3;
    let num_steps = 5000;

    let key = Array::key(42)?;

    // Real state
    let model = RefCell::new(SequenceClassifier::new(input_dim, hidden_dim, num_classes, &key)?);
    let mut optimizer = Adam::new(learning_rate, &model.borrow().parameters_owned())?;

    // --- JIT COMPILE SETUP ---

    let local_model = RefCell::new(SequenceClassifier::new(input_dim, hidden_dim, num_classes, &key)?);
    let local_opt = RefCell::new(Adam::new(learning_rate, &local_model.borrow().parameters_owned())?);

    // We compile a single function that handles Data -> Update
    let compiled_step = mlx::compile(move |inputs: &[Array]| -> Result<Vec<Array>> {
        // inputs[0] is X, inputs[1] is Targets, the rest is flattened state
        let x = &inputs[0];
        let targets = &inputs[1];
        let mut state_iter = inputs[2..].iter();

      
        local_model.borrow_mut().unflatten_state(&mut state_iter);
        local_opt.borrow_mut().unflatten_state(&mut state_iter);

        let mut params = local_model.borrow().parameters_owned();

     
        let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
            let mut m = local_model.borrow_mut();
            m.update_parameters(p);
            let logits = m.forward(x).unwrap(); // Unwrap inside tracing is fine
            Ok(cross_entropy(&logits, targets).unwrap())
        }, &params)?;

       
        local_opt.borrow_mut().update(params.iter_mut().collect(), grads)?;
        local_model.borrow_mut().update_parameters(&params);

      
        let mut outputs = vec![loss];
        outputs.extend(local_model.borrow().flatten_state());
        outputs.extend(local_opt.borrow().flatten_state());
        Ok(outputs)
    }, false)?;

    println!("--- Sequence Classifier Training (JIT) Started ---");

    for step in 1..=num_steps {
        let step_key = Array::key((42 + step) as u64)?;
        let (x, targets) = next_batch(batch_size, seq_len, input_dim, num_classes, &step_key)?;

     
        let mut flat_in = vec![x, targets];
        flat_in.extend(model.borrow().flatten_state());
        flat_in.extend(optimizer.flatten_state());

     
        let flat_out = compiled_step(&flat_in)?;

    
        let mut out_iter = flat_out[1..].iter(); 
        model.borrow_mut().unflatten_state(&mut out_iter);
        optimizer.unflatten_state(&mut out_iter);

       
        Array::eval_all(&flat_out)?;

        if step % 100 == 0 || step == 1 {
            let loss_val: f32 = flat_out[0].item()?;
            println!("Step {:04} | Loss: {:.4}", step, loss_val);
        }
    }

    Ok(())
}