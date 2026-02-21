// examples/benchmark.rs
use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Module, ModuleParams, Optimizer, Sequential, Linear, ReLU, Conv2d, Flatten};
use mlx::nn::{cross_entropy, Adam};
use mlx::tree::TreeFlatten; 
use mlx::nn::transformers::TransformerEncoder;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn bench<F: FnMut() -> Result<()>>(name: &str, warmup: usize, iters: usize, mut f: F) {
    for _ in 0..warmup {
        f().unwrap();
    }

    let start = Instant::now();
    for _ in 0..iters {
        f().unwrap();
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iter = total_ms / iters as f64;
    println!(
        "  {:<40} {:>8.2} ms total | {:>8.3} ms/iter  ({} iters)",
        name, total_ms, per_iter, iters
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Raw Matmul Throughput
// ═══════════════════════════════════════════════════════════════════════════

fn bench_matmul(key: &Array) -> Result<()> {
    println!("\n══ MATMUL THROUGHPUT ══");

    for &size in &[256, 512, 1024, 2048, 4096] {
        let a = Array::random_uniform(&[size, size], -1.0, 1.0, Dtype::Float32, key)?;
        let b = Array::random_uniform(&[size, size], -1.0, 1.0, Dtype::Float32, key)?;

        bench(&format!("matmul {}x{}", size, size), 3, 50, || {
            let c = a.matmul(&b)?;
            c.eval()?;
            Ok(())
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Element-wise Operations
// ═══════════════════════════════════════════════════════════════════════════

fn bench_elementwise(key: &Array) -> Result<()> {
    println!("\n══ ELEMENT-WISE OPS (1M elements) ══");

    let n = 1_000_000;
    let a = Array::random_uniform(&[n], -1.0, 1.0, Dtype::Float32, key)?;
    let b = Array::random_uniform(&[n], -1.0, 1.0, Dtype::Float32, key)?;

    bench("add", 3, 200, || { let c = a.add(&b)?; c.eval()?; Ok(()) });
    bench("multiply", 3, 200, || { let c = a.multiply(&b)?; c.eval()?; Ok(()) });
    bench("exp", 3, 200, || { let c = a.exp()?; c.eval()?; Ok(()) });
    bench("sin", 3, 200, || { let c = a.sin()?; c.eval()?; Ok(()) });
    bench("sqrt", 3, 200, || { let c = a.abs()?.sqrt()?; c.eval()?; Ok(()) });

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. MLP Training Step (Fully JIT Compiled)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_mlp_training(key: &Array) -> Result<()> {
    println!("\n══ MLP TRAINING STEP ══");

    for &(batch, input, hidden, output) in &[
        (32_usize, 128_usize, 256_usize, 10_usize),
        (64, 512, 1024, 100),
        (128, 784, 2048, 10),
    ] {
        let build_model = || -> Result<Sequential> {
            Ok(Sequential::new(vec![
                Box::new(Linear::new(input, hidden, true, key)?),
                Box::new(ReLU::new()),
                Box::new(Linear::new(hidden, hidden, true, key)?),
                Box::new(ReLU::new()),
                Box::new(Linear::new(hidden, output, true, key)?),
            ]))
        };

        let model = RefCell::new(build_model()?);
        let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

        let x = Array::random_uniform(&[batch, input], -1.0, 1.0, Dtype::Float32, key)?;
        let class_range = Array::arange(0.0, output as f64, 1.0, Dtype::Float32)?;
        let raw_labels = Array::random_uniform(&[batch], 0.0, output as f32, Dtype::Float32, key)?.cast(Dtype::Int32)?;
        let targets = raw_labels.reshape(&[batch as i32, 1])?.equal(&class_range.reshape(&[1, output as i32])?)?.cast(Dtype::Float32)?;

        // Dummy state for tracing
        let local_model = RefCell::new(build_model()?);
        let local_opt = RefCell::new(Adam::new(1e-3, &local_model.borrow().parameters_owned())?);
        
        let x_c = x.clone();
        let targets_c = targets.clone();

        let compiled_train_step = mlx::compile(move |flat_inputs: &[Array]| -> Result<Vec<Array>> {
            let mut iter = flat_inputs.iter();
            local_model.borrow_mut().unflatten_state(&mut iter);
            local_opt.borrow_mut().unflatten_state(&mut iter);

            let mut params = local_model.borrow().parameters_owned();
            let (loss, grads) = transforms::value_and_grad(|inner_p: &[Array]| {
                let mut m = local_model.borrow_mut();
                m.update_parameters(inner_p);
                let logits = m.forward(&x_c).unwrap();
                Ok(cross_entropy(&logits, &targets_c).unwrap())
            }, &params)?;

            local_opt.borrow_mut().update(params.iter_mut().collect(), grads)?;
            local_model.borrow_mut().update_parameters(&params);

            let mut outputs = vec![loss];
            outputs.extend(local_model.borrow().flatten_state());
            outputs.extend(local_opt.borrow().flatten_state());
            Ok(outputs)
        }, false)?;

        bench(&format!("MLP B={} [{}->{}->{}]", batch, input, hidden, output), 3, 100, || {
            let mut flat_in = model.borrow().flatten_state();
            flat_in.extend(optimizer.flatten_state());

            let flat_out = compiled_train_step(&flat_in)?;

            let mut out_iter = flat_out[1..].iter();
            model.borrow_mut().unflatten_state(&mut out_iter);
            optimizer.unflatten_state(&mut out_iter);

            Array::eval_all(&flat_out)?;
            Ok(())
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. CNN Training Step (Fully JIT Compiled)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_cnn_training(key: &Array) -> Result<()> {
    println!("\n══ CNN TRAINING STEP (MNIST-like) ══");

    let batch = 32;
    let build_cnn = || -> Result<Sequential> {
        Ok(Sequential::new(vec![
            Box::new(Conv2d::new(1, 16, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
            Box::new(ReLU::new()),
            Box::new(Conv2d::new(16, 32, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
            Box::new(ReLU::new()),
            Box::new(Flatten::new()),
            Box::new(Linear::new(1568, 10, true, key)?),
        ]))
    };

    let model = RefCell::new(build_cnn()?);
    let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

    let x = Array::random_uniform(&[batch, 28, 28, 1], -1.0, 1.0, Dtype::Float32, key)?;
    let class_range = Array::arange(0.0, 10.0, 1.0, Dtype::Float32)?;
    let raw_labels = Array::random_uniform(&[batch], 0.0, 10.0, Dtype::Float32, key)?.cast(Dtype::Int32)?;
    let targets = raw_labels.reshape(&[batch as i32, 1])?.equal(&class_range.reshape(&[1, 10])?)?.cast(Dtype::Float32)?;

    let local_model = RefCell::new(build_cnn()?);
    let local_opt = RefCell::new(Adam::new(1e-3, &local_model.borrow().parameters_owned())?);
    
    let x_c = x.clone();
    let targets_c = targets.clone();

    let compiled_train_step = mlx::compile(move |flat_inputs: &[Array]| -> Result<Vec<Array>> {
        let mut iter = flat_inputs.iter();
        local_model.borrow_mut().unflatten_state(&mut iter);
        local_opt.borrow_mut().unflatten_state(&mut iter);

        let mut params = local_model.borrow().parameters_owned();
        let (loss, grads) = transforms::value_and_grad(|inner_p: &[Array]| {
            let mut m = local_model.borrow_mut();
            m.update_parameters(inner_p);
            let logits = m.forward(&x_c).unwrap();
            Ok(cross_entropy(&logits, &targets_c).unwrap())
        }, &params)?;

        local_opt.borrow_mut().update(params.iter_mut().collect(), grads)?;
        local_model.borrow_mut().update_parameters(&params);

        let mut outputs = vec![loss];
        outputs.extend(local_model.borrow().flatten_state());
        outputs.extend(local_opt.borrow().flatten_state());
        Ok(outputs)
    }, false)?;

    bench("CNN fwd+bwd+update B=32 [28x28x1 -> 10]", 3, 100, || {
        let mut flat_in = model.borrow().flatten_state();
        flat_in.extend(optimizer.flatten_state());

        let flat_out = compiled_train_step(&flat_in)?;

        let mut out_iter = flat_out[1..].iter();
        model.borrow_mut().unflatten_state(&mut out_iter);
        optimizer.unflatten_state(&mut out_iter);

        Array::eval_all(&flat_out)?;
        Ok(())
    });

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Transformer Forward Pass
// ═══════════════════════════════════════════════════════════════════════════

fn bench_transformer(key: &Array) -> Result<()> {
    println!("\n══ TRANSFORMER ENCODER FORWARD ══");

    for &(d_model, n_heads, n_layers, seq_len, batch) in &[
        (64_usize,  4_usize, 2_usize, 32_usize, 16_usize),
        (128,       4,       4,       64,       16),
        (256,       8,       6,       128,      8),
    ] {
        let d_ff = d_model * 4;
        let encoder = Rc::new(TransformerEncoder::new(n_layers, d_model, n_heads, d_ff, 0.0, key)?);
        let x = Array::random_uniform(&[batch, seq_len, d_model], -1.0, 1.0, Dtype::Float32, key)?;

        let enc_c = Rc::clone(&encoder);
        let compiled_fwd = mlx::compile(move |inputs: &[Array]| -> Result<Vec<Array>> {
            let out = enc_c.forward(&inputs[0])?;
            Ok(vec![out])
        }, false)?;

        bench(&format!("Encoder B={} S={} d={} h={} L={}", batch, seq_len, d_model, n_heads, n_layers), 3, 50, || {
            let out = compiled_fwd(&[x.clone()])?;
            Array::eval_all(&out)?;
            Ok(())
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║          mlx-rs JIT Benchmark Suite              ║");
    println!("║          Device: Apple Silicon GPU               ║");
    println!("╚══════════════════════════════════════════════════╝");

    let key = Array::key(42)?;

    bench_matmul(&key)?;
    bench_elementwise(&key)?;
    bench_mlp_training(&key)?;
    bench_cnn_training(&key)?;
    bench_transformer(&key)?;

    println!("\n══ DONE ══");
    Ok(())
}