// examples/benchmark.rs
//
// Benchmark suite for mlx-rs on Apple Silicon.
// Run with: cargo run --release --example benchmark
//
// Compares well against the Python MLX equivalent (bench_mlx.py)
// and PyTorch MPS equivalent (bench_torch.py) included in this repo.

use mlx::{Array, Dtype, Result, transforms, Device, DeviceType};
use mlx::nn::{Module, ModuleParams, Optimizer, Sequential, Linear, ReLU, Conv2d, Flatten};
use mlx::nn::{cross_entropy, Adam};
use std::rc::Rc;
//use mlx::nn::layers::normalization::LayerNorm;
//use mlx::nn::layers::embedding::Embedding;
use mlx::nn::transformers::TransformerEncoder;
use std::cell::RefCell;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Time a closure over `n` iterations, return (total_ms, per_iter_ms).
fn bench<F: FnMut() -> Result<()>>(name: &str, warmup: usize, iters: usize, mut f: F) {
    // Warmup — don't measure
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
        let a = Array::random_uniform(
            &[size, size], -1.0, 1.0, Dtype::Float32, key,
        )?;
        let b = Array::random_uniform(
            &[size, size], -1.0, 1.0, Dtype::Float32, key,
        )?;

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
// 3. MLP Training Step (JIT Compiled)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_mlp_training(key: &Array) -> Result<()> {
    println!("\n══ MLP TRAINING STEP ══");

    for &(batch, input, hidden, output) in &[
        (32_usize, 128_usize, 256_usize, 10_usize),
        (64, 512, 1024, 100),
        (128, 784, 2048, 10),
    ] {
        // Use Rc to share the model into the static JIT closure
        let model = Rc::new(RefCell::new(Sequential::new(vec![
            Box::new(Linear::new(input, hidden, true, key)?),
            Box::new(ReLU::new()),
            Box::new(Linear::new(hidden, hidden, true, key)?),
            Box::new(ReLU::new()),
            Box::new(Linear::new(hidden, output, true, key)?),
        ])));

        let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

        let x = Array::random_uniform(&[batch, input], -1.0, 1.0, Dtype::Float32, key)?;
        let class_range = Array::arange(0.0, output as f64, 1.0, Dtype::Float32)?;
        let raw_labels = Array::random_uniform(&[batch], 0.0, output as f32, Dtype::Float32, key)?
            .cast(Dtype::Int32)?;
        let targets = raw_labels.reshape(&[batch as i32, 1])?
            .equal(&class_range.reshape(&[1, output as i32])?)?
            .cast(Dtype::Float32)?;

        let label = format!(
            "MLP fwd+bwd+update B={} [{}->{}->{}->{}]",
            batch, input, hidden, hidden, output
        );

        // --- NEW: JIT Compile the Forward & Backward pass ---
        let x_c = x.clone();
        let targets_c = targets.clone();
        let model_c = Rc::clone(&model);

        // This compiles the entire computation graph into a single FFI boundary call!
        let  compiled_grad_step = mlx::compile(move |p: &[Array]| -> Result<Vec<Array>> {
        let (loss, grads) = transforms::value_and_grad(|inner_p: &[Array]| {
            let logits = {
                let mut m = model_c.borrow_mut();
                m.update_parameters(inner_p);
                m.forward(&x_c)? 
            };
            cross_entropy(&logits, &targets_c) 
        }, p)?; 

        let mut out = vec![loss];
        out.extend(grads);
        Ok(out)
    }, false)?;

        bench(&label, 3, 100, || {
            let mut params = model.borrow().parameters_owned();

            // 1. ONE FFI call executes the entire compiled graph!
            let outputs = compiled_grad_step(&params)?;
            let loss = outputs[0].clone();
            let grads = outputs[1..].to_vec();

            // 2. Optimizer step remains eager (which is fast enough on its own)
            optimizer.update(params.iter_mut().collect(), grads)?;
            model.borrow_mut().update_parameters(&params);

            // 3. Force lazy evaluation
            let mut to_eval = params;
            to_eval.push(loss);
            Array::eval_all(&to_eval)?;
            Ok(())
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. CNN Training Step (JIT Compiled)
// ═══════════════════════════════════════════════════════════════════════════

fn bench_cnn_training(key: &Array) -> Result<()> {
    println!("\n══ CNN TRAINING STEP (MNIST-like) ══");

    let batch = 32;
    // Use Rc to share the model into the static JIT closure
    let model = Rc::new(RefCell::new(Sequential::new(vec![
        Box::new(Conv2d::new(1, 16, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
        Box::new(ReLU::new()),
        Box::new(Conv2d::new(16, 32, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
        Box::new(ReLU::new()),
        Box::new(Flatten::new()),
        Box::new(Linear::new(1568, 10, true, key)?),
    ])));

    let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

    let x = Array::random_uniform(&[batch, 28, 28, 1], -1.0, 1.0, Dtype::Float32, key)?;
    let class_range = Array::arange(0.0, 10.0, 1.0, Dtype::Float32)?;
    let raw_labels = Array::random_uniform(&[batch], 0.0, 10.0, Dtype::Float32, key)?
        .cast(Dtype::Int32)?;
    let targets = raw_labels.reshape(&[batch as i32, 1])?
        .equal(&class_range.reshape(&[1, 10])?)?
        .cast(Dtype::Float32)?;

    // --- NEW: JIT Compile the Forward & Backward pass ---
    let x_c = x.clone();
    let targets_c = targets.clone();
    let model_c = Rc::clone(&model);
    let  compiled_grad_step = mlx::compile(move |p: &[Array]| -> Result<Vec<Array>> {
        let (loss, grads) = transforms::value_and_grad(|inner_p: &[Array]| {
            let logits = {
                let mut m = model_c.borrow_mut();
                m.update_parameters(inner_p);
                m.forward(&x_c)? // Use ? instead of .unwrap()
            };
            Ok(cross_entropy(&logits, &targets_c).unwrap()) // Removed .unwrap() to return the Result directly
        }, p)?; // Use ? here as well

        let mut out = vec![loss];
        out.extend(grads);
        Ok(out)
    }, false)?;

    // let mut compiled_grad_step = mlx::compile(move |p: &[Array]| -> Result<Vec<Array>> {
    //     let (loss, grads) = transforms::value_and_grad(|inner_p: &[Array]| {
    //         let logits = {
    //             let mut m = model_c.borrow_mut();
    //             m.update_parameters(inner_p);
    //             m.forward(&x_c).unwrap()
    //         };
    //         cross_entropy(&logits, &targets_c).unwrap()
    //     }, p).unwrap();

    //     let mut out = vec![loss];
    //     out.extend(grads);
    //     Ok(out)
    // }, false)?;

    bench("CNN fwd+bwd+update B=32 [28x28x1 -> 10]", 3, 100, || {
        let mut params = model.borrow().parameters_owned();
        
        // 1. ONE FFI call executes the entire compiled graph!
        let outputs = compiled_grad_step(&params)?;
        let loss = outputs[0].clone();
        let grads = outputs[1..].to_vec();

        // 2. Optimizer update
        optimizer.update(params.iter_mut().collect(), grads)?;
        model.borrow_mut().update_parameters(&params);

        // 3. Force lazy evaluation
        let mut to_eval = params;
        to_eval.push(loss);
        Array::eval_all(&to_eval)?;
        Ok(())
    });

    Ok(())
}

// // ═══════════════════════════════════════════════════════════════════════════
// // 3. MLP Training Step
// // ═══════════════════════════════════════════════════════════════════════════

// fn bench_mlp_training(key: &Array) -> Result<()> {
//     println!("\n══ MLP TRAINING STEP ══");

//     for &(batch, input, hidden, output) in &[
//         (32_usize, 128_usize, 256_usize, 10_usize),
//         (64, 512, 1024, 100),
//         (128, 784, 2048, 10),
//     ] {
//         let model = RefCell::new(Sequential::new(vec![
//             Box::new(Linear::new(input, hidden, true, key)?),
//             Box::new(ReLU::new()),
//             Box::new(Linear::new(hidden, hidden, true, key)?),
//             Box::new(ReLU::new()),
//             Box::new(Linear::new(hidden, output, true, key)?),
//         ]));

//         let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

//         let x = Array::random_uniform(&[batch, input], -1.0, 1.0, Dtype::Float32, key)?;
//         let class_range = Array::arange(0.0, output as f64, 1.0, Dtype::Float32)?;
//         let raw_labels = Array::random_uniform(&[batch], 0.0, output as f32, Dtype::Float32, key)?
//             .cast(Dtype::Int32)?;
//         let targets = raw_labels.reshape(&[batch as i32, 1])?
//             .equal(&class_range.reshape(&[1, output as i32])?)?
//             .cast(Dtype::Float32)?;

//         let label = format!(
//             "MLP fwd+bwd+update B={} [{}->{}->{}->{}]",
//             batch, input, hidden, hidden, output
//         );

//         bench(&label, 3, 100, || {
//             let mut params = model.borrow().parameters_owned();
//             let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
//                 let logits = {
//                     let mut m = model.borrow_mut();
//                     m.update_parameters(p);
//                     m.forward(&x)?
//                 };
//                 cross_entropy(&logits, &targets)
//             }, &params)?;
//             optimizer.update(params.iter_mut().collect(), grads)?;
//             model.borrow_mut().update_parameters(&params);
//             let mut to_eval = params;
//             to_eval.push(loss);
//             Array::eval_all(&to_eval)?;
//             Ok(())
//         });
//     }
//     Ok(())
// }

// // ═══════════════════════════════════════════════════════════════════════════
// // 4. CNN Training Step
// // ═══════════════════════════════════════════════════════════════════════════

// fn bench_cnn_training(key: &Array) -> Result<()> {
//     println!("\n══ CNN TRAINING STEP (MNIST-like) ══");

//     let batch = 32;
//     let model = RefCell::new(Sequential::new(vec![
//         Box::new(Conv2d::new(1, 16, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
//         Box::new(ReLU::new()),
//         Box::new(Conv2d::new(16, 32, [3, 3], [2, 2], [1, 1], [1, 1], 1, true, key)?),
//         Box::new(ReLU::new()),
//         Box::new(Flatten::new()),
//         Box::new(Linear::new(1568, 10, true, key)?),
//     ]));

//     let mut optimizer = Adam::new(1e-3, &model.borrow().parameters_owned())?;

//     let x = Array::random_uniform(&[batch, 28, 28, 1], -1.0, 1.0, Dtype::Float32, key)?;
//     let class_range = Array::arange(0.0, 10.0, 1.0, Dtype::Float32)?;
//     let raw_labels = Array::random_uniform(&[batch], 0.0, 10.0, Dtype::Float32, key)?
//         .cast(Dtype::Int32)?;
//     let targets = raw_labels.reshape(&[batch as i32, 1])?
//         .equal(&class_range.reshape(&[1, 10])?)?
//         .cast(Dtype::Float32)?;

//     bench("CNN fwd+bwd+update B=32 [28x28x1 -> 10]", 3, 100, || {
//         let mut params = model.borrow().parameters_owned();
//         let (loss, grads) = transforms::value_and_grad(|p: &[Array]| {
//             let logits = {
//                 let mut m = model.borrow_mut();
//                 m.update_parameters(p);
//                 m.forward(&x)?
//             };
//             cross_entropy(&logits, &targets)
//         }, &params)?;
//         optimizer.update(params.iter_mut().collect(), grads)?;
//         model.borrow_mut().update_parameters(&params);
//         let mut to_eval = params;
//         to_eval.push(loss);
//         Array::eval_all(&to_eval)?;
//         Ok(())
//     });

//     Ok(())
// }

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
        let encoder = TransformerEncoder::new(n_layers, d_model, n_heads, d_ff, 0.0, key)?;

        let x = Array::random_uniform(
            &[batch, seq_len, d_model], -1.0, 1.0, Dtype::Float32, key,
        )?;

        let label = format!(
            "Encoder fwd B={} S={} d={} h={} L={}",
            batch, seq_len, d_model, n_heads, n_layers
        );

        bench(&label, 3, 50, || {
            let out = encoder.forward(&x)?;
            out.eval()?;
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
    println!("║          mlx-rs Benchmark Suite                  ║");
    println!("║          Device: Apple Silicon GPU               ║");
    println!("║          Build: --release                        ║");
    println!("╚══════════════════════════════════════════════════╝");

    let key = Array::key(42)?;

    bench_matmul(&key)?;
    bench_elementwise(&key)?;
    bench_mlp_training(&key)?;
    bench_cnn_training(&key)?;
    bench_transformer(&key)?;

    println!("\n══ DONE ══");
    println!("Compare against: python bench_mlx.py && python bench_torch.py");

    Ok(())
}