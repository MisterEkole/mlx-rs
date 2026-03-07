// mlx/examples/ane_bench.rs
//
// Benchmark: ANE offload vs MLX GPU for Linear::forward
//
// Run with:
//   cargo run --example ane_bench --features ane_offload --release
//
// What this measures:
//   For each (batch_size, in_features, out_features) configuration:
//     1. MLX GPU path  — raw matmul bypassing the ANE routing
//     2. ANE path      — try_linear_forward() directly; reports if unavailable
//     3. Speedup       — GPU mean / ANE mean
//     4. Correctness   — max absolute diff between GPU and ANE outputs
//
//   Additionally:
//     - Compile overhead: first ANE call (includes MIL compile + link)
//                         vs subsequent calls (cache hit, execution only)
//     - Compile budget:   how many compiles remain in the 108-compile window
//
// Methodology:
//   - WARMUP_ITERS iterations discarded before timing begins
//   - BENCH_ITERS iterations timed; mean and min reported
//   - Both paths eval() to flush lazy work before timing
//   - Timings use std::time::Instant (wall clock, not CPU time)
//
// Interpretation guide:
//   If ANE mean < GPU mean:    ANE is faster at this batch size
//   If ANE mean ≈ GPU mean:    conversion overhead is matching the savings
//   If ANE mean > GPU mean:    batch too small; fp16 conversion dominates
//   If ANE = unavailable:      check bridge.m selector names against maderix source

use mlx::{Array, Dtype, Result, Device, DeviceType};
use mlx::nn::layers::linear::Linear;
use mlx::ane;
use std::time::{Duration, Instant};

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS:  usize = 100;

// ── Benchmark configurations ──────────────────────────────────────────────────

struct Config {
    batch:    usize,
    in_feat:  usize,
    out_feat: usize,
    label:    &'static str,
}

fn configs() -> Vec<Config> {
    vec![
        Config { batch: 1,    in_feat: 512,  out_feat: 512,  label: "tiny  (B=1,   512→512)"   },
        Config { batch: 8,    in_feat: 512,  out_feat: 512,  label: "small (B=8,   512→512)"   },
        Config { batch: 32,   in_feat: 512,  out_feat: 2048, label: "med   (B=32,  512→2048)"  },
        Config { batch: 64,   in_feat: 768,  out_feat: 3072, label: "large (B=64,  768→3072)"  },
        Config { batch: 128,  in_feat: 1024, out_feat: 4096, label: "xl    (B=128, 1024→4096)" },
        Config { batch: 256,  in_feat: 1024, out_feat: 4096, label: "xxl   (B=256, 1024→4096)" },
    ]
}

// ── Timing helpers ────────────────────────────────────────────────────────────

struct Timing {
    mean_us: f64,
    min_us:  f64,
    #[allow(dead_code)]
    max_us:  f64,
}

impl Timing {
    fn from_durations(durations: &[Duration]) -> Self {
        let us: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1e6).collect();
        let mean = us.iter().sum::<f64>() / us.len() as f64;
        let min  = us.iter().cloned().fold(f64::INFINITY, f64::min);
        let max  = us.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self { mean_us: mean, min_us: min, max_us: max }
    }
}

// ── GPU baseline — raw matmul bypassing ANE routing ───────────────────────────
//
// We call the matmul directly instead of layer.forward() so the ANE routing
// inside forward() doesn't intercept it. This gives us a pure MLX GPU number
// to compare against even when the ane_offload feature is active.

fn gpu_forward(layer: &Linear, x: &Array) -> Result<Array> {
    let weight_t = layer.weight.transpose(&[])?;
    let mut out  = x.matmul(&weight_t)?;
    if let Some(ref b) = layer.bias {
        out = out.add(b)?;
    }
    Ok(out)
}

fn bench_gpu(layer: &Linear, x: &Array) -> Result<(Timing, Vec<f32>)> {
    // Warmup
    for _ in 0..WARMUP_ITERS {
        let out = gpu_forward(layer, x)?;
        out.eval()?;
    }

    // Benchmark
    let mut durations = Vec::with_capacity(BENCH_ITERS);
    let mut last_out  = Vec::new();

    for i in 0..BENCH_ITERS {
        let t0  = Instant::now();
        let out = gpu_forward(layer, x)?;
        out.eval()?;
        durations.push(t0.elapsed());

        if i == BENCH_ITERS - 1 {
            last_out = out.to_vec_f32()?;
        }
    }

    Ok((Timing::from_durations(&durations), last_out))
}

// ── ANE path — calls try_linear_forward() directly ───────────────────────────

fn bench_ane(layer: &Linear, x: &Array) -> Result<Option<(Duration, Timing, Vec<f32>)>> {
    // First call — includes MIL compile overhead
    let compile_t0 = Instant::now();
    let first_result = ane::try_linear_forward(layer, x)?;
    let compile_elapsed = compile_t0.elapsed();

    let first_out = match first_result {
        Some(arr) => { arr.eval()?; arr }
        None      => return Ok(None),  // ANE not available on this system
    };
    let _first_out_data = first_out.to_vec_f32()?;

    // Warmup with cache already warm (minus 1 since we just did the first call)
    for _ in 0..(WARMUP_ITERS - 1) {
        let out = ane::try_linear_forward(layer, x)?;
        if let Some(arr) = out { arr.eval()?; }
    }

    // Benchmark — all cache hits from here
    let mut durations = Vec::with_capacity(BENCH_ITERS);
    let mut last_out  = Vec::new();

    for i in 0..BENCH_ITERS {
        let t0  = Instant::now();
        let out = ane::try_linear_forward(layer, x)?;
        if let Some(arr) = out {
            arr.eval()?;
            durations.push(t0.elapsed());
            if i == BENCH_ITERS - 1 {
                last_out = arr.to_vec_f32()?;
            }
        } else {
            // ANE fell back mid-benchmark (budget hit?)
            break;
        }
    }

    if durations.is_empty() {
        return Ok(None);
    }

    Ok(Some((compile_elapsed, Timing::from_durations(&durations), last_out)))
}

// ── Correctness check — max absolute difference ───────────────────────────────

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
     .map(|(x, y)| (x - y).abs())
     .fold(0.0_f32, f32::max)
}

// ── Printer ───────────────────────────────────────────────────────────────────

fn print_row(label: &str, gpu: &Timing, ane_result: &Option<(Duration, Timing, Vec<f32>, f32)>) {
    let gpu_str = format!("{:>7.1} µs (min {:>6.1})", gpu.mean_us, gpu.min_us);

    match ane_result {
        None => {
            println!(
                "  {:<38}  GPU: {}   ANE: unavailable (fallback to GPU)",
                label, gpu_str
            );
        }
        Some((compile_dur, ane_timing, _, max_diff)) => {
            let speedup = gpu.mean_us / ane_timing.mean_us;
            let faster_label = if speedup > 1.0 {
                format!("{:.2}× faster", speedup)
            } else {
                format!("{:.2}× slower", 1.0 / speedup)
            };

            println!(
                "  {:<38}  GPU: {}   ANE: {:>7.1} µs (min {:>6.1})   {}   Δ {:.5}   compile: {:.1} ms",
                label,
                gpu_str,
                ane_timing.mean_us,
                ane_timing.min_us,
                faster_label,
                max_diff,
                compile_dur.as_secs_f64() * 1e3,
            );
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    // Set GPU as default so MLX baseline uses Metal
    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║         mlx-rs  ane_offload  Benchmark                          ║");
    println!("║         ANE forward pass  vs  MLX Metal GPU                     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ANE availability check
    let status = ane::status();
    if status.available {
        println!("  ANE:  available   compile budget: {}/{} remaining",
            status.budget_left, mlx::ane::cache::ANE_BUDGET_LIMIT);
    } else {
        println!("  ANE:  NOT available on this system");
        println!("        (AppleNeuralEngine.framework not loaded — check bridge.m)");
        println!("        Benchmarks will still run; ANE column will show 'unavailable'");
    }

    // Probe compile path before the main loop — detect selector mismatches early.
    // If the framework loaded but a test compile immediately falls back, run the
    // full runtime diagnostic so we can see the actual private method names.
    if status.available {
        let probe_key  = Array::key(0xDEAD)?;
        let probe_layer = Linear::new(4, 4, false, &probe_key)?;
        let probe_x    = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, 4], Dtype::Float32)?;
        let probe_result = ane::try_linear_forward(&probe_layer, &probe_x)?;
        if probe_result.is_none() && ane::status().compile_count == 0 {
            println!();
            println!("  WARNING: framework loaded but compile returned NULL.");
            println!("           Running runtime diagnostic to find the real selector names...");
            println!("           (Fix bridge.m using the method list printed below, then rebuild)");
            println!();
            unsafe { ane_bridge::ane_diagnose(); }
        }
    }
    println!();
    println!("  Warmup: {} iters   Benchmark: {} iters", WARMUP_ITERS, BENCH_ITERS);
    println!("  Δ = max absolute difference between GPU and ANE outputs");
    println!();

    let key = Array::key(0xA_E_BE)?;  // seed: 0xAEBE ≈ "ANE bench"

    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!("  Measuring linear forward pass (y = x @ W^T + b, both with bias)");
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!();

    for cfg in configs() {
        // Build layer for this config
        let layer = Linear::new(cfg.in_feat, cfg.out_feat, true, &key)?;
        let x = Array::random_uniform(
            &[cfg.batch, cfg.in_feat],
            -1.0, 1.0,
            Dtype::Float32,
            &key,
        )?;
        x.eval()?;
        layer.weight.eval()?;
        layer.bias.as_ref().map(|b| b.eval()).transpose()?;

        // GPU baseline
        let (gpu_timing, gpu_out) = bench_gpu(&layer, &x)?;

        // ANE path
        let ane_result: Option<(Duration, Timing, Vec<f32>, f32)> =
            match bench_ane(&layer, &x)? {
                None => None,
                Some((compile_dur, ane_timing, ane_out)) => {
                    let diff = max_abs_diff(&gpu_out, &ane_out);
                    Some((compile_dur, ane_timing, ane_out, diff))
                }
            };

        print_row(cfg.label, &gpu_timing, &ane_result);
    }

    println!();
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!("  Compile overhead breakdown (first call vs cached)");
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");
    println!();

    // Dedicated compile overhead measurement — one fresh layer
    let layer_co = Linear::new(512, 2048, true, &key)?;
    let x_co = Array::random_uniform(&[32, 512], -1.0, 1.0, Dtype::Float32, &key)?;
    x_co.eval()?;

    // First call (compile + execute)
    let t_cold = Instant::now();
    let cold_result = ane::try_linear_forward(&layer_co, &x_co)?;
    let cold_elapsed = t_cold.elapsed();

    if cold_result.is_some() {
        // Subsequent calls (cache hit, execute only)
        let mut cached_times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t = Instant::now();
            let r = ane::try_linear_forward(&layer_co, &x_co)?;
            if r.is_some() { cached_times.push(t.elapsed()); }
        }

        let cached_mean_us = cached_times.iter()
            .map(|d| d.as_secs_f64() * 1e6)
            .sum::<f64>() / cached_times.len() as f64;

        let compile_us = cold_elapsed.as_secs_f64() * 1e6 - cached_mean_us;

        println!("  Layer: 512→2048, batch=32");
        println!("  Cold call (compile + execute): {:.1} µs  ({:.1} ms)",
            cold_elapsed.as_secs_f64() * 1e6,
            cold_elapsed.as_secs_f64() * 1e3);
        println!("  Cached call (execute only):    {:.1} µs  (mean over 20 calls)", cached_mean_us);
        println!("  Estimated compile overhead:    {:.1} µs  ({:.2}× of cached execution)",
            compile_us, compile_us / cached_mean_us);
    } else {
        println!("  ANE not available — skipping compile overhead breakdown.");
    }

    println!();
    println!("─────────────────────────────────────────────────────────────────────────────────────────────────────");

    // Final budget report
    let final_status = ane::status();
    println!();
    println!("  Compile budget used:     {}  /  {}",
        final_status.compile_count,
        mlx::ane::cache::ANE_BUDGET_LIMIT);
    println!("  Compile budget remaining: {}", final_status.budget_left);
    println!();
    println!("  Run with RUST_LOG=debug for verbose ANE dispatch logging.");
    println!();

    Ok(())
}
