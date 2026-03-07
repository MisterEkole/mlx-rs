// mlx/examples/gpt2_ane_bench.rs
//
// GPT-2-style inference benchmark with ANE offload.
//
// This example builds a GPT-2 Small architecture with random weights and
// measures how the ANE offload feature performs for inference — the use case
// it was actually built for (frozen weights, repeated forward passes).
//
// What this demonstrates:
//   - ANE compile budget consumed once at warmup, then zero recompiles forever
//   - Warmup latency (first forward pass = all ANE compiles happening)
//   - Cached inference latency (subsequent passes = pure cache hits)
//   - Which parts of the model go to ANE vs stay on MLX GPU
//   - ANE vs GPU speedup breakdown for the individual linear layer sizes
//
// Architecture used (GPT-2 Small class):
//   n_layers=12, d_model=768, n_heads=12, d_ff=3072, vocab_size=1000
//
//   Per transformer block:
//     q_proj   768→768   (ANE — reshaped to [B*T, 768] before forward)
//     k_proj   768→768   (ANE)
//     v_proj   768→768   (ANE)
//     o_proj   768→768   (ANE)
//     fc1      768→3072  (ANE)
//     fc2      3072→768  (ANE)
//     softmax  stays on MLX GPU — not a Linear layer
//     LayerNorm stays on MLX GPU — not a Linear layer
//
//   12 blocks × 6 linears = 72 + 1 lm_head = 73 total ANE compiles at warmup.
//   Budget is 108 — safely within limit for this config.
//
// Run with:
//   cargo run --example gpt2_ane_bench --features ane_offload --release
//
// Without --features ane_offload, the ANE path is never compiled in and
// every forward call uses MLX GPU automatically. This lets you run the
// same binary to compare GPU-only vs ANE-offloaded performance.

use mlx::{Array, Dtype, Result, Device, DeviceType};
use mlx::nn::layers::linear::Linear;
use mlx::nn::layers::normalization::LayerNorm;
use mlx::nn::layers::embedding::Embedding;
use mlx::nn::layers::activations::{gelu, softmax};
use mlx::nn::Module;
use std::time::{Duration, Instant};
use env_logger;

// ── Config ────────────────────────────────────────────────────────────────────

struct GptConfig {
    n_layers:   usize,
    d_model:    usize,
    n_heads:    usize,
    d_ff:       usize,
    vocab_size: usize,
}

impl GptConfig {
    // GPT-2 Small class — 12 layers, 768 dims, 12 heads, 3072 FFN
    // 73 ANE compiles at warmup (within the 108-compile budget)
    fn gpt2_small() -> Self {
        Self { n_layers: 12, d_model: 768, n_heads: 12, d_ff: 3072, vocab_size: 1000 }
    }

    // Smaller variant for faster iteration during development / low-budget testing
    // 25 ANE compiles at warmup
    fn gpt2_tiny() -> Self {
        Self { n_layers: 4, d_model: 384, n_heads: 6, d_ff: 1536, vocab_size: 1000 }
    }

    fn linears_per_block(&self) -> usize { 6 } // q, k, v, o, fc1, fc2
    fn total_linears(&self) -> usize { self.linears_per_block() * self.n_layers + 1 }
}

// ── GPU baseline helper ───────────────────────────────────────────────────────
//
// Calls the raw matmul directly, bypassing Linear::forward() entirely.
// This gives us a pure MLX GPU number regardless of whether ane_offload is on.

fn gpu_linear(layer: &Linear, x: &Array) -> Result<Array> {
    let wt = layer.weight.transpose(&[])?;
    let mut out = x.matmul(&wt)?;
    if let Some(ref b) = layer.bias {
        out = out.add(b)?;
    }
    Ok(out)
}

// ── Timing helpers ────────────────────────────────────────────────────────────

fn mean_us(durations: &[Duration]) -> f64 {
    durations.iter().map(|d| d.as_secs_f64() * 1e6).sum::<f64>() / durations.len() as f64
}

fn min_us(durations: &[Duration]) -> f64 {
    durations.iter().map(|d| d.as_secs_f64() * 1e6).fold(f64::INFINITY, f64::min)
}

// ── MLP (FFN) block ───────────────────────────────────────────────────────────

struct MlpBlock {
    fc1: Linear,   // d_model → d_ff
    fc2: Linear,   // d_ff → d_model
}

impl MlpBlock {
    fn new(d_model: usize, d_ff: usize, key: &Array) -> Result<Self> {
        let (k1, k2) = key.split()?;
        Ok(Self {
            fc1: Linear::new(d_model, d_ff,    true, &k1)?,
            fc2: Linear::new(d_ff,    d_model, true, &k2)?,
        })
    }

    // x: [B, T, D]
    // Flattens to [B*T, D] before each Linear so the ANE path is eligible.
    // The ANE only handles 2D inputs — this reshape is the critical step.
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?;
        let (b, t, d) = (shape[0], shape[1], shape[2]);

        let x_2d = x.reshape(&[b as i32 * t as i32, d as i32])?;

        // fc1: [B*T, d_model] → [B*T, d_ff]   — ANE eligible
        let h = self.fc1.forward(&x_2d)?;
        // gelu: element-wise — stays on MLX GPU
        let h = gelu(&h)?;
        // fc2: [B*T, d_ff] → [B*T, d_model]   — ANE eligible
        let out = self.fc2.forward(&h)?;

        out.reshape(&[b as i32, t as i32, d as i32])
    }
}

// ── Attention block ───────────────────────────────────────────────────────────

struct Attention {
    q_proj: Linear,   // d_model → d_model
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads:  usize,
    head_dim: usize,
}

impl Attention {
    fn new(d_model: usize, n_heads: usize, key: &Array) -> Result<Self> {
        let (k1, rest)  = key.split()?;
        let (k2, rest)  = rest.split()?;
        let (k3, k4)    = rest.split()?;
        let head_dim    = d_model / n_heads;
        Ok(Self {
            q_proj: Linear::new(d_model, d_model, true, &k1)?,
            k_proj: Linear::new(d_model, d_model, true, &k2)?,
            v_proj: Linear::new(d_model, d_model, true, &k3)?,
            o_proj: Linear::new(d_model, d_model, true, &k4)?,
            n_heads,
            head_dim,
        })
    }

    // x: [B, T, D]
    // The four Linear projections (q/k/v/o) each see a [B*T, D] input
    // and are ANE-eligible. The matmul for attention scores and the softmax
    // stay on MLX GPU.
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape()?;
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let (nh, hd) = (self.n_heads, self.head_dim);

        // Flatten → [B*T, D] for ANE-eligible projections
        let x_2d = x.reshape(&[b as i32 * t as i32, d as i32])?;

        let q = self.q_proj.forward(&x_2d)?;   // [B*T, D] — ANE
        let k = self.k_proj.forward(&x_2d)?;   // [B*T, D] — ANE
        let v = self.v_proj.forward(&x_2d)?;   // [B*T, D] — ANE

        // Reshape to [B, T, n_heads, head_dim] → transpose to [B, n_heads, T, head_dim]
        let q = q.reshape(&[b as i32, t as i32, nh as i32, hd as i32])?;
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b as i32, t as i32, nh as i32, hd as i32])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b as i32, t as i32, nh as i32, hd as i32])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        // Attention scores [B, n_heads, T, T] — MLX GPU matmul
        let scale = (hd as f32).sqrt().recip();
        let k_t   = k.transpose_axes(&[0, 1, 3, 2])?;
        let scores = q.matmul(&k_t)?.multiply_scalar(scale)?;
        // Softmax — MLX GPU
        let attn = softmax(&scores, -1)?;

        // Context [B, n_heads, T, head_dim] → [B, T, D]
        let ctx    = attn.matmul(&v)?;
        let ctx    = ctx.transpose_axes(&[0, 2, 1, 3])?;
        let ctx_2d = ctx.reshape(&[b as i32 * t as i32, d as i32])?;

        // Output projection — ANE eligible
        let out = self.o_proj.forward(&ctx_2d)?;
        out.reshape(&[b as i32, t as i32, d as i32])
    }
}

// ── Transformer block ─────────────────────────────────────────────────────────

struct TransformerBlock {
    ln1:  LayerNorm,
    attn: Attention,
    ln2:  LayerNorm,
    mlp:  MlpBlock,
}

impl TransformerBlock {
    fn new(cfg: &GptConfig, key: &Array) -> Result<Self> {
        let (k_attn, k_mlp) = key.split()?;
        Ok(Self {
            ln1:  LayerNorm::new(cfg.d_model, 1e-5)?,
            attn: Attention::new(cfg.d_model, cfg.n_heads, &k_attn)?,
            ln2:  LayerNorm::new(cfg.d_model, 1e-5)?,
            mlp:  MlpBlock::new(cfg.d_model, cfg.d_ff, &k_mlp)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        // Pre-norm attention + residual
        let h = self.ln1.forward(x)?;
        let h = self.attn.forward(&h)?;
        let x = x.add(&h)?;

        // Pre-norm FFN + residual
        let h = self.ln2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        x.add(&h)
    }
}

// ── MiniGPT2 ──────────────────────────────────────────────────────────────────

struct MiniGPT2 {
    tok_emb: Embedding,
    pos_emb: Embedding,
    blocks:  Vec<TransformerBlock>,
    ln_f:    LayerNorm,
    lm_head: Linear,
    cfg:     GptConfig,
}

impl MiniGPT2 {
    fn new(cfg: GptConfig, key: &Array) -> Result<Self> {
        let (k_tok,  rest) = key.split()?;
        let (k_pos,  rest) = rest.split()?;

        let tok_emb = Embedding::new(cfg.vocab_size, cfg.d_model, None, &k_tok)?;
        // Position embedding table: 1024 positions max
        let pos_emb = Embedding::new(1024, cfg.d_model, None, &k_pos)?;

        let mut blocks    = Vec::with_capacity(cfg.n_layers);
        let mut block_key = rest;
        for _ in 0..cfg.n_layers {
            let (kb, next) = block_key.split()?;
            blocks.push(TransformerBlock::new(&cfg, &kb)?);
            block_key = next;
        }

        let (k_lm, _) = block_key.split()?;
        let ln_f    = LayerNorm::new(cfg.d_model, 1e-5)?;
        let lm_head = Linear::new(cfg.d_model, cfg.vocab_size, false, &k_lm)?;

        Ok(Self { tok_emb, pos_emb, blocks, ln_f, lm_head, cfg })
    }

    // input_ids: [B, T]  integer token indices
    // Returns logits: [B, T, vocab_size]
    fn forward(&self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape()?;
        let (b, t) = (shape[0], shape[1]);

        // Token + position embeddings  → [B, T, D]
        let tok = self.tok_emb.forward(input_ids)?;
        let pos_ids = Array::arange(0.0, t as f64, 1.0, Dtype::Int32)?;
        let pos = self.pos_emb.forward(&pos_ids)?;    // [T, D]
        let mut x = tok.add(&pos)?;                   // [B, T, D]  (broadcast)

        // Transformer blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final norm + LM head
        let x = self.ln_f.forward(&x)?;
        // Flatten to 2D for lm_head — ANE eligible
        let x_2d     = x.reshape(&[b as i32 * t as i32, self.cfg.d_model as i32])?;
        let logits_2d = self.lm_head.forward(&x_2d)?;
        logits_2d.reshape(&[b as i32, t as i32, self.cfg.vocab_size as i32])
    }
}

// ── Benchmark helpers ─────────────────────────────────────────────────────────

fn bench_forward(model: &MiniGPT2, input_ids: &Array, n: usize) -> Result<Vec<Duration>> {
    let mut times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0  = Instant::now();
        let out = model.forward(input_ids)?;
        out.eval()?;
        times.push(t0.elapsed());
    }
    Ok(times)
}

// Benchmark a single linear layer — ANE path vs GPU path — for a given input
fn bench_single_linear(
    layer:   &Linear,
    x:       &Array,
    n:       usize,
    label:   &str,
) -> Result<()> {
    // Warmup (1 call — may trigger ANE compile for this layer+batch combo)
    let _ = layer.forward(x)?.eval();
    let _ = gpu_linear(layer, x)?.eval();

    // ANE path (via forward() — routes through ANE when feature is active)
    let mut ane_times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0  = Instant::now();
        let out = layer.forward(x)?;
        out.eval()?;
        ane_times.push(t0.elapsed());
    }

    // GPU path (bypasses ANE routing entirely)
    let mut gpu_times = Vec::with_capacity(n);
    for _ in 0..n {
        let t0  = Instant::now();
        let out = gpu_linear(layer, x)?;
        out.eval()?;
        gpu_times.push(t0.elapsed());
    }

    let ane_mean = mean_us(&ane_times);
    let gpu_mean = mean_us(&gpu_times);
    let speedup  = gpu_mean / ane_mean;

    let direction = if speedup > 1.0 {
        format!("{:.2}× faster than GPU", speedup)
    } else {
        format!("{:.2}× slower than GPU", 1.0 / speedup)
    };

    println!(
        "  {:<32}  ANE: {:>7.1} µs (min {:>6.1})   GPU: {:>7.1} µs (min {:>6.1})   {}",
        label,
        ane_mean, min_us(&ane_times),
        gpu_mean, min_us(&gpu_times),
        direction,
    );
    Ok(())
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    env_logger::init();

    let gpu = Device::new(DeviceType::Gpu);
    gpu.set_default()?;

    let key = Array::key(0x6770_7432)?; // 0x67707432 = "gpt2" in ascii

    // ── Header ────────────────────────────────────────────────────────────────

    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║   mlx-rs  ane_offload  —  GPT-2 Inference Prototype Benchmark   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ── ANE status ────────────────────────────────────────────────────────────

    #[cfg(feature = "ane_offload")]
    {
        let status = mlx::ane::status();
        if status.available {
            println!(
                "  ANE:  available   compile budget: {}/{} remaining",
                status.budget_left, mlx::ane::cache::ANE_BUDGET_LIMIT
            );
        } else {
            println!("  ANE:  NOT available on this system");
            println!("        Linear layers will use the CoreML fallback path.");
        }
    }
    #[cfg(not(feature = "ane_offload"))]
    println!("  ANE:  feature not compiled in — all compute on MLX GPU");

    println!();

    // ── Model config ──────────────────────────────────────────────────────────

    // Use gpt2_tiny (4 layers, 25 compiles) for quick iteration.
    // Switch to gpt2_small() to test the full 73-compile warmup cost.
    let cfg = GptConfig::gpt2_tiny();
    println!("  Model:   GPT-2 Tiny class (use gpt2_small() for full model)");
    println!("  Config:  n_layers={} d_model={} n_heads={} d_ff={} vocab={}",
        cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.vocab_size);
    println!("  Linear layers total: {} ({} per block × {} blocks + 1 lm_head)",
        cfg.total_linears(), cfg.linears_per_block(), cfg.n_layers);
    println!("  ANE eligible linears: all {} (reshaped to 2D [B*T, D] before each)",
        cfg.total_linears());
    println!();
    println!("  ANE compiles needed at warmup: {}  (budget: {})",
        cfg.total_linears(), 108);
    println!();

    // ── Build model ───────────────────────────────────────────────────────────

    print!("  Building model with random weights... ");
    let t_build = Instant::now();
    let model = MiniGPT2::new(cfg, &key)?;
    println!("{:.1} ms", t_build.elapsed().as_secs_f64() * 1e3);

    // ── Input ─────────────────────────────────────────────────────────────────

    // Simulate a single-sample inference request with a sequence of 32 tokens.
    // This is the realistic use case: batch=1, moderate context length.
    let batch_size = 1usize;
    let seq_len    = 32usize;

    // Random token ids in [0, vocab_size)
    let token_data: Vec<i32> = (0..batch_size * seq_len)
        .map(|i| (i % model.cfg.vocab_size) as i32)
        .collect();
    let input_ids = Array::from_slice(&token_data, &[batch_size, seq_len], Dtype::Int32)?;
    input_ids.eval()?;

    println!();
    println!("  Input:   batch={} seq_len={}  (total tokens in flight: {})",
        batch_size, seq_len, batch_size * seq_len);
    println!();

    // ── Pre-eval weights ──────────────────────────────────────────────────────

    // Flush all lazy weight creation to GPU before timing starts.
    // Without this, the first forward would include weight materialisation time.
    for block in &model.blocks {
        block.attn.q_proj.weight.eval()?;
        block.attn.k_proj.weight.eval()?;
        block.attn.v_proj.weight.eval()?;
        block.attn.o_proj.weight.eval()?;
        block.mlp.fc1.weight.eval()?;
        block.mlp.fc2.weight.eval()?;
    }
    model.lm_head.weight.eval()?;
    model.tok_emb.weight.eval()?;
    model.pos_emb.weight.eval()?;

    // ── Warmup pass (first forward — triggers all ANE compiles) ──────────────

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Benchmark 1 — Warmup pass (first forward through the model)");
    println!("─────────────────────────────────────────────────────────────────────");
    println!();
    println!("  With ane_offload active, this triggers all {} ANE compiles.", model.cfg.total_linears());
    println!("  Compile time dominates — this is a one-time startup cost.");
    println!();

    let t_warmup = Instant::now();
    let warmup_out = model.forward(&input_ids)?;
    warmup_out.eval()?;
    let warmup_elapsed = t_warmup.elapsed();

    println!("  First forward (warmup):  {:.1} ms  ({:.1} µs)",
        warmup_elapsed.as_secs_f64() * 1e3,
        warmup_elapsed.as_secs_f64() * 1e6);

    #[cfg(feature = "ane_offload")]
    {
        let after_warmup = mlx::ane::status();
        println!("  ANE compiles consumed:   {}  (budget remaining: {})",
            after_warmup.compile_count,
            after_warmup.budget_left);
    }
    println!();

    // ── Cached inference passes ───────────────────────────────────────────────

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Benchmark 2 — Cached inference passes (all cache hits)");
    println!("─────────────────────────────────────────────────────────────────────");
    println!();
    println!("  Zero recompiles from here — every Linear call is a cache hit.");
    println!();

    const N_CACHED: usize = 50;
    let cached_times = bench_forward(&model, &input_ids, N_CACHED)?;
    let cached_mean  = mean_us(&cached_times);
    let cached_min   = min_us(&cached_times);

    println!("  Cached forward ({} runs):", N_CACHED);
    println!("    mean:  {:>8.1} µs  ({:.1} ms)", cached_mean, cached_mean / 1000.0);
    println!("    min:   {:>8.1} µs  ({:.1} ms)", cached_min,  cached_min  / 1000.0);
    println!();

    let warmup_us = warmup_elapsed.as_secs_f64() * 1e6;
    println!("  Warmup vs cached:");
    println!("    Warmup overhead: {:.1}× of a single cached forward",
        warmup_us / cached_mean);
    println!("    Breakeven at {} cached calls: warmup cost is amortised",
        (warmup_us / cached_mean).ceil() as usize);
    println!();

    // ── Individual linear layers: ANE vs GPU ──────────────────────────────────

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Benchmark 3 — Individual linear layers: ANE vs GPU");
    println!("─────────────────────────────────────────────────────────────────────");
    println!();
    println!("  Input to each layer: [B*T, features] = [{}, features]",
        batch_size * seq_len);
    println!("  This is the exact shape the layers see inside the model.");
    println!();

    let tokens = batch_size * seq_len;
    const N_LINEAR: usize = 100;

    // Attention projection shape: [B*T, d_model] → [B*T, d_model]
    let x_attn = Array::random_uniform(
        &[tokens, model.blocks[0].attn.q_proj.in_features],
        -1.0, 1.0, Dtype::Float32, &key,
    )?;
    x_attn.eval()?;

    // FFN fc1 shape: [B*T, d_model] → [B*T, d_ff]
    let x_ffn = Array::random_uniform(
        &[tokens, model.blocks[0].mlp.fc1.in_features],
        -1.0, 1.0, Dtype::Float32, &key,
    )?;
    x_ffn.eval()?;

    // FFN fc2 shape: [B*T, d_ff] → [B*T, d_model]
    let x_ffn2 = Array::random_uniform(
        &[tokens, model.blocks[0].mlp.fc2.in_features],
        -1.0, 1.0, Dtype::Float32, &key,
    )?;
    x_ffn2.eval()?;

    println!("  {:<32}  {:>38}   {:>38}   Verdict",
        "Layer", "ANE path (forward())", "GPU path (raw matmul)");
    println!("  {}", "─".repeat(130));

    bench_single_linear(
        &model.blocks[0].attn.q_proj, &x_attn, N_LINEAR,
        &format!("q_proj  [{:>4}→{:>4}]  B*T={}", 768, 768, tokens),
    )?;
    bench_single_linear(
        &model.blocks[0].mlp.fc1, &x_ffn, N_LINEAR,
        &format!("fc1     [{:>4}→{:>4}]  B*T={}", 768, 3072, tokens),
    )?;
    bench_single_linear(
        &model.blocks[0].mlp.fc2, &x_ffn2, N_LINEAR,
        &format!("fc2     [{:>4}→{:>4}]  B*T={}", 3072, 768, tokens),
    )?;
    bench_single_linear(
        &model.lm_head, &x_attn, N_LINEAR,
        &format!("lm_head [{:>4}→{:>4}]  B*T={}", 768, model.cfg.vocab_size, tokens),
    )?;
    println!();

    // ── End-to-end throughput vs batch size ───────────────────────────────────

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Benchmark 4 — End-to-end throughput vs batch size");
    println!("─────────────────────────────────────────────────────────────────────");
    println!();
    println!("  Note: changing batch size costs 1 ANE compile per linear layer.");
    println!("  After that first call, it is a cache hit at that batch size.");
    println!();
    println!("  seq_len fixed at {}", seq_len);
    println!();
    println!("  {:<10}  {:<15}  {:<15}  {:<15}",
        "batch", "forward (ms)", "tokens/sec", "note");
    println!("  {}", "─".repeat(60));

    for &b in &[1usize, 2, 4] {
        let td: Vec<i32> = (0..b * seq_len).map(|i| (i % 1000) as i32).collect();
        let ids = Array::from_slice(&td, &[b, seq_len], Dtype::Int32)?;
        ids.eval()?;

        // One warmup call for this batch size (may or may not compile)
        let _ = model.forward(&ids)?.eval();

        // Timed runs
        let times = bench_forward(&model, &ids, 20)?;
        let mean_ms = mean_us(&times) / 1000.0;
        let tps = (b * seq_len) as f64 / (mean_us(&times) / 1e6);

        let note = if b == 1 { "single-sample autoregressive" } else { "batched" };
        println!("  {:<10}  {:<15.2}  {:<15.0}  {}", b, mean_ms, tps, note);
    }
    println!();

    // ── Final budget report ───────────────────────────────────────────────────

    println!("─────────────────────────────────────────────────────────────────────");
    println!("  Final ANE compile budget report");
    println!("─────────────────────────────────────────────────────────────────────");
    println!();

    #[cfg(feature = "ane_offload")]
    {
        let final_status = mlx::ane::status();
        println!("  Budget limit:     {}", mlx::ane::cache::ANE_BUDGET_LIMIT);
        println!("  Compiles used:    {}", final_status.compile_count);
        println!("  Budget remaining: {}", final_status.budget_left);
        println!();
        println!("  Inference conclusion:");
        if final_status.compile_count <= model.cfg.total_linears() + 10 {
            println!("  Compile budget is sustainable for this model at 3+ batch sizes.");
            println!("  All subsequent process runs with the same batch size will consume");
            println!("  0 additional compile budget units — inference is cache-hit-only.");
        }
    }
    #[cfg(not(feature = "ane_offload"))]
    {
        println!("  ANE feature not active — all compute used MLX GPU.");
        println!("  Re-run with --features ane_offload to see ANE numbers.");
    }

    println!();
    println!("  Run with RUST_LOG=debug to see per-call ANE dispatch logs.");
    println!();

    Ok(())
}
