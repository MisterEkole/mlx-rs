# What's New

## `ane_offload` — Apple Neural Engine forward-pass acceleration

Building with `--features ane_offload` routes `Linear::forward` through Apple's Neural Engine instead of MLX's Metal GPU path. Everything else — autograd, optimizer updates, loss — stays on the MLX stack.

---

### Background

Manjeet Singh (maderix) reverse-engineered Apple's ANE and got a 109M-parameter transformer training on it. The project is [maderix/ANE](https://github.com/maderix/ANE). Reading through it raised the question of whether the same approach could be integrated cleanly into mlx-rs. This feature is the result of that investigation.

---

### What it does

Every call to `Linear::forward` now tries to:

1. Materialise the input and weight arrays (flush any pending Metal work)
2. Convert fp32 → fp16 (the ANE operates in fp16)
3. Look up a cached compiled ANE program for this layer's current weights
4. On a cache miss — compile a new ANE program via the Core ML compiler and load it into the ANE daemon
5. Execute via IOSurface I/O on the ANE
6. Convert fp16 → fp32, wrap back into an MLX Array

If anything fails at any step — ANE framework not available, compile budget exhausted, execution error, non-2D input — it falls back silently to the MLX GPU path. The API is identical either way. No error is raised.

---

### The cache design

ANE programs have weights baked at compile time. The cache key is `(layer_id, weight_ptr, batch_size)`. `weight_ptr` is the backing pointer of the weight's MLX handle, which changes exactly once per optimizer step when the optimizer writes `*p = new_array`. The cache misses exactly when the weights change — one recompile per layer per training step, no redundant compiles.

For inference with fixed weights: compile once per layer on the first forward pass, cache hits on every call after that.

---

### The compile budget

The private ANE framework leaks resources and eventually crashes at around 119 compilations per process — this is what maderix documented. The budget limit is set to 108 as a conservative margin. Once the counter hits that, all remaining forward calls fall back to MLX GPU silently. The training loop keeps running.

For a model with N linear layers, you get roughly `108 / N` training steps before hitting the budget. After that you're on GPU for the rest of the run. The feature is most useful for:

- **Inference**: Weights never change — compile once, unlimited cached execution
- **Short fine-tuning runs**: Where total steps fit within the budget
- **Large batch training**: Where the compile cost amortises across enough samples

---

### Architecture

**New crate — `ane-bridge`**

A thin Objective-C layer that loads Apple's private `AppleNeuralEngine.framework` at runtime via `dlopen`, runs the Core ML compiler in-process, and executes compiled programs with IOSurface I/O. The bridge exposes a plain row-major fp16 C interface to Rust.

**New module — `mlx/src/ane/`**

- `convert.rs` — fp32↔fp16 in pure Rust (IEEE 754, round-to-nearest-even)
- `cache.rs` — global thread-safe program cache and compile budget tracking
- `ops.rs` — `try_linear_forward()`, the main dispatch function

**Modified — `mlx/src/nn/layers/linear.rs`**

Added `layer_id: u64` (cfg-gated) and feature-gated routing in `forward()`. The MLX GPU path remains unconditionally as the fallback.

---

### Build and run

```bash
# Normal build — ane_offload is opt-in, nothing changes
cargo build

# Enable ANE offloading
cargo build --features ane_offload

# Per-layer ANE vs GPU benchmark (single linear layer, multiple shapes and batch sizes)
cargo run --example ane_bench --features ane_offload --release

# GPT-2 inference benchmark (full transformer forward pass, warmup + cached timing)
cargo run --example gpt2_ane_bench --features ane_offload --release

# With per-call ANE dispatch logs
RUST_LOG=debug cargo run --example gpt2_ane_bench --features ane_offload --release
```

---

### GPT-2 inference benchmark results

The `gpt2_ane_bench` example runs a GPT-2-style transformer (configurable size) with random weights through four benchmarks:

1. **Warmup pass** — first forward, triggers all ANE compiles
2. **Cached inference** — 50 subsequent passes, all cache hits, zero recompiles
3. **Layer-level ANE vs GPU** — individual linear layers timed on both paths
4. **Throughput at batch sizes 1, 2, 4**

Config used: GPT-2 Tiny class — 4 transformer blocks, d_model=384, d_ff=1536, vocab=1000, 25 total linear layers, batch×seq=32 tokens.

**Warmup vs cached:**

| Pass | Time |
|---|---|
| First forward (25 ANE compiles) | 1002 ms |
| Cached forward — mean (50 runs) | 14.9 ms |
| Cached forward — min | 13.2 ms |
| Warmup amortised after | 68 cached calls |

The warmup cost is a one-time startup expense. After the first forward pass, all 25 layer programs are cached and every subsequent call runs at execution speed only.

**Layer-level ANE vs GPU (batch×seq=32):**

| Layer | Shape | ANE | GPU | Result |
|---|---|---|---|---|
| q_proj | 384→384 | 187 µs | 282 µs | 1.51× faster on ANE |
| fc1 | 384→1536 | 446 µs | 372 µs | 1.20× slower on ANE |
| fc2 | 1536→384 | 450 µs | 370 µs | 1.22× slower on ANE |
| lm_head | 384→1000 | 402 µs | 315 µs | CoreML path (slot 25, beyond 24-slot cap) |

The attention projection runs faster on ANE. The FFN layers run slower because of the CPU round-trip in the current data path — the MLX Metal buffer is read to CPU, converted fp32→fp16, written into an IOSurface, then read back out after ANE execution. On Apple Silicon UMA, the Metal buffer and IOSurface map to the same physical DRAM, so this round-trip is unnecessary. A Metal compute shader for in-GPU conversion will eliminate this.

**End-to-end throughput (seq_len=32 fixed):**

| Batch | Forward time | Tokens/sec |
|---|---|---|
| 1 | 14.4 ms | 2,220 |
| 2 | 18.6 ms | 3,449 |
| 4 | 21.7 ms | 5,911 |

**ANE compile budget used:** 75 of 108 (25 layers × 3 batch sizes). Budget remaining: 33.

---

### Private API notice

This feature uses `_ANEClient`, `_ANEModel`, and `_ANERequest` from Apple's private `AppleNeuralEngine.framework`. These are undocumented and Apple can change them in any OS update. The selector names in `bridge.m` were identified through runtime introspection based on maderix's work. If the ANE path stops working after a macOS update, cross-reference `ane-bridge/src/bridge.m` against [maderix/ANE](https://github.com/maderix/ANE) to check selector names.

A broken ANE path falls back to MLX GPU silently — it will not crash your code.

---

### What's next

**Phase 7** replaces the CPU-side fp32↔fp16 conversion with a Metal compute shader that converts between the MLX Metal buffer and an IOSurface-backed texture directly in GPU memory. On Apple Silicon UMA, both live in the same physical memory pool, making this a zero-copy path. That would make the ANE competitive at smaller batch sizes than it currently is.

**Phase 8** extends offload to `MultiHeadAttention`, fusing Q, K, V projections into a single ANE program to reduce per-call overhead.
