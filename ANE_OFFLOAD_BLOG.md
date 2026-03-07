# Routing Neural Networks Through Apple's Neural Engine in Rust: How I Built ane_offload into mlx-rs

This post walks through how I added an ANE execution path to mlx-rs — a Rust wrapper around Apple's MLX framework — what the implementation actually looks like at the systems level, and where it currently falls short. The feature is called `ane_offload` and it is a Cargo feature flag, not something enabled by default.

---

## Background: Why Add an ANE Path

mlx-rs targets the GPU through MLX, which uses Metal internally. The GPU path works and is already fast on Apple Silicon. The motivation for an ANE path is not that the GPU is inadequate — it's that the ANE is a dedicated matrix engine with a fundamentally different compute profile.

The ANE does one thing: matrix multiplication. No branching, no arbitrary shader programs, just `y = xW + b` mapped onto fixed silicon. For transformer inference — repeated forward passes through frozen weight matrices — that single-purpose design is a good fit. The question worth investigating is whether you can actually route those operations to ANE hardware from application code, and at what cost.

The short answer is yes, you can, but the path is entirely through private APIs and involves more moving pieces than it looks like from the outside.

---

## What the Access Model Looks Like

There is no public API for talking to the ANE directly. Application code cannot write a Metal shader for the ANE, cannot allocate memory that the ANE reads, and has no visibility into whether a Core ML prediction ran on ANE versus CPU versus GPU.

The public path into the ANE is Core ML: compile a `.mlmodel`, load it with `MLModel`, call `predictionFromFeatures:`, and the framework's internal scheduler routes compatible operations to the ANE. You get the result back but you do not control the execution path.

The private path is through `AppleNeuralEngine.framework`, which exposes `_ANEClient`, `_ANEModel`, and `_ANERequest` — Objective-C classes that talk directly to the ANE daemon (`aned`). Core ML uses these internally. They are not documented, not ABI-stable, and can change between macOS releases. The reverse engineering work that identified the relevant selectors and established the execution path was done by Manjeet Singh (maderix) in the [ANE project](https://github.com/maderix/ANE). The implementation here builds on that foundation.

---

## The Execution Pipeline

Before getting into the mlx-rs specifics, it helps to understand what the ANE execution path looks like end-to-end. There are more steps than you would expect.

### Getting a Matrix Multiply onto the ANE

The ANE does not execute arbitrary computation graphs fed to it at runtime. It executes compiled ANE programs — binary blobs produced by Apple's toolchain, stored in a compiled `.mlmodelc` bundle. The format is not public, but the compiler is: `[MLModel compileModelAtURL:error:]` takes a `.mlmodel` binary protobuf and produces the compiled bundle. Core ML's compiler routes `innerProduct` operations to the ANE on Apple Silicon hardware.

The full pipeline for a single linear layer is:

1. Encode a Core ML binary protobuf in memory describing an `InnerProduct` layer with fp16 weights embedded
2. Write it to a temp `.mlmodel` file on disk
3. Call `MLModel.compileModelAtURL:` to produce a `.mlmodelc` bundle
4. Load the bundle with `_ANEModel.modelAtURL:key:` (private)
5. Load the model into the ANE daemon with `_ANEClient.loadModel:options:qos:error:` (private)
6. Create two IOSurfaces for input and output data via the C API
7. Build an `_ANERequest` that binds those surfaces to the model's input/output symbol indices
8. Call `mapIOSurfacesWithModel:request:cacheInference:YES` to pre-wire DMA mappings (private)
9. Call `evaluateWithModel:options:request:qos:error:` per inference call (private)

Steps 1–8 happen once at compile time per unique `(weight matrix, batch size)` combination. Step 9 is the hot path, called on every forward pass.

Step 8 is the one that matters for performance. `mapIOSurfacesWithModel:` tells the ANE daemon to set up persistent DMA mappings between the IOSurfaces and the model's internal memory regions. With `cacheInference:YES`, those mappings stay alive indefinitely. Subsequent `evaluateWithModel:` calls dispatch directly to the ANE hardware without re-negotiating anything — no handshake, no re-mapping, just execution.

### IOSurfaces as the Data Bridge

The ANE daemon runs as a separate system process. It cannot directly access your process's heap. IOSurface is Apple's shared memory mechanism for cross-process GPU/ANE data sharing — both your process and the daemon get a view into the same physical pages through shared memory descriptors.

For a linear layer with input `[batch, in_features]` and output `[batch, out_features]`, the IOSurface dimensions are:
- Input surface: `Width = in_features`, `Height = batch`, 2 bytes per element (fp16)
- Output surface: `Width = out_features`, `Height = batch`, 2 bytes per element (fp16)

This layout matches row-major `[batch, features]` directly. Writing the input and reading the output is a plain memcpy with no transposing. The correct dimensions come from inspecting `model.espresso.shape` in the compiled bundle, which specifies how the ANE program expects its I/O surfaces to be laid out.

---

## How ane_offload Is Built Into mlx-rs

The feature compiles in when `ane_offload` is set as a Cargo feature. With it active, every `Linear` layer gets a unique `layer_id` — a `u64` assigned at construction from an atomically incremented global counter. `Linear::forward` routes through `try_linear_forward` first and falls back to the standard MLX GPU matmul if that returns `Ok(None)`.

The implementation has four layers.

### The FFI Boundary (`ane-bridge`)

A separate crate that compiles `bridge.m` — an Objective-C/C file that owns all interaction with the private ANE APIs. Rust sees a C interface:

```rust
extern "C" {
    fn ane_linear_compile(weights: *const u16, bias: *const u16,
                          in_f: i32, out_f: i32, batch: i32) -> *mut AneProgram;
    fn ane_linear_execute(prog: *mut AneProgram, input: *const u16,
                          output: *mut u16, batch: i32) -> i32;
    fn ane_program_free(prog: *mut AneProgram);
}
```

`AneProgram` is an opaque C struct holding all compiled state for one layer at one batch size: the `MLModel` handle for the CoreML fallback path, the `_ANEModel` and `_ANERequest` handles for the fast path, the two IOSurface refs, stride information for the surfaces, and a `use_fast_path` flag.

### The Program Cache (`ane/cache.rs`)

The cache sits between Rust and the bridge and avoids recompiling when a layer is called with the same weights and batch size. The cache key is:

```rust
pub struct CacheKey {
    pub layer_id:   u64,    // unique per layer instance, never changes
    pub weight_ptr: usize,  // backing pointer of the weight MLX array
    pub batch_size: usize,  // ANE programs are shape-specific
}
```

`weight_ptr` is the field that makes the cache work correctly across training steps. When the optimizer replaces a weight array with a new allocation, the backing pointer changes. That change is a cache miss, which triggers one recompile for the new weights. During inference, weights are never replaced — the pointer is stable, every call after the first forward is a hit, and compile overhead is zero past the warmup pass.

The cache is a `Mutex<HashMap<CacheKey, Entry>>` behind a `OnceLock`. It lives for the process lifetime and is thread-safe.

### The Dispatch Function (`ane/ops.rs`)

`try_linear_forward` is the single entry point from `Linear::forward`:

```rust
pub fn try_linear_forward(layer: &Linear, x: &Array) -> Result<Option<Array>> {
    // 1. Check ANE availability and compile budget
    // 2. eval() the MLX arrays to make them CPU-readable
    // 3. to_vec_f32() to pull data out of the Metal buffer
    // 4. f32 → f16 conversion
    // 5. Build cache key from layer_id, weight_ptr, batch_size
    // 6. get_or_compile → cache hit or bridge compile
    // 7. ane_linear_execute
    // 8. f16 → f32 → Array::from_slice back into MLX
}
```

`Ok(None)` means fall back to MLX GPU. It is returned when ANE is unavailable, the compile budget is exhausted, the fast-path slot limit is reached, the input shape is not 2D, or execution fails. None of these are fatal — the caller always has a working fallback.

### The Objective-C Bridge (`bridge.m`)

`ane_linear_compile` runs the full 9-step pipeline. It builds the Core ML protobuf in memory using a minimal hand-written encoder (no external proto library), calls the compiler, loads via `_ANEModel` and `_ANEClient`, creates IOSurfaces, and calls `mapIOSurfaces`.

All of the fast-path setup is inside a `do{}while(0)` block. If any step fails, a `break` exits the block and the function returns with `use_fast_path = 0` set. The `MLModel` CoreML path is always set up first, so the fallback is always live regardless of what happens in the fast-path block.

`ane_linear_execute` checks `use_fast_path`. If 1: `write_surface`, `evaluateWithModel:`, `read_surface`. If 0: `coreml_execute`, which wraps an `MLMultiArray` around the input and calls `predictionFromFeatures:`.

---

## The macOS 26 IOSurface Regression

On macOS 26, `mapIOSurfacesWithModel:request:cacheInference:error:` throws an unrecognized-selector exception rather than running. Tracing through what it does internally: it calls `-ioSurface` on each surface in the request to get the underlying `IOSurfaceRef` for DMA setup. In macOS 15 and earlier, `IOSurface` objects created via `IOSurfaceCreate()` responded to that selector. In macOS 26 they do not.

The fix is a one-time method injection at startup:

```objc
static void install_iosurface_compat(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        Class cls = NSClassFromString(@"IOSurface");
        SEL sel = sel_registerName("ioSurface");
        if (class_getInstanceMethod(cls, sel)) return;
        IMP imp = imp_implementationWithBlock(^id(id self_) { return self_; });
        class_addMethod(cls, sel, imp, "@@:");
    });
}
```

Returning `self` is correct here. `IOSurfaceRef` is toll-free bridged to the `IOSurface` ObjC class — the object is the surface. `mapIOSurfaces` gets back the same object it passed in, which is sufficient for it to set up the DMA mapping.

---

## The Compile Budget

The private ANE framework enforces a per-process limit on calls to `MLModel.compileModelAtURL:`. The limit is around 119 before the framework starts leaking resources; the implementation halts at 108 as a conservative margin.

Each unique `(layer_id, weight_ptr, batch_size)` combination consumes one budget unit permanently. The budget does not reset.

For inference with frozen weights at a fixed batch size, this is not a problem in practice. The warmup pass compiles all layers once, and every subsequent forward pass is a cache hit. A 25-layer model at one batch size uses 25 of 108 budget units.

For training, the budget is fatal. The optimizer replaces weight arrays after each step. Each replacement is a new pointer, which is a new cache key, which is a new compile. A 25-layer model burns 25 compile units per training step. The budget runs out after 4 steps, after which the ANE path falls back to MLX GPU for the remainder of the process. This is not a bug — it is a consequence of how ANE programs work: weights are baked into the compiled binary at `compileModelAtURL:` time. There is no way to update them without recompiling.

---

## The 24-Slot Daemon Limit

During testing with a GPT-2-style benchmark (25 linear layers at 3 batch sizes = 75 total compiles), the 25th `mapIOSurfaces(YES)` call failed:

```
Error Domain=com.apple.appleneuralengine Code=13
"mapIOSurfacesWithModel:request:cacheInference:error::
 Program IOSurfaces map failure (0x1D)"
```

The ANE daemon maintains a pool of inference-cache DMA mapping slots per `_ANEClient.sharedConnection`. The pool holds 24 slots. The 25th call returns Code=13 regardless of which model it's for. The slots do not release when a model is idle, when its IOSurfaces are unmapped, or when the process that loaded the model exits without calling `unloadModel:`. They are tied to the daemon-side model registration and persist until the daemon restarts.

Because the daemon is a shared system service, slots consumed by a previous process run can reduce the available pool for a new process. Running the same benchmark twice back-to-back without a daemon reset produced 3 failures instead of 1 — the daemon was still holding slots from the first run.

The approaches tried before finding the right fix:

| Approach | Result |
|---|---|
| `cacheInference:NO` | Smaller pool, still exhausted |
| `mapIOSurfaces` per execute call instead of at compile time | Still limited; also adds latency on every call |
| Remove `mapIOSurfaces` entirely | `evaluateWithModel:` fails with `status=0x1D`; the mapping is mandatory |
| `unmapIOSurfaces` after each execute | No effect; slot stays occupied |
| `unloadModel:` on failure to free a slot | Frees one slot but loses the model from the daemon; coverage dropped |

The fix is a hard cap at 24 active fast-path mappings in `bridge.m`. A global counter `g_fast_path_cnt` is incremented each time a model successfully calls `mapIOSurfaces`. At the top of the fast-path setup block, if the counter is already at 24, the block exits without calling `mapIOSurfaces`. The model still compiles fully — `MLModel` is loaded, the CoreML fallback is ready — but `use_fast_path` stays 0:

```c
do {
    if (g_fast_path_cnt >= ANE_FAST_PATH_LIMIT) {
        break;  // silent fallback: CoreML will handle execute calls
    }
    // ... full fast-path setup ...
    prog->use_fast_path = 1;
    g_fast_path_cnt++;
} while (0);
```

This produces zero error logs on any run regardless of prior daemon state. The trade-off is explicit: the first 24 models get direct ANE dispatch, the rest use CoreML. For a GPT-2 Tiny model (25 layers), only the lm_head lands on CoreML — one layer, called once per sequence, negligible impact.

---

## Benchmark Results

Numbers from GPT-2 Tiny (4 transformer layers, 384 model dim, 1536 FFN width), batch×seq = 32 tokens:

```
Layer                    ANE path          GPU path       Verdict
q_proj  [384→384]        187.3 µs          282.0 µs       1.51× faster
fc1     [384→1536]       446.3 µs          371.7 µs       1.20× slower
fc2     [1536→384]       450.2 µs          370.1 µs       1.22× slower
lm_head [384→1000]       401.8 µs          314.8 µs       1.28× slower
```

The attention projection (384→384) is faster on ANE. The FFN layers (384→1536, 1536→384) are slower despite being larger operations that should benefit more from dedicated matrix hardware. The reason is the CPU round-trip.

---

## The CPU Round-Trip Bottleneck

Every `try_linear_forward` call moves data through the following path:

```
MLX Array (Metal buffer, unified memory)
    │  eval() + to_vec_f32()          ← GPU → CPU
    ▼
Vec<f32>
    │  f32_slice_to_f16()             ← CPU conversion
    ▼
Vec<u16>
    │  write_surface()                ← memcpy into IOSurface
    ▼
IOSurface ──► ANE hardware ──► IOSurface
    │  read_surface()                 ← memcpy out of IOSurface
    ▼
Vec<u16>
    │  f16_slice_to_f32()             ← CPU conversion
    ▼
Vec<f32>
    │  Array::from_slice()            ← CPU → new Metal buffer
    ▼
MLX Array (Metal buffer, unified memory)
```

On Apple Silicon, the MLX Metal buffer and the IOSurface share the same physical DRAM. UMA means there is no separate GPU VRAM — both the Metal buffer and the IOSurface are virtual mappings into the same physical pages. Despite this, the current path reads the full tensor out of the Metal buffer to CPU, converts it, writes it into the IOSurface, then after ANE execution reads it back out through the CPU and allocates a new Metal buffer for the output.

For the attention projection at 384×384, the tensor is small, the conversion is short, and the ANE's execution speed more than covers the overhead. For the FFN at 384×1536, the tensor is 4× larger — the round-trip scales with size, the ANE execution time also scales with size, but the round-trip overhead dominates at this batch dimension.

The path to fixing this is a Metal compute shader that converts fp32↔fp16 directly on the GPU and writes the result into an IOSurface-backed Metal texture. On UMA, a shader can read from the MLX Metal buffer and write to an IOSurface because both map to the same physical memory — the shader accesses them through different GPU virtual addresses but moves no data across a physical bus. The CPU never touches the tensor data. The round-trip cost goes to zero.

This shader is not implemented yet. It requires threading Metal device handles and IOSurface-backed texture objects through the Rust/Objective-C FFI boundary, which is more plumbing than the correctness-first prototype stage warranted. The architecture does not need to change to accommodate it — the cache, the `AneProgram` struct, and the execute path are all in the right shape already.

---

## Why Training Does Not Work

The compile budget issue means ANE offload for training is not viable in the current design.

During a training loop:
1. Forward pass — `weight_ptr = 0xAAAA`, cache miss, compile consumed
2. Optimizer step — weight array is replaced, `weight_ptr` is now `0xBBBB`
3. Next forward pass — cache miss on `0xBBBB`, another compile consumed

A 25-layer model burns 25 compile units per training step. The 108-unit budget runs out after 4 steps. From step 5 onwards, `budget_available()` returns false and every linear layer uses the MLX GPU path for the rest of the process.

The fundamental issue is that ANE programs have weights embedded at compile time. The `InnerProduct` layer's weight data is part of the compiled binary written into the daemon by `loadModel:`. There is no runtime API to update those weights without recompiling. Each optimizer step requires a fresh binary.

A design that could support training would need to represent the weight matrix as an IOSurface that the optimizer writes to in-place, and use an ANE program compiled with external weight slots rather than embedded constants. Whether the ANE hardware supports that at all is not clear from the private API surface, and exploring it would require deeper framework inspection than what this prototype covers.

---

## What This Is and Is Not

To be clear about the current state:

**Working correctly:**
- The full compile pipeline from protobuf encoding through `_ANEClient` load and IOSurface setup runs end-to-end
- ANE output matches MLX GPU output within fp16 precision (~1e-3 max absolute difference)
- The fallback chain at every level — slot limit, budget exhaustion, execution failure — produces correct results with no crash and no log spam
- The cache key correctly handles all cases: frozen weights (100% hit rate after warmup), changing weights (one miss per optimizer step), multiple batch sizes (one compile per size per layer)
- The macOS 26 IOSurface regression is handled
- The 24-slot daemon limit is handled with a hard cap and silent fallback

**Not working or not implemented:**
- ANE does not beat MLX GPU on large matrices at small batch sizes due to the CPU round-trip overhead
- Models with more than 24 linear layers have layers 25+ on CoreML's dispatch path rather than the direct `_ANEClient` path
- No persistent compile cache — each process restart recompiles everything from scratch
- Training is not viable due to the compile budget and embedded-weight constraint

---

## Budget and Slot Math

Two limits to track independently:

| Model | Linear layers | Compile budget (1 batch size) | Fast-path slots (1 batch size) |
|---|---|---|---|
| GPT-2 Tiny (4L) | 25 | 25 / 108 — fine | 24 / 24 — lm_head on CoreML |
| GPT-2 Small (12L) | 73 | 73 / 108 — fine | 24 / 24 — 49 layers on CoreML |
| GPT-2 Medium (24L) | 145 | over budget | 24 / 24 |
| Custom model, ≤24 linears | ≤24 | within budget | full fast path |

For a model with ≤24 linear layers at a single fixed batch size, both constraints are satisfied and every layer runs on the `_ANEClient` direct path. GPT-2 Tiny at one batch size misses by one layer (lm_head), which is inconsequential in practice.

---

## What Would Make This Production-Ready

In rough order of impact:

**1. Metal shader for zero-copy data transfer.** This eliminates the CPU round-trip and is the single change that would most likely flip the ANE from slower-than-GPU to faster-than-GPU on the FFN layers. It is a well-understood engineering task, not a research problem.

**2. LRU eviction for fast-path slots.** When all 24 slots are occupied and a new layer needs one, evict the least-recently-used entry: `unmapIOSurfaces`, `unloadModel:`, remove from the fast-path registry. The evicted model keeps its CoreML fallback. This replaces the hard first-24 cap with a sliding window of the 24 most actively used layers.

**3. Persistent compile cache.** Serialize the `.mlmodelc` bundle path and a weight hash to disk. On process restart, reload the existing bundle instead of calling `compileModelAtURL:`. This brings warmup cost for a restarted inference server down from several seconds to near zero.

**4. Selective offload by arithmetic intensity.** Assign the 24 fast-path slots to the layers with the highest `in_features × out_features × batch_size` product. For GPT-2-class models, FFN projection layers dominate — they should get priority over the attention projections.

---

## Summary

The `ane_offload` feature routes `Linear::forward` through the ANE's private `_ANEClient` dispatch path for models with frozen weights and a fixed batch size. The full pipeline — Core ML protobuf encoding, compilation, `_ANEModel` load, IOSurface setup, `mapIOSurfaces` DMA pre-wiring, and `evaluateWithModel:` dispatch — is implemented and working. The macOS 26 IOSurface regression and the 24-slot daemon limit are both handled.

The current bottleneck is the CPU round-trip between the MLX Metal buffer and the ANE IOSurfaces. On UMA hardware, this round-trip is unnecessary — the data never needs to leave physical DRAM — but eliminating it requires a Metal shader that has not been implemented yet. Without it, the ANE wins only on small matrices where execution time dominates over conversion overhead.

The private API work, the IOSurface layout, the cache architecture, the fallback chain, and the daemon slot management are all in place. What remains is performance engineering, not new research.

---

*Implementation: `ane-bridge/src/bridge.m` (Objective-C bridge), `mlx/src/ane/` (Rust cache and dispatch), `mlx/examples/ane_bench.rs` and `gpt2_ane_bench.rs` (benchmarks).
