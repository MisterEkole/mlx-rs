# Rust→C ABI Mismatch in MLX Quantization

**Project:** mlx-rs (Rust bindings for Apple MLX) **Date:** February 2026 **Severity:** Blocking — quantization pipeline completely non-functional **Status:** Resolved

---

## Summary

Calling `mlx_quantize` from Rust FFI consistently failed with:

```
MLX error: [quantize] Invalid quantization mode 'linear'.
```

despite the Rust code explicitly passing `"affine"` as the mode string. The root cause was an **ABI (Application Binary Interface) mismatch** in how Rust and C pass `mlx_optional_int` structs across the FFI boundary. The fix was a thin C wrapper that accepts plain `int` parameters and constructs the structs on the C side.

---

### The Quantization API

The C function signature:

```c
int mlx_quantize(
    mlx_vector_array* res,     // output: [quantized, scales, biases]
    const mlx_array w,         // input weights
    mlx_optional_int group_size, // quantization group size (e.g. 64)
    mlx_optional_int bits,       // bit width (e.g. 4)
    const char* mode,            // quantization mode string ("affine")
    const mlx_stream s           // execution stream
);
```

Where `mlx_optional_int` is:

```c
typedef struct mlx_optional_int_ {
    int value;
    bool has_value;
} mlx_optional_int;
```

### The Rust Binding

```rust
extern "C" {
    pub fn mlx_quantize(
        res: *mut mlx_vector_array,
        w: mlx_array,
        group_size: mlx_optional_int,
        bits: mlx_optional_int,
        mode: *const c_char,
        s: mlx_stream,
    ) -> c_int;
}
```

With the Rust struct:

```rust
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mlx_optional_int_ {
    pub value: ::std::os::raw::c_int,
    pub has_value: bool,
}
```

---

## The Symptom

Every call to `mlx_quantize` from Rust failed with the same error, regardless of what string was passed as the mode:

```
MLX error: [quantize] Invalid quantization mode 'linear'.
```

Key observations:

- The string `"linear"` was never present anywhere in the Rust code
- Changing the Rust string from `"affine"` to `"default"` to a raw byte literal (`b"affine\0"`) had no effect
- The same C library worked perfectly when called from a pure C test program

---

## Debugging Timeline

### Phase 1: Obvious Fixes (Failed)

**Hypothesis:** Wrong mode string. Tried `"default"`, `"affine"`, `b"affine\0".as_ptr()`. All produced the same `"linear"` error. 

**Hypothesis:** Missing `mlx_optional_dtype` parameter on `mlx_dequantize`. The `mlx_dequantize` C signature includes an `mlx_optional_dtype` parameter that was missing from the Rust call. Added it. Error persisted on `mlx_quantize` which doesn't have that parameter. 

**Hypothesis:** Wrong struct field types (`mlx_optional_int`). Tried passing plain `i32` instead of `mlx_optional_int`. Compiler rejected it because the binding expects the struct type. 

### Phase 2: Size Verification

Printed struct sizes from both Rust and C:

| Type | Rust `size_of` | C `sizeof` |
| --- | --- | --- |
| `mlx_optional_int` | 8 | 8 |
| `mlx_array` | 8 | 8 |
| `mlx_vector_array` | 8 | 8 |
| `mlx_stream` | 8 | 8 |

All sizes matched. The `#[repr(C)]` attribute was correctly applied. The struct field types (`c_int` + `bool`) matched the C definition (`int` + `bool`).

### Phase 3: C-Side Debug Instrumentation

Inserted `fprintf` into `mlx-c/mlx/c/ops.cpp` right before the `mlx::core::quantize` call:

```cpp
fprintf(stderr, "C: mode_ptr=%p mode=%s stream_ctx=%p\n",
    (void*)mode, mode, s.ctx);
```

**Critical finding:** The mode pointer (`0x1033b319c`) pointed to a static data segment address containing `"linear"` — a string literal baked into the MLX C++ library. This was NOT the heap address that Rust's `CString::new("affine")` would produce.

The Rust-side debug showed a completely different pointer address for the mode string. The C function was reading the **wrong argument register/stack slot** for the `mode` parameter.

### Phase 4: Pure C Validation

Compiled and ran a minimal C program that called `mlx_quantize` directly:

```c
mlx_optional_int group_size = {64, true};
mlx_optional_int bits = {4, true};
int status = mlx_quantize(&res, arr, group_size, bits, "affine", s);
// Status: 0 — SUCCESS
```

This confirmed the C library was correct. The bug was exclusively in the Rust→C FFI boundary.

### Phase 5: The C Wrapper Fix

Created `quantize_wrapper.cpp` with simple `int` parameters:

```cpp
extern "C" int mlx_quantize_simple(
    mlx_vector_array* res, mlx_array w,
    int group_size, int bits,
    const char* mode, mlx_stream s) {
    mlx_optional_int gs = {group_size, true};
    mlx_optional_int bs = {bits, true};
    return mlx_quantize(res, w, gs, bs, mode, s);
}
```

Updated Rust to call the wrapper. **Result: success.** The wrapper's own debug print confirmed `mode=affine` arrived correctly at the wrapper, and the inner `mlx_quantize` call (now from C→C) also received `"affine"` correctly.

---

## Root Cause Analysis: The ABI Mismatch

An ABI (Application Binary Interface) defines the low-level contract for how compiled code interacts: how function arguments are passed (registers vs. stack), how structs are laid out in memory, how return values are delivered, and how the call stack is managed.

When Rust calls a C function via `extern "C"`, both sides must agree on exactly how every argument is passed. If they disagree, arguments shift — one function writes to register X, but the callee reads from register Y.

### Why Did This Happen?

On ARM64 (Apple Silicon), the calling convention has specific rules for how structs are passed to functions:

1. **Small structs (≤ 16 bytes)** may be passed in registers, but the exact packing depends on the struct's "homogeneous" classification
2. **Structs with mixed types** (like `int` + `bool`) have specific alignment and padding rules that affect which registers they occupy
3. **The `bool` type** is particularly tricky: C's `_Bool` and Rust's `bool` are both 1 byte, but the **padding after `bool`** to fill the 8-byte struct can be handled differently by the Rust and C compilers

The `mlx_optional_int` struct is 8 bytes:

```
Offset 0-3: int value    (4 bytes)
Offset 4:   bool has_value (1 byte)
Offset 5-7: padding       (3 bytes)
```

When two of these structs are passed as function arguments, they occupy specific registers on ARM64. The Rust compiler and the C++ compiler (clang) made **different decisions** about how to pack these structs into registers.

### The Cascading Effect

The `mlx_quantize` signature has 6 parameters:

```
Param 1: mlx_vector_array* res      → register x0
Param 2: mlx_array w                → register x1
Param 3: mlx_optional_int group_size → registers x2/x3 (or different packing)
Param 4: mlx_optional_int bits       → registers x4/x5 (or different packing)
Param 5: const char* mode           → register x6 (or wherever it lands)
Param 6: mlx_stream s               → register x7 (or stack)
```

Because Rust packed the two `mlx_optional_int` structs differently than C expected, parameters 3 and 4 consumed a different number of registers. This shifted parameter 5 (`mode`) to the wrong location. The C function read `mode` from a register that actually contained part of the stream pointer or some other value — which happened to point to the static string `"linear"` in the library's data segment.

### Why "linear" Specifically?

The string `"linear"` is a default/fallback quantization mode string compiled into the MLX C++ library. The corrupted pointer happened to land on this string literal's address in the library's read-only data segment. This was coincidental but consistent — the same register misalignment always produced the same garbage pointer.

### Why C→C Worked But Rust→C Didn't

When the C wrapper calls `mlx_quantize`, both the caller and callee are compiled by the **same compiler** (clang/g++) with the **same ABI interpretation**. The struct packing is identical on both sides. When Rust calls the same function, `rustc`(which uses LLVM but with its own ABI lowering logic) may make subtly different decisions about struct argument passing.

This is a known class of bugs in FFI programming. The Rust Reference notes that passing non-trivial structs by value across FFI boundaries can be problematic, and the `improper_ctypes` lint exists specifically to catch some of these cases.

---

## The Fix

### Architecture

```
Before (broken):
  Rust → mlx_quantize(... mlx_optional_int, mlx_optional_int, char*, ...)
         ↑ ABI mismatch on struct passing shifts all subsequent args

After (working):
  Rust → mlx_quantize_simple(... int, int, char*, ...)  ← plain types, no ABI ambiguity
         → mlx_quantize(... mlx_optional_int, mlx_optional_int, char*, ...)  ← C→C, same compiler
```

### Files Changed

**New: `mlx-c/mlx/c/quantize_wrapper.h`**

```c
int mlx_quantize_simple(
    mlx_vector_array* res, mlx_array w,
    int group_size, int bits,
    const char* mode, mlx_stream s);

int mlx_dequantize_simple(
    mlx_array* res, mlx_array w, mlx_array scales, mlx_array biases,
    int group_size, int bits,
    const char* mode, mlx_stream s);
```

**New: `mlx-c/mlx/c/quantize_wrapper.cpp`**

Thin wrappers that accept plain `int` parameters, construct `mlx_optional_int` on the C side, and forward to the real functions.

**Modified: `mlx-rs/mlx/src/quantization/q_ops.rs`**

Declares `extern "C"` bindings for the `_simple` wrapper functions and calls them instead of the originals.

**Modified: `mlx-c/CMakeLists.txt`**

Added `quantize_wrapper.cpp` to the build.

---

## Why This Wasn't Caught Before

The most established Rust MLX binding — [oxideai/mlx-rs](https://github.com/oxideai/mlx-rs) (256+ stars) — has quantization support but **never hits this bug** because they take a fundamentally different approach:

**oxideai/mlx-rs workflow:**

```rust
Python (mlx_lm.convert) → quantize weights → save .safetensors
                                                       ↓
Rust (mlx-rs) → load pre-quantized .safetensors → inference via mlx_quantized_matmul
```

Their `QuantizedLinear` and `QuantizedEmbedding` modules (PR [#142](https://github.com/oxideai/mlx-rs/pull/142), [#174](https://github.com/oxideai/mlx-rs/pull/174)) consume **pre-quantized weights** and only call `mlx_quantized_matmul` for inference. The actual `mlx_quantize` function is never called from Rust. The quantization step is offloaded to Python.

**Our workflow (fully native):**

`Rust → train → mlx_quantize → mlx_dequantize → save → inference`

We call `mlx_quantize` and `mlx_dequantize` directly from Rust, which is where the `mlx_optional_int` struct-by-value ABI mismatch surfaces. This is the first known attempt at **end-to-end native quantization** from Rust without a Python dependency in the pipeline.

Notably, the oxideai team was already aware of FFI boundary issues — their PR [#116](https://github.com/oxideai/mlx-rs/pull/116) ("Use custom c shim for fallible closure") introduced C shims to work around other Rust→C ABI problems. The pattern of needing C wrappers for non-trivial FFI types is a recurring theme.

This means the bug has remained latent because no Rust binding has attempted to call `mlx_quantize` directly until now. Any future binding author  who attempts native quantization will hit the same issue

---

## References

- [Rust FFI Nomicon — Other Reprs](https://doc.rust-lang.org/nomicon/other-reprs.html)
- [ARM64 Procedure Call Standard (AAPCS64)](https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst)
- [Rust `improper_ctypes` lint](https://doc.rust-lang.org/rustc/lints/listing/warn-by-default.html#improper-ctypes)
- [Apple Silicon ABI documentation](https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms)