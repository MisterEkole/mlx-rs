// ane-bridge/src/lib.rs
//
// Rust-side FFI declarations for the Obj-C bridge that accesses Apple's
// private AppleNeuralEngine.framework.
//
// The bridge presents a simple, row-major fp16 interface to Rust.
// All IOSurface layout handling ([1,C,1,S]) lives inside bridge.m.
//
// All functions return non-zero on failure, NULL on failed allocation.
// Every AneProgram* obtained from ane_linear_compile() must be freed with
// ane_program_free() when no longer needed.

use std::ffi::c_int;

/// Opaque handle to a compiled ANE program.
/// Not Send/Sync — must be used from the thread that compiled it,
/// or behind a Mutex (which our cache already provides).
#[repr(C)]
pub struct AneProgram {
    _opaque: [u8; 0],
}

extern "C" {
    /// Compile a linear layer as a 1×1 convolution MIL program on the ANE.
    ///
    /// `weights`     — fp16, row-major [out_features × in_features]
    /// `bias`        — fp16, [out_features], may be NULL
    /// Returns NULL if the framework is unavailable, the compile limit has been
    /// reached (~119), or compilation fails.
    pub fn ane_linear_compile(
        weights: *const u16,
        bias: *const u16,
        in_features: c_int,
        out_features: c_int,
        batch_size: c_int,
    ) -> *mut AneProgram;

    /// Execute a previously compiled program.
    ///
    /// `input_fp16`  — [batch_size × in_features] fp16, row-major (caller owns)
    /// `output_fp16` — [batch_size × out_features] fp16, row-major (caller allocates)
    ///
    /// Returns 0 on success, non-zero on failure (caller should fall back to MLX).
    pub fn ane_linear_execute(
        prog: *mut AneProgram,
        input_fp16: *const u16,
        output_fp16: *mut u16,
        batch_size: c_int,
    ) -> c_int;

    /// Free a compiled program.
    pub fn ane_program_free(prog: *mut AneProgram);

    /// Number of compilations performed so far in this process.
    /// Use to track proximity to the ~119 per-process limit.
    pub fn ane_compile_count() -> c_int;

    /// Returns 1 if the ANE framework loaded successfully, 0 otherwise.
    pub fn ane_is_available() -> c_int;

    /// Print a full runtime diagnostic to stdout:
    /// - Whether dlopen succeeded
    /// - Every instance/class method on _ANEClient, _ANEInMemoryModelDescriptor, _ANECompiler
    /// - Whether _ANEClient alloc+init produces a live object
    ///
    /// Call this when ane_is_available() == 1 but ane_linear_compile() returns NULL.
    /// The method names in the output are the real selectors — use them to fix bridge.m.
    pub fn ane_diagnose();
}
