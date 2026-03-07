// mlx/src/ane/mod.rs
//
// ANE offload module. Compiled only when --features ane_offload is active.
//
// convert.rs  — fp32 ↔ fp16 conversion (CPU path)
// mil.rs      — MIL text generation for ANE programs
// cache.rs    — program cache keyed by (layer_id, weight_ptr, batch_size)
// ops.rs      — try_linear_forward(), the dispatch entry point

pub mod convert;
pub mod mil;
pub mod cache;
pub mod ops;

pub use ops::try_linear_forward;

use std::sync::atomic::{AtomicU64, Ordering};

static LAYER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Allocate a unique, monotonically-increasing ID for a new layer instance.
/// Called from Linear::new() and Linear::from_weights() when ane_offload is active.
pub fn next_layer_id() -> u64 {
    LAYER_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Human-readable status for diagnostics / tests.
pub struct AneStatus {
    pub available:     bool,
    pub compile_count: usize,
    pub budget_left:   usize,
}

pub fn status() -> AneStatus {
    let available     = unsafe { ane_bridge::ane_is_available() } != 0;
    let compile_count = cache::compile_count();
    let budget_left   = cache::ANE_BUDGET_LIMIT.saturating_sub(compile_count);
    AneStatus { available, compile_count, budget_left }
}
