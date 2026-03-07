// mlx/src/ane/cache.rs
//
// Thread-safe cache for compiled ANE programs.
//
// Cache key: (layer_id, weight_ptr, batch_size)
//   - layer_id   — unique u64 assigned at Linear::new(), never changes
//   - weight_ptr — backing pointer of the weight MLX handle; changes once per
//                  optimizer step when the optimizer writes *p = new_array
//   - batch_size — ANE programs are shape-specific
//
// During training: each optimizer step changes weight_ptr, causing one cache
// miss per layer per step. Budget burns at num_layers × num_steps.
// During inference: weight_ptr is stable, every call after the first is a hit.
//
// Compile budget: the private ANE framework leaks resources beyond ~119
// compilations per process. We halt at ANE_BUDGET_LIMIT and return None.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use ane_bridge as ane;

pub const ANE_BUDGET_LIMIT: usize = 108; // conservative margin below 119

// ── Cache key ─────────────────────────────────────────────────────────────────

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct CacheKey {
    /// Unique ID for the layer instance (assigned at Linear::new())
    pub layer_id: u64,
    /// Backing pointer of weight.handle.ctx — changes on every optimizer step
    pub weight_ptr: usize,
    /// ANE programs are compiled for a specific batch size
    pub batch_size: usize,
}

// ── Cache entry ───────────────────────────────────────────────────────────────

struct Entry {
    prog: *mut ane::AneProgram,
}

// Safety: AneProgram is opaque C memory accessed only while the cache Mutex
// is held. We never alias the pointer across threads.
unsafe impl Send for Entry {}

// ── Global cache singleton ────────────────────────────────────────────────────

struct ProgramCache {
    entries:       HashMap<CacheKey, Entry>,
    compile_count: usize,
}

impl ProgramCache {
    fn new() -> Self {
        Self {
            entries:       HashMap::new(),
            compile_count: 0,
        }
    }

    /// Returns an existing program or compiles a new one.
    /// Returns `None` when the compile budget is exhausted or compilation fails.
    pub fn get_or_compile(
        &mut self,
        key:         CacheKey,
        weights_f16: &[u16],
        bias_f16:    Option<&[u16]>,
        in_f:        usize,
        out_f:       usize,
    ) -> Option<*mut ane::AneProgram> {
        // Cache hit — no recompile needed
        if let Some(entry) = self.entries.get(&key) {
            return Some(entry.prog);
        }

        // Budget check
        if self.compile_count >= ANE_BUDGET_LIMIT {
            return None;
        }

        let bias_ptr = bias_f16.map(|b| b.as_ptr()).unwrap_or(std::ptr::null());

        let prog = unsafe {
            ane::ane_linear_compile(
                weights_f16.as_ptr(),
                bias_ptr,
                in_f  as i32,
                out_f as i32,
                key.batch_size as i32,
            )
        };

        if prog.is_null() {
            return None;
        }

        self.compile_count += 1;
        self.entries.insert(key, Entry { prog });
        Some(prog)
    }

    pub fn compile_count(&self) -> usize {
        self.compile_count
    }
}

impl Drop for ProgramCache {
    fn drop(&mut self) {
        for (_, entry) in self.entries.drain() {
            unsafe { ane::ane_program_free(entry.prog); }
        }
    }
}

// ── Public accessors ──────────────────────────────────────────────────────────

static CACHE: OnceLock<Mutex<ProgramCache>> = OnceLock::new();

fn cache() -> &'static Mutex<ProgramCache> {
    CACHE.get_or_init(|| Mutex::new(ProgramCache::new()))
}

/// Get or compile a program, returning the raw pointer if successful.
/// Returns `None` on budget exhaustion or compile failure.
pub fn get_or_compile(
    key:         CacheKey,
    weights_f16: &[u16],
    bias_f16:    Option<&[u16]>,
    in_f:        usize,
    out_f:       usize,
) -> Option<*mut ane::AneProgram> {
    cache()
        .lock()
        .ok()?
        .get_or_compile(key, weights_f16, bias_f16, in_f, out_f)
}

/// How many programs have been compiled in this process.
/// Once this approaches `ANE_BUDGET_LIMIT` the system falls back to MLX.
pub fn compile_count() -> usize {
    cache()
        .lock()
        .map(|g| g.compile_count())
        .unwrap_or(0)
}

/// Whether the budget allows further compilations.
pub fn budget_available() -> bool {
    compile_count() < ANE_BUDGET_LIMIT
}

/// Whether a program for this key is already compiled and cached.
pub fn is_cached(key: &CacheKey) -> bool {
    cache()
        .lock()
        .map(|g| g.entries.contains_key(key))
        .unwrap_or(false)
}
