// mlx/src/ane/ops.rs
//
// Entry point called from Linear::forward when ane_offload is active.
//
// try_linear_forward() returns:
//   Ok(Some(Array))  — ANE executed; caller uses this result
//   Ok(None)         — ANE unavailable or fell back; caller uses MLX GPU path
//   Err(_)           — hard error reading input (propagated up)
//
// Current data path (CPU round-trip):
//   MLX Array (Metal buffer) → eval + to_vec_f32 → f32→f16 conversion →
//   write_surface → ANE execute → read_surface → f16→f32 → new MLX Array
//
// The CPU round-trip is avoidable on UMA hardware: the Metal buffer and
// IOSurface share physical DRAM. A Metal shader doing in-GPU fp32↔fp16
// conversion would eliminate it without other architectural changes.

use crate::{Array, Result, Error, Dtype};
use crate::nn::layers::linear::Linear;
use super::{
    cache::{self, CacheKey},
    convert::{f32_slice_to_f16, f16_slice_to_f32},
};
use ane_bridge as ane;

#[allow(unused_imports)]
use log::debug;

/// Try to execute a linear forward pass on the ANE.
/// Returns Ok(None) whenever the ANE path is unavailable — no panic, no error.
pub fn try_linear_forward(layer: &Linear, x: &Array) -> Result<Option<Array>> {
    let available = unsafe { ane::ane_is_available() };
    if available == 0 {
        return Ok(None);
    }

    if !cache::budget_available() {
        return Ok(None);
    }

    // eval() flushes pending Metal work before reading data to CPU.
    x.eval()?;
    layer.weight.eval()?;
    if let Some(ref b) = layer.bias {
        b.eval()?;
    }

    let x_data = x.to_vec_f32()?;
    let w_data = layer.weight.to_vec_f32()?;
    let b_data: Option<Vec<f32>> = layer.bias.as_ref()
        .map(|b| b.to_vec_f32())
        .transpose()?;

    let shape = x.shape()?;
    if shape.len() != 2 {
        // ANE path only handles 2D inputs [batch, in_features].
        return Ok(None);
    }
    let batch    = shape[0];
    let in_check = shape[1];
    if in_check != layer.in_features {
        return Err(Error::InvalidShape(format!(
            "ane_offload: input has {in_check} features but layer expects {}",
            layer.in_features
        )));
    }

    let x_f16 = f32_slice_to_f16(&x_data);
    let w_f16 = f32_slice_to_f16(&w_data);
    let b_f16 = b_data.as_deref().map(f32_slice_to_f16);

    // weight_ptr changes when the optimizer replaces the weight array.
    // That change produces a cache miss and exactly one recompile.
    let weight_ptr = layer.weight.handle.ctx as usize;
    let key = CacheKey {
        layer_id:   layer.layer_id,
        weight_ptr,
        batch_size: batch,
    };

    let is_cached = cache::is_cached(&key);
    let prog = match cache::get_or_compile(
        key.clone(),
        &w_f16,
        b_f16.as_deref(),
        layer.in_features,
        layer.out_features,
    ) {
        Some(p) => p,
        None    => return Ok(None),
    };

    if is_cached {
        debug!(
            "ane dispatch  layer={} {}x{} batch={} [cache hit]",
            layer.layer_id, layer.in_features, layer.out_features, batch
        );
    } else {
        debug!(
            "ane compile   layer={} {}x{} batch={} [compile #{}]",
            layer.layer_id, layer.in_features, layer.out_features, batch,
            cache::compile_count()
        );
    }

    let mut output_f16 = vec![0u16; batch * layer.out_features];

    let status = unsafe {
        ane::ane_linear_execute(
            prog,
            x_f16.as_ptr(),
            output_f16.as_mut_ptr(),
            batch as i32,
        )
    };

    if status != 0 {
        debug!(
            "ane execute failed  layer={} {}x{} batch={} status={}",
            layer.layer_id, layer.in_features, layer.out_features, batch, status
        );
        return Ok(None);
    }

    let output_f32 = f16_slice_to_f32(&output_f16);
    let result = Array::from_slice(
        &output_f32,
        &[batch, layer.out_features],
        Dtype::Float32,
    )?;

    Ok(Some(result))
}
