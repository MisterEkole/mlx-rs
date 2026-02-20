use crate::{sys, Array, Error, Result};
use std::ptr;

use super::dist_group::{null_group, DistributedGroup};

/// All-reduce sum across all processes in the group.
///
/// Sums the array `x` across all processes and returns the result on each process.
/// When `group` is `None`, uses the global group.
///
/// # Arguments
/// * `x` - The input array to sum across processes
/// * `group` - Optional distributed group. Uses the global group if `None`.
///
/// # Example
/// ```no_run
/// use mlx::distributed;
///
/// let group = distributed::init(false);
/// let x = mlx::Array::ones(&[10]);
/// let sum = distributed::all_sum(&x, Some(&group)).unwrap();
/// // sum contains the element-wise sum across all processes
/// ```
pub fn all_sum(x: &Array, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_all_sum(
            &mut res,
            x.handle,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_all_sum failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// Gather arrays from all processes and concatenate along the first axis.
///
/// Each process contributes its array `x`, and all processes receive the
/// concatenation of all arrays. Arrays should all have the same shape.
///
/// # Arguments
/// * `x` - The input array to gather
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn all_gather(x: &Array, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_all_gather(
            &mut res,
            x.handle,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_all_gather failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// All-reduce max across all processes in the group.
///
/// Computes the element-wise maximum of `x` across all processes.
///
/// # Arguments
/// * `x` - The input array
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn all_max(x: &Array, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_all_max(
            &mut res,
            x.handle,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_all_max failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// All-reduce min across all processes in the group.
///
/// Computes the element-wise minimum of `x` across all processes.
///
/// # Arguments
/// * `x` - The input array
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn all_min(x: &Array, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_all_min(
            &mut res,
            x.handle,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_all_min failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// Sum-scatter across all processes in the group.
///
/// Performs an all-reduce sum followed by a scatter, so each process receives
/// a portion of the reduced result.
///
/// # Arguments
/// * `x` - The input array
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn sum_scatter(x: &Array, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_sum_scatter(
            &mut res,
            x.handle,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed(
                "mlx_distributed_sum_scatter failed".into(),
            ));
        }
        Ok(Array { handle: res })
    }
}

/// Send an array to another process.
///
/// Sends `x` from the current process to the process with rank `dst` in the group.
/// Returns a "dependency" array that must be evaluated to ensure the send completes.
///
/// # Arguments
/// * `x` - The array to send
/// * `dst` - The destination rank
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn send(x: &Array, dst: i32, group: Option<&DistributedGroup>) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_send(
            &mut res,
            x.handle,
            dst,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_send failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// Receive an array from another process.
///
/// Receives an array with the specified `shape` and `dtype` from the process
/// with rank `src` in the group.
///
/// # Arguments
/// * `shape` - The shape of the expected array
/// * `dtype` - The data type of the expected array
/// * `src` - The source rank to receive from
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn recv(
    shape: &[i32],
    dtype: sys::mlx_dtype,
    src: i32,
    group: Option<&DistributedGroup>,
) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_recv(
            &mut res,
            shape.as_ptr(),
            shape.len(),
            dtype,
            src,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_distributed_recv failed".into()));
        }
        Ok(Array { handle: res })
    }
}

/// Receive an array with the same shape and type as a reference array.
///
/// Receives an array from the process with rank `src`, with shape and dtype
/// matching the reference array `x`.
///
/// # Arguments
/// * `x` - Reference array whose shape and dtype will be used
/// * `src` - The source rank to receive from
/// * `group` - Optional distributed group. Uses the global group if `None`.
pub fn recv_like(
    x: &Array,
    src: i32,
    group: Option<&DistributedGroup>,
) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let grp = group.map_or(null_group(), |g| g.handle);

        let status = sys::mlx_distributed_recv_like(
            &mut res,
            x.handle,
            src,
            grp,
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed(
                "mlx_distributed_recv_like failed".into(),
            ));
        }
        Ok(Array { handle: res })
    }
}