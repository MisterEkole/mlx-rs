use crate::{sys, Array, Error, Result};
use std::ffi::CString;
use std::ptr;

extern "C" {
    fn mlx_quantize_simple(
        res: *mut sys::mlx_vector_array,
        w: sys::mlx_array,
        group_size: i32,
        bits: i32,
        mode: *const std::os::raw::c_char,
        s: sys::mlx_stream,
    ) -> std::os::raw::c_int;

    fn mlx_dequantize_simple(
        res: *mut sys::mlx_array,
        w: sys::mlx_array,
        scales: sys::mlx_array,
        biases: sys::mlx_array,
        group_size: i32,
        bits: i32,
        mode: *const std::os::raw::c_char,
        s: sys::mlx_stream,
    ) -> std::os::raw::c_int;
}

pub fn quantize(array: &Array, bits: i32, group_size: i32) -> Result<(Array, Array, Array)> {
    unsafe {
        let mut res_vec = sys::mlx_vector_array { ctx: ptr::null_mut() };
        let mode = CString::new("affine").unwrap();

        let status = mlx_quantize_simple(
            &mut res_vec,
            array.handle,
            group_size,
            bits,
            mode.as_ptr(),
            Array::default_stream(),
        );

        if status != 0 || res_vec.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_quantize failed".into()));
        }

        let mut q_h = sys::mlx_array { ctx: ptr::null_mut() };
        let mut s_h = sys::mlx_array { ctx: ptr::null_mut() };
        let mut b_h = sys::mlx_array { ctx: ptr::null_mut() };

        sys::mlx_vector_array_get(&mut q_h, res_vec, 0);
        sys::mlx_vector_array_get(&mut s_h, res_vec, 1);
        sys::mlx_vector_array_get(&mut b_h, res_vec, 2);

        sys::mlx_vector_array_free(res_vec);
        Ok((Array { handle: q_h }, Array { handle: s_h }, Array { handle: b_h }))
    }
}

pub fn dequantize(q: &Array, s: &Array, b: &Array, bits: i32, group_size: i32) -> Result<Array> {
    unsafe {
        let mut res = sys::mlx_array { ctx: ptr::null_mut() };
        let mode = CString::new("affine").unwrap();

        let status = mlx_dequantize_simple(
            &mut res,
            q.handle,
            s.handle,
            b.handle,
            group_size,
            bits,
            mode.as_ptr(),
            Array::default_stream(),
        );

        if status != 0 || res.ctx.is_null() {
            return Err(Error::OperationFailed("mlx_dequantize failed".into()));
        }

        Ok(Array { handle: res })
    }
}