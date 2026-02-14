// Fast Fourier Transform (FFT) implementation for MLX-RS

use crate::{Array, Result, sys};
use std::ptr;
use std::os::raw::c_int;

/// Internal helper to resolve n (transform size) from the array shape if not provided.
fn resolve_n(a: &Array, n: Option<i32>, axis: i32) -> Result<i32> {
    match n {
        Some(val) => Ok(val),
        None => {
            let shape = a.shape()?;
            let ndim = shape.len() as i32;
            let target_axis = if axis < 0 { axis + ndim } else { axis };
            if target_axis < 0 || target_axis >= ndim {
                return Err(crate::Error::InvalidShape(format!("Axis {} out of bounds", axis)));
            }
            Ok(shape[target_axis as usize] as i32)
        }
    }
}

// Computes the 1D FFT of an array
pub fn fft(a: &Array, n: Option<i32>, axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let n_val = resolve_n(a, n, axis)?;

        let status = sys::mlx_fft_fft(
            &mut res_handle, 
            a.handle, 
            n_val, 
            axis, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

// Inverse discrete FT
pub fn ifft(a: &Array, n: Option<i32>, axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let n_val = resolve_n(a, n, axis)?;
        let status = sys::mlx_fft_ifft(
            &mut res_handle, 
            a.handle, 
            n_val, 
            axis, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

// 1D Real Discrete FT
pub fn rfft(a: &Array, n: Option<i32>, axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let n_val = resolve_n(a, n, axis)?;
        let status = sys::mlx_fft_rfft(
            &mut res_handle, 
            a.handle, 
            n_val, 
            axis, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

//Inverse Real Discrete FT
pub fn irfft(a: &Array, n: Option<i32>, axis: i32) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let n_val = resolve_n(a, n, axis)?;
        let status = sys::mlx_fft_irfft(
            &mut res_handle, 
            a.handle, 
            n_val, 
            axis, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

// ========================================
// 2D & N Dimensional FFTs
// ========================================

// helper func to unpack Option<&[i32]> for c pointers
fn unpack_slice(slice: Option<&[i32]>) -> (*const c_int, usize) {
    match slice {
        Some(s) => (s.as_ptr() as *const c_int, s.len()),
        None => (ptr::null(), 0),
    }
}

// 2D Discrete FFT
pub fn fft2(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);
        let status = sys::mlx_fft_fft2(
            &mut res_handle, 
            a.handle, 
            n_ptr, 
            n_len, 
            axes_ptr, 
            axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// 2D Inverse Discrete Fourier Transform
pub fn ifft2(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_ifft2(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// 2D Real Discrete Fourier Transform
pub fn rfft2(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_rfft2(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// 2D Inverse Real Discrete Fourier Transform
pub fn irfft2(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_irfft2(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// N-Dimensional Discrete Fourier Transform
pub fn fftn(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_fftn(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// N-Dimensional Inverse Discrete Fourier Transform
pub fn ifftn(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_ifftn(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// N-Dimensional Real Discrete Fourier Transform
pub fn rfftn(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_rfftn(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// N-Dimensional Inverse Real Discrete Fourier Transform
pub fn irfftn(a: &Array, n: Option<&[i32]>, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (n_ptr, n_len) = unpack_slice(n);
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_irfftn(
            &mut res_handle, a.handle, 
            n_ptr, n_len, axes_ptr, axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

// ==========================================
// Shift Operations
// ==========================================

/// Shift the zero-frequency component to the center of the spectrum.
pub fn fftshift(a: &Array, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_fftshift(
            &mut res_handle, 
            a.handle, 
            axes_ptr, 
            axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}

/// The inverse of `fftshift`.
pub fn ifftshift(a: &Array, axes: Option<&[i32]>) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let (axes_ptr, axes_len) = unpack_slice(axes);

        let status = sys::mlx_fft_ifftshift(
            &mut res_handle, 
            a.handle, 
            axes_ptr, 
            axes_len, 
            Array::default_stream()
        );
        a.check_status(status, res_handle)
    }
}