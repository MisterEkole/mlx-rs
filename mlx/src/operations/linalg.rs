use crate ::{Array, Result, sys, Error};
use std::ffi::CString;
//use std::os::raw::c_void;
use std::ptr;

// Computes inv of sq matrix: (A^T A + λI)^(-1) A^T
pub fn inv(a: &Array) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_inv(&mut res_handle, a.handle, Array::default_stream());
        a.check_status(status, res_handle)
    }
}

// cComputes the pseudo-inverse of a matrix: A^T (A A^T + λI)^(-1)
pub fn pinv(a: &Array) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_pinv(&mut res_handle, a.handle, Array::default_stream());
        a.check_status(status, res_handle)
    }
}

// Computes Cholesky decomposition of a positive-definite matrix A = L L^T
pub fn cholesky(a: &Array, upper: bool) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_cholesky(&mut res_handle, a.handle, upper, Array::default_stream());
        a.check_status(status, res_handle)
    }
}

// Solve a linear sys of equations Ax=b

pub fn solve(a: &Array, b: &Array) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_solve(&mut res_handle, a.handle, b.handle, Array::default_stream());
        a.check_status(status, res_handle)
    }
}

// Solve triangular lin sys of eqs
pub fn solve_triangular(a: &Array, b: &Array, upper: bool) -> Result<Array> {
    unsafe {
        let mut res_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_solve_triangular(&mut res_handle, a.handle, b.handle, upper, Array::default_stream());
        a.check_status(status, res_handle)
    }
}

// Computes matrix vector norm

pub fn norm(a: &Array, ord:f64 , axis: &[i32], keepdims: bool) -> Result<Array>{
    unsafe{
        let mut res_handle = sys:: mlx_array{ctx: ptr::null_mut()};
        let axis_pts=if axis .is_empty() { ptr::null() } else { axis.as_ptr() };
        let status = sys::mlx_linalg_norm(&mut res_handle, a.handle,
        ord, axis_pts, axis.len(), keepdims, Array::default_stream());
        a.check_status(status, res_handle)
    }

}

// ========================================
// ======= QR and Eigenvalues =============
// ========================================

// Computes QR Decomposition of a matrix A = QR
pub fn qr(a: &Array)-> Result<(Array, Array)> {
    unsafe {
        let mut q_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let mut r_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_qr(&mut q_handle, &mut r_handle, a.handle, Array::default_stream());
        if status != 0 || q_handle.ctx.is_null() || r_handle.ctx.is_null() {
            Err(Error::OperationFailed("Failed to compute QR decomposition".into()))
        } else {
            Ok((Array { handle: q_handle }, Array { handle: r_handle }))
        }
    }
}

// Computes eigenvalues and eigenvectors of complex hermitian or real symetrix matrix A
pub fn eigh(a: &Array, upper: bool) -> Result<(Array, Array)> {
    unsafe {
        let mut w_handle = sys::mlx_array { ctx: ptr::null_mut() };
        let mut v_handle = sys::mlx_array { ctx: ptr::null_mut() };

        let uplo = CString::new(if upper { "U" } else { "L" }).unwrap();
        let status = sys::mlx_linalg_eigh(&mut w_handle, &mut v_handle, a.handle, uplo.as_ptr(), Array::default_stream());
        if status != 0 || w_handle.ctx.is_null() || v_handle.ctx.is_null() {
            Err(Error::OperationFailed("Failed to compute eigenvalues and eigenvectors".into()))
        } else {
            Ok((Array { handle: w_handle }, Array { handle: v_handle }))
        }
    }
}



/// SVD
/// Returns a vector contint [S] if compute_uv is false, otherwise returns [U, S, V^H]

pub fn svd(a: &Array, compute_uv: bool) -> Result<Vec<Array>>{
    unsafe{
        let mut res_handle= sys::mlx_vector_array_ { ctx: ptr::null_mut() };
        let status = sys::mlx_linalg_svd(&mut res_handle, a.handle, compute_uv, Array::default_stream());
        if status != 0 || res_handle.ctx.is_null(){
            return Err(Error::OperationFailed("Failed to compute SVD".into()));
        }
        let count = sys::mlx_vector_array_size(res_handle);
        let mut result = Vec::with_capacity(count as usize);
        for i in 0..count {
            let mut arr_handle = sys::mlx_array { ctx: ptr::null_mut() };
            let status = sys::mlx_vector_array_get(&mut arr_handle, res_handle, i);
            if status != 0 || arr_handle.ctx.is_null() {
                sys::mlx_vector_array_free(res_handle);
                return Err(Error::OperationFailed(format!("Failed to retrieve SVD component {}", i)));
            }
            result.push(Array { handle: arr_handle });
        }
        sys::mlx_vector_array_free(res_handle);
        Ok(result)
    }

}