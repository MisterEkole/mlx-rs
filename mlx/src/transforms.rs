use crate::{Array, Result, sys};
use std::ffi::c_void;

/// Internal helper to convert Rust Vec<Array> to C mlx_vector_array
unsafe fn vec_to_vector(arrays: &[Array]) -> sys::mlx_vector_array {
    let vec = sys::mlx_vector_array_new();
    for arr in arrays {
        // Use append_value to add the individual array handle
        sys::mlx_vector_array_append_value(vec, arr.handle);
    }
    vec
}

/// Internal helper to convert C mlx_vector_array to Rust Vec<Array>
// unsafe fn vector_to_vec(vec: sys::mlx_vector_array) -> Vec<Array> {
//     let size = sys::mlx_vector_array_size(vec);
//     let mut out = Vec::with_capacity(size);
//     for i in 0..size {
//         let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
//         // use pointer to pointer catch to match signature
//         let status = sys::mlx_vector_array_get(&mut handle as *mut sys::mlx_array, vec, i);
//         if status == 0 && !handle.ctx.is_null() {
//             out.push(Array { handle });
//         }
//     }
//     sys::mlx_vector_array_free(vec);
//     out
// }

/// Helper: Convert and FREE the C-vector (used for VJP results)
unsafe fn vector_to_vec_consume(vec: sys::mlx_vector_array) -> Vec<Array> {
    let size = sys::mlx_vector_array_size(vec);
    let mut out = Vec::with_capacity(size);
    for i in 0..size {
        let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        sys::mlx_vector_array_get(&mut handle as *mut sys::mlx_array, vec, i);
        // We assume the handle returned by get is ready for Rust to own
        out.push(Array { handle });
    }
    sys::mlx_vector_array_free(vec); // Only free what we created or are told to consume
    out
}

/// Helper: Convert WITHOUT freeing (used for trampoline arguments)
unsafe fn vector_to_vec_borrowed(vec: sys::mlx_vector_array) -> Vec<Array> {
    let size = sys::mlx_vector_array_size(vec);
    let mut out = Vec::with_capacity(size);
    for i in 0..size {
        let mut handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        sys::mlx_vector_array_get(&mut handle as *mut sys::mlx_array, vec, i);
        
        // CRITICAL: We create a NEW handle that increments the ref count.
        // If you don't have mlx_retain, your Array::from_handle or .clone() 
        // must handle the C++ reference increment. 
        // For now, we wrap the handle.
        out.push(Array { handle });
    }
    // NOTICE: No free() here. MLX still owns 'vec'.
    out
}

unsafe extern "C" fn trampoline<F>(
    out: *mut sys::mlx_vector_array,
    args: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 
where 
    F: Fn(&[Array]) -> Result<Vec<Array>> 
{
    let closure = &*(payload as *const F);
    
    // We borrow args; we do NOT free them.
    let inputs = vector_to_vec_borrowed(args);

    match closure(&inputs) {
        Ok(outputs) => {
            // We create the output vector. MLX will free this later.
            *out = vec_to_vector(&outputs);
            0
        }
        Err(_) => 1,
    }
}






// /// The trampoline function that MLX-C calls back into.
// unsafe extern "C" fn trampoline<F>(
//     out: *mut sys::mlx_vector_array,
//     args: sys::mlx_vector_array,
//     payload: *mut c_void,
// ) -> i32 
// where 
//     F: Fn(&[Array]) -> Result<Vec<Array>> 
// {
//     let closure = &*(payload as *const F);
//     let inputs = vector_to_vec(args);

//     match closure(&inputs) {
//         Ok(outputs) => {
//             *out = vec_to_vector(&outputs);
//             0
//         }
//         Err(_) => 1,
//     }
// }

/// Computes the Vector-Jacobian Product.
/// f: Function to differentiate
/// primals: The input values at which to evaluate
/// cotangents: The gradient of the output
pub fn vjp<F>(f: F, primals: &[Array], cotangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where 
    F: Fn(&[Array]) -> Result<Vec<Array>> 
{
    unsafe {
        // paylooad pointer to the Rust closure
        let payload = &f as *const F as *mut std::ffi::c_void;
        
        let closure = sys::mlx_closure_new_func_payload(
            Some(trampoline::<F>),
            payload,
            None, // No cleanup function needed since f is local to this scope
        );

        if closure.ctx.is_null() {
            return Err(crate::Error::OperationFailed("Failed to create closure".into()));
        }

        let mut res_values = sys::mlx_vector_array { ctx: std::ptr::null_mut() };
        let mut res_grads = sys::mlx_vector_array { ctx: std::ptr::null_mut() };

        // cumpute the VJP using the MLX C API
        let status = sys::mlx_vjp(
            &mut res_values,
            &mut res_grads,
            closure,
            vec_to_vector(primals),
            vec_to_vector(cotangents),
        );
        sys::mlx_closure_free(closure);

        if status != 0 {
            return Err(crate::Error::OperationFailed("VJP computation failed".into()));
        }
        Ok((vector_to_vec_consume(res_values), vector_to_vec_consume(res_grads)))
    }
}


/// Higher level 'grad' function. 
/// Returns the gradients of the scalar output of 'f' w.r.t the inputs.

pub fn grad<F>(f: F, primals: &[Array]) -> Result<Vec<Array>>
where 
    F: Fn(&[Array]) -> Result<Array> 
{
    // Wrap the scalar function to return a Vec for VJP compatibility
    let wrapped_f = |inputs: &[Array]| -> Result<Vec<Array>> {
        f(inputs).map(|a| vec![a])
    };

    // For a scalar output, the cotangent is just [1.0]
    let cotangents = vec![Array::full(&[], 1.0, crate::Dtype::Float32)?];
    
    let (_, grads) = vjp(wrapped_f, primals, &cotangents)?;
    Ok(grads)
}

/// Computes both the value of the function and its gradients.
pub fn value_and_grad<F>(f: F, primals: &[Array]) -> Result<(Array, Vec<Array>)>
where 
    F: Fn(&[Array]) -> Result<Array> 
{
    let wrapped_f = |inputs: &[Array]| -> Result<Vec<Array>> {
        f(inputs).map(|a| vec![a])
    };

    let cotangents = vec![Array::full(&[], 1.0,crate::Dtype::Float32)?];
    let (values, grads) = vjp(wrapped_f, primals, &cotangents)?;
    
    // values[0] is the result of the original scalar function
    Ok((values[0].clone(), grads))
}