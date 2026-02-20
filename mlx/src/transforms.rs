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
        out.push(Array { handle });
    }
   
    out
}
// trampoline function
unsafe extern "C" fn trampoline<F>(
    out: *mut sys::mlx_vector_array,
    args: sys::mlx_vector_array,
    payload: *mut c_void,
) -> i32 
where 
    F: Fn(&[Array]) -> Result<Vec<Array>> 
{
    let closure = &*(payload as *const F);
    let inputs = vector_to_vec_borrowed(args);

    match closure(&inputs) {
        Ok(outputs) => {
           
            *out = vec_to_vector(&outputs);
            0
        }
        Err(_) => 1,
    }
}


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


// --- Compile Mode Enum ---

/// JIT compilation mode for MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileMode {
    /// Compilation disabled
    Disabled,
    /// Compile but skip simplification passes
    NoSimplify,
    /// Compile but skip fusion passes
    NoFuse,
    /// Full compilation enabled (default)
    Enabled,
}

impl CompileMode {
    fn to_sys(self) -> sys::mlx_compile_mode {
        match self {
            CompileMode::Disabled => sys::mlx_compile_mode__MLX_COMPILE_MODE_DISABLED,
            CompileMode::NoSimplify => sys::mlx_compile_mode__MLX_COMPILE_MODE_NO_SIMPLIFY,
            CompileMode::NoFuse => sys::mlx_compile_mode__MLX_COMPILE_MODE_NO_FUSE,
            CompileMode::Enabled => sys::mlx_compile_mode__MLX_COMPILE_MODE_ENABLED,
        }
    }
}

// --- JIT Compilation ---

/// JIT-compile a function for optimized execution.


pub fn compile<F>(f: F, shapeless: bool) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
    unsafe {
        let f_boxed = Box::new(f);
        let payload = Box::into_raw(f_boxed) as *mut std::ffi::c_void;

        let closure = sys::mlx_closure_new_func_payload(
            Some(trampoline::<F>),
            payload,
            Some(drop_payload::<F>),
        );

        if closure.ctx.is_null() {
            // Clean up the leaked box
            let _ = Box::from_raw(payload as *mut F);
            return Err(crate::Error::OperationFailed("Failed to create closure".into()));
        }

        let mut compiled = sys::mlx_closure { ctx: std::ptr::null_mut() };
        let status = sys::mlx_compile(&mut compiled, closure, shapeless);
        sys::mlx_closure_free(closure);

        if status != 0 || compiled.ctx.is_null() {
            return Err(crate::Error::OperationFailed("mlx_compile failed".into()));
        }

        // Return a callable that invokes the compiled closure
        Ok(move |inputs: &[Array]| -> Result<Vec<Array>> {
            let args = vec_to_vector(inputs);
            let mut res = sys::mlx_vector_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_closure_apply(&mut res, compiled, args);
            sys::mlx_vector_array_free(args);

            if status != 0 || res.ctx.is_null() {
                return Err(crate::Error::OperationFailed("Compiled function call failed".into()));
            }
            Ok(vector_to_vec_consume(res))
        })
    }
}


pub fn set_compile_mode(mode: CompileMode) -> Result<()> {
    let status = unsafe { sys::mlx_set_compile_mode(mode.to_sys()) };
    if status != 0 {
        return Err(crate::Error::OperationFailed("mlx_set_compile_mode failed".into()));
    }
    Ok(())
}


pub fn enable_compile() -> Result<()> {
    let status = unsafe { sys::mlx_enable_compile() };
    if status != 0 {
        return Err(crate::Error::OperationFailed("mlx_enable_compile failed".into()));
    }
    Ok(())
}


pub fn disable_compile() -> Result<()> {
    let status = unsafe { sys::mlx_disable_compile() };
    if status != 0 {
        return Err(crate::Error::OperationFailed("mlx_disable_compile failed".into()));
    }
    Ok(())
}

/// Clear the compilation cache.
pub fn compile_clear_cache() -> Result<()> {
    let status = unsafe { sys::mlx_detail_compile_clear_cache() };
    if status != 0 {
        return Err(crate::Error::OperationFailed("compile_clear_cache failed".into()));
    }
    Ok(())
}

// --- Gradient Checkpointing ---

/// Wrap a function with gradient checkpointing.


pub fn checkpoint<F>(f: F) -> Result<impl Fn(&[Array]) -> Result<Vec<Array>>>
where
    F: Fn(&[Array]) -> Result<Vec<Array>> + 'static,
{
    unsafe {
        let f_boxed = Box::new(f);
        let payload = Box::into_raw(f_boxed) as *mut std::ffi::c_void;

        let closure = sys::mlx_closure_new_func_payload(
            Some(trampoline::<F>),
            payload,
            Some(drop_payload::<F>),
        );

        if closure.ctx.is_null() {
            let _ = Box::from_raw(payload as *mut F);
            return Err(crate::Error::OperationFailed("Failed to create closure".into()));
        }

        let mut checkpointed = sys::mlx_closure { ctx: std::ptr::null_mut() };
        let status = sys::mlx_checkpoint(&mut checkpointed, closure);
        sys::mlx_closure_free(closure);

        if status != 0 || checkpointed.ctx.is_null() {
            return Err(crate::Error::OperationFailed("mlx_checkpoint failed".into()));
        }

        Ok(move |inputs: &[Array]| -> Result<Vec<Array>> {
            let args = vec_to_vector(inputs);
            let mut res = sys::mlx_vector_array { ctx: std::ptr::null_mut() };
            let status = sys::mlx_closure_apply(&mut res, checkpointed, args);
            sys::mlx_vector_array_free(args);

            if status != 0 || res.ctx.is_null() {
                return Err(crate::Error::OperationFailed(
                    "Checkpointed function call failed".into(),
                ));
            }
            Ok(vector_to_vec_consume(res))
        })
    }
}

// --- JVP (Forward-Mode Automatic Differentiation) ---

/// Computes the Jacobian-Vector Product (forward-mode AD).
///
/// Given a function `f`, input `primals`, and `tangents`, computes both
/// the function outputs and the directional derivatives.
///
/// # Returns
/// A tuple of (outputs, output_tangents).
pub fn jvp<F>(f: F, primals: &[Array], tangents: &[Array]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: Fn(&[Array]) -> Result<Vec<Array>>,
{
    unsafe {
        let payload = &f as *const F as *mut std::ffi::c_void;

        let closure = sys::mlx_closure_new_func_payload(
            Some(trampoline::<F>),
            payload,
            None,
        );

        if closure.ctx.is_null() {
            return Err(crate::Error::OperationFailed("Failed to create closure".into()));
        }

        let mut res_values = sys::mlx_vector_array { ctx: std::ptr::null_mut() };
        let mut res_tangents = sys::mlx_vector_array { ctx: std::ptr::null_mut() };

        let status = sys::mlx_jvp(
            &mut res_values,
            &mut res_tangents,
            closure,
            vec_to_vector(primals),
            vec_to_vector(tangents),
        );
        sys::mlx_closure_free(closure);

        if status != 0 {
            return Err(crate::Error::OperationFailed("JVP computation failed".into()));
        }

        Ok((
            vector_to_vec_consume(res_values),
            vector_to_vec_consume(res_tangents),
        ))
    }
}

// --- Async Eval ---

pub fn async_eval(arrays: &[Array]) -> Result<()> {
    unsafe {
        let vec = vec_to_vector(arrays);
        let status = sys::mlx_async_eval(vec);
        sys::mlx_vector_array_free(vec);

        if status != 0 {
            return Err(crate::Error::OperationFailed("mlx_async_eval failed".into()));
        }
        Ok(())
    }
}

// --- Payload cleanup function for long-lived closures ---

/// Cleanup function called by MLX when a closure is freed.
/// This is needed for `compile` and `checkpoint` where the closure outlives
/// the calling scope.
unsafe extern "C" fn drop_payload<F>(payload: *mut c_void) {
    let _ = Box::from_raw(payload as *mut F);
}