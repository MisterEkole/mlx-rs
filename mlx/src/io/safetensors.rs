use crate::{Array, Error, Result, sys};
use std::collections::HashMap;
use std::path::Path;
use std::ffi::{CString, CStr};

/// Save weights to a .safetensors file.
pub fn save_safetensors(path: impl AsRef<Path>, weights: &HashMap<String, Array>) -> Result<()> {
    unsafe {
        let c_path = super::numpy::path_to_cstring(path)?; // Re-use helper
        
        let map = sys::mlx_map_string_to_array_new();
        for (k, v) in weights {
            let c_key = CString::new(k.as_str()).map_err(|_| Error::OperationFailed("Invalid key".into()))?;
            sys::mlx_map_string_to_array_insert(map, c_key.as_ptr(), v.handle);
        }
        let metadata = sys::mlx_map_string_to_string_new();

        let status = sys::mlx_save_safetensors(c_path.as_ptr(), map, metadata);
        
        sys::mlx_map_string_to_array_free(map);
        sys::mlx_map_string_to_string_free(metadata);

        if status != 0 {
            Err(Error::OperationFailed("Failed to save safetensors".into()))
        } else {
            Ok(())
        }
    }
}

/// Load weights from a .safetensors file into a Rust HashMap.
pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    unsafe {
        let c_path = super::numpy::path_to_cstring(path)?;
        let mut map_handle = sys::mlx_map_string_to_array { ctx: std::ptr::null_mut() };
        let mut metadata_handle = sys::mlx_map_string_to_string { ctx: std::ptr::null_mut() };
        
        let status = sys::mlx_load_safetensors(
            &mut map_handle,
            &mut metadata_handle,
            c_path.as_ptr(),
            Array::default_stream(),
        );

        if status != 0 || map_handle.ctx.is_null() {
            return Err(Error::OperationFailed("Failed to load safetensors".into()));
        }

        let mut results = HashMap::new();
        let iterator = sys::mlx_map_string_to_array_iterator_new(map_handle);
        let mut key_ptr: *const i8 = std::ptr::null();
        let mut val_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        
        while sys::mlx_map_string_to_array_iterator_next(&mut key_ptr, &mut val_handle, iterator) == 0 {
            if !key_ptr.is_null() {
                let key = CStr::from_ptr(key_ptr).to_string_lossy().into_owned();
                results.insert(key, Array { handle: val_handle });
            }
        }

        sys::mlx_map_string_to_array_iterator_free(iterator);
        sys::mlx_map_string_to_array_free(map_handle);
        sys::mlx_map_string_to_string_free(metadata_handle);

        Ok(results)
    }
}