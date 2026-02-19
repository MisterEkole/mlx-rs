use crate::{Array, Error, Result, sys};
use std::path::Path;
use std::ffi::CString;
use std::collections::HashMap;

/// Save a single array to a .npy file.
pub fn save_npy(path: impl AsRef<Path>, array: &Array) -> Result<()> {
    unsafe {
        let c_path = path_to_cstring(path)?;
        let status = sys::mlx_save(c_path.as_ptr(), array.handle);
        
        if status != 0 {
            return Err(Error::OperationFailed("Failed to save .npy file".into()));
        }
        Ok(())
    }
}

/// Load a single array from a .npy file.
pub fn load_npy(path: impl AsRef<Path>) -> Result<Array> {
    unsafe {
        let c_path = path_to_cstring(path)?;
        let mut res_handle = sys::mlx_array { ctx: std::ptr::null_mut() };
        let status = sys::mlx_load(&mut res_handle, c_path.as_ptr(), Array::default_stream());

        if status != 0 || res_handle.ctx.is_null() {
            return Err(Error::OperationFailed("Failed to load .npy file".into()));
        }
        Ok(Array { handle: res_handle })
    }
}

/// Save a map of arrays to a compressed .npz file.

pub fn save_npz(path: impl AsRef<Path>, arrays: &HashMap<String, Array>) -> Result<()> {
    use std::io::Write;
    
    let file = std::fs::File::create(path).map_err(|e| Error::OperationFailed(e.to_string()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);

    let temp_dir = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    for (i, (key, array)) in arrays.iter().enumerate() {
        let temp_filename = format!("mlx_npz_{}_{}_{}.npy", timestamp, i, key);
        let temp_path = temp_dir.join(&temp_filename);
        save_npy(&temp_path, array)?;

      
        let data = std::fs::read(&temp_path).map_err(|e| Error::OperationFailed(e.to_string()))?;

    
        let entry_name = if key.ends_with(".npy") { key.clone() } else { format!("{}.npy", key) };
        zip.start_file(entry_name, options).map_err(|e| Error::OperationFailed(e.to_string()))?;
        zip.write_all(&data).map_err(|e| Error::OperationFailed(e.to_string()))?;

        // Cleanup
        let _ = std::fs::remove_file(temp_path);
    }

    zip.finish().map_err(|e| Error::OperationFailed(e.to_string()))?;
    Ok(())
}


// Internal helper for path conversion
pub fn path_to_cstring(path: impl AsRef<Path>) -> Result<CString> {
    let path_str = path.as_ref().to_str()
        .ok_or_else(|| Error::OperationFailed("Invalid Unicode in path".into()))?;
    CString::new(path_str).map_err(|_| Error::OperationFailed("Path contains null byte".into()))
}