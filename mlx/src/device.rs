use crate::sys;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum DeviceType {
    Cpu = 0,
    Gpu = 1,
}

pub struct Device {
    pub(crate) handle: sys::mlx_device,
}

impl Device {
    /// Creates a device of a specific type (CPU or GPU)
    pub fn new(device_type: DeviceType) -> Self {
        unsafe {
            // This function returns the struct directly!
            let handle = sys::mlx_device_new_type(device_type as u32, 0);
            Device { handle }
        }
    }

    /// Fetches the current default device
    pub fn default() -> Self {
        unsafe {
          
            let mut handle = sys::mlx_device { ctx: std::ptr::null_mut() };
            sys::mlx_get_default_device(&mut handle);
            Device { handle }
        }
    }

    /// Sets this device as the global default for all operations
    pub fn set_default(&self) -> crate::Result<()> {
        unsafe {
            let status = sys::mlx_set_default_device(self.handle);
            if status != 0 {
                return Err(crate::Error::OperationFailed);
            }
            Ok(())
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.ctx.is_null() {
            
                sys::mlx_device_free(self.handle); 
            }
        }
    }
}