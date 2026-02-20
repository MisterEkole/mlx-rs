/// A distributed communication group
/// 
/// Represents a group of MLX processes that can communicate with each other.

use crate::sys;
use std::ptr;

pub struct DistributedGroup{
    pub(crate) handle: sys::mlx_distributed_group,
}

impl DistributedGroup{
    pub fn rank(&self) -> i32{
        unsafe{ sys::mlx_distributed_group_rank(self.handle) }
    }

    pub fn size(&self) -> i32{
        unsafe{ sys::mlx_distributed_group_size(self.handle) }
    }

    pub fn group_split(&self, color: i32, key: i32) -> DistributedGroup{
            let handle = unsafe{ sys::mlx_distributed_group_split(self.handle, color, key) };
            DistributedGroup{handle}
        
    }

}

impl Drop for DistributedGroup{
    fn drop(&mut self){
      
    }
}
pub fn is_available() -> bool{
    unsafe{ sys::mlx_distributed_is_available() }
}

pub fn init(strict:bool) -> DistributedGroup{
    let handle =unsafe{ sys::mlx_distributed_init(strict) };
    DistributedGroup{handle}
}

pub(crate) fn null_group() -> sys:: mlx_distributed_group{
    sys::mlx_distributed_group{ctx:ptr::null_mut()}
}
