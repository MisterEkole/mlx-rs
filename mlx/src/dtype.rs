use mlx_sys as sys;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Dtype {
    Bool,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
    BFloat16,
    Complex64,
}

impl Dtype {
    pub(crate) fn to_sys(self) -> sys::mlx_dtype_ {
        match self {
            Dtype::Bool => sys::mlx_dtype__MLX_BOOL,
            Dtype::UInt8 => sys::mlx_dtype__MLX_UINT8,
            Dtype::UInt16 => sys::mlx_dtype__MLX_UINT16,
            Dtype::UInt32 => sys::mlx_dtype__MLX_UINT32,
            Dtype::UInt64 => sys::mlx_dtype__MLX_UINT64,
            Dtype::Int8 => sys::mlx_dtype__MLX_INT8,
            Dtype::Int16 => sys::mlx_dtype__MLX_INT16,
            Dtype::Int32 => sys::mlx_dtype__MLX_INT32,
            Dtype::Int64 => sys::mlx_dtype__MLX_INT64,
            Dtype::Float16 => sys::mlx_dtype__MLX_FLOAT16,
            Dtype::Float32 => sys::mlx_dtype__MLX_FLOAT32,
            Dtype::BFloat16 => sys::mlx_dtype__MLX_BFLOAT16,
            Dtype::Complex64 => sys::mlx_dtype__MLX_COMPLEX64,
        }
    }

    pub(crate) fn from_sys(sys_dtype: sys::mlx_dtype_) -> Self {
        match sys_dtype {
            sys::mlx_dtype__MLX_BOOL => Dtype::Bool,
            sys::mlx_dtype__MLX_UINT8 => Dtype::UInt8,
            sys::mlx_dtype__MLX_UINT16 => Dtype::UInt16,
            sys::mlx_dtype__MLX_UINT32 => Dtype::UInt32,
            sys::mlx_dtype__MLX_UINT64 => Dtype::UInt64,
            sys::mlx_dtype__MLX_INT8 => Dtype::Int8,
            sys::mlx_dtype__MLX_INT16 => Dtype::Int16,
            sys::mlx_dtype__MLX_INT32 => Dtype::Int32,
            sys::mlx_dtype__MLX_INT64 => Dtype::Int64,
            sys::mlx_dtype__MLX_FLOAT16 => Dtype::Float16,
            sys::mlx_dtype__MLX_FLOAT32 => Dtype::Float32,
            sys::mlx_dtype__MLX_BFLOAT16 => Dtype::BFloat16,
            sys::mlx_dtype__MLX_COMPLEX64 => Dtype::Complex64,
            _ => panic!("Unknown MLX dtype value: {}", sys_dtype),
        }
    }
}


impl From<Dtype> for sys::mlx_dtype_ {
    fn from(dtype: Dtype) -> Self {
        dtype.to_sys()
    }
}

impl From<sys::mlx_dtype_> for Dtype {
    fn from(sys_dtype: sys::mlx_dtype_) -> Self {
        Dtype::from_sys(sys_dtype)
    }
}