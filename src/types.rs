use anyhow::Result;

use crate::tensor::DType;

/// Convert a dtype to its suffix string representation.
#[allow(unused)]
pub fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F8 => Ok("f8"),
        DType::BF16 => Ok("bf16"),
        DType::F16 => Ok("f16"),
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        DType::I4 => Ok("i4"),
        DType::I8 => Ok("i8"),
        DType::I16 => Ok("i16"),
        DType::I32 => Ok("i32"),
        DType::I64 => Ok("i64"),
        DType::U4 => Ok("u4"),
        DType::U8 => Ok("u8"),
        DType::U16 => Ok("u16"),
        DType::U32 => Ok("u32"),
        DType::U64 => Ok("u64"),
        DType::Bool => Ok("bool"),
    }
}

/// Metadata about a variable stored in a model file.
#[derive(Debug, Clone)]
pub struct VarInfo {
    pub name: String,
    pub dtype: DType,
    pub dims: Vec<String>,
    pub value_range: Option<(usize, usize)>,
    pub has_data: bool,
}
