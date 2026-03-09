use anyhow::{anyhow, Result};
use bytemuck::{Pod, Zeroable};

use crate::ops::cpu::broadcast::broadcast_strides;
use crate::tensor::{DType, TensorValue};

include!(concat!(env!("OUT_DIR"), "/vulkan_config.rs"));

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct TensorDesc {
    pub rank: u32,
    pub dtype: u32,
    pub elem_bits: u32,
    pub byte_offset: u32,
    pub shape: [u32; MAX_DIMS],
    pub strides: [u32; MAX_DIMS],
}

pub fn dtype_code(dtype: DType) -> u32 {
    match dtype {
        DType::I8 => 1,
        DType::I16 => 2,
        DType::I32 => 3,
        DType::I64 => 4,
        DType::U8 => 5,
        DType::U16 => 6,
        DType::U32 => 7,
        DType::U64 => 8,
        DType::F16 => 9,
        DType::BF16 => 10,
        DType::F32 => 11,
        DType::F64 => 12,
        DType::F8 => 13,
        DType::Bool => 14,
        DType::I4 => 16,
        DType::U4 => 19,
    }
}

/// Encode an accumulation rule as a u32 bit pattern for Vulkan push constants.
/// Uses 8 bits per dtype (codes 0-255). For rules longer than 4 dtypes, only first 4 are encoded.
pub fn encode_acc_rule(rule: &[DType]) -> u32 {
    rule.iter()
        .enumerate()
        .take(4)
        .map(|(i, d)| (dtype_code(*d) as u32).min(0xFF) << (i * 8))
        .fold(0u32, |a, b| a | b)
}

/// Default accumulation dtype for ops that need accumulation but have no accumulation_rules.
/// - Input ≤ 16-bit: Use 32-bit accumulation (i32, u32, f32 as appropriate).
/// - Input 32/64-bit: Use largest available dtype (64-bit if supported).
/// - Vulkan without 64-bit: Always use 32-bit accumulation, even for 64-bit inputs.
#[allow(unused)]
pub fn default_accum_dtype(input_dtype: DType, supports_64bit: bool) -> DType {
    let bits = input_dtype.bit_width();
    if bits <= 16 {
        match input_dtype {
            DType::F8 | DType::BF16 | DType::F16 => DType::F32,
            DType::I4 | DType::I8 | DType::I16 => DType::I32,
            DType::U4 | DType::U8 | DType::U16 => DType::U32,
            _ => DType::F32,
        }
    } else {
        if supports_64bit {
            match input_dtype {
                DType::F64 | DType::I64 | DType::U64 => input_dtype,
                DType::F32 => DType::F64,
                DType::I32 => DType::I64,
                DType::U32 => DType::U64,
                _ => DType::I64,
            }
        } else {
            match input_dtype {
                DType::F32 | DType::F64 => DType::F32,
                DType::I32 | DType::I64 => DType::I32,
                DType::U32 | DType::U64 => DType::U32,
                _ => DType::I32,
            }
        }
    }
}

pub fn build_tensor_desc(
    value: &TensorValue,
    out_rank: usize,
    byte_offset: u32,
) -> Result<TensorDesc> {
    let dtype = value.dtype();
    let shape = value.shape();
    if out_rank > MAX_DIMS {
        return Err(anyhow!(
            "vulkan tensors only support up to {} dims (got {})",
            MAX_DIMS,
            out_rank
        ));
    }
    let strides = broadcast_strides(shape, value.strides(), out_rank);
    let mut desc = TensorDesc::default();
    desc.rank = out_rank as u32;
    desc.dtype = dtype_code(dtype);
    desc.elem_bits = dtype.bit_width() as u32;
    desc.byte_offset = byte_offset;
    let offset = out_rank.saturating_sub(shape.len());
    for i in 0..out_rank {
        if i < offset {
            desc.shape[i] = 1;
        } else {
            desc.shape[i] = shape[i - offset] as u32;
        }
        desc.strides[i] = strides[i] as u32;
    }
    Ok(desc)
}

pub fn build_tensor_desc_broadcast(
    value: &TensorValue,
    out_shape: &[usize],
    byte_offset: u32,
) -> Result<TensorDesc> {
    let dtype = value.dtype();
    let out_rank = out_shape.len();
    if out_rank > MAX_DIMS {
        return Err(anyhow!(
            "vulkan tensors only support up to {} dims (got {})",
            MAX_DIMS,
            out_rank
        ));
    }
    let strides = broadcast_strides(value.shape(), value.strides(), out_rank);
    let mut desc = TensorDesc::default();
    desc.rank = out_rank as u32;
    desc.dtype = dtype_code(dtype);
    desc.elem_bits = dtype.bit_width() as u32;
    desc.byte_offset = byte_offset;
    for i in 0..out_rank {
        desc.shape[i] = out_shape[i] as u32;
        desc.strides[i] = strides[i] as u32;
    }
    Ok(desc)
}
