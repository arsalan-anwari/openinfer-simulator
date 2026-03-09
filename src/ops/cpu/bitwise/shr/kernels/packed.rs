use anyhow::Result;

use crate::graph::OpAttrs;
use crate::ops::cpu::packed_ops::{
    packed_unary_signed, packed_unary_signed_inplace, packed_unary_unsigned,
    packed_unary_unsigned_inplace,
};
use crate::tensor::{I4, U4, Tensor};

use super::common::shr_bits;



pub fn shr_i4_packed(attrs: &OpAttrs, a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    packed_unary_signed(a, out, 4, |v| v.wrapping_shr(bits))
}

pub fn shr_i4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<I4>) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    packed_unary_signed_inplace(a, 4, |v| v.wrapping_shr(bits))
}



pub fn shr_u4_packed(attrs: &OpAttrs, a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    packed_unary_unsigned(a, out, 4, |v| v.wrapping_shr(bits))
}

pub fn shr_u4_packed_inplace(attrs: &OpAttrs, a: &mut Tensor<U4>) -> Result<()> {
    let bits = shr_bits(attrs)? as u32;
    packed_unary_unsigned_inplace(a, 4, |v| v.wrapping_shr(bits))
}
