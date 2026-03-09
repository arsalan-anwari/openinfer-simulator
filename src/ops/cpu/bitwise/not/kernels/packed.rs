use anyhow::Result;

use crate::ops::cpu::packed_ops::{
    packed_unary_signed, packed_unary_signed_inplace, packed_unary_unsigned,
    packed_unary_unsigned_inplace,
};
use crate::tensor::{I4, U4, Tensor};



pub fn not_i4_packed(a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed(a, out, 4, |v| !v)
}



pub fn not_u4_packed(a: &Tensor<U4>, out: &mut Tensor<U4>) -> Result<()> {
    packed_unary_unsigned(a, out, 4, |v| !v)
}



pub fn not_i4_packed_inplace(a: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed_inplace(a, 4, |v| !v)
}



pub fn not_u4_packed_inplace(a: &mut Tensor<U4>) -> Result<()> {
    packed_unary_unsigned_inplace(a, 4, |v| !v)
}
