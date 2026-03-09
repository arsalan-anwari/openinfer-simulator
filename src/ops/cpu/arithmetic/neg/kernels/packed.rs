use anyhow::Result;

use crate::ops::cpu::packed_ops::{
    packed_unary_signed, packed_unary_signed_inplace,
};
use crate::tensor::{I4, Tensor};



pub fn neg_i4_packed(a: &Tensor<I4>, out: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed(a, out, 4, |v| v.wrapping_neg())
}



pub fn neg_i4_packed_inplace(a: &mut Tensor<I4>) -> Result<()> {
    packed_unary_signed_inplace(a, 4, |v| v.wrapping_neg())
}
