use anyhow::{anyhow, Result};

use crate::ops::cpu::packed_cpu::PackedBits;
use crate::ops::cpu::packed_ops::{packed_compare_signed, packed_compare_unsigned};
use crate::tensor::{I4, U4, Tensor};

fn ensure_same_shape<T>(a: &Tensor<T>, b: &Tensor<T>, out: &Tensor<bool>) -> Result<()> {
    if a.shape() != b.shape() {
        return Err(anyhow!(
            "input shapes {:?} and {:?} must match",
            a.shape(),
            b.shape()
        ));
    }
    if out.shape() != a.shape() {
        return Err(anyhow!(
            "output shape {:?} does not match input shape {:?}",
            out.shape(),
            a.shape()
        ));
    }
    Ok(())
}

fn gt_packed_signed<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b, out)?;
    packed_compare_signed(a, b, out, width, |l, r| l > r)
}

fn gt_packed_unsigned<T: PackedBits>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    out: &mut Tensor<bool>,
    width: u8,
) -> Result<()> {
    ensure_same_shape(a, b, out)?;
    packed_compare_unsigned(a, b, out, width, |l, r| l > r)
}



pub fn gt_i4_packed(a: &Tensor<I4>, b: &Tensor<I4>, out: &mut Tensor<bool>) -> Result<()> {
    gt_packed_signed(a, b, out, 4)
}



pub fn gt_u4_packed(a: &Tensor<U4>, b: &Tensor<U4>, out: &mut Tensor<bool>) -> Result<()> {
    gt_packed_unsigned(a, b, out, 4)
}
