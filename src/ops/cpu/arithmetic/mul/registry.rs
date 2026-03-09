use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_same_input, KernelFn, OpKey, OpMode};

use super::kernel::{mul_inplace_dispatch, mul_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_same_input(OpKind::Mul, |mode| match mode {
        OpMode::Normal => Some(mul_normal_dispatch),
        OpMode::Inplace => Some(mul_inplace_dispatch),
    })
    .expect("failed to build mul cpu entries")
});
