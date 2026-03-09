use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_with_accumulation, KernelFn, OpKey, OpMode};

use super::kernel::{matmul_inplace_dispatch, matmul_normal_dispatch};

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_with_accumulation(OpKind::Matmul, |mode| match mode {
        OpMode::Normal => Some(matmul_normal_dispatch),
        OpMode::Inplace => Some(matmul_inplace_dispatch),
    })
    .expect("failed to build matmul cpu entries")
});
