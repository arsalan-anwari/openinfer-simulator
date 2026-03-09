use once_cell::sync::Lazy;

use crate::ops::registry::{build_op_entries_with_accumulation, KernelFn, OpKey, OpMode};

use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_with_accumulation(crate::graph::OpKind::Matmul, |mode| match mode {
        OpMode::Normal => Some(kernel::matmul_normal_dispatch),
        OpMode::Inplace => Some(kernel::matmul_inplace_dispatch),
    })
    .expect("failed to build matmul vulkan entries")
});
