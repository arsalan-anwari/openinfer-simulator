use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_with_accumulation, KernelFn, OpKey, OpMode};

use super::kernel::sum_axis_normal_dispatch;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_with_accumulation(OpKind::SumAxis, |mode| match mode {
        OpMode::Normal => Some(sum_axis_normal_dispatch),
        OpMode::Inplace => None,
    })
    .expect("failed to build sum_axis cpu entries")
});
