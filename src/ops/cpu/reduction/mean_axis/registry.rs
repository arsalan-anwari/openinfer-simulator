use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{build_op_entries_with_accumulation, KernelFn, OpKey, OpMode};

use super::kernel::mean_axis_normal_dispatch;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_op_entries_with_accumulation(OpKind::MeanAxis, |mode| match mode {
        OpMode::Normal => Some(mean_axis_normal_dispatch),
        OpMode::Inplace => None,
    })
    .expect("failed to build mean_axis cpu entries")
});
