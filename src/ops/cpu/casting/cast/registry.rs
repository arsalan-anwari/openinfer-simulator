use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;

use crate::graph::OpKind;
use crate::ops::registry::{KernelFn, OpKey, OpMode};
use crate::op_defs::op_schema;
use super::kernel;

pub static ENTRIES: Lazy<Vec<(OpKey, KernelFn)>> = Lazy::new(|| {
    build_cast_entries().unwrap_or_else(|err| panic!("cast entry build failed: {}", err))
});

fn build_cast_entries() -> Result<Vec<(OpKey, KernelFn)>> {
    let kind = OpKind::Cast;
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let output_dtypes = schema
        .output_dtypes_from_attr()
        .ok_or_else(|| anyhow!("op {:?} has from_attr output but no output dtypes", kind))?;

    let mut entries = Vec::new();
    for in_dtype in schema.input_tensor_types {
        for &out_dtype in output_dtypes {
            if !kernel::is_allowed_cast(*in_dtype, out_dtype) {
                continue;
            }
            let key = OpKey {
                kind,
                mode: OpMode::Normal,
                broadcast: false,
                inputs: vec![*in_dtype; schema.input_count],
                out0: out_dtype,
                acc_rule: None,
            };
            entries.push((key, kernel::cast_normal_dispatch as KernelFn));
        }
    }
    Ok(entries)
}
