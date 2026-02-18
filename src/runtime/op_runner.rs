use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::ops::{lookup_kernel, OpKey, OpMode};
use crate::op_defs::op_schema;
use crate::runtime::state::SharedTensor;
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

/// Execute a single op kernel given inputs and optional output storage.
pub fn exec_op(
    device: Device,
    op: OpKind,
    attrs: &OpAttrs,
    inputs: &[TensorValue],
    output: Option<&SharedTensor>,
    is_inplace: bool,
) -> Result<()> {
    for input in inputs {
        ensure_layout_supported(input)?;
    }
    let schema = op_schema(op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
    let input_dtypes = inputs.iter().map(|tensor| tensor.dtype()).collect::<Vec<_>>();
    let is_accumulate = schema.accumulate.allow() && attrs.items.iter().any(|attr| attr.name == "acc");
    let is_broadcast = schema.broadcast.allow()
        && inputs
            .windows(2)
            .any(|pair| pair[0].shape() != pair[1].shape());
    let is_inplace = schema.inplace.allow() && is_inplace;
    let output_dtype = if is_accumulate {
        acc_dtype(attrs)?
    } else {
        schema.type_rule.output_dtype(&input_dtypes, attrs)?
    };
    let mode = if is_accumulate {
        OpMode::Accumulate
    } else if is_inplace {
        OpMode::Inplace
    } else {
        OpMode::Normal
    };

    let key = OpKey {
        kind: op,
        mode,
        broadcast: is_broadcast,
        inputs: input_dtypes.clone(),
        out0: output_dtype,
    };

    let kernel = lookup_kernel(device, key)?;
    let mut output_guard = match output {
        Some(shared) => Some(
            shared
                .lock()
                .map_err(|_| anyhow!("output tensor lock poisoned"))?,
        ),
        None => None,
    };
    if let Some(out) = output_guard.as_ref() {
        ensure_layout_supported(out)?;
    }
    kernel(attrs, inputs, output_guard.as_deref_mut())
}

fn ensure_layout_supported(value: &TensorValue) -> Result<()> {
    if value.has_negative_strides() {
        return Err(anyhow!(
            "non-contiguous execution is not yet supported: negative strides for dtype {:?}",
            value.dtype()
        ));
    }
    if !value.is_contiguous_layout() {
        return Err(anyhow!(
            "non-contiguous execution is not yet supported: layout must be contiguous with zero offset for dtype {:?}",
            value.dtype()
        ));
    }
    Ok(())
}

fn acc_dtype(attrs: &OpAttrs) -> Result<DType> {
    attrs
        .items
        .iter()
        .find(|attr| attr.name == "acc")
        .ok_or_else(|| anyhow!("missing acc attribute"))
        .and_then(|attr| match &attr.value {
            AttrValue::DType(dtype) => Ok(*dtype),
            _ => Err(anyhow!("acc attribute must be a dtype")),
        })
}
