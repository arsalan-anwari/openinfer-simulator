use anyhow::{anyhow, Result};

use crate::graph::{OpAttrs, OpKind};
use crate::ops::{lookup_kernel, OpKey, OpMode};
use crate::op_defs::{acc_list, op_schema, supports_tuple};
use crate::runtime::state::SharedTensor;
use crate::simulator::Device;
use crate::tensor::TensorValue;

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
    let requested_acc = if is_accumulate {
        acc_list(attrs)?
    } else {
        Vec::new()
    };
    let is_broadcast = schema.broadcast.allow()
        && inputs
            .windows(2)
            .any(|pair| pair[0].shape() != pair[1].shape());
    let is_inplace = schema.inplace.allow() && is_inplace;
    let mode = if is_accumulate {
        OpMode::Accumulate
    } else if is_inplace {
        OpMode::Inplace
    } else {
        OpMode::Normal
    };
    let mut output_guard = match output {
        Some(shared) => Some(
            shared
                .lock()
                .map_err(|_| anyhow!("output tensor lock poisoned"))?,
        ),
        None => None,
    };
    let output_dtype = if let Some(out) = output_guard.as_ref() {
        out.dtype()
    } else if mode == OpMode::Accumulate {
        requested_acc
            .last()
            .copied()
            .ok_or_else(|| anyhow!("missing acc dtype list for accumulate mode"))?
    } else {
        schema.type_rule.output_dtype(&input_dtypes, attrs)?
    };
    if !supports_tuple(
        schema,
        &input_dtypes,
        &requested_acc,
        output_dtype,
        mode == OpMode::Accumulate,
    ) {
        return Err(anyhow!(
            "unsupported op typing tuple at runtime for {}: inputs={:?}, acc={:?}, out={:?}, mode={:?}",
            op,
            input_dtypes,
            requested_acc,
            output_dtype,
            mode
        ));
    }

    let key = OpKey {
        kind: op,
        mode,
        broadcast: is_broadcast,
        inputs: input_dtypes.clone(),
        out0: output_dtype,
    };

    let kernel = lookup_kernel(device, key)?;
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
