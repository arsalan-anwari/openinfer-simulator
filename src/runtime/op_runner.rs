use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttr, OpAttrs, OpKind};
use crate::ops::{lookup_kernel, OpKey, OpMode};
use crate::op_defs::{op_schema, OpSchema, supports_pattern};
use crate::runtime::state::SharedTensor;
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

/// Resolve acc_rule from attrs or schema default for dispatch.
/// When acc is omitted, returns None so output_dtype uses output_type (Same) for backward compatibility.
pub(crate) fn resolve_acc_rule(
    schema: &OpSchema,
    _input_dtypes: &[DType],
    attrs: &OpAttrs,
) -> Option<Vec<DType>> {
    if schema.accumulation_rules.is_empty() {
        return None;
    }
    if let Some(attr) = attrs.items.iter().find(|a| a.name == "acc") {
        if let AttrValue::DTypeList(rule) = &attr.value {
            return Some(rule.clone());
        }
    }
    // When acc is omitted: return None so output_dtype uses output_type (Same).
    None
}

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
    let is_broadcast = schema.supports_broadcast
        && inputs
            .windows(2)
            .any(|pair| pair[0].shape() != pair[1].shape());
    let is_inplace = schema.supports_inplace && is_inplace;
    let mode = if is_inplace {
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
    } else {
        schema.output_dtype(&input_dtypes, attrs)?
    };
    if !supports_pattern(schema, &input_dtypes, output_dtype, attrs) {
        return Err(anyhow!(
            "unsupported op typing tuple at runtime for {}: inputs={:?}, out={:?}",
            op,
            input_dtypes,
            output_dtype
        ));
    }

    let acc_rule = resolve_acc_rule(op_schema(op).unwrap(), &input_dtypes, attrs);
    let key = OpKey {
        kind: op,
        mode,
        broadcast: is_broadcast,
        inputs: input_dtypes.clone(),
        out0: output_dtype,
        acc_rule: acc_rule.clone(),
    };

    // Ensure attrs has acc when acc_rule is set, so kernels can read it
    let attrs_to_use = if let Some(rule) = &acc_rule {
        if !attrs.items.iter().any(|a| a.name == "acc") {
            let mut items = attrs.items.clone();
            items.push(OpAttr {
                name: "acc".to_string(),
                value: AttrValue::DTypeList(rule.clone()),
            });
            OpAttrs { items }
        } else {
            attrs.clone()
        }
    } else {
        attrs.clone()
    };

    let kernel = lookup_kernel(device, key)?;
    if let Some(out) = output_guard.as_ref() {
        ensure_layout_supported(out)?;
    }
    kernel(&attrs_to_use, inputs, output_guard.as_deref_mut())
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
