use anyhow::{anyhow, Result};

use crate::graph::OpAttrs;
use crate::op_defs::{op_schema, OutputType};
use crate::simulator::Device;
use crate::tensor::{DType, TensorValue};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpMode {
    Normal,
    Inplace,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OpKey {
    pub kind: crate::graph::OpKind,
    pub mode: OpMode,
    pub broadcast: bool,
    pub inputs: Vec<DType>,
    pub out0: DType,
}

pub type KernelFn = fn(&OpAttrs, &[TensorValue], Option<&mut TensorValue>) -> Result<()>;

#[allow(unused)]
pub fn op_supports_dtype(
    kind: crate::graph::OpKind,
    _mode: OpMode,
    in0: DType,
    out0: DType,
    attrs: &OpAttrs,
) -> bool {
    let schema = match op_schema(kind) {
        Some(schema) => schema,
        None => return false,
    };
    let input_dtypes = vec![in0; schema.input_count];
    crate::op_defs::supports_pattern(schema, &input_dtypes, out0, attrs)
}

pub fn build_op_entries_same_input(
    kind: crate::graph::OpKind,
    kernel_for_mode: impl Fn(OpMode) -> Option<KernelFn>,
) -> Result<Vec<(OpKey, KernelFn)>> {
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let input_count = schema.input_count;
    let broadcast_flags: &[bool] = if schema.supports_broadcast {
        &[false, true]
    } else {
        &[false]
    };

    let mut entries = Vec::new();
    for in_dtype in schema.input_tensor_types {
        let out_dtype = match &schema.output_type {
            OutputType::Same => *in_dtype,
            OutputType::Fixed(dtype) => *dtype,
            OutputType::FromAttr(_) => {
                return Err(anyhow!(
                    "op {:?} has from_attr output, use build_op_entries_with_outputs",
                    kind
                ))
            }
        };
        for &broadcast in broadcast_flags {
            let normal_key = OpKey {
                kind,
                mode: OpMode::Normal,
                broadcast,
                inputs: vec![*in_dtype; input_count],
                out0: out_dtype,
            };
            if let Some(kernel) = kernel_for_mode(OpMode::Normal) {
                entries.push((normal_key, kernel));
            }
            if schema.supports_inplace {
                let inplace_key = OpKey {
                    kind,
                    mode: OpMode::Inplace,
                    broadcast,
                    inputs: vec![*in_dtype; input_count],
                    out0: out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Inplace) {
                    entries.push((inplace_key, kernel));
                }
            }
        }
    }
    Ok(entries)
}

#[allow(unused)]
pub fn build_op_entries_with_outputs(
    kind: crate::graph::OpKind,
    output_dtypes: &[DType],
    kernel_for_mode: impl Fn(OpMode) -> Option<KernelFn>,
) -> Result<Vec<(OpKey, KernelFn)>> {
    let schema = op_schema(kind).ok_or_else(|| anyhow!("missing op schema {:?}", kind))?;
    let input_count = schema.input_count;
    let broadcast_flags: &[bool] = if schema.supports_broadcast {
        &[false, true]
    } else {
        &[false]
    };

    let mut entries = Vec::new();
    for in_dtype in schema.input_tensor_types {
        for &out_dtype in output_dtypes {
            for &broadcast in broadcast_flags {
                let normal_key = OpKey {
                    kind,
                    mode: OpMode::Normal,
                    broadcast,
                    inputs: vec![*in_dtype; input_count],
                    out0: out_dtype,
                };
                if let Some(kernel) = kernel_for_mode(OpMode::Normal) {
                    entries.push((normal_key, kernel));
                }
                if schema.supports_inplace {
                    let inplace_key = OpKey {
                        kind,
                        mode: OpMode::Inplace,
                        broadcast,
                        inputs: vec![*in_dtype; input_count],
                        out0: out_dtype,
                    };
                    if let Some(kernel) = kernel_for_mode(OpMode::Inplace) {
                        entries.push((inplace_key, kernel));
                    }
                }
            }
        }
    }
    Ok(entries)
}

pub fn lookup_kernel(device: Device, key: OpKey) -> Result<KernelFn> {
    match device {
        Device::Cpu => crate::ops::cpu::registry::lookup_kernel(key),
        Device::Vulkan => {
            #[cfg(feature = "vulkan")]
            {
                crate::ops::vulkan::registry::lookup_kernel(key)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                Err(anyhow!("device {:?} requires the vulkan feature", device))
            }
        }
    }
}
