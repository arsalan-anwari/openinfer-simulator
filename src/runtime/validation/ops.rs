use anyhow::{anyhow, Result};

use crate::graph::{MemoryKind, OpAttrs, OpKind};
use crate::op_defs::{op_schema, supports_pattern};
use super::attrs;
use super::context::ValidationContext;

pub fn validate_op(
    ctx: &ValidationContext,
    op: OpKind,
    attrs: &OpAttrs,
    inputs: &[String],
    output: &str,
) -> Result<()> {
    let schema = op_schema(op).ok_or_else(|| anyhow!("unsupported op {}", op))?;
    if inputs.len() != schema.input_count {
        return Err(anyhow!(
            "op {} expects {} inputs, got {}",
            op,
            schema.input_count,
            inputs.len()
        ));
    }
    if schema.output_count != 1 {
        return Err(anyhow!(
            "op {} expects {} outputs, got 1",
            op,
            schema.output_count
        ));
    }
    if output.trim().is_empty() {
        return Err(anyhow!("op {} missing output", op));
    }

    let mut input_dtypes = Vec::with_capacity(inputs.len());
    let mut input_shapes = Vec::with_capacity(inputs.len());
    for input in inputs {
        if !ctx.has_var(input) {
            return Err(anyhow!("unknown input variable {}", input));
        }
        if let Some(decl) = ctx.decl_for(input) {
            if decl.kind == MemoryKind::Persistent {
                return Err(anyhow!(
                    "persistent cache {} must be read via cache.read",
                    input
                ));
            }
        }
        input_dtypes.push(ctx.var_dtype(input)?);
        input_shapes.push(ctx.var_shape(input)?);
    }

    if !schema.supports_broadcast && input_shapes.windows(2).any(|pair| pair[0] != pair[1]) {
        return Err(anyhow!("op {} does not allow broadcast inputs", op));
    }

    let output_decl = ctx.decl_for(output);
    let output_dtype = if let Some(decl) = output_decl {
        match decl.kind {
            MemoryKind::Constant => {
                return Err(anyhow!("cannot write to constant memory: {}", output));
            }
            MemoryKind::Persistent => {
                return Err(anyhow!(
                    "persistent cache {} must be written via cache.write",
                    output
                ));
            }
            _ => {}
        }
        decl.dtype
    } else if !ctx.temps.contains(output) {
        return Err(anyhow!("unknown output variable {}", output));
    } else {
        ctx.var_dtype(output)?
    };

    let inferred = schema.output_dtype(&input_dtypes, attrs)?;
    if inferred != output_dtype {
        return Err(anyhow!(
            "op {} output dtype mismatch for {}: expected {:?}, got {:?}",
            op,
            output,
            output_dtype,
            inferred
        ));
    }

    let is_inplace = inputs.iter().any(|name| name == output);
    if is_inplace && !schema.supports_inplace {
        return Err(anyhow!("op {} does not support inplace writes", op));
    }

    if !input_dtypes.is_empty() {
        if !supports_pattern(schema, &input_dtypes, output_dtype, attrs) {
            return Err(anyhow!(
                "unsupported op typing tuple for {}: inputs={:?}, out={:?}",
                op,
                input_dtypes,
                output_dtype
            ));
        }
    }

    attrs::validate_attrs(ctx, op, attrs, schema.parameter_types)?;
    Ok(())
}

pub fn validate_transfer(ctx: &ValidationContext, src: &str, dst: &str) -> Result<()> {
    if !ctx.has_var(src) {
        return Err(anyhow!("unknown transfer source {}", src));
    }
    if !ctx.has_var(dst) {
        return Err(anyhow!("unknown transfer destination {}", dst));
    }
    if let Some(decl) = ctx.decl_for(dst) {
        if matches!(decl.kind, MemoryKind::Constant | MemoryKind::Persistent) {
            return Err(anyhow!("cannot transfer into {}", dst));
        }
    }
    let src_dtype = ctx.var_dtype(src)?;
    let dst_dtype = ctx.var_dtype(dst)?;
    if src_dtype != dst_dtype {
        return Err(anyhow!(
            "transfer dtype mismatch {} -> {}: {:?} vs {:?}",
            src,
            dst,
            src_dtype,
            dst_dtype
        ));
    }
    let src_shape = ctx.var_shape(src)?;
    let dst_shape = ctx.var_shape(dst)?;
    if src_shape != dst_shape {
        return Err(anyhow!(
            "transfer shape mismatch {} -> {}: {:?} vs {:?}",
            src,
            dst,
            src_shape,
            dst_shape
        ));
    }
    Ok(())
}
