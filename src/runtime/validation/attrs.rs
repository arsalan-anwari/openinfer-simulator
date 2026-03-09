use std::collections::HashSet;

use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::op_defs::{ParamDef, ParamKind};

use super::context::ValidationContext;

pub fn validate_attrs(
    ctx: &ValidationContext,
    op: OpKind,
    attrs: &OpAttrs,
    allowed: &[ParamDef],
) -> Result<()> {
    let mut seen = HashSet::new();
    for attr in &attrs.items {
        if !seen.insert(attr.name.as_str()) {
            return Err(anyhow!("duplicate {} setting: {}", op, attr.name));
        }
        let def = allowed
            .iter()
            .find(|def| def.name == attr.name)
            .ok_or_else(|| anyhow!("unsupported {} setting: {}", op, attr.name))?;
        if !param_kind_matches(&def.kind, &attr.value) {
            return Err(anyhow!(
                "unsupported {} setting type: {}",
                op,
                attr.name
            ));
        }
        match (&def.kind, &attr.value) {
            (ParamKind::DTypes(_), AttrValue::Var(name)) => {
                if !ctx.has_var(name) {
                    return Err(anyhow!("unknown attribute variable: {}", name));
                }
                if !ctx.is_scalar_var(name)? {
                    return Err(anyhow!("attribute {} must be scalar", name));
                }
            }
            (ParamKind::String, AttrValue::Var(name)) => {
                if !ctx.model.has_metadata_string(name) {
                    return Err(anyhow!("unknown attribute string: {}", name));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn param_kind_matches(kind: &ParamKind, value: &AttrValue) -> bool {
    match kind {
        ParamKind::DTypes(_) => matches!(
            value,
            AttrValue::Float(_)
                | AttrValue::Double(_)
                | AttrValue::Int(_)
                | AttrValue::UInt(_)
                | AttrValue::Bool(_)
                | AttrValue::Var(_)
                | AttrValue::DType(_)
        ),
        ParamKind::IntList => matches!(value, AttrValue::IntList(_)),
        ParamKind::Bool => matches!(value, AttrValue::Bool(_) | AttrValue::Var(_)),
        ParamKind::String => matches!(value, AttrValue::Str(_) | AttrValue::Var(_)),
        ParamKind::Scalar(_) => matches!(
            value,
            AttrValue::Float(_)
                | AttrValue::Double(_)
                | AttrValue::Int(_)
                | AttrValue::UInt(_)
                | AttrValue::Bool(_)
                | AttrValue::Var(_)
        ),
    }
}
