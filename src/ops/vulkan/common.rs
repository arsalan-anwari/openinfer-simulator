use std::collections::HashMap;

use anyhow::{anyhow, Result};

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::op_defs::{op_schema, ParamKind, ScalarAttrKind};
use crate::tensor::DType;

#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarAttrBits {
    pub f64_bits: u64,
    pub f32_bits: u32,
    pub i64_bits: u64,
    pub u64_bits: u64,
    pub bool_u32: u32,
    pub u8: u8,
}

impl ScalarAttrBits {
    pub fn as_f64(self) -> f64 {
        f64::from_bits(self.f64_bits)
    }

    pub fn as_i64(self) -> i64 {
        self.i64_bits as i64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DTypeClass {
    Float,
    Signed,
    Unsigned,
    Bool,
}

pub fn collect_scalar_attr_bits(
    op: OpKind,
    dtype: DType,
    attrs: &OpAttrs,
) -> Result<HashMap<String, ScalarAttrBits>> {
    let schema = op_schema(op).ok_or_else(|| anyhow!("missing schema for {}", op))?;
    let dtype_class = dtype_class(dtype);
    let mut map = HashMap::new();
    for def in schema.parameter_types {
        if !def.kind.is_scalar() {
            continue;
        }
        if let Some(attr) = attrs.items.iter().find(|attr| attr.name == def.name) {
            let allowed = allowed_scalar_kinds(&def.kind);
            let bits = scalar_bits_from_attr(allowed, dtype_class, &attr.value)?;
            map.insert(def.name.to_string(), bits);
        }
    }
    Ok(map)
}

fn allowed_scalar_kinds(kind: &ParamKind) -> &'static [ScalarAttrKind] {
    const ALL: &[ScalarAttrKind] = &[
        ScalarAttrKind::Float,
        ScalarAttrKind::Int,
        ScalarAttrKind::UInt,
        ScalarAttrKind::Bool,
    ];
    match kind {
        ParamKind::DTypes(_) => ALL,
        ParamKind::Bool => &[ScalarAttrKind::Bool],
        ParamKind::Scalar(d) => match d {
            DType::F8 | DType::BF16 | DType::F16 | DType::F32 | DType::F64 => {
                &[ScalarAttrKind::Float]
            }
            DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::I4 => {
                &[ScalarAttrKind::Int]
            }
            DType::U8 | DType::U16 | DType::U32 | DType::U64 | DType::U4 => {
                &[ScalarAttrKind::UInt]
            }
            DType::Bool => &[ScalarAttrKind::Bool],
        },
        _ => &[],
    }
}

fn scalar_bits_from_attr(
    allowed_kinds: &[ScalarAttrKind],
    dtype_class: DTypeClass,
    value: &AttrValue,
) -> Result<ScalarAttrBits> {
    let value_kind = scalar_kind_from_value(value)?;
    if !allowed_kinds.is_empty() && !allowed_kinds.contains(&value_kind) {
        return Err(anyhow!("unsupported scalar attr kind {:?}", value_kind));
    }
    let mut bits = ScalarAttrBits::default();
    match dtype_class {
        DTypeClass::Float => {
            let f64_value = match value {
                AttrValue::Float(v) => *v as f64,
                AttrValue::Double(v) => *v,
                AttrValue::Int(v) => *v as f64,
                AttrValue::UInt(v) => *v as f64,
                AttrValue::Bool(v) => {
                    if *v {
                        1.0
                    } else {
                        0.0
                    }
                }
                AttrValue::Str(_) | AttrValue::IntList(_) | AttrValue::DTypeList(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
                AttrValue::Var(_) | AttrValue::DType(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
            };
            bits.f64_bits = f64_value.to_bits();
            bits.f32_bits = (f64_value as f32).to_bits();
        }
        DTypeClass::Signed => {
            let i64_value = match value {
                AttrValue::Int(v) => *v,
                AttrValue::UInt(v) => *v as i64,
                AttrValue::Bool(v) => {
                    if *v {
                        1
                    } else {
                        0
                    }
                }
                AttrValue::Float(_) | AttrValue::Double(_) => {
                    return Err(anyhow!("integer attrs must be int/uint/bool"))
                }
                AttrValue::Str(_) | AttrValue::IntList(_) | AttrValue::DTypeList(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
                AttrValue::Var(_) | AttrValue::DType(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
            };
            bits.i64_bits = i64_value as u64;
            bits.u8 = i64_value as u8;
        }
        DTypeClass::Unsigned => {
            let u64_value = match value {
                AttrValue::UInt(v) => *v,
                AttrValue::Int(v) => {
                    if *v < 0 {
                        return Err(anyhow!("unsigned attrs must be non-negative"));
                    }
                    *v as u64
                }
                AttrValue::Bool(v) => {
                    if *v {
                        1
                    } else {
                        0
                    }
                }
                AttrValue::Float(_) | AttrValue::Double(_) => {
                    return Err(anyhow!("unsigned attrs must be int/uint/bool"))
                }
                AttrValue::Str(_) | AttrValue::IntList(_) | AttrValue::DTypeList(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
                AttrValue::Var(_) | AttrValue::DType(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
            };
            bits.u64_bits = u64_value;
            bits.u8 = u64_value as u8;
        }
        DTypeClass::Bool => {
            let bool_value = match value {
                AttrValue::Bool(v) => *v,
                AttrValue::Int(v) => *v != 0,
                AttrValue::UInt(v) => *v != 0,
                AttrValue::Float(_) | AttrValue::Double(_) => {
                    return Err(anyhow!("bool attrs must be bool/int/uint"))
                }
                AttrValue::Str(_) | AttrValue::IntList(_) | AttrValue::DTypeList(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
                AttrValue::Var(_) | AttrValue::DType(_) => {
                    return Err(anyhow!("unsupported scalar attr value for vulkan"))
                }
            };
            bits.bool_u32 = bool_value as u32;
        }
    }
    Ok(bits)
}

fn scalar_kind_from_value(value: &AttrValue) -> Result<ScalarAttrKind> {
    match value {
        AttrValue::Float(_) | AttrValue::Double(_) => Ok(ScalarAttrKind::Float),
        AttrValue::Int(_) => Ok(ScalarAttrKind::Int),
        AttrValue::UInt(_) => Ok(ScalarAttrKind::UInt),
        AttrValue::Bool(_) => Ok(ScalarAttrKind::Bool),
        AttrValue::Str(_) | AttrValue::IntList(_) | AttrValue::DTypeList(_) => {
            Err(anyhow!("unsupported scalar attr value for vulkan"))
        }
        AttrValue::Var(_) | AttrValue::DType(_) => {
            Err(anyhow!("unsupported scalar attr value for vulkan"))
        }
    }
}

fn dtype_class(dtype: DType) -> DTypeClass {
    match dtype {
        DType::F8 | DType::BF16 | DType::F16 | DType::F32 | DType::F64 => DTypeClass::Float,
        DType::I8
        | DType::I16
        | DType::I32
        | DType::I64
        | DType::I4 => DTypeClass::Signed,
        DType::U8
        | DType::U16
        | DType::U32
        | DType::U64
        | DType::U4 => DTypeClass::Unsigned,
        DType::Bool => DTypeClass::Bool,
    }
}
