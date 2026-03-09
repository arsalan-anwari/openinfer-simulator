use anyhow::{anyhow, Result};
use once_cell::sync::OnceCell;
use serde::Deserialize;

use crate::graph::{AttrValue, OpAttrs, OpKind};
use crate::tensor::DType;

/// Output type rule for an op.
#[derive(Debug, Clone)]
pub enum OutputType {
    Same,
    Fixed(DType),
    FromAttr(&'static str),
}

/// Parameter kind for op attributes.
#[derive(Debug, Clone)]
pub enum ParamKind {
    /// Scalar or DType param with allowed dtypes.
    DTypes(&'static [DType]),
    /// IntList (e.g. axes).
    IntList,
    /// Boolean param.
    Bool,
    /// String param.
    String,
    /// Single scalar type (e.g. u64 for bits).
    Scalar(DType),
}

impl ParamKind {
    /// True if this param is a scalar (for Vulkan push constants).
    pub fn is_scalar(&self) -> bool {
        matches!(self, ParamKind::DTypes(_) | ParamKind::Bool | ParamKind::Scalar(_))
    }
}

/// Scalar attribute value kind (for Vulkan validation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // used when vulkan feature is enabled
pub enum ScalarAttrKind {
    Float,
    Int,
    UInt,
    Bool,
}

/// Definition of a single op parameter.
#[derive(Debug, Clone)]
pub struct ParamDef {
    pub name: &'static str,
    pub kind: ParamKind,
}

/// Static schema definition for an op.
#[derive(Debug, Clone)]
pub struct OpSchema {
    pub kind: OpKind,
    pub name: &'static str,
    pub input_count: usize,
    pub output_count: usize,
    pub input_tensor_types: &'static [DType],
    pub output_type: OutputType,
    pub parameter_types: &'static [ParamDef],
    pub supports_broadcast: bool,
    pub supports_inplace: bool,
}

impl OpSchema {
    /// Infer output dtype from inputs and attributes.
    pub fn output_dtype(&self, input_dtypes: &[DType], attrs: &OpAttrs) -> Result<DType> {
        match &self.output_type {
            OutputType::Same => input_dtypes
                .first()
                .copied()
                .ok_or_else(|| anyhow!("missing input dtype")),
            OutputType::Fixed(dtype) => Ok(*dtype),
            OutputType::FromAttr(attr) => attrs
                .items
                .iter()
                .find(|item| item.name == *attr)
                .ok_or_else(|| anyhow!("missing {} attribute", attr))
                .and_then(|item| match &item.value {
                    AttrValue::DType(dtype) => Ok(*dtype),
                    _ => Err(anyhow!("{} attribute must be a dtype", attr)),
                }),
        }
    }

    /// For output_type FromAttr, return the allowed output dtypes from the attr's parameter_types.
    pub fn output_dtypes_from_attr(&self) -> Option<&'static [DType]> {
        match &self.output_type {
            OutputType::FromAttr(attr_name) => self
                .parameter_types
                .iter()
                .find(|p| p.name == *attr_name)
                .and_then(|p| match &p.kind {
                    ParamKind::DTypes(dtypes) => Some(*dtypes),
                    _ => None,
                }),
            _ => None,
        }
    }
}

#[derive(Debug)]
struct OpRegistry {
    schemas: Vec<OpSchema>,
}

static REGISTRY: OnceCell<OpRegistry> = OnceCell::new();

#[derive(Debug, Deserialize)]
struct OpsFile {
    version: u32,
    ops: Vec<OpSchemaJson>,
}

#[derive(Debug, Deserialize)]
struct OpSchemaJson {
    name: String,
    #[serde(default)]
    #[allow(dead_code)]
    category: String,
    input_count: usize,
    output_count: usize,
    input_tensor_types: Vec<String>,
    parameter_types: Vec<ParamTypeJson>,
    output_type: String,
    #[serde(default)]
    supports_broadcast: bool,
    #[serde(default)]
    supports_inplace: bool,
}

#[derive(Debug, Deserialize)]
struct ParamTypeJson {
    name: String,
    kind: serde_json::Value,
}

fn registry() -> &'static OpRegistry {
    REGISTRY.get_or_init(|| {
        load_registry().unwrap_or_else(|err| panic!("ops registry init failed: {err}"))
    })
}

fn load_registry() -> Result<OpRegistry> {
    let json = include_str!("../ops.json");
    let file: OpsFile = serde_json::from_str(json)?;
    if file.version != 1 {
        return Err(anyhow!("unsupported ops.json version {}", file.version));
    }

    let mut schemas = Vec::with_capacity(file.ops.len());
    for op in file.ops {
        let kind = match OpKind::from_name(&op.name) {
            Ok(k) => k,
            Err(_) => continue,
        };

        let input_tensor_types: Vec<DType> = op
            .input_tensor_types
            .iter()
            .map(|s| DType::from_ident(s))
            .collect::<Result<Vec<_>>>()?;
        let input_tensor_types: &'static [DType] =
            Box::leak(input_tensor_types.into_boxed_slice());

        let output_type = parse_output_type(&op.output_type, &op.parameter_types)?;

        let param_defs: Vec<ParamDef> = op
            .parameter_types
            .iter()
            .map(|p| parse_param_def(p))
            .collect::<Result<Vec<_>>>()?;
        let parameter_types: &'static [ParamDef] =
            Box::leak(param_defs.into_boxed_slice());

        let name_static: &'static str = Box::leak(op.name.into_boxed_str());

        schemas.push(OpSchema {
            kind,
            name: name_static,
            input_count: op.input_count,
            output_count: op.output_count,
            input_tensor_types,
            output_type,
            parameter_types,
            supports_broadcast: op.supports_broadcast,
            supports_inplace: op.supports_inplace,
        });
    }
    Ok(OpRegistry { schemas })
}

fn parse_output_type(
    output_type: &str,
    parameter_types: &[ParamTypeJson],
) -> Result<OutputType> {
    match output_type {
        "same" => Ok(OutputType::Same),
        "from_attr" => {
            let attr = parameter_types
                .iter()
                .find(|p| {
                    if let Some(arr) = p.kind.as_array() {
                        arr.iter().any(|v| v.as_str().is_some())
                    } else {
                        false
                    }
                })
                .map(|p| p.name.as_str())
                .ok_or_else(|| anyhow!("from_attr requires parameter with dtype array (e.g. cast 'to')"))?;
            Ok(OutputType::FromAttr(Box::leak(attr.to_string().into_boxed_str())))
        }
        dtype_str => {
            let dtype = DType::from_ident(dtype_str)?;
            Ok(OutputType::Fixed(dtype))
        }
    }
}

fn parse_param_def(p: &ParamTypeJson) -> Result<ParamDef> {
    let name_static: &'static str = Box::leak(p.name.clone().into_boxed_str());
    let kind = match &p.kind {
        serde_json::Value::Array(arr) => {
            let dtypes: Vec<DType> = arr
                .iter()
                .filter_map(|v| v.as_str())
                .map(|s| DType::from_ident(s))
                .collect::<Result<Vec<_>>>()?;
            ParamKind::DTypes(Box::leak(dtypes.into_boxed_slice()))
        }
        serde_json::Value::String(s) => match s.as_str() {
            "i64[]" | "i32[]" => ParamKind::IntList,
            "bool" => ParamKind::Bool,
            "string" => ParamKind::String,
            scalar => ParamKind::Scalar(DType::from_ident(scalar)?),
        },
        _ => return Err(anyhow!("invalid parameter kind for {}", p.name)),
    };
    Ok(ParamDef {
        name: name_static,
        kind,
    })
}

/// Returns true if an op schema supports the provided typing tuple.
pub fn supports_pattern(
    schema: &OpSchema,
    input_dtypes: &[DType],
    output_dtype: DType,
    attrs: &OpAttrs,
) -> bool {
    if input_dtypes.len() != schema.input_count {
        return false;
    }
    if !input_dtypes
        .iter()
        .all(|d| schema.input_tensor_types.contains(d))
    {
        return false;
    }
    match &schema.output_type {
        OutputType::Same => {
            input_dtypes.first().map_or(false, |d| *d == output_dtype)
        }
        OutputType::Fixed(d) => output_dtype == *d,
        OutputType::FromAttr(attr) => attrs
            .items
            .iter()
            .find(|item| item.name == *attr)
            .and_then(|item| {
                if let AttrValue::DType(d) = &item.value {
                    Some(*d == output_dtype)
                } else {
                    None
                }
            })
            .unwrap_or(false),
    }
}

/// Lookup the schema for a specific op kind.
pub fn op_schema(kind: OpKind) -> Option<&'static OpSchema> {
    registry().schemas.iter().find(|op| op.kind == kind)
}

/// Initialize the global op registry (idempotent).
pub fn init_ops_registry() {
    let _ = registry();
}
