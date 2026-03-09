//! Lazy `.oinf` model loader.
//!
//! The loader memory-maps the model file, validates headers, and loads tensor
//! payloads only on demand.
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;

use crate::runtime::tensor_store::{MappedSlice, TensorRef, TensorStore};
use crate::tensor::{
    BF16, DType, F16, F8, I4, QuantParams, QuantScale, QuantScheme, QuantZeroPoint, U4, Tensor,
    TensorValue,
};
use crate::types::VarInfo;

const MAGIC: &[u8; 5] = b"OINF\0";
const HEADER_SIZE: usize = 69;

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct MetadataInfo {
    value_type: u32,
    value_offset: u64,
    value_nbytes: u64,
    dims: Vec<u64>,
}

/// Loads `.oinf` model files and exposes tensors/metadata.
#[derive(Debug, Clone)]
pub struct ModelLoader {
    #[allow(dead_code)]
    path: PathBuf,
    sizes: HashMap<String, usize>,
    vars: HashMap<String, VarInfo>,
    #[allow(dead_code)]
    metadata: HashMap<String, MetadataInfo>,
    tensor_quant: HashMap<String, Option<QuantParams>>,
    mmap: Arc<Mmap>,
    tensor_store: TensorStore,
}

impl ModelLoader {
    /// Open an `.oinf` model file from disk.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::ModelLoader;
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// # Ok(()) }
    /// ```
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).with_context(|| "open model file")?;
        let mmap = unsafe { Mmap::map(&file).with_context(|| "mmap model file")? };
        let data = &mmap[..];
        if data.len() < HEADER_SIZE {
            return Err(anyhow!("file too small for OINF header"));
        }

        let mut cursor = 0usize;
        let magic = read_bytes(data, &mut cursor, 5)?;
        if magic != MAGIC {
            return Err(anyhow!("invalid OINF magic"));
        }
        let version = read_u32(data, &mut cursor)?;
        if version != 2 {
            return Err(anyhow!("unsupported OINF version {}", version));
        }
        let _flags = read_u32(data, &mut cursor)?;
        let n_sizevars = read_u32(data, &mut cursor)? as usize;
        let n_metadata = read_u32(data, &mut cursor)? as usize;
        let n_tensors = read_u32(data, &mut cursor)? as usize;
        let _reserved = read_u32(data, &mut cursor)?;
        let offset_sizevars = read_u64(data, &mut cursor)? as usize;
        let offset_metadata = read_u64(data, &mut cursor)? as usize;
        let offset_tensors = read_u64(data, &mut cursor)? as usize;
        let offset_data = read_u64(data, &mut cursor)? as usize;
        let file_size = read_u64(data, &mut cursor)? as usize;

        if file_size != data.len() {
            return Err(anyhow!("file size mismatch"));
        }
        let offsets = vec![
            offset_sizevars,
            offset_metadata,
            offset_tensors,
            offset_data,
            file_size,
        ];
        let mut sorted = offsets.clone();
        sorted.sort_unstable();
        if offsets != sorted {
            return Err(anyhow!("OINF offsets are not ascending"));
        }
        for off in offsets.iter().take(4) {
            if *off % 8 != 0 {
                return Err(anyhow!("OINF section offset not aligned"));
            }
            if *off > file_size {
                return Err(anyhow!("OINF section offset out of bounds"));
            }
        }

        let mut sizes = HashMap::new();
        let mut size_cursor = offset_sizevars;
        for _ in 0..n_sizevars {
            let name = read_string(data, &mut size_cursor)?;
            if sizes.contains_key(&name) {
                return Err(anyhow!("duplicate sizevar {}", name));
            }
            let value = read_u64_at(data, size_cursor)?;
            size_cursor += 8;
            sizes.insert(name, value as usize);
        }

        let mut metadata = HashMap::new();
        let mut meta_cursor = offset_metadata;
        for _ in 0..n_metadata {
            let key = read_string(data, &mut meta_cursor)?;
            if metadata.contains_key(&key) {
                return Err(anyhow!("duplicate metadata key {}", key));
            }
            let value_type = read_u32_at(data, meta_cursor)?;
            let flags = read_u32_at(data, meta_cursor + 4)?;
            let value_nbytes = read_u64_at(data, meta_cursor + 8)?;
            let value_offset = read_u64_at(data, meta_cursor + 16)?;
            meta_cursor += 24;
            if flags != 0 {
                return Err(anyhow!("metadata flags must be 0"));
            }
            if value_offset % 8 != 0 {
                return Err(anyhow!("metadata value offset not aligned"));
            }
            let value_end = value_offset
                .checked_add(value_nbytes)
                .ok_or_else(|| anyhow!("metadata value offset overflow"))?;
            if value_end as usize > file_size {
                return Err(anyhow!("metadata value out of bounds"));
            }

            let mut dims = Vec::new();
            if value_type == ValueType::NDARRAY {
                let mut cursor = value_offset as usize;
                let element_type = read_u32(data, &mut cursor)?;
                let ndim = read_u32(data, &mut cursor)? as usize;
                if !ValueType::is_scalar(element_type) {
                    return Err(anyhow!("metadata ndarray has invalid element type"));
                }
                for _ in 0..ndim {
                    dims.push(read_u64(data, &mut cursor)?);
                }
            }

            metadata.insert(
                key,
                MetadataInfo {
                    value_type,
                    value_offset,
                    value_nbytes,
                    dims,
                },
            );
        }

        let mut vars = HashMap::new();
        let mut tensor_quant: HashMap<String, Option<QuantParams>> = HashMap::new();
        let mut tensor_cursor = offset_tensors;
        for _ in 0..n_tensors {
            let name = read_string(data, &mut tensor_cursor)?;
            if vars.contains_key(&name) {
                return Err(anyhow!("duplicate tensor name {}", name));
            }
            let dtype_raw = read_u32(data, &mut tensor_cursor)?;
            let ndim = read_u32(data, &mut tensor_cursor)? as usize;
            let flags = read_u32(data, &mut tensor_cursor)?;
            let mut dims = Vec::new();
            for _ in 0..ndim {
                dims.push(read_u64(data, &mut tensor_cursor)?);
            }
            let data_nbytes = read_u64(data, &mut tensor_cursor)? as usize;
            let data_offset = read_u64(data, &mut tensor_cursor)? as usize;
            let quant_nbytes = read_u64(data, &mut tensor_cursor)? as usize;
            let quant_offset = read_u64(data, &mut tensor_cursor)? as usize;

            let dtype = ValueType::to_dtype(dtype_raw)?;
            let has_data = (flags & 1) != 0;
            let has_quant = (flags & 2) != 0;
            if flags & !0x3 != 0 {
                return Err(anyhow!("tensor flags contain unsupported bits"));
            }
            if has_data {
                if data_offset % 8 != 0 {
                    return Err(anyhow!("tensor data offset not aligned"));
                }
                if data_offset < offset_data {
                    return Err(anyhow!("tensor data offset precedes data section"));
                }
                if data_offset + data_nbytes > file_size {
                    return Err(anyhow!("tensor data out of bounds"));
                }
            } else if data_offset != 0 || data_nbytes != 0 {
                return Err(anyhow!("tensor without data must have zero offset/size"));
            }
            if has_quant {
                if quant_offset % 8 != 0 {
                    return Err(anyhow!("tensor quant offset not aligned"));
                }
                if quant_offset < offset_data {
                    return Err(anyhow!("tensor quant offset precedes data section"));
                }
                if quant_offset + quant_nbytes > file_size {
                    return Err(anyhow!("tensor quant payload out of bounds"));
                }
            } else if quant_offset != 0 || quant_nbytes != 0 {
                return Err(anyhow!("tensor without quant must have zero quant offset/size"));
            }

            let dims_str = dims.iter().map(|d| d.to_string()).collect();
            let value_range = if has_data {
                Some((data_offset, data_offset + data_nbytes))
            } else {
                None
            };
            let dims_usize = dims
                .iter()
                .map(|d| usize::try_from(*d).map_err(|_| anyhow!("tensor dim exceeds usize for {}", name)))
                .collect::<Result<Vec<_>>>()?;
            let quant = if has_quant {
                Some(parse_quant_params(data, quant_offset, quant_nbytes, &dims_usize, &name)?)
            } else {
                None
            };
            vars.insert(
                name.clone(),
                VarInfo {
                    name: name.clone(),
                    dtype,
                    dims: dims_str,
                    value_range,
                    has_data,
                },
            );
            tensor_quant.insert(name, quant);
        }

        let mmap = Arc::new(mmap);
        let tensor_store = build_tensor_store(&sizes, &vars, mmap.clone())?;

        Ok(Self {
            path,
            sizes,
            vars,
            metadata,
            tensor_quant,
            mmap,
            tensor_store,
        })
    }

    /// Lookup a size variable by name.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::ModelLoader;
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let b = model.size_of("B")?;
    /// # Ok(()) }
    /// ```
    pub fn size_of(&self, name: &str) -> Result<usize> {
        self.sizes
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("unknown size: {}", name))
    }

    /// Resolve a product of dimension strings into a length.
    pub fn resolve_len(&self, dims: &[String]) -> Result<usize> {
        let mut total = 1usize;
        for dim in dims {
            total = total.saturating_mul(self.resolve_dim_value(dim)?);
        }
        Ok(total)
    }

    /// Resolve dimension strings into a concrete shape.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::ModelLoader;
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let shape = model.resolve_shape(&["B".into(), "D".into()])?;
    /// # Ok(()) }
    /// ```
    pub fn resolve_shape(&self, dims: &[String]) -> Result<Vec<usize>> {
        let mut shape = Vec::with_capacity(dims.len());
        for dim in dims {
            shape.push(self.resolve_dim_value(dim)?);
        }
        Ok(shape)
    }

    /// Resolve a single dimension expression (literal, sizevar, or product).
    pub fn resolve_dim_value(&self, dim: &str) -> Result<usize> {
        if let Ok(val) = dim.parse::<usize>() {
            return Ok(val);
        }
        let trimmed = dim.trim();
        if let Some((left, right)) = trimmed.split_once('*') {
            let left = left.trim();
            let right = right.trim();
            let left_val = match left.parse::<usize>() {
                Ok(value) => value,
                Err(_) => self.size_of(left)?,
            };
            let right_val = match right.parse::<usize>() {
                Ok(value) => value,
                Err(_) => self.size_of(right)?,
            };
            return Ok(left_val.saturating_mul(right_val));
        }
        self.size_of(trimmed)
    }

    /// Fetch variable metadata by name.
    pub fn var_info(&self, name: &str) -> Option<&VarInfo> {
        self.vars.get(name)
    }

    /// Access the underlying tensor store.
    pub fn tensor_store(&self) -> &TensorStore {
        &self.tensor_store
    }

    /// Load a tensor payload by name from the mapped file.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::ModelLoader;
    /// # fn main() -> anyhow::Result<()> {
    /// let model = ModelLoader::open("model.oinf")?;
    /// let tensor = model.load_tensor("w1")?;
    /// # Ok(()) }
    /// ```
    pub fn load_tensor(&self, name: &str) -> Result<TensorValue> {
        let info = self
            .vars
            .get(name)
            .ok_or_else(|| anyhow!("unknown variable: {}", name))?;
        if !info.has_data {
            return Err(anyhow!("no data found for {}", name));
        }
        let range = info
            .value_range
            .ok_or_else(|| anyhow!("missing data range for {}", name))?;
        let data = &self.mmap[range.0..range.1];
        let mut tensor = tensor_value_from_bytes(info, data)?;
        if let Some(quant) = self.tensor_quant.get(name) {
            tensor.set_quant(quant.clone());
        }
        Ok(tensor)
    }

    /// Load a metadata tensor by name, if present.
    pub fn load_metadata_tensor(&self, name: &str) -> Result<Option<TensorValue>> {
        let info = match self.metadata.get(name) {
            Some(info) => info,
            None => return Ok(None),
        };
        let data = &self.mmap[..];
        let start = info.value_offset as usize;
        let end = start + info.value_nbytes as usize;
        if end > data.len() {
            return Err(anyhow!("metadata value out of bounds for {}", name));
        }

        if info.value_type == ValueType::STRING {
            return Err(anyhow!("metadata {} is a string, not a tensor", name));
        }

        if info.value_type == ValueType::NDARRAY {
            let mut cursor = start;
            let element_type = read_u32(data, &mut cursor)?;
            let ndim = read_u32(data, &mut cursor)? as usize;
            let mut dims = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                dims.push(read_u64(data, &mut cursor)?);
            }
            let dtype = ValueType::to_dtype(element_type)?;
            let var_info = VarInfo {
                name: name.to_string(),
                dtype,
                dims: dims.iter().map(|d| d.to_string()).collect(),
                value_range: None,
                has_data: true,
            };
            let payload = &data[cursor..end];
            return tensor_value_from_bytes(&var_info, payload).map(Some);
        }

        let dtype = ValueType::to_dtype(info.value_type)?;
        let var_info = VarInfo {
            name: name.to_string(),
            dtype,
            dims: Vec::new(),
            value_range: None,
            has_data: true,
        };
        let payload = &data[start..end];
        tensor_value_from_bytes(&var_info, payload).map(Some)
    }

    /// True if a named metadata entry is a string.
    pub fn has_metadata_string(&self, name: &str) -> bool {
        self.metadata
            .get(name)
            .map(|info| info.value_type == ValueType::STRING)
            .unwrap_or(false)
    }

    /// Load a metadata string by name, if present.
    pub fn load_metadata_string(&self, name: &str) -> Result<Option<String>> {
        let info = match self.metadata.get(name) {
            Some(info) => info,
            None => return Ok(None),
        };
        if info.value_type != ValueType::STRING {
            return Ok(None);
        }
        let data = &self.mmap[..];
        let start = info.value_offset as usize;
        let end = start + info.value_nbytes as usize;
        if end > data.len() {
            return Err(anyhow!("metadata value out of bounds for {}", name));
        }
        if info.value_nbytes < 4 {
            return Err(anyhow!("metadata string too small for {}", name));
        }

        let len = read_u32_at(data, start)? as usize;
        let payload_end = start + 4 + len;
        if payload_end > end {
            return Err(anyhow!("metadata string payload out of bounds for {}", name));
        }
        let raw = &data[start + 4..payload_end];
        let text = std::str::from_utf8(raw).context("invalid UTF-8 string")?;
        let padded = align_up(4 + len, 8);
        if start + padded > end {
            return Err(anyhow!("metadata string padding out of bounds for {}", name));
        }
        Ok(Some(text.to_string()))
    }
}

fn build_tensor_store(
    sizes: &HashMap<String, usize>,
    vars: &HashMap<String, VarInfo>,
    mmap: Arc<Mmap>,
) -> Result<TensorStore> {
    let mut tensors = HashMap::new();
    for (name, info) in vars {
        let shape = resolve_shape(sizes, &info.dims)?;
        let data = info.value_range.map(|(start, end)| {
            MappedSlice::new(mmap.clone(), start..end)
        });
        tensors.insert(
            name.clone(),
            TensorRef {
                name: name.clone(),
                dtype: info.dtype,
                dims: info.dims.clone(),
                shape,
                data,
            },
        );
    }
    Ok(TensorStore::new(tensors))
}

fn resolve_shape(sizes: &HashMap<String, usize>, dims: &[String]) -> Result<Vec<usize>> {
    let mut shape = Vec::with_capacity(dims.len());
    for dim in dims {
        shape.push(resolve_dim_value(sizes, dim)?);
    }
    Ok(shape)
}

fn resolve_dim_value(sizes: &HashMap<String, usize>, dim: &str) -> Result<usize> {
    if let Ok(val) = dim.parse::<usize>() {
        return Ok(val);
    }
    let trimmed = dim.trim();
    if let Some((left, right)) = trimmed.split_once('*') {
        let left = left.trim();
        let right = right.trim();
        let left_val = match left.parse::<usize>() {
            Ok(value) => value,
            Err(_) => sizes
                .get(left)
                .copied()
                .ok_or_else(|| anyhow!("unknown size: {}", left))?,
        };
        let right_val = match right.parse::<usize>() {
            Ok(value) => value,
            Err(_) => sizes
                .get(right)
                .copied()
                .ok_or_else(|| anyhow!("unknown size: {}", right))?,
        };
        return Ok(left_val.saturating_mul(right_val));
    }
    sizes
        .get(trimmed)
        .copied()
        .ok_or_else(|| anyhow!("unknown size: {}", trimmed))
}

fn read_bytes<'a>(data: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    if *cursor + len > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    let out = &data[*cursor..*cursor + len];
    *cursor += len;
    Ok(out)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32> {
    let bytes = read_bytes(data, cursor, 4)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64> {
    let bytes = read_bytes(data, cursor, 8)?;
    Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
}

fn read_u32_at(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()))
}

fn read_u64_at(data: &[u8], offset: usize) -> Result<u64> {
    if offset + 8 > data.len() {
        return Err(anyhow!("unexpected EOF"));
    }
    Ok(u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()))
}

fn read_string(data: &[u8], cursor: &mut usize) -> Result<String> {
    let len = read_u32(data, cursor)? as usize;
    let bytes = read_bytes(data, cursor, len)?;
    let s = std::str::from_utf8(bytes).context("invalid UTF-8 string")?;
    let padded = align_up(4 + len, 8);
    let consumed = 4 + len;
    if padded > consumed {
        let skip = padded - consumed;
        if *cursor + skip > data.len() {
            return Err(anyhow!("unexpected EOF"));
        }
        *cursor += skip;
    }
    Ok(s.to_string())
}

fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) / alignment * alignment
}

fn tensor_value_from_bytes(info: &VarInfo, bytes: &[u8]) -> Result<TensorValue> {
    match info.dtype {
        DType::I8 => tensor_from_bytes::<i8>(info, bytes).map(TensorValue::I8),
        DType::I16 => tensor_from_bytes::<i16>(info, bytes).map(TensorValue::I16),
        DType::I32 => tensor_from_bytes::<i32>(info, bytes).map(TensorValue::I32),
        DType::I64 => tensor_from_bytes::<i64>(info, bytes).map(TensorValue::I64),
        DType::U8 => tensor_from_bytes::<u8>(info, bytes).map(TensorValue::U8),
        DType::U16 => tensor_from_bytes::<u16>(info, bytes).map(TensorValue::U16),
        DType::U32 => tensor_from_bytes::<u32>(info, bytes).map(TensorValue::U32),
        DType::U64 => tensor_from_bytes::<u64>(info, bytes).map(TensorValue::U64),
        DType::F16 => tensor_from_bits::<u16, F16>(info, bytes, |bits| F16 { bits }).map(TensorValue::F16),
        DType::BF16 => tensor_from_bits::<u16, BF16>(info, bytes, |bits| BF16 { bits }).map(TensorValue::BF16),
        DType::F8 => tensor_from_bits::<u8, F8>(info, bytes, |bits| F8 { bits }).map(TensorValue::F8),
        DType::F32 => tensor_from_bytes::<f32>(info, bytes).map(TensorValue::F32),
        DType::F64 => tensor_from_bytes::<f64>(info, bytes).map(TensorValue::F64),
        DType::Bool => tensor_from_bytes::<bool>(info, bytes).map(TensorValue::Bool),
        DType::I4 => tensor_from_bits::<u8, I4>(info, bytes, |bits| I4 { bits }).map(TensorValue::I4),
        DType::U4 => tensor_from_bits::<u8, U4>(info, bytes, |bits| U4 { bits }).map(TensorValue::U4),
    }
}

fn tensor_from_bytes<T: Copy>(info: &VarInfo, bytes: &[u8]) -> Result<Tensor<T>> {
    let shape = info
        .dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("invalid tensor dims for {}", info.name))?;
    let len = shape.iter().product::<usize>();
    let expected = len * std::mem::size_of::<T>();
    if bytes.len() != expected {
        return Err(anyhow!(
            "tensor {} byte length mismatch: expected {}, got {}",
            info.name,
            expected,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(len);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let end = cursor + std::mem::size_of::<T>();
        let value = read_t::<T>(&bytes[cursor..end])?;
        out.push(value);
        cursor = end;
    }
    Tensor::from_vec_with_opts(
        out,
        crate::tensor::TensorOptions {
            shape: Some(shape),
            ..crate::tensor::TensorOptions::default()
        },
    )
}

fn tensor_from_bits<B: Copy, T>(
    info: &VarInfo,
    bytes: &[u8],
    map: fn(B) -> T,
) -> Result<Tensor<T>> {
    let shape = info
        .dims
        .iter()
        .map(|dim| dim.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| anyhow!("invalid tensor dims for {}", info.name))?;
    let len = shape.iter().product::<usize>();
    if bytes.is_empty() && len == 0 {
        return Tensor::from_vec_with_opts(
            Vec::new(),
            crate::tensor::TensorOptions {
                shape: Some(shape),
                ..crate::tensor::TensorOptions::default()
            },
        );
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let end = cursor + std::mem::size_of::<B>();
        let value = read_t::<B>(&bytes[cursor..end])?;
        out.push(map(value));
        cursor = end;
    }
    Tensor::from_vec_with_opts(
        out,
        crate::tensor::TensorOptions {
            shape: Some(shape),
            allow_len_mismatch: true,
            ..crate::tensor::TensorOptions::default()
        },
    )
}

fn read_t<T: Copy>(bytes: &[u8]) -> Result<T> {
    let mut value = std::mem::MaybeUninit::<T>::uninit();
    let len = std::mem::size_of::<T>();
    if bytes.len() != len {
        return Err(anyhow!("invalid byte length"));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), value.as_mut_ptr() as *mut u8, len);
        Ok(value.assume_init())
    }
}

struct ValueType;

impl ValueType {
    const I8: u32 = 1;
    const I16: u32 = 2;
    const I32: u32 = 3;
    const I64: u32 = 4;
    const U8: u32 = 5;
    const U16: u32 = 6;
    const U32: u32 = 7;
    const U64: u32 = 8;
    const F16: u32 = 9;
    const F32: u32 = 10;
    const F64: u32 = 11;
    const BOOL: u32 = 12;
    const BITSET: u32 = 13;
    #[allow(dead_code)]
    const STRING: u32 = 14;
    const NDARRAY: u32 = 15;
    const BF16: u32 = 16;
    const F8: u32 = 17;
    const I4: u32 = 18;
    const U4: u32 = 21;

    fn is_scalar(value_type: u32) -> bool {
        value_type >= Self::I8 && value_type <= Self::U4
    }

    fn to_dtype(value_type: u32) -> Result<DType> {
        Ok(match value_type {
            Self::I8 => DType::I8,
            Self::I16 => DType::I16,
            Self::I32 => DType::I32,
            Self::I64 => DType::I64,
            Self::U8 => DType::U8,
            Self::U16 => DType::U16,
            Self::U32 => DType::U32,
            Self::U64 => DType::U64,
            Self::F16 => DType::F16,
            Self::F32 => DType::F32,
            Self::F64 => DType::F64,
            Self::BOOL => DType::Bool,
            Self::BITSET => return Err(anyhow!("unsupported tensor dtype {}", value_type)),
            Self::BF16 => DType::BF16,
            Self::F8 => DType::F8,
            Self::I4 => DType::I4,
            Self::U4 => DType::U4,
            _ => return Err(anyhow!("unknown tensor dtype {}", value_type)),
        })
    }
}

fn parse_quant_params(
    data: &[u8],
    offset: usize,
    nbytes: usize,
    dims: &[usize],
    name: &str,
) -> Result<QuantParams> {
    if nbytes < 48 {
        return Err(anyhow!("tensor quant payload too small for {}", name));
    }
    let end = offset
        .checked_add(nbytes)
        .ok_or_else(|| anyhow!("tensor quant payload overflow for {}", name))?;
    if end > data.len() {
        return Err(anyhow!("tensor quant payload out of bounds for {}", name));
    }
    let payload = &data[offset..end];
    let scheme = read_u32_at(payload, 0)?;
    let scale_mode = read_u32_at(payload, 4)?;
    let zp_mode = read_u32_at(payload, 8)?;
    let reserved = read_u32_at(payload, 12)?;
    if reserved != 0 {
        return Err(anyhow!("tensor quant reserved must be 0 for {}", name));
    }
    let scale_axis = read_u64_at(payload, 16)? as usize;
    let scale_count = read_u64_at(payload, 24)? as usize;
    let zp_axis = read_u64_at(payload, 32)? as usize;
    let zp_count = read_u64_at(payload, 40)? as usize;
    let scale_bytes = scale_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("scale byte count overflow for {}", name))?;
    let zp_bytes = zp_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("zero-point byte count overflow for {}", name))?;
    let expected = align_up(48 + scale_bytes + zp_bytes, 8);
    if expected != nbytes {
        return Err(anyhow!("tensor quant payload size mismatch for {}", name));
    }

    let scale_start = 48usize;
    let scale_end = scale_start + scale_bytes;
    let zp_end = scale_end + zp_bytes;
    let scale_values = read_f32_vec(&payload[scale_start..scale_end])?;
    let zp_values = read_i32_vec(&payload[scale_end..zp_end])?;

    let quant_scheme = match scheme {
        1 => QuantScheme::Symmetric,
        2 => QuantScheme::Asymmetric,
        _ => return Err(anyhow!("invalid quant scheme for {}", name)),
    };

    let scale = match scale_mode {
        1 => {
            if scale_axis != 0 || scale_count != 1 {
                return Err(anyhow!("per-tensor scale requires axis=0,count=1 for {}", name));
            }
            QuantScale::PerTensor(*scale_values.first().ok_or_else(|| anyhow!("missing per-tensor scale for {}", name))?)
        }
        2 => {
            if scale_axis >= dims.len() {
                return Err(anyhow!("per-channel scale axis out of range for {}", name));
            }
            if scale_count != dims[scale_axis] {
                return Err(anyhow!("per-channel scale count mismatch for {}", name));
            }
            QuantScale::PerChannel {
                axis: scale_axis,
                values: scale_values,
            }
        }
        _ => return Err(anyhow!("invalid quant scale mode for {}", name)),
    };

    let zero_point = match zp_mode {
        0 => {
            if zp_count != 0 {
                return Err(anyhow!("zero-point none mode requires count=0 for {}", name));
            }
            None
        }
        1 => {
            if scale_mode != 1 {
                return Err(anyhow!("per-tensor zero-point requires per-tensor scale for {}", name));
            }
            if zp_axis != 0 || zp_count != 1 {
                return Err(anyhow!("per-tensor zero-point requires axis=0,count=1 for {}", name));
            }
            let value = *zp_values.first().ok_or_else(|| anyhow!("missing per-tensor zero-point for {}", name))?;
            Some(QuantZeroPoint::PerTensor(value))
        }
        2 => {
            if scale_mode != 2 {
                return Err(anyhow!("per-channel zero-point requires per-channel scale for {}", name));
            }
            if zp_axis >= dims.len() {
                return Err(anyhow!("per-channel zero-point axis out of range for {}", name));
            }
            if zp_axis != scale_axis {
                return Err(anyhow!("per-channel zero-point axis must match scale axis for {}", name));
            }
            if zp_count != dims[zp_axis] {
                return Err(anyhow!("per-channel zero-point count mismatch for {}", name));
            }
            Some(QuantZeroPoint::PerChannel {
                axis: zp_axis,
                values: zp_values,
            })
        }
        _ => return Err(anyhow!("invalid quant zero-point mode for {}", name)),
    };

    if matches!(quant_scheme, QuantScheme::Symmetric) && zero_point.is_some() {
        return Err(anyhow!("symmetric quantization cannot include zero-point for {}", name));
    }

    Ok(QuantParams {
        scheme: quant_scheme,
        scale,
        zero_point,
    })
}

fn read_f32_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!("invalid f32 payload length"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        out.push(f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
        cursor += 4;
    }
    Ok(out)
}

fn read_i32_vec(bytes: &[u8]) -> Result<Vec<i32>> {
    if bytes.len() % 4 != 0 {
        return Err(anyhow!("invalid i32 payload length"));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        out.push(i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()));
        cursor += 4;
    }
    Ok(out)
}
