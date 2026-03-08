use anyhow::Result;
use openinfer::{ModelLoader, QuantScale, QuantZeroPoint};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn align_up(value: usize) -> usize {
    (value + 7) & !7
}

fn encode_string(value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let bytes = value.as_bytes();
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    out.resize(align_up(out.len()), 0);
    out
}

fn quant_payload_per_tensor(scale: f32, zero_point: i32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&2u32.to_le_bytes()); // asymmetric
    out.extend_from_slice(&1u32.to_le_bytes()); // scale per-tensor
    out.extend_from_slice(&1u32.to_le_bytes()); // zp per-tensor
    out.extend_from_slice(&0u32.to_le_bytes()); // reserved
    out.extend_from_slice(&0u64.to_le_bytes()); // scale_axis
    out.extend_from_slice(&1u64.to_le_bytes()); // scale_count
    out.extend_from_slice(&0u64.to_le_bytes()); // zp_axis
    out.extend_from_slice(&1u64.to_le_bytes()); // zp_count
    out.extend_from_slice(&scale.to_le_bytes());
    out.extend_from_slice(&zero_point.to_le_bytes());
    out.resize(align_up(out.len()), 0);
    out
}

fn build_model_bytes() -> Vec<u8> {
    let header_size = 69usize;
    let header_padded = align_up(header_size);
    let tensor_name = encode_string("q");
    let dims = [4u64];
    let data_payload = vec![1u8, 2, 3, 4];
    let quant_payload = quant_payload_per_tensor(0.5, 3);

    let mut tensor_table_zero = Vec::new();
    tensor_table_zero.extend_from_slice(&tensor_name);
    tensor_table_zero.extend_from_slice(&5u32.to_le_bytes()); // u8
    tensor_table_zero.extend_from_slice(&1u32.to_le_bytes()); // ndim
    tensor_table_zero.extend_from_slice(&3u32.to_le_bytes()); // HAS_DATA | HAS_QUANT
    tensor_table_zero.extend_from_slice(&dims[0].to_le_bytes());
    tensor_table_zero.extend_from_slice(&(data_payload.len() as u64).to_le_bytes());
    tensor_table_zero.extend_from_slice(&0u64.to_le_bytes()); // data_offset placeholder
    tensor_table_zero.extend_from_slice(&(quant_payload.len() as u64).to_le_bytes());
    tensor_table_zero.extend_from_slice(&0u64.to_le_bytes()); // quant_offset placeholder

    let offset_sizevars = header_padded;
    let offset_metadata = align_up(offset_sizevars);
    let offset_tensors = align_up(offset_metadata);
    let offset_data = align_up(offset_tensors + tensor_table_zero.len());

    let data_offset = offset_data;
    let quant_offset = align_up(data_offset + data_payload.len());
    let file_size = align_up(quant_offset + quant_payload.len());

    let mut tensor_table = Vec::new();
    tensor_table.extend_from_slice(&tensor_name);
    tensor_table.extend_from_slice(&5u32.to_le_bytes());
    tensor_table.extend_from_slice(&1u32.to_le_bytes());
    tensor_table.extend_from_slice(&3u32.to_le_bytes());
    tensor_table.extend_from_slice(&dims[0].to_le_bytes());
    tensor_table.extend_from_slice(&(data_payload.len() as u64).to_le_bytes());
    tensor_table.extend_from_slice(&(data_offset as u64).to_le_bytes());
    tensor_table.extend_from_slice(&(quant_payload.len() as u64).to_le_bytes());
    tensor_table.extend_from_slice(&(quant_offset as u64).to_le_bytes());

    let mut header = Vec::new();
    header.extend_from_slice(b"OINF\0");
    header.extend_from_slice(&2u32.to_le_bytes()); // canonical format
    header.extend_from_slice(&0u32.to_le_bytes()); // flags
    header.extend_from_slice(&0u32.to_le_bytes()); // n_sizevars
    header.extend_from_slice(&0u32.to_le_bytes()); // n_metadata
    header.extend_from_slice(&1u32.to_le_bytes()); // n_tensors
    header.extend_from_slice(&0u32.to_le_bytes()); // reserved
    header.extend_from_slice(&(offset_sizevars as u64).to_le_bytes());
    header.extend_from_slice(&(offset_metadata as u64).to_le_bytes());
    header.extend_from_slice(&(offset_tensors as u64).to_le_bytes());
    header.extend_from_slice(&(offset_data as u64).to_le_bytes());
    header.extend_from_slice(&(file_size as u64).to_le_bytes());
    header.resize(header_padded, 0);

    let mut out = Vec::new();
    out.extend_from_slice(&header);
    out.resize(offset_tensors, 0);
    out.extend_from_slice(&tensor_table);
    out.resize(offset_data, 0);
    out.extend_from_slice(&data_payload);
    out.resize(quant_offset, 0);
    out.extend_from_slice(&quant_payload);
    out.resize(file_size, 0);
    out
}

fn write_temp_model(bytes: &[u8]) -> Result<PathBuf> {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    path.push(format!("openinfer_quant_test_{}.oinf", nanos));
    fs::write(&path, bytes)?;
    Ok(path)
}

fn read_u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut out = [0u8; 8];
    out.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_le_bytes(out)
}

fn first_tensor_quant_offset(bytes: &[u8]) -> usize {
    let offset_tensors = read_u64_at(bytes, 45) as usize;
    let mut cursor = offset_tensors;
    let name_len = {
        let mut out = [0u8; 4];
        out.copy_from_slice(&bytes[cursor..cursor + 4]);
        u32::from_le_bytes(out) as usize
    };
    let string_padded = align_up(4 + name_len);
    cursor += string_padded;
    cursor += 12; // dtype, ndim, flags
    cursor += 8; // one dim
    let quant_offset_offset = cursor + 24; // data_nbytes,data_offset,quant_nbytes
    read_u64_at(bytes, quant_offset_offset) as usize
}

#[test]
fn model_loader_attaches_quant_params() -> Result<()> {
    let bytes = build_model_bytes();
    let path = write_temp_model(&bytes)?;
    let model = ModelLoader::open(&path)?;
    let tensor = model.load_tensor("q")?;
    let quant = tensor.quant().expect("expected quant metadata on tensor");
    match &quant.scale {
        QuantScale::PerTensor(v) => assert!((*v - 0.5).abs() < 1e-6),
        _ => panic!("expected per-tensor scale"),
    }
    match quant.zero_point.as_ref() {
        Some(QuantZeroPoint::PerTensor(v)) => assert_eq!(*v, 3),
        _ => panic!("expected per-tensor zero-point"),
    }
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn model_loader_rejects_legacy_version() -> Result<()> {
    let mut bytes = build_model_bytes();
    bytes[5..9].copy_from_slice(&1u32.to_le_bytes());
    let path = write_temp_model(&bytes)?;
    let err = ModelLoader::open(&path).expect_err("expected legacy version rejection");
    assert!(err.to_string().contains("unsupported OINF version"));
    fs::remove_file(path)?;
    Ok(())
}

#[test]
fn model_loader_rejects_malformed_quant_axis() -> Result<()> {
    let mut bytes = build_model_bytes();
    let quant_offset = first_tensor_quant_offset(&bytes);
    bytes[quant_offset + 16..quant_offset + 24].copy_from_slice(&9u64.to_le_bytes());
    let path = write_temp_model(&bytes)?;
    let err = ModelLoader::open(&path).expect_err("expected malformed quant axis rejection");
    assert!(err.to_string().contains("per-tensor scale requires axis=0,count=1"));
    fs::remove_file(path)?;
    Ok(())
}
