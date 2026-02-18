use anyhow::Result;
use openinfer::{QuantParams, QuantScale, QuantScheme, Tensor, TensorOptions, TensorValue};

#[test]
fn tensor_strides_match_default_layout() -> Result<()> {
    let data = vec![0.0f32; 6];
    let tensor = Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(vec![2, 3]),
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.strides(), &[3, 1]);
    Ok(())
}

#[test]
fn tensor_allows_len_mismatch_for_packed_shapes() -> Result<()> {
    let data = vec![1u8; 2];
    let tensor = Tensor::from_vec_with_opts(
        data,
        TensorOptions {
            shape: Some(vec![3]),
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.shape(), &[3]);
    Ok(())
}

#[test]
fn tensor_supports_negative_stride_with_offset() -> Result<()> {
    let tensor = Tensor::from_vec_with_opts(
        vec![10i32, 20, 30],
        TensorOptions {
            shape: Some(vec![3]),
            strides: Some(vec![-1]),
            offset_elems: 2,
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.at(&[0]), &30);
    assert_eq!(tensor.at(&[1]), &20);
    assert_eq!(tensor.at(&[2]), &10);
    Ok(())
}

#[test]
fn tensor_supports_non_zero_offset_contiguous_view() -> Result<()> {
    let tensor = Tensor::from_vec_with_opts(
        vec![0i32, 1, 2, 3, 4, 5, 6],
        TensorOptions {
            shape: Some(vec![2, 2]),
            strides: Some(vec![2, 1]),
            offset_elems: 2,
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )?;
    assert_eq!(tensor.at(&[0, 0]), &2);
    assert_eq!(tensor.at(&[1, 1]), &5);
    Ok(())
}

#[test]
fn tensor_rejects_out_of_bounds_view_metadata() {
    let err = Tensor::from_vec_with_opts(
        vec![1u8, 2, 3, 4],
        TensorOptions {
            shape: Some(vec![2, 2]),
            strides: Some(vec![2, 1]),
            offset_elems: 1,
            ..TensorOptions::default()
        },
    )
    .expect_err("expected out-of-bounds validation error");
    assert!(err.to_string().contains("exceeds backing length"));
}

#[test]
fn tensor_rejects_negative_reachable_storage_index() {
    let err = Tensor::from_vec_with_opts(
        vec![1u8, 2, 3],
        TensorOptions {
            shape: Some(vec![2]),
            strides: Some(vec![-1]),
            offset_elems: 0,
            allow_len_mismatch: true,
            ..TensorOptions::default()
        },
    )
    .expect_err("expected negative index validation error");
    assert!(err.to_string().contains("negative reachable storage index"));
}

#[test]
fn tensor_value_preserves_quant_metadata_roundtrip() -> Result<()> {
    let quant = QuantParams {
        scheme: QuantScheme::Asymmetric,
        scale: QuantScale::PerTensor(0.125),
        zero_point: None,
    };
    let tensor = Tensor::from_vec_with_opts(
        vec![1u8, 2, 3, 4],
        TensorOptions {
            shape: Some(vec![2, 2]),
            quant: Some(quant.clone()),
            ..TensorOptions::default()
        },
    )?;
    let value = TensorValue::from(tensor);
    assert_eq!(value.quant(), Some(&quant));
    Ok(())
}

#[test]
fn logical_index_mapping_matches_manual_offsets() -> Result<()> {
    let data: Vec<i32> = (0..24).collect();
    let cases = vec![
        (vec![2, 3], vec![3, 1], 0usize),
        (vec![2, 3], vec![-3, 1], 3usize),
        (vec![2, 3], vec![3, -1], 2usize),
    ];
    for (shape, strides, offset) in cases {
        let tensor = Tensor::from_vec_with_opts(
            data.clone(),
            TensorOptions {
                shape: Some(shape.clone()),
                strides: Some(strides.clone()),
                offset_elems: offset,
                allow_len_mismatch: true,
                ..TensorOptions::default()
            },
        )?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                let abs = (offset as isize) + (i as isize) * strides[0] + (j as isize) * strides[1];
                let abs = usize::try_from(abs).expect("manual offset became negative");
                assert_eq!(tensor.at(&[i, j]), &data[abs]);
            }
        }
    }
    Ok(())
}
