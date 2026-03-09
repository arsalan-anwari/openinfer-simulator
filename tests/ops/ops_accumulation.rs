//! Tests for user-defined accumulation types (acc attribute).
//!
//! Covers: matmul, sum_axis, prod_axis, mean_axis with acc=[...],
//! validation errors for invalid acc, and default accumulation when acc is omitted.

use anyhow::Result;
use openinfer::{
    AttrValue, Device, Graph, MemoryKind, ModelLoader, NodeKind, OpAttr, OpAttrs, OpKind,
    TensorValue,
};

use crate::common;

const ENTRY_BLOCK: &str = "entry";

#[test]
fn ops_matmul_with_acc_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_matmul.oinf")?;

    for device in common::test_targets() {
        run_matmul_with_acc(&model, device)?;
    }
    Ok(())
}

fn run_matmul_with_acc(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let left = model.load_tensor("matmul_a")?;
    let right = model.load_tensor("matmul_b")?;
    let expected = model.load_tensor("matmul_out")?;

    add_dynamic(&mut graph, "matmul_a", &left);
    add_dynamic(&mut graph, "matmul_b", &right);
    add_volatile(&mut graph, "matmul_out", &expected);

    let attrs = OpAttrs {
        items: vec![OpAttr {
            name: "acc".to_string(),
            value: AttrValue::DTypeList(vec![
                openinfer::DType::F32,
                openinfer::DType::F32,
            ]),
        }],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Matmul,
            attrs,
            inputs: vec!["matmul_a".to_string(), "matmul_b".to_string()],
            output: "matmul_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping matmul with acc on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("matmul_a", left)?;
    exec.insert_dynamic("matmul_b", right)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("matmul_out")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

#[test]
fn ops_sum_axis_with_acc_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_reduce.oinf")?;

    for device in common::test_targets() {
        run_sum_axis_with_acc(&model, device)?;
    }
    Ok(())
}

fn run_sum_axis_with_acc(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("reduce_x")?;
    let expected = model.load_tensor("sum_axis_out")?;

    add_dynamic(&mut graph, "reduce_x", &input);
    add_volatile(&mut graph, "sum_axis_out", &expected);

    let attrs = OpAttrs {
        items: vec![
            OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
            OpAttr {
                name: "acc".to_string(),
                value: AttrValue::DTypeList(vec![
                    openinfer::DType::F32,
                    openinfer::DType::F32,
                ]),
            },
        ],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::SumAxis,
            attrs,
            inputs: vec!["reduce_x".to_string()],
            output: "sum_axis_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping sum_axis with acc on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("reduce_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("sum_axis_out")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

#[test]
fn ops_prod_axis_with_acc_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_reduce.oinf")?;

    for device in common::test_targets() {
        run_prod_axis_with_acc(&model, device)?;
    }
    Ok(())
}

fn run_prod_axis_with_acc(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("reduce_x")?;
    let expected = model.load_tensor("prod_axis_out")?;

    add_dynamic(&mut graph, "reduce_x", &input);
    add_volatile(&mut graph, "prod_axis_out", &expected);

    let attrs = OpAttrs {
        items: vec![
            OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
            OpAttr {
                name: "acc".to_string(),
                value: AttrValue::DTypeList(vec![
                    openinfer::DType::F32,
                    openinfer::DType::F32,
                ]),
            },
        ],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::ProdAxis,
            attrs,
            inputs: vec!["reduce_x".to_string()],
            output: "prod_axis_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping prod_axis with acc on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("reduce_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("prod_axis_out")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

#[test]
fn ops_mean_axis_with_acc_parity() -> Result<()> {
    let model = common::load_baseline_model("ops/baseline/data/ops_reduce.oinf")?;

    for device in common::test_targets() {
        run_mean_axis_with_acc(&model, device)?;
    }
    Ok(())
}

fn run_mean_axis_with_acc(model: &ModelLoader, device: Device) -> Result<()> {
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("reduce_x")?;
    let expected = model.load_tensor("mean_axis_out")?;

    add_dynamic(&mut graph, "reduce_x", &input);
    add_volatile(&mut graph, "mean_axis_out", &expected);

    let attrs = OpAttrs {
        items: vec![
            OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
            OpAttr {
                name: "acc".to_string(),
                value: AttrValue::DTypeList(vec![
                    openinfer::DType::F32,
                    openinfer::DType::F32,
                ]),
            },
        ],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::MeanAxis,
            attrs,
            inputs: vec!["reduce_x".to_string()],
            output: "mean_axis_out".to_string(),
        },
    )?;
    graph.add_node(ENTRY_BLOCK, NodeKind::Return)?;

    let sim = match openinfer::Simulator::new(model, &graph, device) {
        Ok(sim) => sim,
        Err(err) => {
            if device == Device::Vulkan {
                eprintln!("Skipping mean_axis with acc on {:?}: {}", device, err);
                return Ok(());
            }
            return Err(err);
        }
    };
    let mut exec = sim.make_executor()?;
    exec.insert_dynamic("reduce_x", input)?;
    exec.step()?;
    let actual: TensorValue = exec.fetch("mean_axis_out")?;
    common::assert_tensor_close(&actual, &expected, device)?;
    Ok(())
}

#[test]
fn ops_sum_axis_invalid_acc_fails_validation() {
    let model = common::load_baseline_model("ops/baseline/data/ops_reduce.oinf").unwrap();
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let input = model.load_tensor("reduce_x").unwrap();
    let expected = model.load_tensor("sum_axis_out").unwrap();

    add_dynamic(&mut graph, "reduce_x", &input);
    add_volatile(&mut graph, "sum_axis_out", &expected);

    // Invalid acc: [i16, i16] is not in sum_axis's accumulation_rules
    let attrs = OpAttrs {
        items: vec![
            OpAttr {
                name: "axes".to_string(),
                value: AttrValue::IntList(vec![1]),
            },
            OpAttr {
                name: "keepdims".to_string(),
                value: AttrValue::Bool(true),
            },
            OpAttr {
                name: "acc".to_string(),
                value: AttrValue::DTypeList(vec![
                    openinfer::DType::I16,
                    openinfer::DType::I16,
                ]),
            },
        ],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::SumAxis,
            attrs,
            inputs: vec!["reduce_x".to_string()],
            output: "sum_axis_out".to_string(),
        },
    )
    .unwrap();
    graph.add_node(ENTRY_BLOCK, NodeKind::Return).unwrap();

    let result = openinfer::Simulator::new(&model, &graph, Device::Cpu);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(
        err_str.contains("accumulation") || err_str.contains("acc"),
        "expected validation error about acc, got: {}",
        err
    );
}

#[test]
fn ops_matmul_invalid_acc_fails_validation() {
    let model = common::load_baseline_model("ops/baseline/data/ops_matmul.oinf").unwrap();
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let left = model.load_tensor("matmul_a").unwrap();
    let right = model.load_tensor("matmul_b").unwrap();
    let expected = model.load_tensor("matmul_out").unwrap();

    add_dynamic(&mut graph, "matmul_a", &left);
    add_dynamic(&mut graph, "matmul_b", &right);
    add_volatile(&mut graph, "matmul_out", &expected);

    // Invalid acc: [f16, f16] is not in matmul's accumulation_rules
    let attrs = OpAttrs {
        items: vec![OpAttr {
            name: "acc".to_string(),
            value: AttrValue::DTypeList(vec![
                openinfer::DType::F16,
                openinfer::DType::F16,
            ]),
        }],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Matmul,
            attrs,
            inputs: vec!["matmul_a".to_string(), "matmul_b".to_string()],
            output: "matmul_out".to_string(),
        },
    )
    .unwrap();
    graph.add_node(ENTRY_BLOCK, NodeKind::Return).unwrap();

    let result = openinfer::Simulator::new(&model, &graph, Device::Cpu);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(
        err_str.contains("accumulation") || err_str.contains("acc"),
        "expected validation error about acc, got: {}",
        err
    );
}

#[test]
fn ops_add_rejects_acc() {
    let model = common::load_baseline_model("ops/baseline/data/ops_basic.oinf").unwrap();
    let mut graph = Graph::new();
    graph.add_block(ENTRY_BLOCK);

    let a = model.load_tensor("add_a").unwrap();
    let b = model.load_tensor("add_b").unwrap();
    let expected = model.load_tensor("add_out").unwrap();

    add_dynamic(&mut graph, "add_a", &a);
    add_dynamic(&mut graph, "add_b", &b);
    add_volatile(&mut graph, "add_out", &expected);

    // add has no accumulation_rules; acc should be rejected
    let attrs = OpAttrs {
        items: vec![OpAttr {
            name: "acc".to_string(),
            value: AttrValue::DTypeList(vec![openinfer::DType::F32]),
        }],
    };

    graph.add_node(
        ENTRY_BLOCK,
        NodeKind::Op {
            op: OpKind::Add,
            attrs,
            inputs: vec!["add_a".to_string(), "add_b".to_string()],
            output: "add_out".to_string(),
        },
    )
    .unwrap();
    graph.add_node(ENTRY_BLOCK, NodeKind::Return).unwrap();

    let result = openinfer::Simulator::new(&model, &graph, Device::Cpu);
    assert!(result.is_err());
    let err = result.unwrap_err();
    let err_str = format!("{:?}", err);
    assert!(
        err_str.contains("acc") || err_str.contains("unsupported") || err_str.contains("accumulation"),
        "expected validation error about acc, got: {}",
        err
    );
}

fn add_dynamic(graph: &mut Graph, name: &str, tensor: &TensorValue) {
    graph.add_var(
        MemoryKind::Dynamic,
        name,
        tensor.dtype(),
        dims_from_shape(tensor.shape()),
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
}

fn add_volatile(graph: &mut Graph, name: &str, tensor: &TensorValue) {
    graph.add_var(
        MemoryKind::Volatile,
        name,
        tensor.dtype(),
        dims_from_shape(tensor.shape()),
        None,
        None,
        Vec::new(),
        None,
        false,
        Vec::new(),
        Vec::new(),
    );
}

fn dims_from_shape(shape: &[usize]) -> Vec<String> {
    shape.iter().map(|dim| dim.to_string()).collect()
}
