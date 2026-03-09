//! Accumulation types demo: using the `acc` attribute with the graph!{} DSL.
//!
//! This example shows how to control accumulation dtypes for matmul and sum_axis
//! via the `acc=[f32, f32]` attribute. Ops that support accumulation (matmul,
//! sum_axis, prod_axis, mean_axis) accept `acc` to choose the accumulation type.
//!
//! Generate the model first:
//!   python openinfer-oinf/examples/accumulation_demo_oinf.py
//!
//! Then run:
//!   cargo run --example accumulation_demo
//!   cargo run --example accumulation_demo --features vulkan -- --target=vulkan

use openinfer::{
    fetch_executor, graph, insert_executor, ModelLoader, Random, Simulator, Tensor, TensorOptions,
};
mod util;
use util::{repo_root, select_device};

fn main() -> anyhow::Result<()> {
    let model_path = repo_root().join("res/models/accumulation_demo.oinf");
    let model = ModelLoader::open(model_path)?;

    let g = graph! {
        dynamic {
            x: f32[B, D];
        }

        constant {
            w: f32[D, H];
        }

        volatile {
            h: f32[B, H];
            y: f32[B, 1];
        }

        block entry {
            // matmul with explicit acc=[f32, f32] (accumulate and output in f32)
            op matmul(x, w, acc=[f32,f32]) >> h;
            // sum_axis with acc=[f32,f32], reduce along axis 1
            op sum_axis(h, axes=[1], keepdims=true, acc=[f32,f32]) >> y;
            return;
        }
    };

    let sim = Simulator::new(&model, &g, select_device()?)?;
    let mut exec = sim.make_executor()?;

    let b = model.size_of("B")?;
    let d = model.size_of("D")?;
    let x = Random::<f32>::generate_with_seed_opts(
        42,
        (-1.0, 1.0),
        b * d,
        TensorOptions {
            shape: Some(vec![b, d]),
            ..TensorOptions::default()
        },
    )?;

    insert_executor!(exec, { x: x });
    exec.step()?;

    fetch_executor!(exec, { y: Tensor<f32> });
    openinfer::log!("sum_axis output y[0..4] = {:?}", &y.data[..4.min(y.len())]);

    Ok(())
}
