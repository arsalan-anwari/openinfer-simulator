## openinfer-simulator

Host-side simulator and runtime for OpenInfer graphs. This crate validates
graph correctness, scheduling logic, and memory layouts without depending on
target devices.

Key responsibilities:
- Load `.oinf` model packages and resolve sizevars/tensors.
- Execute graphs in a deterministic, inspectable simulator.
- Provide traces and validation diagnostics for synthesis.

Docs: https://github.com/arsalan-awnari/openinfer/tree/main/docs/sphinx/modules/openinfer-simulator
