## openinfer-simulator

Host-side simulator and runtime for OpenInfer graphs. This crate validates
graph correctness, scheduling logic, and memory layouts without depending on
target devices.

Key responsibilities:
- Load `.oinf` model packages and resolve sizevars/tensors.
- Execute graphs in a deterministic, inspectable simulator.
- Provide traces and validation diagnostics for synthesis.

### Build
```bash
cargo check
cargo build
```

Vulkan (optional):
```bash
cargo build --features vulkan
```

### Tests
```bash
cargo test
```

### Examples
```bash
cargo run --example mlp_regression
```

### Model files
Graph tests and examples load `.oinf` model files from `res/models`. In the main
repo (https://github.com/arsalan-anwari/openinfer), these are synced from `openinfer-oinf` via:
```bash
./scripts/sync_models.sh
```

Docs: docs.open-infer.nl
