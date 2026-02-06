#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
try:
    from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf  # type: ignore[import-not-found]
except ImportError:
    sys.path.insert(0, str(ROOT.parent / "openinfer-oinf"))
    from dataclass_to_oinf import SizeVar, TensorSpec, write_oinf  # noqa: E402


@dataclass
class ModelSpec:
    sizevars: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)


def write_model(name: str, spec: ModelSpec) -> None:
    output_dir = ROOT / "res" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.oinf"
    write_oinf(spec, str(path))
    print(f"Wrote {path}")


def build_models() -> None:
    rng = np.random.default_rng(42)

    write_model(
        "minimal_model",
        ModelSpec(sizevars={"B": SizeVar(4)}),
    )

    write_model(
        "cache_scalar_model",
        ModelSpec(),
    )

    write_model(
        "cache_table_model",
        ModelSpec(sizevars={"D": SizeVar(4)}),
    )

    write_model(
        "cache_auto_dim_model",
        ModelSpec(sizevars={"D": SizeVar(4), "H": SizeVar(3)}),
    )

    write_model(
        "cache_weight_update_model",
        ModelSpec(sizevars={"D": SizeVar(4)}),
    )

    d = 4
    b = 3
    write_model(
        "branching_model",
        ModelSpec(
            sizevars={"B": SizeVar(b), "D": SizeVar(d)},
            tensors={
                "w": TensorSpec(rng.normal(scale=0.2, size=(d, d)).astype(np.float32)),
            },
        ),
    )

    num_layers = 2
    num_heads = 3
    loop_tensors = {}
    for layer in range(num_layers):
        for head in range(num_heads):
            key = f"attn.{head}.qkv.{layer}"
            loop_tensors[key] = TensorSpec(
                rng.normal(scale=0.2, size=(d, 3 * d)).astype(np.float32)
            )
    write_model(
        "loop_model",
        ModelSpec(
            sizevars={
                "D": SizeVar(d),
                "num_layers": SizeVar(num_layers),
                "num_heads": SizeVar(num_heads),
            },
            tensors=loop_tensors,
        ),
    )

    write_model(
        "yield_model",
        ModelSpec(
            sizevars={"B": SizeVar(b), "D": SizeVar(d)},
            tensors={
                "w": TensorSpec(rng.normal(scale=0.2, size=(d, d)).astype(np.float32)),
                "bias": TensorSpec(rng.normal(scale=0.05, size=(d,)).astype(np.float32)),
            },
        ),
    )

    prefix_tensors = {f"W.{idx}": TensorSpec(rng.normal(scale=0.1, size=(d,)).astype(np.float32)) for idx in range(11)}
    prefix_tensors["QKV.0.0"] = TensorSpec(rng.normal(scale=0.1, size=(d,)).astype(np.float32))
    prefix_tensors["QKV.1.2"] = TensorSpec(rng.normal(scale=0.1, size=(d,)).astype(np.float32))
    write_model(
        "prefix_table_model",
        ModelSpec(sizevars={"D": SizeVar(d)}, tensors=prefix_tensors),
    )

    write_model(
        "reference_model",
        ModelSpec(
            sizevars={"B": SizeVar(4)},
            tensors={
                "state.0": TensorSpec(np.zeros((4,), dtype=np.float32)),
                "weight.0": TensorSpec(rng.normal(scale=0.1, size=(4,)).astype(np.float32)),
                "bias.0": TensorSpec(np.array(0.25, dtype=np.float32)),
            },
        ),
    )

    write_model(
        "attrs_from_model",
        ModelSpec(
            sizevars={"B": SizeVar(5)},
            metadata={"rounding_mode": "trunc"},
            tensors={
                "alpha": TensorSpec(np.array(0.1, dtype=np.float32)),
                "clamp_max": TensorSpec(np.array(1.7, dtype=np.float32)),
            },
        ),
    )


def main() -> None:
    build_models()


if __name__ == "__main__":
    main()
