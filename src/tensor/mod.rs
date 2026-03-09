//! Tensor and dtype utilities.
//!
//! Provides tensor containers (`Tensor`/`TensorValue`) and dtype helpers used
//! across the runtime and graph layers.
//!
//! ## Highlights
//! - `Tensor<T>`: owned tensor container with shape/stride metadata.
//! - `TensorValue`: enum for runtime dtype dispatch.
//! - `DType`: dtype definitions and packing utilities.
mod scalar;
mod scalar_value;
mod shape;
mod tensor;
mod value;

pub use scalar::{BF16, F16, F8, I4, U4};
pub use scalar_value::ScalarValue;
#[allow(unused_imports)]
pub use shape::{compute_strides, numel};
#[allow(unused_imports)]
pub use tensor::{Tensor, TensorOptions, TensorView};
#[allow(unused_imports)]
pub use value::{
    DType, QuantParams, QuantScale, QuantScheme, QuantZeroPoint, TensorElement, TensorValue,
};
