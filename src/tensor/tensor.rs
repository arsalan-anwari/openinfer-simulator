//! Tensor container and views.
//!
//! `Tensor<T>` owns flat storage with shape/stride metadata and provides view
//! access for slicing and indexing.
use anyhow::{anyhow, Result};
use std::cell::UnsafeCell;
use std::ops::Index;

use super::shape::{
    compute_strides, is_contiguous, linear_to_indices, numel, offset_for, validate_view,
    validate_view_metadata, view_parts,
};
use super::value::QuantParams;

/// Tensor construction options (shape/stride overrides).
#[derive(Debug, Clone, Default)]
pub struct TensorOptions {
    /// Optional explicit shape.
    pub shape: Option<Vec<usize>>,
    /// Optional explicit strides.
    pub strides: Option<Vec<isize>>,
    /// Optional explicit storage offset.
    pub offset_elems: usize,
    /// Allow length mismatch when using packed types.
    pub allow_len_mismatch: bool,
    /// Optional quantization metadata.
    pub quant: Option<QuantParams>,
}

/// Borrowed view into tensor data with shape/stride metadata.
#[derive(Debug, Clone)]
pub struct TensorView<T> {
    data: *const T,
    storage_len: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset_elems: usize,
    quant: Option<QuantParams>,
}

impl<T> TensorView<T> {
    fn new(
        data: *const T,
        storage_len: usize,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset_elems: usize,
        quant: Option<QuantParams>,
    ) -> Self {
        Self {
            data,
            storage_len,
            shape,
            strides,
            offset_elems,
            quant,
        }
    }

    /// Return the shape of this view.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the strides of this view.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the storage offset (in logical elements).
    pub fn offset_elems(&self) -> usize {
        self.offset_elems
    }

    /// Return optional quantization metadata.
    pub fn quant(&self) -> Option<&QuantParams> {
        self.quant.as_ref()
    }

    /// Return the logical element count.
    pub fn len(&self) -> usize {
        numel(&self.shape)
    }

    /// Access a value by multidimensional indices.
    pub fn at(&self, indices: &[usize]) -> &T {
        let offset = offset_for(&self.shape, &self.strides, self.offset_elems, indices)
            .unwrap_or_else(|err| panic!("tensor view index error: {}", err));
        if offset >= self.storage_len {
            panic!(
                "tensor view index error: reachable storage index {} exceeds backing length {}",
                offset, self.storage_len
            );
        }
        unsafe { &*self.data.add(offset) }
    }

    /// Return a contiguous slice if the view is contiguous.
    pub fn as_slice(&self) -> Option<&[T]> {
        if !is_contiguous(&self.shape, &self.strides) {
            return None;
        }
        let len = self.len();
        if len == 0 {
            return Some(&[]);
        }
        if self.offset_elems.checked_add(len)? > self.storage_len {
            return None;
        }
        unsafe { Some(std::slice::from_raw_parts(self.data.add(self.offset_elems), len)) }
    }

    /// Collect the view into a contiguous vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        if let Some(slice) = self.as_slice() {
            return slice.to_vec();
        }
        let mut out = Vec::with_capacity(self.len());
        for idx in 0..self.len() {
            let coords = linear_to_indices(idx, &self.shape);
            out.push(self.at(&coords).clone());
        }
        out
    }
}

/// Owned tensor container with shape and stride metadata.
#[derive(Debug)]
pub struct Tensor<T> {
    pub data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset_elems: usize,
    quant: Option<QuantParams>,
    // Indexing caches a view; this is not thread-safe.
    view_cache: UnsafeCell<TensorView<T>>,
}

// Tensor owns its backing storage; moving between threads is safe when it is
// not accessed concurrently.
unsafe impl<T: Send> Send for Tensor<T> {}

impl<T: Clone> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        let data = self.data.clone();
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let offset_elems = self.offset_elems;
        let quant = self.quant.clone();
        let data_ptr = data.as_ptr();
        let storage_len = data.len();
        Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            offset_elems,
            quant: quant.clone(),
            view_cache: UnsafeCell::new(TensorView::new(
                data_ptr,
                storage_len,
                shape,
                strides,
                offset_elems,
                quant,
            )),
        }
    }
}

impl<T> Tensor<T> {
    /// Build a tensor from a flat data vector.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::Tensor;
    /// # fn main() -> anyhow::Result<()> {
    /// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0])?;
    /// # Ok(()) }
    /// ```
    pub fn from_vec(data: Vec<T>) -> Result<Self> {
        Self::from_vec_with_opts(data, TensorOptions::default())
    }

    /// Build a tensor with explicit options.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::{Tensor, TensorOptions};
    /// # fn main() -> anyhow::Result<()> {
    /// let t = Tensor::from_vec_with_opts(
    ///     vec![1.0f32, 2.0, 3.0, 4.0],
    ///     TensorOptions { shape: Some(vec![2, 2]), ..TensorOptions::default() },
    /// )?;
    /// # Ok(()) }
    /// ```
    pub fn from_vec_with_opts(data: Vec<T>, opts: TensorOptions) -> Result<Self> {
        let TensorOptions {
            shape,
            strides,
            offset_elems,
            allow_len_mismatch,
            quant,
        } = opts;
        let shape = match shape {
            Some(shape) => shape,
            None => vec![data.len()],
        };
        let expected = numel(&shape);
        if !allow_len_mismatch && expected != data.len() {
            return Err(anyhow!(
                "tensor shape {:?} expects {} values, got {}",
                shape,
                expected,
                data.len()
            ));
        }
        if shape.is_empty() && !allow_len_mismatch && data.len() != 1 {
            return Err(anyhow!(
                "scalar tensor expects 1 value, got {}",
                data.len()
            ));
        }
        let strides = match strides {
            Some(strides) => {
                if strides.len() != shape.len() {
                    return Err(anyhow!(
                        "tensor strides length {} does not match shape length {}",
                        strides.len(),
                        shape.len()
                    ));
                }
                strides
            }
            None => compute_strides(&shape),
        };
        if !allow_len_mismatch {
            validate_view(&shape, &strides, offset_elems, data.len())?;
        } else {
            // Packed storage can have physical length < logical numel.
            validate_view_metadata(&shape, &strides, offset_elems)?;
        }
        let data_ptr = data.as_ptr();
        let storage_len = data.len();
        Ok(Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            offset_elems,
            quant: quant.clone(),
            view_cache: UnsafeCell::new(TensorView::new(
                data_ptr,
                storage_len,
                shape,
                strides,
                offset_elems,
                quant,
            )),
        })
    }

    /// Create a scalar tensor from a single value.
    ///
    /// # Example
    /// ```no_run
    /// # use openinfer::Tensor;
    /// let t = Tensor::from_scalar(3.14f32);
    /// ```
    pub fn from_scalar(value: T) -> Self {
        let data = vec![value];
        let data_ptr = data.as_ptr();
        let shape = Vec::new();
        let strides = compute_strides(&shape);
        Self {
            data,
            shape: shape.clone(),
            strides: strides.clone(),
            offset_elems: 0,
            quant: None,
            view_cache: UnsafeCell::new(TensorView::new(
                data_ptr,
                1,
                shape,
                strides,
                0,
                None,
            )),
        }
    }

    /// Create a tensor, panicking on invalid shape.
    pub fn new(data: Vec<T>) -> Self {
        Tensor::from_vec(data)
            .unwrap_or_else(|err| panic!("tensor creation failed: {}", err))
    }

    /// Return the raw data length.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the tensor strides.
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Return the storage offset (in logical elements).
    pub fn offset_elems(&self) -> usize {
        self.offset_elems
    }

    /// Return optional quantization metadata.
    pub fn quant(&self) -> Option<&QuantParams> {
        self.quant.as_ref()
    }

    /// Replace quantization metadata.
    pub fn set_quant(&mut self, quant: Option<QuantParams>) {
        self.quant = quant.clone();
        unsafe {
            (*self.view_cache.get()).quant = quant;
        }
    }

    /// True when tensor view maps to standard contiguous storage.
    pub fn is_contiguous(&self) -> bool {
        self.offset_elems == 0 && is_contiguous(&self.shape, &self.strides)
    }

    /// True when any logical stride is negative.
    pub fn has_negative_strides(&self) -> bool {
        self.strides.iter().any(|stride| *stride < 0)
    }

    /// Return the logical element count.
    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    /// Access a value by multidimensional indices.
    pub fn at(&self, indices: &[usize]) -> &T {
        let offset = offset_for(&self.shape, &self.strides, self.offset_elems, indices)
            .unwrap_or_else(|err| panic!("tensor index error: {}", err));
        if offset >= self.data.len() {
            panic!(
                "tensor index error: reachable storage index {} exceeds backing length {}",
                offset,
                self.data.len()
            );
        }
        &self.data[offset]
    }

    /// Create a view starting at the provided indices.
    pub fn view(&self, indices: &[usize]) -> TensorView<T> {
        let (offset, shape, strides) =
            view_parts(&self.shape, &self.strides, self.offset_elems, indices)
                .unwrap_or_else(|err| panic!("tensor view error: {}", err));
        TensorView::new(
            self.data.as_ptr(),
            self.data.len(),
            shape,
            strides,
            offset,
            self.quant.clone(),
        )
    }

    /// Clone the tensor data into a vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.clone()
    }
}

impl<T, const N: usize> Index<[usize; N]> for Tensor<T> {
    type Output = TensorView<T>;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        let view = self.view(&index);
        unsafe {
            *self.view_cache.get() = view;
            &*self.view_cache.get()
        }
    }
}
