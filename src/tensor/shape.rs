use anyhow::{anyhow, Result};

/// Compute the number of elements for a shape.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>()
}

/// Compute contiguous row-major strides for a shape.
pub fn compute_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1isize;
    for (idx, dim) in shape.iter().rev().enumerate() {
        let i = shape.len() - 1 - idx;
        strides[i] = stride;
        stride = stride.saturating_mul(*dim as isize);
    }
    strides
}

pub(crate) fn is_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }
    strides == compute_strides(shape)
}

pub(crate) fn reachable_index_range(shape: &[usize], strides: &[isize]) -> Result<(isize, isize)> {
    if shape.len() != strides.len() {
        return Err(anyhow!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    if shape.is_empty() {
        return Ok((0, 0));
    }
    if shape.iter().any(|dim| *dim == 0) {
        // Empty logical view: no reachable elements.
        return Ok((0, 0));
    }
    let mut min_rel = 0isize;
    let mut max_rel = 0isize;
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        let extent = (*dim as isize)
            .checked_sub(1)
            .ok_or_else(|| anyhow!("invalid zero dim extent"))?;
        let term = stride
            .checked_mul(extent)
            .ok_or_else(|| anyhow!("stride multiplication overflow"))?;
        if term < 0 {
            min_rel = min_rel
                .checked_add(term)
                .ok_or_else(|| anyhow!("minimum reachable offset overflow"))?;
        } else {
            max_rel = max_rel
                .checked_add(term)
                .ok_or_else(|| anyhow!("maximum reachable offset overflow"))?;
        }
    }
    Ok((min_rel, max_rel))
}

pub(crate) fn validate_view(
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    storage_len: usize,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(anyhow!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let logical_numel = numel(shape);
    if logical_numel == 0 {
        if offset > storage_len {
            return Err(anyhow!(
                "view offset {} exceeds storage length {}",
                offset,
                storage_len
            ));
        }
        return Ok(());
    }
    let (min_rel, max_rel) = reachable_index_range(shape, strides)?;
    let base = offset as isize;
    let min_abs = base
        .checked_add(min_rel)
        .ok_or_else(|| anyhow!("minimum reachable offset overflow"))?;
    let max_abs = base
        .checked_add(max_rel)
        .ok_or_else(|| anyhow!("maximum reachable offset overflow"))?;
    if min_abs < 0 {
        return Err(anyhow!("negative reachable storage index {}", min_abs));
    }
    if max_abs < 0 {
        return Err(anyhow!("negative reachable storage index {}", max_abs));
    }
    if max_abs >= storage_len as isize {
        return Err(anyhow!(
            "reachable storage index {} exceeds backing length {}",
            max_abs,
            storage_len
        ));
    }
    Ok(())
}

pub(crate) fn validate_view_metadata(
    shape: &[usize],
    strides: &[isize],
    offset: usize,
) -> Result<()> {
    if shape.len() != strides.len() {
        return Err(anyhow!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let (min_rel, _) = reachable_index_range(shape, strides)?;
    let min_abs = (offset as isize)
        .checked_add(min_rel)
        .ok_or_else(|| anyhow!("minimum reachable offset overflow"))?;
    if min_abs < 0 {
        return Err(anyhow!("negative reachable storage index {}", min_abs));
    }
    Ok(())
}

pub(crate) fn offset_for(
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    indices: &[usize],
) -> Result<usize> {
    if shape.len() != indices.len() {
        return Err(anyhow!(
            "expected {} indices, got {}",
            shape.len(),
            indices.len()
        ));
    }
    if shape.len() != strides.len() {
        return Err(anyhow!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let mut rel = 0isize;
    for ((dim, stride), idx) in shape.iter().zip(strides.iter()).zip(indices.iter()) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        let term = (*idx as isize)
            .checked_mul(*stride)
            .ok_or_else(|| anyhow!("index multiplication overflow"))?;
        rel = rel
            .checked_add(term)
            .ok_or_else(|| anyhow!("index accumulation overflow"))?;
    }
    let abs = (offset as isize)
        .checked_add(rel)
        .ok_or_else(|| anyhow!("absolute offset overflow"))?;
    if abs < 0 {
        return Err(anyhow!("negative reachable storage index {}", abs));
    }
    Ok(abs as usize)
}

pub(crate) fn view_parts(
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    indices: &[usize],
) -> Result<(usize, Vec<usize>, Vec<isize>)> {
    if indices.len() > shape.len() {
        return Err(anyhow!(
            "too many indices: got {}, shape has {} dims",
            indices.len(),
            shape.len()
        ));
    }
    if shape.len() != strides.len() {
        return Err(anyhow!(
            "shape rank {} does not match stride rank {}",
            shape.len(),
            strides.len()
        ));
    }
    let mut rel = 0isize;
    for (idx, (dim, stride)) in indices.iter().zip(shape.iter().zip(strides.iter())) {
        if *idx >= *dim {
            return Err(anyhow!("index {} out of bounds for dim {}", idx, dim));
        }
        let term = (*idx as isize)
            .checked_mul(*stride)
            .ok_or_else(|| anyhow!("index multiplication overflow"))?;
        rel = rel
            .checked_add(term)
            .ok_or_else(|| anyhow!("index accumulation overflow"))?;
    }
    let abs = (offset as isize)
        .checked_add(rel)
        .ok_or_else(|| anyhow!("absolute offset overflow"))?;
    if abs < 0 {
        return Err(anyhow!("negative reachable storage index {}", abs));
    }
    Ok((
        abs as usize,
        shape[indices.len()..].to_vec(),
        strides[indices.len()..].to_vec(),
    ))
}

pub(crate) fn linear_to_indices(linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut rem = linear;
    let mut out = Vec::with_capacity(shape.len());
    let strides = compute_strides(shape);
    for (dim, stride) in shape.iter().zip(strides.iter()) {
        if *stride == 0 {
            out.push(0);
        } else {
            let stride = *stride as usize;
            let coord = rem / stride;
            out.push(coord.min(dim.saturating_sub(1)));
            rem %= stride;
        }
    }
    out
}
