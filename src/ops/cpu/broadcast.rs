use anyhow::{anyhow, Result};

pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let out_rank = a.len().max(b.len());
    let mut out = vec![1usize; out_rank];
    for i in 0..out_rank {
        let a_dim = dim_from_right(a, out_rank, i);
        let b_dim = dim_from_right(b, out_rank, i);
        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            out[i] = a_dim.max(b_dim);
        } else {
            return Err(anyhow!(
                "incompatible broadcast dims {} and {}",
                a_dim,
                b_dim
            ));
        }
    }
    Ok(out)
}

pub fn broadcast_strides(shape: &[usize], strides: &[isize], out_rank: usize) -> Vec<isize> {
    let mut out = vec![0isize; out_rank];
    let offset = out_rank.saturating_sub(shape.len());
    for i in 0..out_rank {
        if i < offset {
            out[i] = 0;
            continue;
        }
        let src_i = i - offset;
        let dim = shape[src_i];
        out[i] = if dim == 1 { 0 } else { strides[src_i] };
    }
    out
}

pub fn for_each_broadcast_index(
    out_shape: &[usize],
    out_strides: &[isize],
    a_strides: &[isize],
    b_strides: &[isize],
    mut f: impl FnMut(usize, usize, usize),
) {
    if out_shape.is_empty() {
        f(0, 0, 0);
        return;
    }
    let mut index = vec![0usize; out_shape.len()];
    let total = out_shape.iter().copied().product::<usize>();
    for _ in 0..total {
        let out_offset = linear_offset(&index, out_strides);
        let a_offset = linear_offset(&index, a_strides);
        let b_offset = linear_offset(&index, b_strides);
        f(out_offset, a_offset, b_offset);
        bump_index(&mut index, out_shape);
    }
}

fn dim_from_right(shape: &[usize], out_rank: usize, idx: usize) -> usize {
    let offset = out_rank.saturating_sub(shape.len());
    if idx < offset {
        1
    } else {
        shape[idx - offset]
    }
}

fn linear_offset(indices: &[usize], strides: &[isize]) -> usize {
    let mut offset = 0isize;
    for (idx, stride) in indices.iter().zip(strides.iter()) {
        let term = (*idx as isize)
            .checked_mul(*stride)
            .expect("broadcast offset multiplication overflow");
        offset = offset
            .checked_add(term)
            .expect("broadcast offset accumulation overflow");
    }
    usize::try_from(offset).expect("broadcast produced negative storage offset")
}

fn bump_index(index: &mut [usize], shape: &[usize]) {
    for (i, dim) in shape.iter().enumerate().rev() {
        index[i] += 1;
        if index[i] < *dim {
            return;
        }
        index[i] = 0;
    }
}
