//! V2 Input Preparation Pipeline
//!
//! This module provides high-performance input preparation for the model forward pass.
//! It replaces the Python-heavy `_prepare_inputs` method with vectorized Rust operations.
//!
//! Key functions:
//! - prepare_decode_inputs_v2: Fast decode input preparation
//! - prepare_prefill_inputs_v2: Prefill input preparation
//! - compute_slot_mapping_batch: Vectorized slot mapping
//! - build_attn_metadata: Build attention metadata struct

use numpy::{PyArray1, ToPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Prepare decode inputs for a batch of sequences.
///
/// This is the V2 fast path for decode - it computes all required tensors
/// in a single Rust call, avoiding Python loop overhead.
///
/// Args:
///     token_ids: [num_reqs] - The next token ID for each sequence
///     num_computed_tokens: [num_reqs] - Tokens already computed per sequence
///     block_table: [num_reqs, max_blocks] - Physical block indices
///     block_size: Tokens per block
///
/// Returns:
///     Tuple of (input_ids, positions, seq_lens, query_start_loc, slot_mapping, max_seq_len)
#[pyfunction]
pub fn prepare_decode_inputs_v2<'py>(
    py: Python<'py>,
    token_ids: PyReadonlyArray1<i64>,
    num_computed_tokens: PyReadonlyArray1<i64>,
    block_table: PyReadonlyArray2<i64>,
    block_size: i64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,  // input_ids
    Bound<'py, PyArray1<i64>>,  // positions
    Bound<'py, PyArray1<i64>>,  // seq_lens
    Bound<'py, PyArray1<i64>>,  // query_start_loc
    Bound<'py, PyArray1<i64>>,  // slot_mapping
    i64,                         // max_seq_len
)> {
    let tokens = token_ids.as_array();
    let computed = num_computed_tokens.as_array();
    let blocks = block_table.as_array();

    let num_reqs = tokens.len();
    let max_blocks = blocks.shape()[1];

    // Pre-allocate all output vectors
    let mut input_ids: Vec<i64> = Vec::with_capacity(num_reqs);
    let mut positions: Vec<i64> = Vec::with_capacity(num_reqs);
    let mut seq_lens: Vec<i64> = Vec::with_capacity(num_reqs);
    let mut query_start_loc: Vec<i64> = Vec::with_capacity(num_reqs + 1);
    let mut slot_mapping: Vec<i64> = Vec::with_capacity(num_reqs);

    let mut max_seq_len: i64 = 0;
    let mut query_start: i64 = 0;

    query_start_loc.push(0);

    for i in 0..num_reqs {
        // Input token ID
        input_ids.push(tokens[i]);

        // Position is the current computed count (0-indexed)
        let pos = computed[i];
        positions.push(pos);

        // Sequence length after this decode step
        let seq_len = computed[i] + 1;
        seq_lens.push(seq_len);

        if seq_len > max_seq_len {
            max_seq_len = seq_len;
        }

        // Query start location
        query_start += 1;
        query_start_loc.push(query_start);

        // Slot mapping for KV cache
        let logical_block = pos / block_size;
        let block_offset = pos % block_size;
        let block_idx = (logical_block as usize).min(max_blocks - 1);
        let physical_block = blocks[[i, block_idx]];
        let slot = physical_block * block_size + block_offset;
        slot_mapping.push(slot);
    }

    Ok((
        input_ids.as_slice().to_pyarray_bound(py),
        positions.as_slice().to_pyarray_bound(py),
        seq_lens.as_slice().to_pyarray_bound(py),
        query_start_loc.as_slice().to_pyarray_bound(py),
        slot_mapping.as_slice().to_pyarray_bound(py),
        max_seq_len,
    ))
}

/// Prepare prefill inputs for a batch of sequences.
///
/// Handles variable-length prefill with multiple tokens per sequence.
///
/// Args:
///     token_ids_flat: [total_tokens] - Flattened token IDs
///     token_indices: [total_tokens] - Indices into flat token_ids
///     num_computed_tokens: [num_reqs] - Already computed tokens
///     num_scheduled_tokens: [num_reqs] - Tokens to process this step
///     block_table: [num_reqs, max_blocks] - Physical block indices
///     block_size: Tokens per block
///
/// Returns:
///     Tuple of tensors for model forward pass
#[pyfunction]
pub fn prepare_prefill_inputs_v2<'py>(
    py: Python<'py>,
    token_ids_flat: PyReadonlyArray1<i64>,
    token_indices: PyReadonlyArray1<i64>,
    num_computed_tokens: PyReadonlyArray1<i64>,
    num_scheduled_tokens: PyReadonlyArray1<i64>,
    block_table: PyReadonlyArray2<i64>,
    block_size: i64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,  // input_ids (selected tokens)
    Bound<'py, PyArray1<i64>>,  // positions
    Bound<'py, PyArray1<i64>>,  // seq_lens
    Bound<'py, PyArray1<i64>>,  // query_start_loc
    Bound<'py, PyArray1<i64>>,  // slot_mapping
    i64,                         // max_seq_len
    i64,                         // max_query_len
)> {
    let tokens_flat = token_ids_flat.as_array();
    let indices = token_indices.as_array();
    let computed = num_computed_tokens.as_array();
    let scheduled = num_scheduled_tokens.as_array();
    let blocks = block_table.as_array();

    let num_reqs = computed.len();
    let max_blocks = blocks.shape()[1];
    let total_tokens: i64 = scheduled.iter().sum();

    // Pre-allocate output vectors
    let mut input_ids: Vec<i64> = Vec::with_capacity(total_tokens as usize);
    let mut positions: Vec<i64> = Vec::with_capacity(total_tokens as usize);
    let mut seq_lens: Vec<i64> = Vec::with_capacity(num_reqs);
    let mut query_start_loc: Vec<i64> = Vec::with_capacity(num_reqs + 1);
    let mut slot_mapping: Vec<i64> = Vec::with_capacity(total_tokens as usize);

    let mut max_seq_len: i64 = 0;
    let mut max_query_len: i64 = 0;
    let mut token_offset: usize = 0;

    query_start_loc.push(0);

    for req_idx in 0..num_reqs {
        let num_tokens = scheduled[req_idx];
        let start_pos = computed[req_idx];
        let seq_len = start_pos + num_tokens;

        seq_lens.push(seq_len);
        if seq_len > max_seq_len {
            max_seq_len = seq_len;
        }
        if num_tokens > max_query_len {
            max_query_len = num_tokens;
        }

        // Query start location
        query_start_loc.push(query_start_loc.last().unwrap() + num_tokens);

        // Process each token in this sequence
        for token_idx in 0..num_tokens {
            let pos = start_pos + token_idx;

            // Get token ID from flat array using indices
            let flat_idx = indices[token_offset] as usize;
            input_ids.push(tokens_flat[flat_idx]);
            token_offset += 1;

            // Position
            positions.push(pos);

            // Slot mapping
            let logical_block = pos / block_size;
            let block_offset = pos % block_size;
            let block_idx = (logical_block as usize).min(max_blocks - 1);
            let physical_block = blocks[[req_idx, block_idx]];
            let slot = physical_block * block_size + block_offset;
            slot_mapping.push(slot);
        }
    }

    Ok((
        input_ids.as_slice().to_pyarray_bound(py),
        positions.as_slice().to_pyarray_bound(py),
        seq_lens.as_slice().to_pyarray_bound(py),
        query_start_loc.as_slice().to_pyarray_bound(py),
        slot_mapping.as_slice().to_pyarray_bound(py),
        max_seq_len,
        max_query_len,
    ))
}

/// Compute slot mapping for a batch of positions.
///
/// Vectorized slot computation for arbitrary positions.
///
/// Args:
///     req_indices: [num_tokens] - Request index for each token
///     positions: [num_tokens] - Position for each token
///     block_table: [num_reqs, max_blocks] - Physical block indices
///     block_size: Tokens per block
///
/// Returns:
///     slot_mapping: [num_tokens] - Physical slot for each token
#[pyfunction]
pub fn compute_slot_mapping_batch<'py>(
    py: Python<'py>,
    req_indices: PyReadonlyArray1<i64>,
    positions: PyReadonlyArray1<i64>,
    block_table: PyReadonlyArray2<i64>,
    block_size: i64,
) -> Bound<'py, PyArray1<i64>> {
    let req_idx_arr = req_indices.as_array();
    let pos_arr = positions.as_array();
    let blocks = block_table.as_array();

    let num_tokens = pos_arr.len();
    let max_blocks = blocks.shape()[1];

    let mut slot_mapping: Vec<i64> = Vec::with_capacity(num_tokens);

    for i in 0..num_tokens {
        let req_idx = req_idx_arr[i] as usize;
        let pos = pos_arr[i];

        let logical_block = pos / block_size;
        let block_offset = pos % block_size;
        let block_idx = (logical_block as usize).min(max_blocks - 1);
        let physical_block = blocks[[req_idx, block_idx]];
        let slot = physical_block * block_size + block_offset;
        slot_mapping.push(slot);
    }

    slot_mapping.as_slice().to_pyarray_bound(py)
}

/// Build attention metadata dictionary.
///
/// Creates all the metadata needed for the attention layer.
///
/// Args:
///     num_reqs: Number of requests
///     num_tokens: Total tokens in batch
///     seq_lens: Sequence lengths
///     query_start_loc: Query start locations
///     max_seq_len: Maximum sequence length
///     is_decode: Whether this is a decode step
///
/// Returns:
///     Dictionary with attention metadata fields
#[pyfunction]
pub fn build_attn_metadata<'py>(
    py: Python<'py>,
    num_reqs: usize,
    num_tokens: usize,
    seq_lens: PyReadonlyArray1<i64>,
    query_start_loc: PyReadonlyArray1<i64>,
    max_seq_len: i64,
    is_decode: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new_bound(py);

    metadata.set_item("num_reqs", num_reqs)?;
    metadata.set_item("num_input_tokens", num_tokens)?;
    metadata.set_item("max_seq_len", max_seq_len)?;
    metadata.set_item("is_decode", is_decode)?;

    // For decode, query_len = 1 for all sequences
    if is_decode {
        metadata.set_item("max_query_len", 1)?;
        metadata.set_item("query_lens", vec![1i64; num_reqs])?;
    } else {
        // For prefill, compute from query_start_loc
        let qsl = query_start_loc.as_array();
        let mut query_lens: Vec<i64> = Vec::with_capacity(num_reqs);
        let mut max_query_len: i64 = 0;

        for i in 0..num_reqs {
            let qlen = qsl[i + 1] - qsl[i];
            query_lens.push(qlen);
            if qlen > max_query_len {
                max_query_len = qlen;
            }
        }

        metadata.set_item("max_query_len", max_query_len)?;
        metadata.set_item("query_lens", query_lens)?;
    }

    Ok(metadata)
}

/// Compute logits indices for sampling.
///
/// For decode, this is simply [0, 1, 2, ...] since each sequence has 1 token.
/// For prefill, this points to the last token of each sequence.
#[pyfunction]
pub fn compute_logits_indices<'py>(
    py: Python<'py>,
    query_start_loc: PyReadonlyArray1<i64>,
    num_reqs: usize,
    is_decode: bool,
) -> Bound<'py, PyArray1<i64>> {
    if is_decode {
        // For decode, logits indices are 0, 1, 2, ... (each request has 1 token)
        let indices: Vec<i64> = (0..num_reqs as i64).collect();
        indices.as_slice().to_pyarray_bound(py)
    } else {
        // For prefill, logits indices point to last token of each sequence
        let qsl = query_start_loc.as_array();
        let mut indices: Vec<i64> = Vec::with_capacity(num_reqs);

        for i in 0..num_reqs {
            // Last token index for this sequence
            indices.push(qsl[i + 1] - 1);
        }

        indices.as_slice().to_pyarray_bound(py)
    }
}

/// Compute request indices for each token in the batch.
///
/// Expands request indices based on query lengths.
#[pyfunction]
pub fn compute_req_indices<'py>(
    py: Python<'py>,
    query_start_loc: PyReadonlyArray1<i64>,
    num_reqs: usize,
) -> Bound<'py, PyArray1<i64>> {
    let qsl = query_start_loc.as_array();
    let total_tokens = qsl[num_reqs] as usize;

    let mut req_indices: Vec<i64> = Vec::with_capacity(total_tokens);

    for req_idx in 0..num_reqs {
        let start = qsl[req_idx] as usize;
        let end = qsl[req_idx + 1] as usize;
        for _ in start..end {
            req_indices.push(req_idx as i64);
        }
    }

    req_indices.as_slice().to_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_inputs_logic() {
        // Test the logic without numpy
        let tokens = vec![100, 200, 300];
        let computed = vec![10, 20, 30];
        let block_size = 16;

        let mut positions = Vec::new();
        let mut seq_lens = Vec::new();
        let mut slot_mapping = Vec::new();

        for i in 0..3 {
            positions.push(computed[i]);
            seq_lens.push(computed[i] + 1);

            let pos = computed[i];
            let logical_block = pos / block_size;
            let block_offset = pos % block_size;
            // Assume physical_block = logical_block for test
            let slot = logical_block * block_size + block_offset;
            slot_mapping.push(slot);
        }

        assert_eq!(positions, vec![10, 20, 30]);
        assert_eq!(seq_lens, vec![11, 21, 31]);
        // slot = pos for this test since physical = logical
        assert_eq!(slot_mapping, vec![10, 20, 30]);
    }
}
