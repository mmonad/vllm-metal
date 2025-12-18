//! V2 Batch State Manager
//!
//! This module provides GPU-persistent batch state management, eliminating
//! the need to rebuild batch metadata on every decode iteration.
//!
//! Key V2 optimizations:
//! - Maintains request state in Rust (no Python dict overhead)
//! - Incremental updates instead of full reconstruction
//! - Pre-computed slot mappings that update in-place
//! - Fast request ID comparison and lookup

use numpy::{PyArray1, PyArray2, ToPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

/// Request state for a single sequence
#[derive(Clone, Debug)]
pub struct RequestState {
    /// Unique request ID
    pub req_id: String,
    /// Index in the current batch
    pub batch_idx: usize,
    /// Number of tokens computed so far
    pub num_computed_tokens: i64,
    /// Number of tokens scheduled this step
    pub num_scheduled_tokens: i64,
    /// Whether this is a prefill request
    pub is_prefill: bool,
    /// Block indices assigned to this request
    pub block_indices: Vec<i64>,
}

/// V2 Batch State Manager
///
/// Maintains persistent batch state across decode iterations,
/// eliminating Python overhead from repeated state reconstruction.
#[pyclass]
pub struct BatchStateManager {
    /// Maximum number of requests in a batch
    max_num_reqs: usize,
    /// Maximum sequence length
    max_model_len: usize,
    /// Block size for paged attention
    block_size: i64,
    /// Maximum blocks per sequence
    max_num_blocks_per_req: usize,

    /// Current request states (indexed by batch position)
    requests: Vec<Option<RequestState>>,
    /// Map from request ID to batch index
    req_id_to_idx: HashMap<String, usize>,
    /// Number of active requests
    num_active_reqs: usize,

    /// Pre-allocated arrays for decode outputs
    /// These are updated in-place to avoid allocation
    positions_buffer: Vec<i64>,
    seq_lens_buffer: Vec<i64>,
    query_start_loc_buffer: Vec<i64>,
    slot_mapping_buffer: Vec<i64>,

    /// Block table (flattened: [max_num_reqs * max_num_blocks_per_req])
    block_table: Vec<i64>,

    /// Statistics for debugging
    decode_count: u64,
    cache_hits: u64,
    cache_misses: u64,
}

#[pymethods]
impl BatchStateManager {
    /// Create a new batch state manager
    #[new]
    fn new(
        max_num_reqs: usize,
        max_model_len: usize,
        block_size: i64,
        max_num_blocks_per_req: usize,
    ) -> Self {
        let mut requests = Vec::with_capacity(max_num_reqs);
        for _ in 0..max_num_reqs {
            requests.push(None);
        }

        BatchStateManager {
            max_num_reqs,
            max_model_len,
            block_size,
            max_num_blocks_per_req,
            requests,
            req_id_to_idx: HashMap::with_capacity(max_num_reqs),
            num_active_reqs: 0,
            positions_buffer: vec![0; max_num_reqs],
            seq_lens_buffer: vec![0; max_num_reqs],
            query_start_loc_buffer: vec![0; max_num_reqs + 1],
            slot_mapping_buffer: vec![0; max_num_reqs * 16], // Allow some spec decode tokens
            block_table: vec![0; max_num_reqs * max_num_blocks_per_req],
            decode_count: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Check if the current batch matches the given request IDs
    /// Returns true if all request IDs match in order
    fn batch_matches(&self, req_ids: Vec<String>) -> bool {
        if req_ids.len() != self.num_active_reqs {
            return false;
        }

        for (i, req_id) in req_ids.iter().enumerate() {
            match &self.requests[i] {
                Some(req) if &req.req_id == req_id => continue,
                _ => return false,
            }
        }
        true
    }

    /// Add a new request to the batch
    fn add_request(
        &mut self,
        req_id: String,
        num_computed_tokens: i64,
        num_scheduled_tokens: i64,
        is_prefill: bool,
        block_indices: Vec<i64>,
    ) -> usize {
        // Find first empty slot
        let batch_idx = self
            .requests
            .iter()
            .position(|r| r.is_none())
            .unwrap_or(self.num_active_reqs);

        if batch_idx >= self.max_num_reqs {
            panic!("Batch full, cannot add request");
        }

        // Store block indices in block table
        for (i, &block_idx) in block_indices.iter().enumerate() {
            if i < self.max_num_blocks_per_req {
                self.block_table[batch_idx * self.max_num_blocks_per_req + i] = block_idx;
            }
        }

        let state = RequestState {
            req_id: req_id.clone(),
            batch_idx,
            num_computed_tokens,
            num_scheduled_tokens,
            is_prefill,
            block_indices,
        };

        self.requests[batch_idx] = Some(state);
        self.req_id_to_idx.insert(req_id, batch_idx);
        self.num_active_reqs += 1;

        batch_idx
    }

    /// Remove a request from the batch
    fn remove_request(&mut self, req_id: &str) -> bool {
        if let Some(&batch_idx) = self.req_id_to_idx.get(req_id) {
            self.requests[batch_idx] = None;
            self.req_id_to_idx.remove(req_id);
            self.num_active_reqs -= 1;
            true
        } else {
            false
        }
    }

    /// Clear all requests (reset batch)
    fn clear(&mut self) {
        for req in &mut self.requests {
            *req = None;
        }
        self.req_id_to_idx.clear();
        self.num_active_reqs = 0;
    }

    /// Update computed tokens for a decode step
    /// This is the fast path - just increments counters
    fn increment_decode(&mut self) {
        for req in &mut self.requests {
            if let Some(ref mut state) = req {
                state.num_computed_tokens += 1;
            }
        }
        self.decode_count += 1;
    }

    /// Update block indices for a request (when new blocks are allocated)
    fn update_blocks(&mut self, req_id: &str, block_indices: Vec<i64>) -> bool {
        if let Some(&batch_idx) = self.req_id_to_idx.get(req_id) {
            if let Some(ref mut state) = self.requests[batch_idx] {
                // Update block table
                for (i, &block_idx) in block_indices.iter().enumerate() {
                    if i < self.max_num_blocks_per_req {
                        self.block_table[batch_idx * self.max_num_blocks_per_req + i] = block_idx;
                    }
                }
                state.block_indices = block_indices;
                return true;
            }
        }
        false
    }

    /// Get the number of active requests
    fn num_requests(&self) -> usize {
        self.num_active_reqs
    }

    /// Prepare decode inputs for all active requests
    /// Returns (positions, seq_lens, query_start_loc, slot_mapping, max_seq_len)
    fn prepare_decode_inputs<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<i64>>,
        i64,
    )> {
        let num_reqs = self.num_active_reqs;
        if num_reqs == 0 {
            // Return empty arrays
            let empty: [i64; 0] = [];
            let one: [i64; 1] = [0];
            return Ok((
                empty.to_pyarray_bound(py),
                empty.to_pyarray_bound(py),
                one.to_pyarray_bound(py),
                empty.to_pyarray_bound(py),
                0,
            ));
        }

        let mut max_seq_len: i64 = 0;
        let mut query_start: i64 = 0;
        let mut slot_idx = 0;

        // Process each active request
        for i in 0..self.max_num_reqs {
            if let Some(ref state) = self.requests[i] {
                let seq_len = state.num_computed_tokens + 1; // After this decode
                self.seq_lens_buffer[i] = seq_len;
                self.positions_buffer[i] = state.num_computed_tokens; // Current position

                if seq_len > max_seq_len {
                    max_seq_len = seq_len;
                }

                // Query start location
                self.query_start_loc_buffer[i] = query_start;
                query_start += 1; // 1 token per request for decode

                // Compute slot mapping
                let pos = state.num_computed_tokens;
                let logical_block = pos / self.block_size;
                let block_offset = pos % self.block_size;
                let block_table_idx = i * self.max_num_blocks_per_req + logical_block as usize;
                let physical_block = self.block_table[block_table_idx];
                let slot = physical_block * self.block_size + block_offset;
                self.slot_mapping_buffer[slot_idx] = slot;
                slot_idx += 1;
            }
        }
        self.query_start_loc_buffer[num_reqs] = query_start;

        // Create output arrays (copy from buffers)
        let positions = &self.positions_buffer[..num_reqs];
        let seq_lens = &self.seq_lens_buffer[..num_reqs];
        let query_start_loc = &self.query_start_loc_buffer[..=num_reqs];
        let slot_mapping = &self.slot_mapping_buffer[..slot_idx];

        Ok((
            positions.to_pyarray_bound(py),
            seq_lens.to_pyarray_bound(py),
            query_start_loc.to_pyarray_bound(py),
            slot_mapping.to_pyarray_bound(py),
            max_seq_len,
        ))
    }

    /// Get the block table as a numpy array
    fn get_block_table<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        let rows = self.num_active_reqs;
        let cols = self.max_num_blocks_per_req;

        let mut data: Vec<i64> = Vec::with_capacity(rows * cols);
        for req_idx in 0..rows {
            let start = req_idx * cols;
            let end = start + cols;
            data.extend_from_slice(&self.block_table[start..end]);
        }

        // Create 2D array using ndarray
        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .expect("Failed to create array");
        arr.to_pyarray_bound(py)
    }

    /// Get statistics as a dictionary
    fn get_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = PyDict::new_bound(py);
        stats.set_item("decode_count", self.decode_count)?;
        stats.set_item("cache_hits", self.cache_hits)?;
        stats.set_item("cache_misses", self.cache_misses)?;
        stats.set_item("num_active_reqs", self.num_active_reqs)?;
        Ok(stats)
    }

    /// Sync state from scheduler output
    /// This handles the transition from scheduler to model runner
    fn sync_from_scheduler(
        &mut self,
        req_ids: Vec<String>,
        num_computed_tokens: PyReadonlyArray1<i64>,
        num_scheduled_tokens: PyReadonlyArray1<i64>,
        block_tables: PyReadonlyArray2<i64>,
    ) -> bool {
        let computed = num_computed_tokens.as_array();
        let scheduled = num_scheduled_tokens.as_array();
        let blocks = block_tables.as_array();

        // Check if batch matches
        if self.batch_matches(req_ids.clone()) {
            // Fast path: batch matches, just update counts
            self.cache_hits += 1;
            for (i, req_id) in req_ids.iter().enumerate() {
                if let Some(&batch_idx) = self.req_id_to_idx.get(req_id) {
                    if let Some(ref mut state) = self.requests[batch_idx] {
                        state.num_computed_tokens = computed[i];
                        state.num_scheduled_tokens = scheduled[i];
                    }
                }
            }
            return true;
        }

        // Slow path: rebuild batch state
        self.cache_misses += 1;
        self.clear();

        let num_blocks = blocks.shape()[1];
        for (i, req_id) in req_ids.iter().enumerate() {
            let mut block_indices: Vec<i64> = Vec::with_capacity(num_blocks);
            for j in 0..num_blocks {
                block_indices.push(blocks[[i, j]]);
            }

            self.add_request(
                req_id.clone(),
                computed[i],
                scheduled[i],
                scheduled[i] > 1, // is_prefill if more than 1 token
                block_indices,
            );
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_state_basic() {
        let mut mgr = BatchStateManager::new(4, 2048, 16, 128);

        // Add a request
        let idx = mgr.add_request(
            "req_1".to_string(),
            10,
            1,
            false,
            vec![0, 1, 2],
        );
        assert_eq!(idx, 0);
        assert_eq!(mgr.num_requests(), 1);

        // Check batch matches
        assert!(mgr.batch_matches(vec!["req_1".to_string()]));
        assert!(!mgr.batch_matches(vec!["req_2".to_string()]));

        // Increment decode
        mgr.increment_decode();
        // The increment just updates internal state
    }
}
