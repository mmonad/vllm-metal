//! V2 Block Table Manager
//!
//! Provides GPU-persistent block table management with incremental updates.
//! This eliminates the need to rebuild block tables on every forward pass.
//!
//! Key optimizations:
//! - In-place block table updates (only send diffs)
//! - Vectorized slot mapping computation
//! - Pre-computed indices for common access patterns

use numpy::{PyArray1, PyArray2, ToPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// Tracks changes to block tables for incremental GPU updates
#[derive(Clone, Debug)]
pub struct BlockTableDiff {
    /// Request index
    pub req_idx: usize,
    /// Block index within the request
    pub block_idx: usize,
    /// Old physical block (-1 if new)
    pub old_block: i64,
    /// New physical block
    pub new_block: i64,
}

/// V2 Block Table Manager
///
/// Manages paged attention block tables with support for:
/// - Incremental updates (diff-based)
/// - Vectorized slot mapping
/// - Multi-group support for GQA
#[pyclass]
pub struct BlockTableManager {
    /// Number of KV cache groups (for GQA)
    num_kv_cache_groups: usize,
    /// Block size (tokens per block)
    block_size: i64,
    /// Maximum number of requests
    max_num_reqs: usize,
    /// Maximum blocks per request
    max_num_blocks_per_req: usize,

    /// Block tables for each group: [group][req_idx * max_blocks + block_idx]
    block_tables: Vec<Vec<i64>>,

    /// Pending diffs for each group (accumulated between GPU syncs)
    pending_diffs: Vec<Vec<BlockTableDiff>>,

    /// Cached slot mappings (invalidated on block table change)
    cached_slot_mapping: Option<Vec<i64>>,
    /// Sequence lengths when slot mapping was cached
    cached_seq_lens: Option<Vec<i64>>,

    /// Statistics
    total_updates: u64,
    diff_updates: u64,
    full_updates: u64,
}

#[pymethods]
impl BlockTableManager {
    /// Create a new block table manager
    #[new]
    fn new(
        num_kv_cache_groups: usize,
        block_size: i64,
        max_num_reqs: usize,
        max_num_blocks_per_req: usize,
    ) -> Self {
        let table_size = max_num_reqs * max_num_blocks_per_req;
        let mut block_tables = Vec::with_capacity(num_kv_cache_groups);
        let mut pending_diffs = Vec::with_capacity(num_kv_cache_groups);

        for _ in 0..num_kv_cache_groups {
            block_tables.push(vec![0; table_size]);
            pending_diffs.push(Vec::new());
        }

        BlockTableManager {
            num_kv_cache_groups,
            block_size,
            max_num_reqs,
            max_num_blocks_per_req,
            block_tables,
            pending_diffs,
            cached_slot_mapping: None,
            cached_seq_lens: None,
            total_updates: 0,
            diff_updates: 0,
            full_updates: 0,
        }
    }

    /// Get a block from the table
    fn get_block(&self, group: usize, req_idx: usize, block_idx: usize) -> i64 {
        let idx = req_idx * self.max_num_blocks_per_req + block_idx;
        self.block_tables[group][idx]
    }

    /// Set a block in the table (records diff for incremental sync)
    fn set_block(&mut self, group: usize, req_idx: usize, block_idx: usize, physical_block: i64) {
        let idx = req_idx * self.max_num_blocks_per_req + block_idx;
        let old_block = self.block_tables[group][idx];

        if old_block != physical_block {
            // Record the diff
            self.pending_diffs[group].push(BlockTableDiff {
                req_idx,
                block_idx,
                old_block,
                new_block: physical_block,
            });

            // Update local table
            self.block_tables[group][idx] = physical_block;

            // Invalidate cached slot mapping
            self.cached_slot_mapping = None;
            self.cached_seq_lens = None;
        }

        self.total_updates += 1;
    }

    /// Update block table from a full numpy array
    /// Returns true if there were changes
    fn update_from_array(&mut self, group: usize, block_table: PyReadonlyArray2<i64>) -> bool {
        let arr = block_table.as_array();
        let shape = arr.shape();
        let num_reqs = shape[0];
        let num_blocks = shape[1].min(self.max_num_blocks_per_req);

        let mut changed = false;

        for req_idx in 0..num_reqs {
            for block_idx in 0..num_blocks {
                let new_block = arr[[req_idx, block_idx]];
                let idx = req_idx * self.max_num_blocks_per_req + block_idx;
                let old_block = self.block_tables[group][idx];

                if old_block != new_block {
                    self.pending_diffs[group].push(BlockTableDiff {
                        req_idx,
                        block_idx,
                        old_block,
                        new_block,
                    });
                    self.block_tables[group][idx] = new_block;
                    changed = true;
                }
            }
        }

        if changed {
            self.cached_slot_mapping = None;
            self.cached_seq_lens = None;
            self.full_updates += 1;
        }

        changed
    }

    /// Get pending diffs for a group and clear them
    fn get_and_clear_diffs<'py>(&mut self, py: Python<'py>, group: usize) -> PyResult<(
        Bound<'py, PyArray1<i64>>,  // req_indices
        Bound<'py, PyArray1<i64>>,  // block_indices
        Bound<'py, PyArray1<i64>>,  // new_blocks
    )> {
        let diffs = std::mem::take(&mut self.pending_diffs[group]);
        let n = diffs.len();

        let mut req_indices: Vec<i64> = Vec::with_capacity(n);
        let mut block_indices: Vec<i64> = Vec::with_capacity(n);
        let mut new_blocks: Vec<i64> = Vec::with_capacity(n);

        for diff in diffs {
            req_indices.push(diff.req_idx as i64);
            block_indices.push(diff.block_idx as i64);
            new_blocks.push(diff.new_block);
        }

        self.diff_updates += n as u64;

        Ok((
            req_indices.as_slice().to_pyarray_bound(py),
            block_indices.as_slice().to_pyarray_bound(py),
            new_blocks.as_slice().to_pyarray_bound(py),
        ))
    }

    /// Check if there are pending diffs
    fn has_pending_diffs(&self, group: usize) -> bool {
        !self.pending_diffs[group].is_empty()
    }

    /// Compute slot mapping for decode (1 token per sequence)
    /// Returns slot indices for the current token position
    fn compute_decode_slot_mapping<'py>(
        &mut self,
        py: Python<'py>,
        group: usize,
        num_computed_tokens: PyReadonlyArray1<i64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let computed = num_computed_tokens.as_array();
        let num_reqs = computed.len();

        let mut slots: Vec<i64> = Vec::with_capacity(num_reqs);

        for req_idx in 0..num_reqs {
            let pos = computed[req_idx]; // Current position (0-indexed)
            let logical_block = pos / self.block_size;
            let block_offset = pos % self.block_size;

            let block_table_idx = req_idx * self.max_num_blocks_per_req + logical_block as usize;
            let physical_block = self.block_tables[group][block_table_idx];

            let slot = physical_block * self.block_size + block_offset;
            slots.push(slot);
        }

        slots.as_slice().to_pyarray_bound(py)
    }

    /// Compute slot mapping for prefill (multiple tokens per sequence)
    /// Returns slot indices for all tokens being prefilled
    fn compute_prefill_slot_mapping<'py>(
        &mut self,
        py: Python<'py>,
        group: usize,
        num_computed_tokens: PyReadonlyArray1<i64>,
        num_scheduled_tokens: PyReadonlyArray1<i64>,
    ) -> Bound<'py, PyArray1<i64>> {
        let computed = num_computed_tokens.as_array();
        let scheduled = num_scheduled_tokens.as_array();
        let num_reqs = computed.len();

        // Calculate total slots needed
        let total_tokens: i64 = scheduled.iter().sum();
        let mut slots: Vec<i64> = Vec::with_capacity(total_tokens as usize);

        for req_idx in 0..num_reqs {
            let start_pos = computed[req_idx];
            let num_tokens = scheduled[req_idx];

            for token_offset in 0..num_tokens {
                let pos = start_pos + token_offset;
                let logical_block = pos / self.block_size;
                let block_offset = pos % self.block_size;

                let block_table_idx = req_idx * self.max_num_blocks_per_req + logical_block as usize;
                let physical_block = self.block_tables[group][block_table_idx];

                let slot = physical_block * self.block_size + block_offset;
                slots.push(slot);
            }
        }

        slots.as_slice().to_pyarray_bound(py)
    }

    /// Get the full block table for a group as numpy array
    fn get_block_table<'py>(&self, py: Python<'py>, group: usize, num_reqs: usize) -> Bound<'py, PyArray2<i64>> {
        let rows = num_reqs.min(self.max_num_reqs);
        let cols = self.max_num_blocks_per_req;

        let mut data: Vec<i64> = Vec::with_capacity(rows * cols);
        for req_idx in 0..rows {
            let start = req_idx * cols;
            let end = start + cols;
            data.extend_from_slice(&self.block_tables[group][start..end]);
        }

        let arr = ndarray::Array2::from_shape_vec((rows, cols), data)
            .expect("Failed to create array");
        arr.to_pyarray_bound(py)
    }

    /// Clear block table for a specific request
    fn clear_request(&mut self, req_idx: usize) {
        for group in 0..self.num_kv_cache_groups {
            let start = req_idx * self.max_num_blocks_per_req;
            let end = start + self.max_num_blocks_per_req;
            for i in start..end {
                self.block_tables[group][i] = 0;
            }
        }
        self.cached_slot_mapping = None;
        self.cached_seq_lens = None;
    }

    /// Clear all block tables
    fn clear_all(&mut self) {
        for group in 0..self.num_kv_cache_groups {
            self.block_tables[group].fill(0);
            self.pending_diffs[group].clear();
        }
        self.cached_slot_mapping = None;
        self.cached_seq_lens = None;
    }

    /// Get statistics
    fn get_stats(&self) -> (u64, u64, u64) {
        (self.total_updates, self.diff_updates, self.full_updates)
    }
}

/// Compute slot mapping for a batch of sequences (standalone function)
/// This is a pure function version for use without BlockTableManager
#[pyfunction]
pub fn compute_slot_mapping_vectorized<'py>(
    py: Python<'py>,
    block_table: PyReadonlyArray2<i64>,
    positions: PyReadonlyArray1<i64>,
    block_size: i64,
) -> Bound<'py, PyArray1<i64>> {
    let bt = block_table.as_array();
    let pos = positions.as_array();
    let num_tokens = pos.len();
    let max_blocks = bt.shape()[1];

    let mut slots: Vec<i64> = Vec::with_capacity(num_tokens);

    for i in 0..num_tokens {
        let position = pos[i];
        let logical_block = position / block_size;
        let block_offset = position % block_size;

        // Get physical block from block table
        // For batched case, we need req_idx - assume positions are ordered by request
        let req_idx = i.min(bt.shape()[0] - 1);
        let physical_block = bt[[req_idx, (logical_block as usize).min(max_blocks - 1)]];

        let slot = physical_block * block_size + block_offset;
        slots.push(slot);
    }

    slots.as_slice().to_pyarray_bound(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_table_basic() {
        let mut mgr = BlockTableManager::new(1, 16, 4, 128);

        // Set some blocks
        mgr.set_block(0, 0, 0, 100);
        mgr.set_block(0, 0, 1, 101);
        mgr.set_block(0, 1, 0, 200);

        assert_eq!(mgr.get_block(0, 0, 0), 100);
        assert_eq!(mgr.get_block(0, 0, 1), 101);
        assert_eq!(mgr.get_block(0, 1, 0), 200);

        // Check pending diffs
        assert!(mgr.has_pending_diffs(0));
    }
}
