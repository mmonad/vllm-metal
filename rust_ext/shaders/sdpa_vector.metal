//
// SDPA Vector Kernel for vLLM Metal
//
// Scaled dot-product attention optimized for decode.
// Uses online softmax with max tracking for numerical stability.
//
// Key optimizations:
// - Simdgroup-based threading (32 threads)
// - Process keys/values in blocks of BN
// - Online softmax: track max and sum incrementally
// - Coalesced memory access patterns
//

#include <metal_stdlib>
using namespace metal;

// Block sizes (tuned for Apple Silicon)
constant int BN = 32;    // Keys/values per block
constant int BD = 32;    // Features processed per simdgroup

// Attention parameters passed from host
struct AttentionParams {
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int seq_len;
    int num_queries;
    float scale;
    int gqa_ratio;  // num_heads / num_kv_heads
};

// ============================================================================
// SDPA Vector Kernel (Decode)
//
// For decode, each query has length 1, so this is optimized for that case.
// Each simdgroup handles one query head.
// ============================================================================

template<typename T, int HEAD_DIM>
[[kernel]] void sdpa_vector(
    device const T* queries     [[buffer(0)]],  // [num_queries, num_heads, head_dim]
    device const T* keys        [[buffer(1)]],  // [num_queries, seq_len, num_kv_heads, head_dim]
    device const T* values      [[buffer(2)]],  // [num_queries, seq_len, num_kv_heads, head_dim]
    device T* output            [[buffer(3)]],  // [num_queries, num_heads, head_dim]
    constant AttentionParams& params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // tid.x = query index (batch element)
    // tid.y = head index
    const int query_idx = tid.x;
    const int head_idx = tid.y;

    if (query_idx >= params.num_queries || head_idx >= params.num_heads) {
        return;
    }

    // For GQA: map head index to KV head
    const int kv_head_idx = head_idx / params.gqa_ratio;

    // Pointers to this query/head
    device const T* q = queries + (query_idx * params.num_heads + head_idx) * HEAD_DIM;
    device const T* k_base = keys + query_idx * params.seq_len * params.num_kv_heads * HEAD_DIM
                            + kv_head_idx * HEAD_DIM;
    device const T* v_base = values + query_idx * params.seq_len * params.num_kv_heads * HEAD_DIM
                            + kv_head_idx * HEAD_DIM;
    device T* out = output + (query_idx * params.num_heads + head_idx) * HEAD_DIM;

    // Load query into registers (each thread handles HEAD_DIM/32 elements)
    T q_reg[HEAD_DIM / 32];
    for (int d = simd_lid; d < HEAD_DIM; d += 32) {
        q_reg[d / 32] = q[d] * T(params.scale);
    }

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Accumulator for output
    float acc[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        acc[i] = 0.0f;
    }

    // Process keys/values in blocks
    const int seq_len = params.seq_len;
    const int kv_stride = params.num_kv_heads * HEAD_DIM;

    for (int k_start = 0; k_start < seq_len; k_start += BN) {
        int k_end = min(k_start + BN, seq_len);

        // Process each key in this block
        for (int k = k_start; k < k_end; k++) {
            device const T* key = k_base + k * kv_stride;

            // Compute Q·K dot product using simd reduction
            float score = 0.0f;
            for (int d = simd_lid; d < HEAD_DIM; d += 32) {
                score += float(q_reg[d / 32]) * float(key[d]);
            }
            score = simd_sum(score);

            // Online softmax update (only lane 0 has the full score)
            if (simd_lid == 0) {
                float old_max = max_score;
                max_score = max(max_score, score);

                // Rescale previous sum
                float scale_factor = exp(old_max - max_score);
                sum_exp = sum_exp * scale_factor + exp(score - max_score);
            }

            // Broadcast max_score and compute weight
            float m = simd_broadcast(max_score, 0);
            float weight = exp(score - m);

            // Accumulate weighted value
            device const T* val = v_base + k * kv_stride;
            for (int d = simd_lid; d < HEAD_DIM; d += 32) {
                acc[d / 32] += weight * float(val[d]);
            }
        }
    }

    // Normalize by sum and write output
    float inv_sum = simd_broadcast(1.0f / sum_exp, 0);
    for (int d = simd_lid; d < HEAD_DIM; d += 32) {
        out[d] = T(acc[d / 32] * inv_sum);
    }
}

// ============================================================================
// Paged Attention Kernel (Decode with KV Cache)
//
// For vLLM's paged attention with block tables.
// ============================================================================

struct PagedAttentionParams {
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int num_blocks;
    float scale;
    int gqa_ratio;
};

template<typename T, int HEAD_DIM, int BLOCK_SIZE>
[[kernel]] void paged_attention_decode(
    device const T* queries          [[buffer(0)]],  // [batch, num_heads, head_dim]
    device const T* key_cache        [[buffer(1)]],  // [num_blocks, block_size, num_kv_heads, head_dim]
    device const T* value_cache      [[buffer(2)]],  // [num_blocks, block_size, num_kv_heads, head_dim]
    device const int* block_tables   [[buffer(3)]],  // [batch, max_blocks]
    device const int* seq_lens       [[buffer(4)]],  // [batch]
    device T* output                 [[buffer(5)]],  // [batch, num_heads, head_dim]
    constant PagedAttentionParams& params [[buffer(6)]],
    constant int& max_blocks_per_seq [[buffer(7)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int batch_idx = tid.x;
    const int head_idx = tid.y;

    const int seq_len = seq_lens[batch_idx];
    if (seq_len == 0) return;

    const int kv_head_idx = head_idx / params.gqa_ratio;

    // Query pointer
    device const T* q = queries + (batch_idx * params.num_heads + head_idx) * HEAD_DIM;
    device T* out = output + (batch_idx * params.num_heads + head_idx) * HEAD_DIM;

    // Block table for this sequence
    device const int* block_table = block_tables + batch_idx * max_blocks_per_seq;

    // Load and scale query
    T q_reg[HEAD_DIM / 32];
    for (int d = simd_lid; d < HEAD_DIM; d += 32) {
        q_reg[d / 32] = q[d] * T(params.scale);
    }

    // Online softmax state
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float acc[HEAD_DIM / 32];
    for (int i = 0; i < HEAD_DIM / 32; i++) {
        acc[i] = 0.0f;
    }

    // Process each block
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int kv_head_stride = HEAD_DIM;
    const int block_stride = BLOCK_SIZE * params.num_kv_heads * HEAD_DIM;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int physical_block = block_table[block_idx];
        const int tokens_in_block = min(BLOCK_SIZE, seq_len - block_idx * BLOCK_SIZE);

        device const T* k_block = key_cache + physical_block * block_stride + kv_head_idx * kv_head_stride;
        device const T* v_block = value_cache + physical_block * block_stride + kv_head_idx * kv_head_stride;

        // Process each token in block
        for (int t = 0; t < tokens_in_block; t++) {
            device const T* key = k_block + t * params.num_kv_heads * HEAD_DIM;
            device const T* val = v_block + t * params.num_kv_heads * HEAD_DIM;

            // Q·K dot product
            float score = 0.0f;
            for (int d = simd_lid; d < HEAD_DIM; d += 32) {
                score += float(q_reg[d / 32]) * float(key[d]);
            }
            score = simd_sum(score);

            // Online softmax
            if (simd_lid == 0) {
                float old_max = max_score;
                max_score = max(max_score, score);
                float scale_factor = exp(old_max - max_score);
                sum_exp = sum_exp * scale_factor + exp(score - max_score);
            }

            float m = simd_broadcast(max_score, 0);
            float weight = exp(score - m);

            // Accumulate
            for (int d = simd_lid; d < HEAD_DIM; d += 32) {
                acc[d / 32] += weight * float(val[d]);
            }
        }
    }

    // Write normalized output
    float inv_sum = simd_broadcast(1.0f / sum_exp, 0);
    for (int d = simd_lid; d < HEAD_DIM; d += 32) {
        out[d] = T(acc[d / 32] * inv_sum);
    }
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

// Float16 kernels for common head dimensions
template [[host_name("sdpa_vector_f16_64")]]
[[kernel]] void sdpa_vector<half, 64>(
    device const half*, device const half*, device const half*,
    device half*, constant AttentionParams&,
    uint3, uint, uint);

template [[host_name("sdpa_vector_f16_96")]]
[[kernel]] void sdpa_vector<half, 96>(
    device const half*, device const half*, device const half*,
    device half*, constant AttentionParams&,
    uint3, uint, uint);

template [[host_name("sdpa_vector_f16_128")]]
[[kernel]] void sdpa_vector<half, 128>(
    device const half*, device const half*, device const half*,
    device half*, constant AttentionParams&,
    uint3, uint, uint);

// Float32 kernels (for debugging/accuracy)
template [[host_name("sdpa_vector_f32_64")]]
[[kernel]] void sdpa_vector<float, 64>(
    device const float*, device const float*, device const float*,
    device float*, constant AttentionParams&,
    uint3, uint, uint);

template [[host_name("sdpa_vector_f32_128")]]
[[kernel]] void sdpa_vector<float, 128>(
    device const float*, device const float*, device const float*,
    device float*, constant AttentionParams&,
    uint3, uint, uint);

// Paged attention kernels
template [[host_name("paged_attention_f16_64_16")]]
[[kernel]] void paged_attention_decode<half, 64, 16>(
    device const half*, device const half*, device const half*,
    device const int*, device const int*, device half*,
    constant PagedAttentionParams&, constant int&,
    uint3, uint, uint);

template [[host_name("paged_attention_f16_128_16")]]
[[kernel]] void paged_attention_decode<half, 128, 16>(
    device const half*, device const half*, device const half*,
    device const int*, device const int*, device half*,
    constant PagedAttentionParams&, constant int&,
    uint3, uint, uint);
