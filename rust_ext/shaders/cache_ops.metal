//
// KV Cache Operations for vLLM Metal
//
// Operations for paged attention KV cache management:
// - reshape_and_cache: Store new K/V into cache blocks
// - copy_blocks: Copy cache blocks between sequences
// - swap_blocks: Swap blocks between GPU and CPU (for unified memory)
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Reshape and Cache: Store new K/V tokens into paged cache
//
// Stores computed key/value vectors into their assigned cache slots.
// Used during both prefill and decode.
// ============================================================================

template<typename T>
[[kernel]] void reshape_and_cache(
    device const T* key             [[buffer(0)]],   // [num_tokens, num_kv_heads, head_dim]
    device const T* value           [[buffer(1)]],   // [num_tokens, num_kv_heads, head_dim]
    device T* key_cache             [[buffer(2)]],   // [num_blocks, block_size, num_kv_heads, head_dim]
    device T* value_cache           [[buffer(3)]],   // [num_blocks, block_size, num_kv_heads, head_dim]
    device const int* slot_mapping  [[buffer(4)]],   // [num_tokens]
    constant int& num_tokens        [[buffer(5)]],
    constant int& num_kv_heads      [[buffer(6)]],
    constant int& head_dim          [[buffer(7)]],
    constant int& block_size        [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int token_idx = tid.x;
    const int head_idx = tid.y;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    // Get the slot for this token
    const int slot = slot_mapping[token_idx];

    // Compute block and offset within block
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    // Source pointers
    device const T* key_src = key + (token_idx * num_kv_heads + head_idx) * head_dim;
    device const T* value_src = value + (token_idx * num_kv_heads + head_idx) * head_dim;

    // Destination pointers in cache
    // Cache layout: [num_blocks, block_size, num_kv_heads, head_dim]
    const int cache_idx = ((block_idx * block_size + block_offset) * num_kv_heads + head_idx) * head_dim;
    device T* key_dst = key_cache + cache_idx;
    device T* value_dst = value_cache + cache_idx;

    // Copy head_dim elements
    for (int d = 0; d < head_dim; d++) {
        key_dst[d] = key_src[d];
        value_dst[d] = value_src[d];
    }
}

// ============================================================================
// Copy Blocks: Copy cache blocks between sequences (for beam search)
// ============================================================================

template<typename T>
[[kernel]] void copy_blocks(
    device T* key_cache             [[buffer(0)]],   // [num_blocks, block_size, num_kv_heads, head_dim]
    device T* value_cache           [[buffer(1)]],   // [num_blocks, block_size, num_kv_heads, head_dim]
    device const int2* block_mapping [[buffer(2)]],  // [(src_block, dst_block), ...]
    constant int& num_pairs         [[buffer(3)]],
    constant int& block_size        [[buffer(4)]],
    constant int& num_kv_heads      [[buffer(5)]],
    constant int& head_dim          [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const int pair_idx = tid;
    if (pair_idx >= num_pairs) return;

    const int src_block = block_mapping[pair_idx].x;
    const int dst_block = block_mapping[pair_idx].y;

    const int block_elements = block_size * num_kv_heads * head_dim;

    // Copy key cache block
    device const T* key_src = key_cache + src_block * block_elements;
    device T* key_dst = key_cache + dst_block * block_elements;

    for (int i = 0; i < block_elements; i++) {
        key_dst[i] = key_src[i];
    }

    // Copy value cache block
    device const T* value_src = value_cache + src_block * block_elements;
    device T* value_dst = value_cache + dst_block * block_elements;

    for (int i = 0; i < block_elements; i++) {
        value_dst[i] = value_src[i];
    }
}

// ============================================================================
// Gather Cached: Gather K/V from cache for specific positions
//
// Used when we need to extract specific cached values.
// ============================================================================

template<typename T>
[[kernel]] void gather_cached(
    device const T* cache           [[buffer(0)]],   // [num_blocks, block_size, num_kv_heads, head_dim]
    device T* output                [[buffer(1)]],   // [num_tokens, num_kv_heads, head_dim]
    device const int* slot_mapping  [[buffer(2)]],   // [num_tokens]
    constant int& num_tokens        [[buffer(3)]],
    constant int& num_kv_heads      [[buffer(4)]],
    constant int& head_dim          [[buffer(5)]],
    constant int& block_size        [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int token_idx = tid.x;
    const int head_idx = tid.y;

    if (token_idx >= num_tokens || head_idx >= num_kv_heads) return;

    const int slot = slot_mapping[token_idx];
    const int block_idx = slot / block_size;
    const int block_offset = slot % block_size;

    device const T* src = cache + ((block_idx * block_size + block_offset) * num_kv_heads + head_idx) * head_dim;
    device T* dst = output + (token_idx * num_kv_heads + head_idx) * head_dim;

    for (int d = 0; d < head_dim; d++) {
        dst[d] = src[d];
    }
}

// ============================================================================
// Initialize Cache Block: Zero-initialize cache blocks
// ============================================================================

template<typename T>
[[kernel]] void init_cache_block(
    device T* key_cache             [[buffer(0)]],
    device T* value_cache           [[buffer(1)]],
    constant int& block_idx         [[buffer(2)]],
    constant int& block_size        [[buffer(3)]],
    constant int& num_kv_heads      [[buffer(4)]],
    constant int& head_dim          [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const int block_elements = block_size * num_kv_heads * head_dim;
    const int offset = block_idx * block_elements;

    if (tid < (uint)block_elements) {
        key_cache[offset + tid] = T(0);
        value_cache[offset + tid] = T(0);
    }
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

template [[host_name("reshape_and_cache_f16")]]
[[kernel]] void reshape_and_cache<half>(
    device const half*, device const half*, device half*, device half*,
    device const int*, constant int&, constant int&, constant int&, constant int&,
    uint2);

template [[host_name("reshape_and_cache_f32")]]
[[kernel]] void reshape_and_cache<float>(
    device const float*, device const float*, device float*, device float*,
    device const int*, constant int&, constant int&, constant int&, constant int&,
    uint2);

template [[host_name("copy_blocks_f16")]]
[[kernel]] void copy_blocks<half>(
    device half*, device half*, device const int2*, constant int&,
    constant int&, constant int&, constant int&,
    uint);

template [[host_name("gather_cached_f16")]]
[[kernel]] void gather_cached<half>(
    device const half*, device half*, device const int*,
    constant int&, constant int&, constant int&, constant int&,
    uint2);

template [[host_name("init_cache_block_f16")]]
[[kernel]] void init_cache_block<half>(
    device half*, device half*, constant int&, constant int&, constant int&, constant int&,
    uint);
