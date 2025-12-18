//
// GEMV Kernel for vLLM Metal
//
// Optimized matrix-vector multiply for decode (single token).
// Uses simdgroup reductions for efficient dot products.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// GEMV Kernel: y = Ax (Matrix-Vector Multiply)
//
// For decode: projects hidden states through weight matrices.
// A: [M, K], x: [K], y: [M]
// ============================================================================

template<typename T, int BLOCK_M, int BLOCK_K>
[[kernel]] void gemv(
    device const T* A       [[buffer(0)]],  // [M, K]
    device const T* x       [[buffer(1)]],  // [K]
    device T* y             [[buffer(2)]],  // [M]
    constant int& M         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // Each threadgroup handles BLOCK_M rows
    const int row_start = bid * BLOCK_M;
    const int row_end = min(row_start + BLOCK_M, M);

    // Shared memory for x vector tile
    threadgroup T x_shared[BLOCK_K];

    // Each thread accumulates one row
    const int row = row_start + tid;
    float acc = 0.0f;

    // Process K in blocks
    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        // Cooperatively load x tile
        for (int i = tid; i < BLOCK_K && (k_start + i) < K; i += threads_per_group) {
            x_shared[i] = x[k_start + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product for this row
        if (row < row_end) {
            device const T* A_row = A + row * K + k_start;
            int k_len = min(BLOCK_K, K - k_start);

            for (int k = 0; k < k_len; k++) {
                acc += float(A_row[k]) * float(x_shared[k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < row_end) {
        y[row] = T(acc);
    }
}

// ============================================================================
// Batched GEMV: Y = AX for multiple vectors
//
// A: [M, K], X: [batch, K], Y: [batch, M]
// ============================================================================

template<typename T>
[[kernel]] void batched_gemv(
    device const T* A       [[buffer(0)]],  // [M, K]
    device const T* X       [[buffer(1)]],  // [batch, K]
    device T* Y             [[buffer(2)]],  // [batch, M]
    constant int& M         [[buffer(3)]],
    constant int& K         [[buffer(4)]],
    constant int& batch     [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int b = tid.x;  // Batch index
    const int row = tid.y; // Output row

    if (b >= batch || row >= M) return;

    device const T* x = X + b * K;
    device const T* A_row = A + row * K;

    // Compute dot product with simd reduction
    float acc = 0.0f;
    for (int k = simd_lid; k < K; k += 32) {
        acc += float(A_row[k]) * float(x[k]);
    }
    acc = simd_sum(acc);

    // Write result (only lane 0)
    if (simd_lid == 0) {
        Y[b * M + row] = T(acc);
    }
}

// ============================================================================
// GEMV with bias: y = Ax + b
// ============================================================================

template<typename T>
[[kernel]] void gemv_bias(
    device const T* A       [[buffer(0)]],  // [M, K]
    device const T* x       [[buffer(1)]],  // [K]
    device const T* bias    [[buffer(2)]],  // [M]
    device T* y             [[buffer(3)]],  // [M]
    constant int& M         [[buffer(4)]],
    constant int& K         [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    const int row = tid / 32;  // Each simdgroup handles one row

    if (row >= M) return;

    device const T* A_row = A + row * K;

    // Compute dot product
    float acc = 0.0f;
    for (int k = simd_lid; k < K; k += 32) {
        acc += float(A_row[k]) * float(x[k]);
    }
    acc = simd_sum(acc);

    // Add bias and write
    if (simd_lid == 0) {
        y[row] = T(acc + float(bias[row]));
    }
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

template [[host_name("gemv_f16_64_256")]]
[[kernel]] void gemv<half, 64, 256>(
    device const half*, device const half*, device half*,
    constant int&, constant int&,
    uint, uint, uint);

template [[host_name("gemv_f16_128_256")]]
[[kernel]] void gemv<half, 128, 256>(
    device const half*, device const half*, device half*,
    constant int&, constant int&,
    uint, uint, uint);

template [[host_name("batched_gemv_f16")]]
[[kernel]] void batched_gemv<half>(
    device const half*, device const half*, device half*,
    constant int&, constant int&, constant int&,
    uint3, uint);

template [[host_name("batched_gemv_f32")]]
[[kernel]] void batched_gemv<float>(
    device const float*, device const float*, device float*,
    constant int&, constant int&, constant int&,
    uint3, uint);

template [[host_name("gemv_bias_f16")]]
[[kernel]] void gemv_bias<half>(
    device const half*, device const half*, device const half*, device half*,
    constant int&, constant int&,
    uint, uint);
