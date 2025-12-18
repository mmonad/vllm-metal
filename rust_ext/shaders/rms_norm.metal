//
// RMS Normalization Kernel for vLLM Metal
//
// Root Mean Square Layer Normalization used in LLaMA and other models.
// y = x * rsqrt(mean(x^2) + eps) * weight
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RMS Norm: Standard Implementation
//
// Each threadgroup handles one row (hidden_size elements)
// Uses simdgroup reductions for computing mean of squares
// ============================================================================

template<typename T>
[[kernel]] void rms_norm(
    device const T* input       [[buffer(0)]],  // [batch, hidden_size]
    device const T* weight      [[buffer(1)]],  // [hidden_size]
    device T* output            [[buffer(2)]],  // [batch, hidden_size]
    constant int& batch_size    [[buffer(3)]],
    constant int& hidden_size   [[buffer(4)]],
    constant float& eps         [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (bid >= (uint)batch_size) return;

    device const T* in = input + bid * hidden_size;
    device T* out = output + bid * hidden_size;

    // Compute sum of squares using parallel reduction
    float local_sum = 0.0f;
    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        float val = float(in[i]);
        local_sum += val * val;
    }

    // Simdgroup reduction
    local_sum = simd_sum(local_sum);

    // Threadgroup reduction (if needed)
    threadgroup float shared_sum[32];  // Max 32 simdgroups
    uint simd_idx = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) {
        shared_sum[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction in first simdgroup
    if (simd_idx == 0) {
        uint num_simd = (threads_per_group + 31) / 32;
        float total = 0.0f;
        if (simd_lane < num_simd) {
            total = shared_sum[simd_lane];
        }
        total = simd_sum(total);

        if (simd_lane == 0) {
            shared_sum[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute scale factor
    float sum_sq = shared_sum[0];
    float rms = rsqrt(sum_sq / float(hidden_size) + eps);

    // Apply normalization and weight
    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        out[i] = T(float(in[i]) * rms * float(weight[i]));
    }
}

// ============================================================================
// RMS Norm Inplace: Modify input directly
// ============================================================================

template<typename T>
[[kernel]] void rms_norm_inplace(
    device T* data              [[buffer(0)]],  // [batch, hidden_size]
    device const T* weight      [[buffer(1)]],  // [hidden_size]
    constant int& batch_size    [[buffer(2)]],
    constant int& hidden_size   [[buffer(3)]],
    constant float& eps         [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (bid >= (uint)batch_size) return;

    device T* row = data + bid * hidden_size;

    // Compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        float val = float(row[i]);
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    uint simd_idx = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) {
        shared_sum[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint num_simd = (threads_per_group + 31) / 32;
        float total = 0.0f;
        if (simd_lane < num_simd) {
            total = shared_sum[simd_lane];
        }
        total = simd_sum(total);
        if (simd_lane == 0) {
            shared_sum[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        row[i] = T(float(row[i]) * rms * float(weight[i]));
    }
}

// ============================================================================
// Fused Add + RMS Norm: residual = input + residual; output = rms_norm(residual)
//
// Common pattern in transformer layers
// ============================================================================

template<typename T>
[[kernel]] void fused_add_rms_norm(
    device const T* input       [[buffer(0)]],  // [batch, hidden_size]
    device T* residual          [[buffer(1)]],  // [batch, hidden_size] - modified inplace
    device const T* weight      [[buffer(2)]],  // [hidden_size]
    device T* output            [[buffer(3)]],  // [batch, hidden_size]
    constant int& batch_size    [[buffer(4)]],
    constant int& hidden_size   [[buffer(5)]],
    constant float& eps         [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (bid >= (uint)batch_size) return;

    device const T* in = input + bid * hidden_size;
    device T* res = residual + bid * hidden_size;
    device T* out = output + bid * hidden_size;

    // First pass: add input to residual and compute sum of squares
    float local_sum = 0.0f;
    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        float val = float(in[i]) + float(res[i]);
        res[i] = T(val);  // Update residual
        local_sum += val * val;
    }

    local_sum = simd_sum(local_sum);

    threadgroup float shared_sum[32];
    uint simd_idx = tid / 32;
    uint simd_lane = tid % 32;

    if (simd_lane == 0) {
        shared_sum[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        uint num_simd = (threads_per_group + 31) / 32;
        float total = 0.0f;
        if (simd_lane < num_simd) {
            total = shared_sum[simd_lane];
        }
        total = simd_sum(total);
        if (simd_lane == 0) {
            shared_sum[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms = rsqrt(shared_sum[0] / float(hidden_size) + eps);

    // Second pass: apply normalization
    for (uint i = tid; i < (uint)hidden_size; i += threads_per_group) {
        out[i] = T(float(res[i]) * rms * float(weight[i]));
    }
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

template [[host_name("rms_norm_f16")]]
[[kernel]] void rms_norm<half>(
    device const half*, device const half*, device half*,
    constant int&, constant int&, constant float&,
    uint, uint, uint);

template [[host_name("rms_norm_f32")]]
[[kernel]] void rms_norm<float>(
    device const float*, device const float*, device float*,
    constant int&, constant int&, constant float&,
    uint, uint, uint);

template [[host_name("rms_norm_inplace_f16")]]
[[kernel]] void rms_norm_inplace<half>(
    device half*, device const half*,
    constant int&, constant int&, constant float&,
    uint, uint, uint);

template [[host_name("fused_add_rms_norm_f16")]]
[[kernel]] void fused_add_rms_norm<half>(
    device const half*, device half*, device const half*, device half*,
    constant int&, constant int&, constant float&,
    uint, uint, uint);
