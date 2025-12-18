//
// Rotary Position Embedding (RoPE) Kernel for vLLM Metal
//
// Applies rotary embeddings to query and key vectors.
// Supports both standard RoPE and NeoX-style interleaved RoPE.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// RoPE: Apply rotary position embeddings
//
// Standard RoPE: split head_dim in half, rotate pairs
// q_out[..., 2*i] = q[..., 2*i] * cos - q[..., 2*i+1] * sin
// q_out[..., 2*i+1] = q[..., 2*i] * sin + q[..., 2*i+1] * cos
// ============================================================================

template<typename T>
[[kernel]] void rope_forward(
    device const T* input       [[buffer(0)]],  // [batch, seq_len, num_heads, head_dim]
    device const float* cos     [[buffer(1)]],  // [max_seq_len, head_dim/2]
    device const float* sin     [[buffer(2)]],  // [max_seq_len, head_dim/2]
    device T* output            [[buffer(3)]],  // [batch, seq_len, num_heads, head_dim]
    constant int& batch_size    [[buffer(4)]],
    constant int& seq_len       [[buffer(5)]],
    constant int& num_heads     [[buffer(6)]],
    constant int& head_dim      [[buffer(7)]],
    constant int& offset        [[buffer(8)]],  // Position offset for KV cache
    uint3 tid [[thread_position_in_grid]]
) {
    const int b = tid.x;
    const int s = tid.y;
    const int h = tid.z;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    const int half_dim = head_dim / 2;
    const int pos = offset + s;

    // Get pointers
    device const T* in = input + ((b * seq_len + s) * num_heads + h) * head_dim;
    device T* out = output + ((b * seq_len + s) * num_heads + h) * head_dim;
    device const float* cos_row = cos + pos * half_dim;
    device const float* sin_row = sin + pos * half_dim;

    // Apply rotation to each pair
    for (int d = 0; d < half_dim; d++) {
        float x0 = float(in[2 * d]);
        float x1 = float(in[2 * d + 1]);
        float c = cos_row[d];
        float s = sin_row[d];

        out[2 * d] = T(x0 * c - x1 * s);
        out[2 * d + 1] = T(x0 * s + x1 * c);
    }
}

// ============================================================================
// RoPE Inplace: Apply rotary embeddings in place
// ============================================================================

template<typename T>
[[kernel]] void rope_inplace(
    device T* data              [[buffer(0)]],  // [batch, seq_len, num_heads, head_dim]
    device const float* cos     [[buffer(1)]],  // [max_seq_len, head_dim/2]
    device const float* sin     [[buffer(2)]],  // [max_seq_len, head_dim/2]
    constant int& batch_size    [[buffer(3)]],
    constant int& seq_len       [[buffer(4)]],
    constant int& num_heads     [[buffer(5)]],
    constant int& head_dim      [[buffer(6)]],
    constant int& offset        [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const int b = tid.x;
    const int s = tid.y;
    const int h = tid.z;

    if (b >= batch_size || s >= seq_len || h >= num_heads) return;

    const int half_dim = head_dim / 2;
    const int pos = offset + s;

    device T* data_ptr = data + ((b * seq_len + s) * num_heads + h) * head_dim;
    device const float* cos_row = cos + pos * half_dim;
    device const float* sin_row = sin + pos * half_dim;

    for (int d = 0; d < half_dim; d++) {
        float x0 = float(data_ptr[2 * d]);
        float x1 = float(data_ptr[2 * d + 1]);
        float c = cos_row[d];
        float s = sin_row[d];

        data_ptr[2 * d] = T(x0 * c - x1 * s);
        data_ptr[2 * d + 1] = T(x0 * s + x1 * c);
    }
}

// ============================================================================
// RoPE for Decode: Single position
// ============================================================================

template<typename T>
[[kernel]] void rope_decode(
    device T* q                 [[buffer(0)]],  // [batch, num_heads, head_dim]
    device T* k                 [[buffer(1)]],  // [batch, num_kv_heads, head_dim]
    device const float* cos     [[buffer(2)]],  // [max_seq_len, head_dim/2]
    device const float* sin     [[buffer(3)]],  // [max_seq_len, head_dim/2]
    device const int* positions [[buffer(4)]],  // [batch]
    constant int& batch_size    [[buffer(5)]],
    constant int& num_heads     [[buffer(6)]],
    constant int& num_kv_heads  [[buffer(7)]],
    constant int& head_dim      [[buffer(8)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int b = tid.x;
    const int h = tid.y;

    if (b >= batch_size) return;

    const int half_dim = head_dim / 2;
    const int pos = positions[b];

    device const float* cos_row = cos + pos * half_dim;
    device const float* sin_row = sin + pos * half_dim;

    // Apply RoPE to query head
    if (h < num_heads) {
        device T* q_ptr = q + (b * num_heads + h) * head_dim;
        for (int d = 0; d < half_dim; d++) {
            float x0 = float(q_ptr[2 * d]);
            float x1 = float(q_ptr[2 * d + 1]);
            float c = cos_row[d];
            float s = sin_row[d];
            q_ptr[2 * d] = T(x0 * c - x1 * s);
            q_ptr[2 * d + 1] = T(x0 * s + x1 * c);
        }
    }

    // Apply RoPE to key head (if within KV head range)
    if (h < num_kv_heads) {
        device T* k_ptr = k + (b * num_kv_heads + h) * head_dim;
        for (int d = 0; d < half_dim; d++) {
            float x0 = float(k_ptr[2 * d]);
            float x1 = float(k_ptr[2 * d + 1]);
            float c = cos_row[d];
            float s = sin_row[d];
            k_ptr[2 * d] = T(x0 * c - x1 * s);
            k_ptr[2 * d + 1] = T(x0 * s + x1 * c);
        }
    }
}

// ============================================================================
// Precompute cos/sin tables
// ============================================================================

[[kernel]] void precompute_freqs(
    device float* cos_out       [[buffer(0)]],  // [max_seq_len, head_dim/2]
    device float* sin_out       [[buffer(1)]],  // [max_seq_len, head_dim/2]
    constant int& max_seq_len   [[buffer(2)]],
    constant int& head_dim      [[buffer(3)]],
    constant float& base        [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int pos = tid.x;
    const int d = tid.y;

    if (pos >= max_seq_len || d >= head_dim / 2) return;

    // Compute frequency: 1 / (base^(2d/head_dim))
    float freq = 1.0f / pow(base, float(2 * d) / float(head_dim));
    float angle = float(pos) * freq;

    cos_out[pos * (head_dim / 2) + d] = cos(angle);
    sin_out[pos * (head_dim / 2) + d] = sin(angle);
}

// ============================================================================
// Kernel Instantiations
// ============================================================================

template [[host_name("rope_forward_f16")]]
[[kernel]] void rope_forward<half>(
    device const half*, device const float*, device const float*,
    device half*, constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3);

template [[host_name("rope_forward_f32")]]
[[kernel]] void rope_forward<float>(
    device const float*, device const float*, device const float*,
    device float*, constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3);

template [[host_name("rope_inplace_f16")]]
[[kernel]] void rope_inplace<half>(
    device half*, device const float*, device const float*,
    constant int&, constant int&, constant int&, constant int&, constant int&,
    uint3);

template [[host_name("rope_decode_f16")]]
[[kernel]] void rope_decode<half>(
    device half*, device half*, device const float*, device const float*, device const int*,
    constant int&, constant int&, constant int&, constant int&,
    uint2);
