#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// 32 token rows x 32 output columns, k=32 per step. Four SIMD groups each
// accumulate a 16x16 quadrant in FP32; matrices multiply half inputs and half
// dequantized weights. The scale/bias affine expression is evaluated in float
// before casting, matching the stored FP16 activations/weights used by MLX QMM.
kernel void prism_affine2_gemm(
    device const half *input [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &outputs [[buffer(6)]],
    constant uint &rows [[buffer(7)]],
    constant uint &outputStride [[buffer(8)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half a[32 * 32];
    threadgroup half b[32 * 32];
    threadgroup float c[32 * 32];
    const uint tokenBase = group.y * 32u;
    const uint outputBase = group.x * 32u;
    const uint wm = columns / 16u;
    const uint gm = columns / 128u;
    const uint sm = (simd >> 1) * 16u;
    const uint sn = (simd & 1u) * 16u;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kBase = 0; kBase < columns; kBase += 32u) {
        // Activations A: [token, k].
        for (uint i = tid; i < 1024u; i += 128u) {
            const uint token = tokenBase + i / 32u;
            const uint k = kBase + i % 32u;
            a[i] = token < rows ? input[token * columns + k] : half(0);
        }
        // Weight tile B: [k, output], transposed during the cooperative load.
        // Each thread dequantizes one 16-column word at a time (64 words/tile).
        if (tid < 64u) {
            const uint out = outputBase + tid / 2u;
            const uint k = kBase + (tid & 1u) * 16u;
            const uint qg = min(out, outputs - 1u) * gm + k / 128u;
            const uint packed = weights[min(out, outputs - 1u) * wm + k / 16u];
            const float scale = float(scales[qg]), bias = float(biases[qg]);
            for (uint j = 0; j < 16u; ++j) {
                const float code = float((packed >> (j * 2u)) & 3u);
                b[((tid & 1u) * 16u + j) * 32u + tid / 2u] = half(fma(code, scale, bias));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 32u; k += 8u) {
            simdgroup_half8x8 a0, a1, b0, b1;
            simdgroup_load(a0, a + sm * 32u + k, 32);
            simdgroup_load(a1, a + (sm + 8u) * 32u + k, 32);
            simdgroup_load(b0, b + k * 32u + sn, 32);
            simdgroup_load(b1, b + k * 32u + sn + 8u, 32);
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c00, c + sm * 32u + sn, 32);
    simdgroup_store(c01, c + sm * 32u + sn + 8u, 32);
    simdgroup_store(c10, c + (sm + 8u) * 32u + sn, 32);
    simdgroup_store(c11, c + (sm + 8u) * 32u + sn + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 1024u; i += 128u) {
        const uint token = tokenBase + i / 32u;
        const uint out = outputBase + i % 32u;
        if (token < rows && out < outputs) output[token * outputStride + out] = half(c[i]);
    }
}
