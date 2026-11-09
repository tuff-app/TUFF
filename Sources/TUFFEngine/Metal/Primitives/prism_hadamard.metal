#include <metal_stdlib>
using namespace metal;

// One block per threadgroup. The checkpoint signs have one entry per input
// feature and are shared by all rows. The transform accumulates in float,
// matching Prism's reference loader before returning to half precision.
kernel void prism_hadamard_fp16(
    device const half *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &block [[buffer(4)]],
    constant uint &inverse [[buffer(5)]],
    constant uint &inputStride [[buffer(6)]],
    constant uint &outputStride [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    threadgroup float values[4096];
    const uint blocksPerRow = width / block;
    const uint row = group / blocksPerRow;
    const uint signBase = (group % blocksPerRow) * block;
    const uint inputBase = row * inputStride + signBase;
    const uint outputBase = row * outputStride + signBase;
    for (uint i = tid; i < block; i += threads) {
        float value = float(input[inputBase + i]);
        values[i] = inverse ? value : value * signs[signBase + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < block; stride <<= 1) {
        for (uint pair = tid; pair < block / 2; pair += threads) {
            uint lo = (pair / stride) * (2 * stride) + pair % stride;
            float a = values[lo];
            float b = values[lo + stride];
            values[lo] = a + b;
            values[lo + stride] = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = rsqrt(float(block));
    for (uint i = tid; i < block; i += threads) {
        float value = values[i] * scale;
        output[outputBase + i] = half(inverse ? value * signs[signBase + i] : value);
    }
}

// 512-wide blocks as three radix-8 passes of eight elements per thread (64
// threads): bits 0-2 in registers, then bits 3-5 and 6-8 after a threadgroup
// exchange, instead of nine barrier-separated radix-2 stages. The result is
// the same orthonormal transform, with a different float summation order.
static inline void prism_wht8(thread float *v) {
    for (uint s = 1; s < 8u; s <<= 1) {
        for (uint i = 0; i < 8u; ++i) {
            if ((i & s) == 0) {
                const float a = v[i], b = v[i + s];
                v[i] = a + b;
                v[i + s] = a - b;
            }
        }
    }
}

static inline void prism_hadamard512_block(
    device const half *input, device const float *signs, device half *output,
    uint tid, bool inverse, threadgroup float *values) {
    float v[8];
    for (uint j = 0; j < 8u; ++j) {
        const uint idx = tid * 8u + j;
        const float x = float(input[idx]);
        v[j] = inverse ? x : x * signs[idx];
    }
    prism_wht8(v);
    for (uint j = 0; j < 8u; ++j) values[tid * 8u + j] = v[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        const uint base = ((tid >> 3) << 6) + (tid & 7u);
        for (uint j = 0; j < 8u; ++j) v[j] = values[base + (j << 3)];
        prism_wht8(v);
        for (uint j = 0; j < 8u; ++j) values[base + (j << 3)] = v[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 8u; ++j) v[j] = values[tid + (j << 6)];
    prism_wht8(v);
    const float scale = rsqrt(512.0f);
    for (uint j = 0; j < 8u; ++j) {
        const uint idx = tid + (j << 6);
        const float value = v[j] * scale;
        output[idx] = half(inverse ? value * signs[idx] : value);
    }
}

kernel void prism_hadamard512_fp16(
    device const half *input [[buffer(0)]],
    device const float *signs [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant uint &block [[buffer(4)]],
    constant uint &inverse [[buffer(5)]],
    constant uint &inputStride [[buffer(6)]],
    constant uint &outputStride [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[512];
    const uint blocksPerRow = width / 512u;
    const uint row = group / blocksPerRow;
    const uint signBase = (group % blocksPerRow) * 512u;
    prism_hadamard512_block(input + row * inputStride + signBase, signs + signBase,
                            output + row * outputStride + signBase, tid, inverse != 0u, values);
}

// Forward transform of two independent inputs of the same width in one dispatch.
kernel void prism_hadamard512_dual(
    device const half *inputA [[buffer(0)]],
    device const half *inputB [[buffer(1)]],
    device const float *signsA [[buffer(2)]],
    device const float *signsB [[buffer(3)]],
    device half *outputA [[buffer(4)]],
    device half *outputB [[buffer(5)]],
    constant uint &width [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float values[512];
    const uint blocksPerRow = width / 512u;
    const bool second = group >= blocksPerRow;
    const uint base = (second ? group - blocksPerRow : group) * 512u;
    if (second) {
        prism_hadamard512_block(inputB + base, signsB + base, outputB + base, tid, false, values);
    } else {
        prism_hadamard512_block(inputA + base, signsA + base, outputA + base, tid, false, values);
    }
}

// MLX affine 2-bit: sixteen low-first codes per uint32, with one FP16
// scale and bias per 128 columns. Each lane owns one 32-bit word (16 columns)
// per iteration and four output rows per SIMD group share the lane's input
// values; dot(x, s*q + b) = s*dot(x, q) + b*sum(x) lets the scale and bias be
// applied once per word. Two SIMD groups per threadgroup cover eight rows.
kernel void prism_affine2_gemv(
    device const half *input [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &outputs [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint firstRow = (group * 2u + simd) * 4u;
    if (firstRow >= outputs) return;
    const uint groups = columns / 128u;
    const uint words = columns / 16u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint rowWords[4], rowGroups[4];
    for (uint r = 0; r < 4u; ++r) {
        const uint row = min(firstRow + r, outputs - 1u);
        rowWords[r] = row * words;
        rowGroups[r] = row * groups;
    }
    for (uint word = lane; word < words; word += 32u) {
        const device half4 *xp = (const device half4 *)(input + word * 16u);
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        const float xsum = dot(x0 + x1 + x2 + x3, float4(1.0f));
        const uint quantGroup = word / 8u;
        for (uint r = 0; r < 4u; ++r) {
            const uint packed = weights[rowWords[r] + word];
            const float4 c0 = float4(uint4(packed, packed >> 2, packed >> 4, packed >> 6) & 3u);
            const float4 c1 = float4(uint4(packed >> 8, packed >> 10, packed >> 12, packed >> 14) & 3u);
            const float4 c2 = float4(uint4(packed >> 16, packed >> 18, packed >> 20, packed >> 22) & 3u);
            const float4 c3 = float4(uint4(packed >> 24, packed >> 26, packed >> 28, packed >> 30) & 3u);
            const float codeDot = dot(c0, x0) + dot(c1, x1) + dot(c2, x2) + dot(c3, x3);
            sums[r] = fma(float(scales[rowGroups[r] + quantGroup]), codeDot,
                          fma(float(biases[rowGroups[r] + quantGroup]), xsum, sums[r]));
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0 && firstRow + r < outputs) output[firstRow + r] = half(sum);
    }
}

static inline float prism_code_dot(uint packed, float4 x0, float4 x1, float4 x2, float4 x3) {
    const float4 c0 = float4(uint4(packed, packed >> 2, packed >> 4, packed >> 6) & 3u);
    const float4 c1 = float4(uint4(packed >> 8, packed >> 10, packed >> 12, packed >> 14) & 3u);
    const float4 c2 = float4(uint4(packed >> 16, packed >> 18, packed >> 20, packed >> 22) & 3u);
    const float4 c3 = float4(uint4(packed >> 24, packed >> 26, packed >> 28, packed >> 30) & 3u);
    return dot(c0, x0) + dot(c1, x1) + dot(c2, x2) + dot(c3, x3);
}

// Two words (32 columns, one quantization group half) per lane per step with
// 8-byte weight loads. Requires columns % 1024 == 0.
static inline void prism_gemv_wide_core(
    device const half *input, device const uint *weights,
    device const half *scales, device const half *biases, device half *output,
    uint columns, uint outputs, uint firstRow, uint lane) {
    const uint groups = columns / 128u;
    const uint words = columns / 16u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint rowWords[4], rowGroups[4];
    for (uint r = 0; r < 4u; ++r) {
        const uint row = min(firstRow + r, outputs - 1u);
        rowWords[r] = row * words;
        rowGroups[r] = row * groups;
    }
    for (uint word = lane * 2u; word < words; word += 64u) {
        const device half4 *xp = (const device half4 *)(input + word * 16u);
        const float4 a0 = float4(xp[0]), a1 = float4(xp[1]), a2 = float4(xp[2]), a3 = float4(xp[3]);
        const float4 b0 = float4(xp[4]), b1 = float4(xp[5]), b2 = float4(xp[6]), b3 = float4(xp[7]);
        const float xsum = dot(a0 + a1 + a2 + a3 + b0 + b1 + b2 + b3, float4(1.0f));
        const uint quantGroup = word / 8u;
        uint2 packed[4];
        float s[4], b[4];
        for (uint r = 0; r < 4u; ++r) {
            packed[r] = *(const device uint2 *)(weights + rowWords[r] + word);
            s[r] = float(scales[rowGroups[r] + quantGroup]);
            b[r] = float(biases[rowGroups[r] + quantGroup]);
        }
        for (uint r = 0; r < 4u; ++r) {
            const float codeDot = prism_code_dot(packed[r].x, a0, a1, a2, a3)
                                + prism_code_dot(packed[r].y, b0, b1, b2, b3);
            sums[r] = fma(s[r], codeDot, fma(b[r], xsum, sums[r]));
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0 && firstRow + r < outputs) output[firstRow + r] = half(sum);
    }
}

kernel void prism_affine2_gemv_wide(
    device const half *input [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &outputs [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint firstRow = (group * 2u + simd) * 4u;
    if (firstRow >= outputs) return;
    prism_gemv_wide_core(input, weights, scales, biases, output, columns, outputs, firstRow, lane);
}

// Two projections in one dispatch: rows [0, outputsA) use set A, the next
// outputsB rows set B. outputsA must be a multiple of 8 so a threadgroup never
// straddles the boundary. Requires columns % 1024 == 0.
kernel void prism_affine2_gemv_dual(
    device const half *inputA [[buffer(0)]],
    device const half *inputB [[buffer(1)]],
    device const uint *weightsA [[buffer(2)]],
    device const uint *weightsB [[buffer(3)]],
    device const half *scalesA [[buffer(4)]],
    device const half *scalesB [[buffer(5)]],
    device const half *biasesA [[buffer(6)]],
    device const half *biasesB [[buffer(7)]],
    device half *outputA [[buffer(8)]],
    device half *outputB [[buffer(9)]],
    constant uint &columns [[buffer(10)]],
    constant uint &outputsA [[buffer(11)]],
    constant uint &outputsB [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint firstRow = (group * 2u + simd) * 4u;
    if (firstRow >= outputsA + outputsB) return;
    if (firstRow < outputsA) {
        prism_gemv_wide_core(inputA, weightsA, scalesA, biasesA, outputA, columns, outputsA, firstRow, lane);
    } else {
        prism_gemv_wide_core(inputB, weightsB, scalesB, biasesB, outputB, columns, outputsB,
                             firstRow - outputsA, lane);
    }
}

// Two FP32 GDN scalar projections (alpha, beta) in one dispatch: one
// threadgroup per output row of either matrix.
kernel void prism_f32_gemv_dual(
    device const half *input [[buffer(0)]],
    device const float *weightsA [[buffer(1)]],
    device const float *weightsB [[buffer(2)]],
    device float *outputA [[buffer(3)]],
    device float *outputB [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &outputs [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const bool second = group >= outputs;
    const uint row = second ? group - outputs : group;
    device const float *weights = second ? weightsB : weightsA;
    float sum = 0.0f;
    for (uint col = lane * 4u; col < columns; col += 128u) {
        const float4 w = *(device const float4 *)(weights + row * columns + col);
        const float4 x = float4(*(device const half4 *)(input + col));
        sum += dot(w, x);
    }
    sum = simd_sum(sum);
    if (lane == 0) (second ? outputB : outputA)[row] = sum;
}

// Four SIMD groups split the columns of the same four rows and combine through
// threadgroup memory, so each lane issues far fewer dependent loads.
kernel void prism_affine2_gemv_split(
    device const half *input [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &outputs [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[4][4];
    const uint firstRow = group * 4u;
    const uint groups = columns / 128u;
    const uint words = columns / 16u;
    const uint span = words / 4u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint rowWords[4], rowGroups[4];
    for (uint r = 0; r < 4u; ++r) {
        const uint row = min(firstRow + r, outputs - 1u);
        rowWords[r] = row * words;
        rowGroups[r] = row * groups;
    }
    for (uint word = simd * span + lane; word < (simd + 1u) * span; word += 32u) {
        const device half4 *xp = (const device half4 *)(input + word * 16u);
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        const float xsum = dot(x0 + x1 + x2 + x3, float4(1.0f));
        const uint quantGroup = word / 8u;
        for (uint r = 0; r < 4u; ++r) {
            const float codeDot = prism_code_dot(weights[rowWords[r] + word], x0, x1, x2, x3);
            sums[r] = fma(float(scales[rowGroups[r] + quantGroup]), codeDot,
                          fma(float(biases[rowGroups[r] + quantGroup]), xsum, sums[r]));
        }
    }
    for (uint r = 0; r < 4u; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0) partial[simd][r] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane < 4u && firstRow + lane < outputs) {
        output[firstRow + lane] = half(partial[0][lane] + partial[1][lane] + partial[2][lane] + partial[3][lane]);
    }
}

// Each SIMD group shares a dequantized weight across four token rows. The
// lane-wise accumulation and SIMD reduction match the single-row GEMV.
kernel void prism_affine2_batched(
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
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x * 8u + simd;
    const uint token = group.y * 4u;
    if (row >= outputs) return;
    const uint groups = columns / 128u;
    const uint words = columns / 16u;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint col = lane; col < columns; col += 32u) {
        uint word = weights[row * words + col / 16u];
        float code = float((word >> (2u * (col % 16u))) & 3u);
        uint quantGroup = row * groups + col / 128u;
        float value = fma(code, float(scales[quantGroup]), float(biases[quantGroup]));
        for (uint t = 0; t < 4u; ++t) {
            if (token + t < rows)
                sums[t] = fma(float(input[(token + t) * columns + col]), value, sums[t]);
        }
    }
    for (uint t = 0; t < 4u; ++t) {
        float sum = simd_sum(sums[t]);
        if (lane == 0 && token + t < rows) output[(token + t) * outputStride + row] = half(sum);
    }
}

// Dequantize selected rows of the rotated token table. The caller follows
// this with prism_hadamard_fp16 in inverse mode to restore embeddings.
kernel void prism_affine2_embedding(
    device const int *tokens [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &vocab [[buffer(6)]],
    constant uint &count [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count * columns) return;
    uint tokenIndex = index / columns;
    uint column = index % columns;
    int token = tokens[tokenIndex];
    if (token < 0 || uint(token) >= vocab) { output[index] = half(0); return; }
    uint row = uint(token);
    uint code = (weights[row * (columns / 16) + column / 16] >> (2 * (column % 16))) & 3u;
    uint quantGroup = row * (columns / 128) + column / 128;
    output[index] = half(fma(float(code), float(scales[quantGroup]), float(biases[quantGroup])));
}

// The small GDN alpha/beta projections in the published MLX pack are F32,
// not quantized. Preserve their stored precision and FP32 projection output.
kernel void prism_f32_gemv(
    device const half *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint col = lane; col < columns; col += 32) {
        sum = fma(float(input[col]), weights[row * columns + col], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

// Per-row RMSNorm with the publisher's F32 weights, shared across rows.
kernel void prism_f32_rmsnorm(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[32];
    float sum = 0.0f;
    for (uint col = tid; col < width; col += threads) {
        float value = float(input[row * width + col]);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float value = lane < ((threads + 31) / 32) ? partial[lane] : 0.0f;
        value = simd_sum(value);
        if (lane == 0) partial[0] = rsqrt(value / float(width) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < width; col += threads) {
        output[row * width + col] = half(float(input[row * width + col]) * partial[0] * weight[col]);
    }
}
// Draft FP32 Prism GDN decode kernels; not yet integrated.

static inline float prism_gdn_silu(float x) { return x / (1.0f + exp(-x)); }

static inline float prism_gdn_softplus(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

kernel void prism_gdn_conv_mix_decode(
    device half*        tail     [[buffer(0)]],   // [K-1, C] shifted in place
    device const half*  qkv      [[buffer(1)]],   // [C] current raw row
    device const float* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [C] silu(conv)
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    if (tid >= C) return;

    const uint history = K - 1u;
    float acc = float(qkv[tid]) * float(conv_w[tid * K + (K - 1u)]);
    for (uint j = 0; j < history; ++j) {
        acc = fma(float(tail[j * C + tid]), float(conv_w[tid * K + j]), acc);
    }
    out[tid] = half(prism_gdn_silu(acc));

    for (uint j = 0; j + 1u < history; ++j) {
        tail[j * C + tid] = tail[(j + 1u) * C + tid];
    }
    if (history > 0u) {
        tail[(history - 1u) * C + tid] = qkv[tid];
    }
}

kernel void prism_gdn_delta_step_decode(
    device const half*   conv_out [[buffer(0)]],   // [C] normed q,k + raw v
    device const float*  a_proj   [[buffer(1)]],   // [Hv]
    device const float*  b_proj   [[buffer(2)]],   // [Hv]
    device const float* A_log    [[buffer(3)]],   // [Hv]
    device const float* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = h / (Hv / Hk);
    device const half* q = conv_out + hk * Dk;
    device const half* k = conv_out + Hk * Dk + hk * Dk;
    device const half* v = conv_out + 2u * Hk * Dk + h * Dv;

    const float g = exp(-exp(float(A_log[h]))
                        * prism_gdn_softplus(float(a_proj[h]) + float(dt_bias[h])));
    const float beta = 1.0f / (1.0f + exp(-float(b_proj[h])));

    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;

    float s[8];
    float kv = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = srow[idx] * g;
        kv = fma(s[i], float(k[idx]), kv);
    }
    kv = simd_sum(kv);

    const float delta = (float(v[dv]) - kv) * beta;

    float out = 0.0f;
    for (uint i = 0; i < nPerLane; ++i) {
        const uint idx = lane * nPerLane + i;
        s[i] = fma(float(k[idx]), delta, s[i]);
        out = fma(s[i], float(q[idx]), out);
        srow[idx] = s[i];
    }
    out = simd_sum(out);
    if (lane == 0) y[h * Dv + dv] = half(out);
}

kernel void prism_gdn_gated_norm(
    device const half*   y        [[buffer(0)]],   // [T, Hv * Dv]
    device const half*   z        [[buffer(1)]],   // [T, Hv * Dv]
    device const float* weight   [[buffer(2)]],   // [Dv]
    device half*         out      [[buffer(3)]],   // [T, Hv * Dv]
    constant uint&       vHeads   [[buffer(4)]],
    constant uint&       valueDim [[buffer(5)]],
    constant uint&       gateKind [[buffer(6)]],   // 0 = silu, 1 = sigmoid
    uint2 tg  [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]],
    uint  simd_lane [[thread_index_in_simdgroup]],
    uint  simd_idx  [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float partial[32];
    const uint tid = tpos.x;
    const uint Hv = vHeads;
    const uint Dv = valueDim;
    const uint head = tg.x;
    const uint row = tg.y;
    if (head >= Hv) return;

    const uint base = row * Hv * Dv + head * Dv;

    float sumsq = 0.0f;
    for (uint i = tid; i < Dv; i += 128u) {
        const float x = float(y[base + i]);
        sumsq = fma(x, x, sumsq);
    }
    sumsq = simd_sum(sumsq);
    if (simd_lane == 0) partial[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 4u; ++i) total += partial[i];
        partial[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = partial[0] / float(Dv);
    const float invRms = rsqrt(mean + 1e-6f);

    for (uint i = tid; i < Dv; i += 128u) {
        const float normed = float(y[base + i]) * invRms * float(weight[i]);
        const float zv = float(z[base + i]);
        const float gate = (gateKind == 1u) ? (1.0f / (1.0f + exp(-zv)))
                                            : prism_gdn_silu(zv);
        out[base + i] = half(normed * gate);
    }
}

// Prefill versions of the Prism GDN steps above. Each runs a whole chunk of
// rows in one dispatch instead of one dispatch per token, and performs the
// same per-token arithmetic in the same order, so the output matches the
// decode kernels run row by row.

// prism_f32_gemv over several token rows: threadgroup (output, token).
kernel void prism_f32_gemv_rows(
    device const half *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &columns [[buffer(3)]],
    constant uint &outputs [[buffer(4)]],
    constant uint &inputStride [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x;
    const uint token = group.y;
    device const half *x = input + token * inputStride;
    float sum = 0.0f;
    for (uint col = lane; col < columns; col += 32) {
        sum = fma(float(x[col]), weights[row * columns + col], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) output[token * outputs + row] = sum;
}

// prism_gdn_conv_mix_decode over `rows` consecutive rows. The history the
// decode kernel shifts through `tail` is kept in registers and written back
// once at the end, holding the same half values.
kernel void prism_gdn_conv_mix_prefill(
    device half*        tail     [[buffer(0)]],   // [K-1, C]
    device const half*  qkv      [[buffer(1)]],   // [T, C]
    device const float* conv_w  [[buffer(2)]],   // [C, K]
    device half*        out      [[buffer(3)]],   // [T, C] silu(conv)
    constant uint&      channels [[buffer(4)]],
    constant uint&      taps     [[buffer(5)]],
    constant uint&      rows     [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    const uint C = channels;
    const uint K = taps;
    if (tid >= C) return;
    const uint history = K - 1u;
    half past[7];
    for (uint j = 0; j < history; ++j) past[j] = tail[j * C + tid];
    for (uint t = 0; t < rows; ++t) {
        const half x = qkv[t * C + tid];
        float acc = float(x) * float(conv_w[tid * K + (K - 1u)]);
        for (uint j = 0; j < history; ++j) {
            acc = fma(float(past[j]), float(conv_w[tid * K + j]), acc);
        }
        out[t * C + tid] = half(prism_gdn_silu(acc));
        for (uint j = 0; j + 1u < history; ++j) past[j] = past[j + 1u];
        if (history > 0u) past[history - 1u] = x;
    }
    for (uint j = 0; j < history; ++j) tail[j * C + tid] = past[j];
}

// prism_gdn_delta_step_decode over `rows` consecutive rows. The state row is
// kept in registers between tokens; the decode kernel stores and reloads the
// same FP32 values.
kernel void prism_gdn_delta_step_prefill(
    device const half*   conv_out [[buffer(0)]],   // [T, C]
    device const float*  a_proj   [[buffer(1)]],   // [T, Hv]
    device const float*  b_proj   [[buffer(2)]],   // [T, Hv]
    device const float* A_log    [[buffer(3)]],   // [Hv]
    device const float* dt_bias  [[buffer(4)]],   // [Hv]
    device float*        state    [[buffer(5)]],   // [Hv, Dv, Dk]
    device half*         y        [[buffer(6)]],   // [T, Hv * Dv]
    constant uint&       kHeads   [[buffer(7)]],
    constant uint&       vHeads   [[buffer(8)]],
    constant uint&       keyDim   [[buffer(9)]],
    constant uint&       valueDim [[buffer(10)]],
    constant uint&       rows     [[buffer(11)]],
    constant uint&       rowStride [[buffer(12)]],  // C
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tpos [[thread_position_in_threadgroup]]
) {
    const uint Hk = kHeads;
    const uint Hv = vHeads;
    const uint Dk = keyDim;
    const uint Dv = valueDim;
    const uint h = tg.x;
    const uint dv = tg.y * 4u + tpos.y;
    const uint lane = tpos.x;
    if (h >= Hv || dv >= Dv) return;

    const uint hk = h / (Hv / Hk);
    const uint nPerLane = Dk / 32u;
    device float* srow = state + (uint(h) * Dv + dv) * Dk;
    float s[8];
    for (uint i = 0; i < nPerLane; ++i) s[i] = srow[lane * nPerLane + i];
    const float aScale = exp(float(A_log[h]));
    const float bias = float(dt_bias[h]);

    for (uint t = 0; t < rows; ++t) {
        device const half* row = conv_out + t * rowStride;
        device const half* q = row + hk * Dk;
        device const half* k = row + Hk * Dk + hk * Dk;
        device const half* v = row + 2u * Hk * Dk + h * Dv;
        const float g = exp(-aScale * prism_gdn_softplus(float(a_proj[t * Hv + h]) + bias));
        const float beta = 1.0f / (1.0f + exp(-float(b_proj[t * Hv + h])));

        float kv = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] = s[i] * g;
            kv = fma(s[i], float(k[idx]), kv);
        }
        kv = simd_sum(kv);
        const float delta = (float(v[dv]) - kv) * beta;
        float out = 0.0f;
        for (uint i = 0; i < nPerLane; ++i) {
            const uint idx = lane * nPerLane + i;
            s[i] = fma(float(k[idx]), delta, s[i]);
            out = fma(s[i], float(q[idx]), out);
        }
        out = simd_sum(out);
        if (lane == 0) y[t * Hv * Dv + h * Dv + dv] = half(out);
    }
    for (uint i = 0; i < nPerLane; ++i) srow[lane * nPerLane + i] = s[i];
}
