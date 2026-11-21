#include <metal_stdlib>
using namespace metal;

#if defined(__HAVE_TENSOR__)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

// Fixed at 64, unlike the decode kernels. This module is compiled as its own
// private library, where a function constant would force every pipeline
// through the specialized-function API, and it is only reached from chunked
// prefill — which Qwen4-Exp does not use yet. It follows when prefill does.
constant constexpr uint kW4A8GroupSize = 64;

constant constexpr int kMPPAffineTileM = 64;
constant constexpr int kMPPAffineTileN = 32;
constant constexpr int kMPPAffineTileK = 64;

kernel void mpp_prefill_affine_threadgroup_f16(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0,
        int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0,
            int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        // TensorOps does not make threadgroup-memory lifetime visible across
        // SIMD groups; keep the next group from overwriting B while another
        // SIMD group can still be consuming the current tile.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

// The tiled MPP matmul for every affine layout TUFF ships: INT4 at group 32
// (Qwen 3.8 Flash Next) or 64, and INT8 at group 64 (Gemma 4 12B and MiniMax
// M2.7 dense and shared MLPs). Each thread dequantizes 8 consecutive weights of
// a 64-wide K tile from one 4- or 8-byte run and one scale and bias, so a
// group smaller than the tile needs no second pass, and the matmul accumulates
// in place across K tiles. TileM is 64, 32, 16 or 8: routed experts see a
// handful of rows each, and a 64-row tile would spend most of its matmul on
// padding. On a 16 GB M2 the smaller tiles run a 26B expert's 16 rows 3.4x and
// Qwen 3.6's 8 rows 4.4x faster than the 64-row tile. Output is Y[M, N] = X[M, K] * W[N, K]^T,
// the layout of the INT4 group-64 kernel above, which stays as it is.
//
// TileK is the K depth of one dequantized tile and one matmul call. The
// 128 x 16 tile with a 256-deep K step serves whole 128-row blocks of a
// prefill chunk: on a 16 GB M2 it runs the dense projections 10-15% faster
// than the 64 x 32 x 64 tile. Every output still accumulates its products in
// ascending K order into one FP32 accumulator, and the results matched the
// 64-row tile bit for bit on that GPU.
template <uint Bits, uint GroupSize, int TileM, int TileN, int TileK = kMPPAffineTileK>
static inline void mpp_prefill_affine_qmm_f16_impl(
    device const uint8_t* packedWeights,
    device const bfloat* scales,
    device const bfloat* biases,
    device half* activations,
    device half* output,
    uint M, uint N, uint K,
    uint3 tgid, uint lid, uint threads,
    threadgroup half* weightTile) {
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, TileK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(TileK, TileN),
        array<int32_t, 2>({1, TileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * TileM);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K * Bits / 8u;
    const uint groupsPerRow = K / GroupSize;
    const uint kTiles = K / uint(TileK);
    constexpr uint chunksPerRow = uint(TileK) / 8u;
    for (uint tile = 0; tile < kTiles; ++tile) {
        for (uint chunk = lid; chunk < uint(TileN) * chunksPerRow; chunk += threads) {
            const uint localN = chunk / chunksPerRow;
            const uint localK = (chunk % chunksPerRow) * 8u;
            const uint globalN = tgid.x * uint(TileN) + localN;
            threadgroup half4* destination =
                (threadgroup half4*)(weightTile + localN * uint(TileK) + localK);
            if (globalN < N) {
                const uint globalK = tile * uint(TileK) + localK;
                const uint group = globalN * groupsPerRow + globalK / GroupSize;
                const float scale = float(scales[group]);
                const float bias = float(biases[group]);
                float4 low, high;
                if (Bits == 4u) {
                    // Byte-aligned: .gturbo tensors are not 4-byte aligned.
                    const uint packed = as_type<uint>(uchar4(*(device const packed_uchar4*)(
                        packedWeights + globalN * rowBytes + (globalK >> 1))));
                    low = float4(float(packed & 0xfu), float((packed >> 4) & 0xfu),
                                 float((packed >> 8) & 0xfu), float((packed >> 12) & 0xfu));
                    high = float4(float((packed >> 16) & 0xfu), float((packed >> 20) & 0xfu),
                                  float((packed >> 24) & 0xfu), float(packed >> 28));
                } else {
                    device const packed_uchar4* packed = (device const packed_uchar4*)(
                        packedWeights + globalN * rowBytes + globalK);
                    low = float4(uchar4(packed[0]));
                    high = float4(uchar4(packed[1]));
                }
                destination[0] = half4(fma(low, scale, bias));
                destination[1] = half4(fma(high, scale, bias));
            } else {
                destination[0] = half4(0.0h);
                destination[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor tileSourceA(
            activations + tile * uint(TileK),
            dextents<int32_t, 2>(TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = tileSourceA.slice(0, int32_t(tgid.y) * TileM);
        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(TileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(TileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

#define TUFF_MPP_AFFINE_QMM_KERNEL(NAME, BITS, GROUP, TM, TN, TK)              \
kernel void NAME(                                                              \
    device const uint8_t* packedWeights [[buffer(0)]],                         \
    device const bfloat* scales         [[buffer(1)]],                         \
    device const bfloat* biases         [[buffer(2)]],                         \
    device half* activations            [[buffer(3)]],                         \
    device half* output                 [[buffer(4)]],                         \
    constant uint& M                    [[buffer(5)]],                         \
    constant uint& N                    [[buffer(6)]],                         \
    constant uint& K                    [[buffer(7)]],                         \
    uint3 tgid                          [[threadgroup_position_in_grid]],      \
    uint3 lid3                          [[thread_position_in_threadgroup]],    \
    uint3 threads3                      [[threads_per_threadgroup]]) {         \
    threadgroup half weightTile[(TN) * (TK)];                                  \
    mpp_prefill_affine_qmm_f16_impl<BITS, GROUP, TM, TN, TK>(                  \
        packedWeights, scales, biases, activations, output, M, N, K,           \
        tgid, lid3.x, threads3.x, weightTile);                                 \
}

#define TUFF_MPP_AFFINE_QMM_LAYOUT(NAME, BITS, GROUP)                          \
    TUFF_MPP_AFFINE_QMM_KERNEL(NAME, BITS, GROUP, 64, 32, kMPPAffineTileK)     \
    TUFF_MPP_AFFINE_QMM_KERNEL(NAME##_m32, BITS, GROUP, 32, 32, kMPPAffineTileK) \
    TUFF_MPP_AFFINE_QMM_KERNEL(NAME##_m16, BITS, GROUP, 16, 64, kMPPAffineTileK) \
    TUFF_MPP_AFFINE_QMM_KERNEL(NAME##_m8, BITS, GROUP, 8, 64, kMPPAffineTileK) \
    TUFF_MPP_AFFINE_QMM_KERNEL(NAME##_m128, BITS, GROUP, 128, 16, 256)

TUFF_MPP_AFFINE_QMM_LAYOUT(mpp_prefill_affine_qmm_f16_w4g32, 4u, 32u)
TUFF_MPP_AFFINE_QMM_LAYOUT(mpp_prefill_affine_qmm_f16_w4g64, 4u, 64u)
TUFF_MPP_AFFINE_QMM_LAYOUT(mpp_prefill_affine_qmm_f16_w8g64, 8u, 64u)
TUFF_MPP_AFFINE_QMM_LAYOUT(mpp_prefill_affine_qmm_f16_w8g32, 8u, 32u)

#undef TUFF_MPP_AFFINE_QMM_LAYOUT
#undef TUFF_MPP_AFFINE_QMM_KERNEL

// GPT-OSS MXFP4 experts as a batched MPP matmul: Y[M, N] = X[M, K] * W[N, K]^T
// + bias[N]. Each weight is an E2M1 nibble (low nibble first) times its
// 32-value block's UE8M0 scale, 2^(e - 127), exactly as `mxfp4_gemv_simd`
// decodes it; this module compiles on its own, so it keeps its own decoder.
static inline float mpp_mxfp4_e2m1(uint code) {
    constexpr float magnitudes[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitudes[code & 7u];
    return (code & 8u) == 0u ? value : -value;
}

// GPT-OSS experts: 8 codes per thread from one 4-byte run and one UE8M0
// scale, accumulated in place, at the same M tiles as the affine kernels.
// FP32 stores keep large down-projection partials finite before route weighting.
template <int TileM, int TileN, typename Output>
static inline void mpp_prefill_mxfp4_qmm_f16_impl(
    device const uint8_t* packedWeights,
    device const uint8_t* scales,
    device const bfloat* bias,
    device half* activations,
    device Output* output,
    uint M, uint N, uint K, uint hasBias,
    uint3 tgid, uint lid, uint threads,
    threadgroup half* weightTile) {
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, kMPPAffineTileK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, TileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * TileM);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint blocksPerRow = K / 32u;
    constexpr uint chunksPerRow = uint(kMPPAffineTileK) / 8u;
    for (uint tile = 0; tile < K / uint(kMPPAffineTileK); ++tile) {
        for (uint chunk = lid; chunk < uint(TileN) * chunksPerRow; chunk += threads) {
            const uint localN = chunk / chunksPerRow;
            const uint localK = (chunk % chunksPerRow) * 8u;
            const uint globalN = tgid.x * uint(TileN) + localN;
            threadgroup half4* destination =
                (threadgroup half4*)(weightTile + localN * uint(kMPPAffineTileK) + localK);
            if (globalN < N) {
                const uint globalK = tile * uint(kMPPAffineTileK) + localK;
                const uint packed = as_type<uint>(uchar4(*(device const packed_uchar4*)(
                    packedWeights + globalN * rowBytes + (globalK >> 1))));
                const float scale = as_type<float>(
                    uint(scales[globalN * blocksPerRow + globalK / 32u]) << 23u);
                destination[0] = half4(float4(mpp_mxfp4_e2m1(packed),
                                              mpp_mxfp4_e2m1(packed >> 4),
                                              mpp_mxfp4_e2m1(packed >> 8),
                                              mpp_mxfp4_e2m1(packed >> 12)) * scale);
                destination[1] = half4(float4(mpp_mxfp4_e2m1(packed >> 16),
                                              mpp_mxfp4_e2m1(packed >> 20),
                                              mpp_mxfp4_e2m1(packed >> 24),
                                              mpp_mxfp4_e2m1(packed >> 28)) * scale);
            } else {
                destination[0] = half4(0.0h);
                destination[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor tileSourceA(
            activations + tile * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = tileSourceA.slice(0, int32_t(tgid.y) * TileM);
        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(TileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(TileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            const float withBias = accumulator[element]
                + (hasBias != 0u ? float(bias[globalN]) : 0.0f);
            output[globalM * N + globalN] = Output(withBias);
        }
    }
}

#define TUFF_MPP_MXFP4_QMM_KERNEL(NAME, TM, TN, Output)                         \
kernel void NAME(                                                              \
    device const uint8_t* packedWeights [[buffer(0)]],                         \
    device const uint8_t* scales        [[buffer(1)]],                         \
    device const bfloat* bias           [[buffer(2)]],                         \
    device half* activations            [[buffer(3)]],                         \
    device Output* output               [[buffer(4)]],                         \
    constant uint& M                    [[buffer(5)]],                         \
    constant uint& N                    [[buffer(6)]],                         \
    constant uint& K                    [[buffer(7)]],                         \
    constant uint& hasBias              [[buffer(8)]],                         \
    uint3 tgid                          [[threadgroup_position_in_grid]],      \
    uint3 lid3                          [[thread_position_in_threadgroup]],    \
    uint3 threads3                      [[threads_per_threadgroup]]) {         \
    threadgroup half weightTile[(TN) * kMPPAffineTileK];                        \
    mpp_prefill_mxfp4_qmm_f16_impl<TM, TN, Output>(                              \
        packedWeights, scales, bias, activations, output, M, N, K, hasBias,    \
        tgid, lid3.x, threads3.x, weightTile);                                 \
}

TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f16, 64, 32, half)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f16_m32, 32, 32, half)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f16_m16, 16, 64, half)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f16_m8, 8, 64, half)

TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f32, 64, 32, float)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f32_m32, 32, 32, float)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f32_m16, 16, 64, float)
TUFF_MPP_MXFP4_QMM_KERNEL(mpp_prefill_mxfp4_qmm_f32_m8, 8, 64, float)

#undef TUFF_MPP_MXFP4_QMM_KERNEL

// Inkling 2 experts, in the publisher's two float formats, at the same tiles.
//
// NVFP4 (Format 0): E2M1 codes eight to a word, one E4M3 scale per 16 values,
// and one factor for the whole expert. MXFP8 (Format 1): E4M3 codes one to a
// byte with one power-of-two scale per 32 values. A dequantized weight is a
// small number either way, so the tile holds halves; the products of codes
// and scales on the way there are formed in FP32.
static inline float mpp_inkling_e4m3(uint code) {
    const uint exponent = (code >> 3u) & 15u;
    const uint mantissa = code & 7u;
    const uint sign = (code & 128u) << 24u;
    if (exponent == 0u) {
        const float magnitude = float(mantissa) * 0.001953125f;
        return sign != 0u ? -magnitude : magnitude;
    }
    return as_type<float>(sign | ((exponent + 120u) << 23u) | (mantissa << 20u));
}

template <int TileM, int TileN, typename Output, int Format>
static inline void mpp_prefill_inkling_qmm_f16_impl(
    device const uint8_t* codes,
    device const uint8_t* scales,
    device half* activations,
    device Output* output,
    uint M, uint N, uint K, float factor,
    uint3 tgid, uint lid, uint threads,
    threadgroup half* weightTile) {
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, kMPPAffineTileK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, TileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * TileM);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    constexpr uint chunksPerRow = uint(kMPPAffineTileK) / 8u;
    for (uint tile = 0; tile < K / uint(kMPPAffineTileK); ++tile) {
        for (uint chunk = lid; chunk < uint(TileN) * chunksPerRow; chunk += threads) {
            const uint localN = chunk / chunksPerRow;
            const uint localK = (chunk % chunksPerRow) * 8u;
            const uint globalN = tgid.x * uint(TileN) + localN;
            threadgroup half4* destination =
                (threadgroup half4*)(weightTile + localN * uint(kMPPAffineTileK) + localK);
            if (globalN < N) {
                const uint globalK = tile * uint(kMPPAffineTileK) + localK;
                if (Format == 0) {
                    const uint packed = as_type<uint>(uchar4(*(device const packed_uchar4*)(
                        codes + size_t(globalN) * (K / 2u) + (globalK >> 1))));
                    const float scale = mpp_inkling_e4m3(
                        uint(scales[size_t(globalN) * (K / 16u) + globalK / 16u])) * factor;
                    destination[0] = half4(float4(mpp_mxfp4_e2m1(packed),
                                                  mpp_mxfp4_e2m1(packed >> 4),
                                                  mpp_mxfp4_e2m1(packed >> 8),
                                                  mpp_mxfp4_e2m1(packed >> 12)) * scale);
                    destination[1] = half4(float4(mpp_mxfp4_e2m1(packed >> 16),
                                                  mpp_mxfp4_e2m1(packed >> 20),
                                                  mpp_mxfp4_e2m1(packed >> 24),
                                                  mpp_mxfp4_e2m1(packed >> 28)) * scale);
                } else {
                    device const packed_uchar4* bytes =
                        (device const packed_uchar4*)(codes + size_t(globalN) * K + globalK);
                    const uchar4 low = uchar4(bytes[0]), high = uchar4(bytes[1]);
                    const float scale = as_type<float>(
                        max(uint(scales[size_t(globalN) * (K / 32u) + globalK / 32u]), 1u) << 23u);
                    destination[0] = half4(float4(mpp_inkling_e4m3(low.x), mpp_inkling_e4m3(low.y),
                                                  mpp_inkling_e4m3(low.z), mpp_inkling_e4m3(low.w)) * scale);
                    destination[1] = half4(float4(mpp_inkling_e4m3(high.x), mpp_inkling_e4m3(high.y),
                                                  mpp_inkling_e4m3(high.z), mpp_inkling_e4m3(high.w)) * scale);
                }
            } else {
                destination[0] = half4(0.0h);
                destination[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor tileSourceA(
            activations + tile * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = tileSourceA.slice(0, int32_t(tgid.y) * TileM);
        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(TileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(TileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[size_t(globalM) * N + globalN] = Output(accumulator[element]);
        }
    }
}

// Each kernel is written out so the kernel index, which reads names from
// this source, finds it.
#define TUFF_MPP_INKLING_QMM_ARGS                                              \
    device const uint8_t* codes         [[buffer(0)]],                         \
    device const uint8_t* scales        [[buffer(1)]],                         \
    device half* activations            [[buffer(2)]],                         \
    device float* output                [[buffer(3)]],                         \
    constant uint& M                    [[buffer(4)]],                         \
    constant uint& N                    [[buffer(5)]],                         \
    constant uint& K                    [[buffer(6)]],                         \
    constant float& factor              [[buffer(7)]],                         \
    uint3 tgid                          [[threadgroup_position_in_grid]],      \
    uint3 lid3                          [[thread_position_in_threadgroup]],    \
    uint3 threads3                      [[threads_per_threadgroup]]
#define TUFF_MPP_INKLING_QMM_BODY(TM, TN, Format)                              \
    threadgroup half weightTile[(TN) * kMPPAffineTileK];                        \
    mpp_prefill_inkling_qmm_f16_impl<TM, TN, float, Format>(                    \
        codes, scales, activations, output, M, N, K, factor,                   \
        tgid, lid3.x, threads3.x, weightTile);

kernel void mpp_prefill_nvfp4_qmm_f32(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(64, 32, 0) }
kernel void mpp_prefill_nvfp4_qmm_f32_m32(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(32, 32, 0) }
kernel void mpp_prefill_nvfp4_qmm_f32_m16(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(16, 64, 0) }
kernel void mpp_prefill_nvfp4_qmm_f32_m8(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(8, 64, 0) }

kernel void mpp_prefill_mxfp8_qmm_f32(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(64, 32, 1) }
kernel void mpp_prefill_mxfp8_qmm_f32_m32(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(32, 32, 1) }
kernel void mpp_prefill_mxfp8_qmm_f32_m16(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(16, 64, 1) }
kernel void mpp_prefill_mxfp8_qmm_f32_m8(TUFF_MPP_INKLING_QMM_ARGS) { TUFF_MPP_INKLING_QMM_BODY(8, 64, 1) }

#undef TUFF_MPP_INKLING_QMM_ARGS
#undef TUFF_MPP_INKLING_QMM_BODY

kernel void mpp_prefill_affine_threadgroup_f16_input_bf16_output(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device bfloat* output               [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineTileM, kMPPAffineTileN, kMPPAffineTileK,
        false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[kMPPAffineTileN * kMPPAffineTileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, kMPPAffineTileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * kMPPAffineTileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < groupsPerRow; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineTileN * kMPPAffineTileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineTileK);
            const uint localK = linear % uint(kMPPAffineTileK);
            const uint globalN = tgid.x * uint(kMPPAffineTileN) + localN;
            if (globalN < N) {
                const uint globalK = group * uint(kMPPAffineTileK) + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu) : uint(packed >> 4);
                const float scale = float(scales[globalN * groupsPerRow + group]);
                const float bias = float(biases[globalN * groupsPerRow + group]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + group * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(0, int32_t(tgid.y) * kMPPAffineTileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineTileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineTileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = bfloat(accumulator[element]);
        }
    }
}

// GPT-OSS's resident BF16 projections (q, k, v, o) for a prefill chunk.
// The batched GEMV gave each (token, output row) its own SIMD group, so every
// weight row was re-read once per token. Here each weight tile is read once
// per tile of tokens: 8 BF16 weights per thread, widened to FP16 in
// threadgroup memory, with the bias added at the store.
template <int TileM, int TileN>
static inline void mpp_prefill_bf16_mm_f16_impl(
    device const uint8_t* weightBytes,
    device const bfloat* bias,
    device half* activations,
    device half* output,
    uint M, uint N, uint K, uint hasBias,
    uint3 tgid, uint lid, uint threads,
    threadgroup half* weightTile) {
    constexpr auto descriptor = matmul2d_descriptor(
        TileM, TileN, kMPPAffineTileK,
        false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineTileK, TileN),
        array<int32_t, 2>({1, kMPPAffineTileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineTileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * TileM);
    auto accumulator = operation.template get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    constexpr uint chunksPerRow = uint(kMPPAffineTileK) / 8u;
    for (uint tile = 0; tile < K / uint(kMPPAffineTileK); ++tile) {
        for (uint chunk = lid; chunk < uint(TileN) * chunksPerRow; chunk += threads) {
            const uint localN = chunk / chunksPerRow;
            const uint localK = (chunk % chunksPerRow) * 8u;
            const uint globalN = tgid.x * uint(TileN) + localN;
            threadgroup half4* destination =
                (threadgroup half4*)(weightTile + localN * uint(kMPPAffineTileK) + localK);
            if (globalN < N) {
                const uint globalK = tile * uint(kMPPAffineTileK) + localK;
                // Byte-aligned reads: BF16 tensors are only 2-byte aligned.
                device const packed_ushort4* source = (device const packed_ushort4*)(
                    weightBytes + (ulong(globalN) * K + globalK) * 2u);
                const uint4 low = uint4(ushort4(source[0])) << 16u;
                const uint4 high = uint4(ushort4(source[1])) << 16u;
                destination[0] = half4(as_type<float4>(low));
                destination[1] = half4(as_type<float4>(high));
            } else {
                destination[0] = half4(0.0h);
                destination[1] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor tileSourceA(
            activations + tile * uint(kMPPAffineTileK),
            dextents<int32_t, 2>(kMPPAffineTileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = tileSourceA.slice(0, int32_t(tgid.y) * TileM);
        operation.run(tileA, tileB, accumulator);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(TileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(TileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            const float withBias = accumulator[element]
                + (hasBias != 0u ? float(bias[globalN]) : 0.0f);
            output[globalM * N + globalN] = half(withBias);
        }
    }
}

#define TUFF_MPP_BF16_MM_KERNEL(NAME, TM, TN)                                  \
kernel void NAME(                                                              \
    device const uint8_t* weights       [[buffer(0)]],                         \
    device const bfloat* bias           [[buffer(1)]],                         \
    device half* activations            [[buffer(2)]],                         \
    device half* output                 [[buffer(3)]],                         \
    constant uint& M                    [[buffer(4)]],                         \
    constant uint& N                    [[buffer(5)]],                         \
    constant uint& K                    [[buffer(6)]],                         \
    constant uint& hasBias              [[buffer(7)]],                         \
    uint3 tgid                          [[threadgroup_position_in_grid]],      \
    uint3 lid3                          [[thread_position_in_threadgroup]],    \
    uint3 threads3                      [[threads_per_threadgroup]]) {         \
    threadgroup half weightTile[(TN) * kMPPAffineTileK];                        \
    mpp_prefill_bf16_mm_f16_impl<TM, TN>(                                      \
        weights, bias, activations, output, M, N, K, hasBias,                  \
        tgid, lid3.x, threads3.x, weightTile);                                 \
}

TUFF_MPP_BF16_MM_KERNEL(mpp_prefill_bf16_mm_f16, 64, 32)
TUFF_MPP_BF16_MM_KERNEL(mpp_prefill_bf16_mm_f16_m32, 32, 32)

#undef TUFF_MPP_BF16_MM_KERNEL

#if defined(TUFF_VISION_TENSOROPS)

kernel void mpp_vision_linear_bf16(
    device bfloat* activations [[buffer(0)]],
    device const bfloat* weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 lid3 [[thread_position_in_threadgroup]],
    uint3 threads3 [[threads_per_threadgroup]]) {
    constexpr int tileM = 64;
    constexpr int tileN = 32;
    constexpr int tileK = 64;
    constexpr auto descriptor = matmul2d_descriptor(
        tileM, tileN, tileK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_bfloat_tensor = tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_bfloat_tensor = tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;
    threadgroup bfloat weightTile[tileN * tileK];
    threadgroup_bfloat_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(tileK, tileN),
        array<int32_t, 2>({1, tileK}));
    // Extent is the K that remains, not a fixed tile: MPP clips loads to the
    // declared extents, and that clipping is the only bound. `down_proj` passes
    // K = 4304, which is 67 tiles plus 16, so a hardcoded 64 let the tail group
    // read past each row — currently landing inside an unexplained 64-element
    // pad in VisionScratch rather than crashing.
    device_bfloat_tensor firstA(
        activations,
        dextents<int32_t, 2>(min(int32_t(tileK), int32_t(K)), M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(0, int32_t(tgid.y) * tileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint group = 0; group < (K + tileK - 1u) / tileK; ++group) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid; linear < uint(tileN * tileK); linear += threads) {
            const uint localN = linear / uint(tileK);
            const uint localK = linear % uint(tileK);
            const uint globalN = tgid.x * uint(tileN) + localN;
            const uint globalK = group * uint(tileK) + localK;
            weightTile[linear] = globalN < N && globalK < K
                ? weights[globalN * K + globalK] : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_bfloat_tensor groupA(
            activations + group * uint(tileK),
            dextents<int32_t, 2>(
                min(int32_t(tileK), int32_t(K) - int32_t(group) * int32_t(tileK)), M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(0, int32_t(tgid.y) * tileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(tileN) + uint(position[0]);
        const uint globalM = tgid.y * uint(tileM) + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = bfloat(accumulator[element]);
        }
    }
}

#endif

constant constexpr int kMPPAffineApple10TileM = 64;
constant constexpr int kMPPAffineApple10TileN = 64;
constant constexpr int kMPPAffineApple10TileK = 128;

kernel void mpp_prefill_affine_threadgroup_f16_apple10_v1(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device half* activations            [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineApple10TileM, kMPPAffineApple10TileN,
        kMPPAffineApple10TileK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_half_tensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_half_tensor = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>;

    threadgroup half weightTile[
        kMPPAffineApple10TileN * kMPPAffineApple10TileK];
    threadgroup_half_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, kMPPAffineApple10TileN),
        array<int32_t, 2>({1, kMPPAffineApple10TileK}));
    device_half_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0, int32_t(tgid.y) * kMPPAffineApple10TileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint tileKStart = 0; tileKStart < K;
         tileKStart += uint(kMPPAffineApple10TileK)) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineApple10TileN * kMPPAffineApple10TileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineApple10TileK);
            const uint localK = linear % uint(kMPPAffineApple10TileK);
            const uint globalN = tgid.x * uint(kMPPAffineApple10TileN) + localN;
            if (globalN < N) {
                const uint globalK = tileKStart + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint affineGroup = globalK / kW4A8GroupSize;
                const float scale = float(scales[
                    globalN * groupsPerRow + affineGroup]);
                const float bias = float(biases[
                    globalN * groupsPerRow + affineGroup]);
                weightTile[linear] = half(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_half_tensor groupA(
            activations + tileKStart,
            dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0, int32_t(tgid.y) * kMPPAffineApple10TileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineApple10TileN)
            + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineApple10TileM)
            + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

kernel void mpp_prefill_affine_threadgroup_bf16_apple10_v1(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device bfloat* activations          [[buffer(3)]],
    device half* output                 [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineApple10TileM, kMPPAffineApple10TileN,
        kMPPAffineApple10TileK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_bfloat_tensor =
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_bfloat_tensor =
        tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;

    threadgroup bfloat weightTile[
        kMPPAffineApple10TileN * kMPPAffineApple10TileK];
    threadgroup_bfloat_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, kMPPAffineApple10TileN),
        array<int32_t, 2>({1, kMPPAffineApple10TileK}));
    device_bfloat_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0, int32_t(tgid.y) * kMPPAffineApple10TileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint tileKStart = 0; tileKStart < K;
         tileKStart += uint(kMPPAffineApple10TileK)) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineApple10TileN * kMPPAffineApple10TileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineApple10TileK);
            const uint localK = linear % uint(kMPPAffineApple10TileK);
            const uint globalN = tgid.x * uint(kMPPAffineApple10TileN) + localN;
            if (globalN < N) {
                const uint globalK = tileKStart + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint affineGroup = globalK / kW4A8GroupSize;
                const float scale = float(scales[
                    globalN * groupsPerRow + affineGroup]);
                const float bias = float(biases[
                    globalN * groupsPerRow + affineGroup]);
                weightTile[linear] = bfloat(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_bfloat_tensor groupA(
            activations + tileKStart,
            dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0, int32_t(tgid.y) * kMPPAffineApple10TileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineApple10TileN)
            + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineApple10TileM)
            + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = half(accumulator[element]);
        }
    }
}

// Unified Gemma's first image projection is followed by BF16 LayerNorm. Its
// raw accumulator legitimately exceeds FP16's 65,504 limit, so narrowing here
// produces infinities that poison whole image rows. Keep the same Apple10
// contraction but store BF16 for that stage.
kernel void mpp_prefill_affine_threadgroup_bf16_output_apple10_v1(
    device const uint8_t* packedWeights [[buffer(0)]],
    device const bfloat* scales         [[buffer(1)]],
    device const bfloat* biases         [[buffer(2)]],
    device bfloat* activations          [[buffer(3)]],
    device bfloat* output               [[buffer(4)]],
    constant uint& M                    [[buffer(5)]],
    constant uint& N                    [[buffer(6)]],
    constant uint& K                    [[buffer(7)]],
    uint3 tgid                          [[threadgroup_position_in_grid]],
    uint3 lid3                          [[thread_position_in_threadgroup]],
    uint3 threads3                      [[threads_per_threadgroup]]) {
    constexpr auto descriptor = matmul2d_descriptor(
        kMPPAffineApple10TileM, kMPPAffineApple10TileN,
        kMPPAffineApple10TileK, false, true, false);
    matmul2d<descriptor, execution_simdgroups<4>> operation;

    using device_bfloat_tensor =
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_bfloat_tensor =
        tensor<threadgroup bfloat, dextents<int32_t, 2>, tensor_inline>;

    threadgroup bfloat weightTile[
        kMPPAffineApple10TileN * kMPPAffineApple10TileK];
    threadgroup_bfloat_tensor tileB(
        weightTile,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, kMPPAffineApple10TileN),
        array<int32_t, 2>({1, kMPPAffineApple10TileK}));
    device_bfloat_tensor firstA(
        activations,
        dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
        array<int32_t, 2>({1, int32_t(K)}));
    auto firstTileA = firstA.slice(
        0, int32_t(tgid.y) * kMPPAffineApple10TileM);
    auto accumulator = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    auto groupProduct = operation.get_destination_cooperative_tensor<
        decltype(firstTileA), decltype(tileB), float>();
    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        accumulator[element] = 0.0f;
    }

    const uint rowBytes = K / 2u;
    const uint groupsPerRow = K / kW4A8GroupSize;
    const uint lid = lid3.x;
    const uint threads = threads3.x;
    for (uint tileKStart = 0; tileKStart < K;
         tileKStart += uint(kMPPAffineApple10TileK)) {
        for (int element = 0; element < groupProduct.get_capacity(); ++element) {
            groupProduct[element] = 0.0f;
        }
        for (uint linear = lid;
             linear < uint(kMPPAffineApple10TileN * kMPPAffineApple10TileK);
             linear += threads) {
            const uint localN = linear / uint(kMPPAffineApple10TileK);
            const uint localK = linear % uint(kMPPAffineApple10TileK);
            const uint globalN = tgid.x * uint(kMPPAffineApple10TileN) + localN;
            if (globalN < N) {
                const uint globalK = tileKStart + localK;
                const uint8_t packed = packedWeights[
                    globalN * rowBytes + (globalK >> 1)];
                const uint q = (globalK & 1u) == 0u
                    ? uint(packed & 0x0fu)
                    : uint(packed >> 4);
                const uint affineGroup = globalK / kW4A8GroupSize;
                const float scale = float(scales[
                    globalN * groupsPerRow + affineGroup]);
                const float bias = float(biases[
                    globalN * groupsPerRow + affineGroup]);
                weightTile[linear] = bfloat(fma(float(q), scale, bias));
            } else {
                weightTile[linear] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device_bfloat_tensor groupA(
            activations + tileKStart,
            dextents<int32_t, 2>(kMPPAffineApple10TileK, M),
            array<int32_t, 2>({1, int32_t(K)}));
        auto tileA = groupA.slice(
            0, int32_t(tgid.y) * kMPPAffineApple10TileM);
        operation.run(tileA, tileB, groupProduct);
        for (int element = 0; element < accumulator.get_capacity(); ++element) {
            accumulator[element] += groupProduct[element];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0; element < accumulator.get_capacity(); ++element) {
        if (!accumulator.is_valid_element(element)) continue;
        const auto position = accumulator.get_multidimensional_index(element);
        const uint globalN = tgid.x * uint(kMPPAffineApple10TileN)
            + uint(position[0]);
        const uint globalM = tgid.y * uint(kMPPAffineApple10TileM)
            + uint(position[1]);
        if (globalM < M && globalN < N) {
            output[globalM * N + globalN] = bfloat(accumulator[element]);
        }
    }
}

#if defined(TUFF_VISION_TENSOROPS)

constant constexpr int kVisionAttentionQueries = 16;
constant constexpr int kVisionAttentionKeys = 64;
constant constexpr int kVisionAttentionHeadDim = 80;
// The QK contraction needs a K that is a multiple of 16, so it keeps 80. The PV
// matmul carries the head dimension as N, which only needs a multiple of 8, so
// it contracts over the 72 real lanes; lanes 72 to 79 hold zero V and their
// outputs were discarded anyway.
constant constexpr int kVisionAttentionRealHeadDim = 72;

/// `inputRowStride` is the per-head row count of the padded Q/K/V buffers, which
/// may exceed `sequenceLength` when the projections write every padded GEMM row.
/// `unpaddedOutputHeadDim` of 0 keeps the padded head-major output store; a
/// nonzero value stores row-major `(row * numHeads + head) * headDim + lane`
/// directly, which removes the separate unpad dispatch.
struct VisionAttentionLayout {
    uint inputRowStride;
    uint unpaddedOutputHeadDim;
    /// Softmax phase: 0 runs the original 16-thread serial phase; 1 spreads
    /// max, exp and sum across all 128 threads, which makes the tile sum a tree
    /// reduction and is therefore not bit-identical; 2 spreads max and exp but
    /// keeps the tile sum sequential per query, which is bit-identical to 0.
    uint parallelSoftmax;
    float scoreScale;
};

/// `PVOutputDim` is how many head lanes the PV matmul produces. Threadgroup
/// memory is passed in because MSL forbids declaring it in a non-kernel function.
template <int PVOutputDim>
inline void mpp_vision_attention_bf16_impl(
    device bfloat* Q, device bfloat* K, device bfloat* V, device bfloat* O,
    constant uint& sequenceLength,
    constant uint& numHeads,
    constant VisionAttentionLayout& layout,
    uint3 tg, uint lid, ushort simdLane, ushort simdIndex,
    threadgroup float* scoreTile,
    threadgroup float* weightTile,
    threadgroup float* rowMax,
    threadgroup float* rowSum,
    threadgroup float* oldScale,
    threadgroup float* nextMaxTile) {
    constexpr auto qkDescriptor = matmul2d_descriptor(
        kVisionAttentionQueries,
        kVisionAttentionKeys,
        kVisionAttentionHeadDim,
        false, true, false);
    constexpr auto pvDescriptor = matmul2d_descriptor(
        kVisionAttentionQueries,
        PVOutputDim,
        kVisionAttentionKeys,
        false, false, false);
    matmul2d<qkDescriptor, execution_simdgroups<4>> qkOperation;
    matmul2d<pvDescriptor, execution_simdgroups<4>> pvOperation;

    using device_bfloat_tensor =
        tensor<device bfloat, dextents<int32_t, 2>, tensor_inline>;
    using threadgroup_float_tensor =
        tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

    const uint queryStart = tg.x * uint(kVisionAttentionQueries);
    const uint head = tg.y;
    if (queryStart >= sequenceLength || head >= numHeads) return;
    const uint rowStride = uint(kVisionAttentionHeadDim);
    const uint validQueries = min(
        uint(kVisionAttentionQueries), sequenceLength - queryStart);

    const uint inputRows = layout.inputRowStride == 0u
        ? sequenceLength : layout.inputRowStride;
    const uint headBase = head * inputRows * uint(kVisionAttentionHeadDim);
    device_bfloat_tensor queryTensor(
        Q + headBase,
        dextents<int32_t, 2>(kVisionAttentionHeadDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    device_bfloat_tensor keyTensor(
        K + headBase,
        dextents<int32_t, 2>(kVisionAttentionHeadDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    device_bfloat_tensor valueTensor(
        V + headBase,
        dextents<int32_t, 2>(PVOutputDim, int32_t(sequenceLength)),
        array<int32_t, 2>({1, int32_t(rowStride)}));
    threadgroup_float_tensor weights(
        weightTile,
        dextents<int32_t, 2>(
            kVisionAttentionKeys, kVisionAttentionQueries),
        array<int32_t, 2>({1, kVisionAttentionKeys}));

    auto querySlice = queryTensor.slice(0, int32_t(queryStart));
    auto firstValueSlice = valueTensor.slice(0, 0);
    auto outputAccumulator =
        pvOperation.template get_destination_cooperative_tensor<
            decltype(weights), decltype(firstValueSlice), float>();
    for (int element = 0;
         element < outputAccumulator.get_capacity();
         ++element) {
        if (outputAccumulator.is_valid_element(element)) {
            outputAccumulator[element] = 0.0f;
        }
    }
    if (lid < uint(kVisionAttentionQueries)) {
        rowMax[lid] = -INFINITY;
        rowSum[lid] = 0.0f;
        oldScale[lid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint keyStart = 0; keyStart < sequenceLength;
         keyStart += uint(kVisionAttentionKeys)) {
        auto keySlice = keyTensor.slice(0, int32_t(keyStart));
        auto scores = qkOperation.template get_destination_cooperative_tensor<
            decltype(querySlice), decltype(keySlice), float>();
        for (int element = 0; element < scores.get_capacity(); ++element) {
            if (scores.is_valid_element(element)) scores[element] = 0.0f;
        }
        qkOperation.run(querySlice, keySlice, scores);
        for (int element = 0; element < scores.get_capacity(); ++element) {
            if (!scores.is_valid_element(element)) continue;
            const auto position = scores.get_multidimensional_index(element);
            scoreTile[
                uint(position[1]) * uint(kVisionAttentionKeys)
                + uint(position[0])] = scores[element] * layout.scoreScale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint visible = min(
            uint(kVisionAttentionKeys), sequenceLength - keyStart);
        if (layout.parallelSoftmax != 0u) {
            // Eight lanes per query, eight consecutive keys each. The lane groups
            // sit inside one simdgroup, so an XOR butterfly over masks 1, 2 and 4
            // reduces within a group without touching its neighbours.
            constexpr uint keysPerLane =
                uint(kVisionAttentionKeys) / 8u;
            const uint lane = uint(simdLane);
            const uint query = uint(simdIndex) * 4u + lane / 8u;
            const uint keyBase = (lane % 8u) * keysPerLane;
            const uint row = query * uint(kVisionAttentionKeys);

            float tileMax = -INFINITY;
            if (query < validQueries) {
                for (uint offset = 0; offset < keysPerLane; ++offset) {
                    const uint key = keyBase + offset;
                    if (key < visible) tileMax = max(tileMax, scoreTile[row + key]);
                }
            }
            for (uint mask = 1u; mask < 8u; mask <<= 1) {
                tileMax = max(tileMax, simd_shuffle_xor(tileMax, ushort(mask)));
            }
            const float previousMax = rowMax[query];
            const float previousSum = rowSum[query];
            const float nextMax = max(previousMax, tileMax);
            const float scale = previousSum > 0.0f
                ? fast::exp(previousMax - nextMax)
                : 0.0f;
            // Mode 2 hands the running max to the serial summation threads, which
            // own different queries than the lanes that computed it.
            if (layout.parallelSoftmax == 2u && lane % 8u == 0u) {
                nextMaxTile[query] = nextMax;
            }
            float tileSum = 0.0f;
            for (uint offset = 0; offset < keysPerLane; ++offset) {
                const uint key = keyBase + offset;
                const float weight = query < validQueries && key < visible
                    ? fast::exp(scoreTile[row + key] - nextMax)
                    : 0.0f;
                weightTile[row + key] = weight;
                tileSum += weight;
            }
            if (layout.parallelSoftmax == 1u) {
                for (uint mask = 1u; mask < 8u; mask <<= 1) {
                    tileSum += simd_shuffle_xor(tileSum, ushort(mask));
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (lane % 8u == 0u) {
                    oldScale[query] = scale;
                    rowSum[query] = previousSum * scale + tileSum;
                    rowMax[query] = nextMax;
                }
            } else {
                // Mode 2: the weights are already written, so re-sum them in the
                // original key order. Identical addends in identical order keep
                // this bit-identical to the serial phase.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lid < uint(kVisionAttentionQueries)) {
                    const uint serialRow = lid * uint(kVisionAttentionKeys);
                    float serialSum = 0.0f;
                    for (uint key = 0; key < uint(kVisionAttentionKeys); ++key) {
                        serialSum += weightTile[serialRow + key];
                    }
                    oldScale[lid] = rowSum[lid] > 0.0f
                        ? fast::exp(rowMax[lid] - nextMaxTile[lid])
                        : 0.0f;
                    rowSum[lid] = rowSum[lid] * oldScale[lid] + serialSum;
                    rowMax[lid] = nextMaxTile[lid];
                }
            }
        } else if (lid < uint(kVisionAttentionQueries)) {
            const uint query = lid;
            float tileMax = -INFINITY;
            if (query < validQueries) {
                for (uint key = 0; key < visible; ++key) {
                    tileMax = max(
                        tileMax,
                        scoreTile[query * uint(kVisionAttentionKeys) + key]);
                }
            }
            const float nextMax = max(rowMax[query], tileMax);
            const float scale = rowSum[query] > 0.0f
                ? fast::exp(rowMax[query] - nextMax)
                : 0.0f;
            float tileSum = 0.0f;
            for (uint key = 0; key < uint(kVisionAttentionKeys); ++key) {
                const float weight = query < validQueries && key < visible
                    ? fast::exp(
                        scoreTile[query * uint(kVisionAttentionKeys) + key]
                        - nextMax)
                    : 0.0f;
                weightTile[query * uint(kVisionAttentionKeys) + key] = weight;
                tileSum += weight;
            }
            oldScale[query] = scale;
            rowSum[query] = rowSum[query] * scale + tileSum;
            rowMax[query] = nextMax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto valueSlice = valueTensor.slice(0, int32_t(keyStart));
        auto outputProduct = pvOperation.template get_destination_cooperative_tensor<
            decltype(weights), decltype(valueSlice), float>();
        for (int element = 0;
             element < outputProduct.get_capacity();
             ++element) {
            if (outputProduct.is_valid_element(element)) {
                outputProduct[element] = 0.0f;
            }
        }
        pvOperation.run(weights, valueSlice, outputProduct);
        for (int element = 0;
             element < outputAccumulator.get_capacity();
             ++element) {
            if (!outputAccumulator.is_valid_element(element)
                || !outputProduct.is_valid_element(element)) continue;
            const auto position =
                outputAccumulator.get_multidimensional_index(element);
            const uint query = uint(position[1]);
            outputAccumulator[element] = fma(
                1.0f,
                outputProduct[element],
                outputAccumulator[element] * oldScale[query]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int element = 0;
         element < outputAccumulator.get_capacity();
         ++element) {
        if (!outputAccumulator.is_valid_element(element)) continue;
        const auto position =
            outputAccumulator.get_multidimensional_index(element);
        const uint dimension = uint(position[0]);
        const uint query = uint(position[1]);
        if (query >= validQueries) continue;
        const uint outputHeadDim = layout.unpaddedOutputHeadDim;
        if (outputHeadDim == 0u) {
            if (dimension < uint(kVisionAttentionHeadDim)) {
                O[(head * sequenceLength + queryStart + query) * rowStride
                  + dimension] =
                    bfloat(outputAccumulator[element] / rowSum[query]);
            }
        } else if (dimension < outputHeadDim) {
            O[(queryStart + query) * numHeads * outputHeadDim
              + head * outputHeadDim + dimension] =
                bfloat(outputAccumulator[element] / rowSum[query]);
        }
    }
}

kernel void mpp_vision_attention_bf16(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant VisionAttentionLayout& layout [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    ushort simdLane [[thread_index_in_simdgroup]],
    ushort simdIndex [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scoreTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float weightTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float rowMax[kVisionAttentionQueries];
    threadgroup float rowSum[kVisionAttentionQueries];
    threadgroup float oldScale[kVisionAttentionQueries];
    threadgroup float nextMaxTile[kVisionAttentionQueries];
    mpp_vision_attention_bf16_impl<kVisionAttentionRealHeadDim>(
        Q, K, V, O, sequenceLength, numHeads, layout, tg, lid, simdLane,
        simdIndex, scoreTile, weightTile, rowMax, rowSum, oldScale, nextMaxTile);
}

kernel void mpp_vision_attention_bf16_pv80(
    device bfloat* Q [[buffer(0)]],
    device bfloat* K [[buffer(1)]],
    device bfloat* V [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    constant uint& sequenceLength [[buffer(4)]],
    constant uint& numHeads [[buffer(5)]],
    constant VisionAttentionLayout& layout [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    ushort simdLane [[thread_index_in_simdgroup]],
    ushort simdIndex [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scoreTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float weightTile[
        kVisionAttentionQueries * kVisionAttentionKeys];
    threadgroup float rowMax[kVisionAttentionQueries];
    threadgroup float rowSum[kVisionAttentionQueries];
    threadgroup float oldScale[kVisionAttentionQueries];
    threadgroup float nextMaxTile[kVisionAttentionQueries];
    mpp_vision_attention_bf16_impl<kVisionAttentionHeadDim>(
        Q, K, V, O, sequenceLength, numHeads, layout, tg, lid, simdLane,
        simdIndex, scoreTile, weightTile, rowMax, rowSum, oldScale, nextMaxTile);
}

#endif

#endif // __HAVE_TENSOR__
