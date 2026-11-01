#include <metal_stdlib>
using namespace metal;

// Flint keeps its router in BF16. One SIMD group computes a row, accumulating
// in FP32, as the reference does before the sigmoid and top-k selection.
kernel void flint_router_bf16(
    device const bfloat* weight [[buffer(0)]],
    device const half* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    uint expert [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (expert >= experts) return;
    float sum = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32u) {
        sum = fma(float(weight[expert * hidden_size + i]), float(hidden[i]), sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) logits[expert] = sum;
}

// Grid x selects the expert and grid y the row, so a prefill chunk routes all
// of its rows in one dispatch; decode dispatches a single row.
kernel void flint_router_bf16_float(
    device const bfloat* weight [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg.x;
    if (expert >= experts) return;
    device const float* x = hidden + tg.y * hidden_size;
    float sum = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32u)
        sum = fma(float(weight[expert * hidden_size + i]), x[i], sum);
    sum = simd_sum(sum);
    if (lane == 0u) logits[tg.y * experts + expert] = sum;
}

// OptiQ stores the router as group-64 affine bytes. Its input remains FP32,
// matching the unquantized Flint router and avoiding an extra half rounding.
kernel void flint_router_int8_float(
    device const uchar* weight [[buffer(0)]],
    device const float* hidden [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    device const bfloat* scales [[buffer(5)]],
    device const bfloat* biases [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint expert = tg.x;
    if (expert >= experts) return;
    device const float* x = hidden + tg.y * hidden_size;
    float sum = 0.0f;
    for (uint i = lane; i < hidden_size; i += 32u) {
        uint group = expert * (hidden_size / 64u) + i / 64u;
        float w = fma(float(scales[group]), float(weight[expert * hidden_size + i]),
                      float(biases[group]));
        sum = fma(w, x[i], sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) logits[tg.y * experts + expert] = sum;
}

// FP32 residual. The half mirror exists only for decode-time tracing; prefill
// rows leave it out (FC_FLINT_MIRROR false).
constant bool FC_FLINT_MIRROR [[function_constant(120)]];

kernel void flint_residual_add(device float* residual [[buffer(0)]],
                               device const half* delta [[buffer(1)]],
                               device half* mirror [[buffer(2)]],
                               constant uint& count [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float value = residual[i] + float(delta[i]);
    residual[i] = value;
    if (FC_FLINT_MIRROR) mirror[i] = half(value);
}

// One threadgroup per row: residual += delta, then RMS-normalize into a half
// row and an FP32 row. Bit-identical to flint_residual_add followed by
// flint_residual_norm: every thread sums the same strided elements in the same
// order, and re-reads only values it wrote itself.
kernel void flint_residual_add_norm(
    device float* residual [[buffer(0)]],
    device const half* delta [[buffer(1)]],
    device half* mirror [[buffer(2)]],
    device const bfloat* weight [[buffer(3)]],
    device half* output [[buffer(4)]],
    device float* wide_output [[buffer(5)]],
    constant uint& count [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    const uint base = row * count;
    threadgroup float partial[8];
    float sum = 0.0f;
    for (uint i = tid; i < count; i += 256u) {
        const float value = residual[base + i] + float(delta[base + i]);
        residual[base + i] = value;
        if (FC_FLINT_MIRROR) mirror[base + i] = half(value);
        sum = fma(value, value, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0u) partial[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        float total = simd_sum(lane < 8u ? partial[lane] : 0.0f);
        if (lane == 0u) partial[0] = rsqrt(total / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < count; i += 256u) {
        const float value = residual[base + i] * partial[0] * float(weight[i]);
        output[base + i] = half(value);
        wide_output[base + i] = value;
    }
}

kernel void flint_residual_norm(
    device const float* residual [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device half* output [[buffer(2)]],
    device float* wide_output [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint group [[simdgroup_index_in_threadgroup]]) {
    const uint base = row * count;
    threadgroup float partial[8];
    float sum = 0.0f;
    for (uint i = tid; i < count; i += 256u)
        sum = fma(residual[base + i], residual[base + i], sum);
    sum = simd_sum(sum);
    if (lane == 0u) partial[group] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group == 0u) {
        float total = simd_sum(lane < 8u ? partial[lane] : 0.0f);
        if (lane == 0u) partial[0] = rsqrt(total / float(count) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < count; i += 256u) {
        float value = residual[base + i] * partial[0] * float(weight[i]);
        output[base + i] = half(value);
        wide_output[base + i] = value;
    }
}

// The router uses sigmoid scores, selects k without a correction bias, then
// normalizes just those probabilities. A softmax over the top logits is not
// equivalent. Lower index wins exact ties for deterministic inference.
//
// One SIMD group per row: each lane owns experts lane, lane+32, ... and the
// group extracts the k best by repeated (score, lowest index) reduction, which
// yields the same ordered set the sequential insertion search produced.
kernel void flint_router_select(
    device const float* logits [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    device half* weights [[buffer(2)]],
    constant uint& experts [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (k == 0u || k > 8u || k > experts || experts > 512u) return;
    logits += row * experts;
    indices += row * k;
    weights += row * k;
    float local[16];
    for (uint j = 0; j < 16u; ++j) {
        const uint e = lane + 32u * j;
        local[j] = e < experts ? 1.0f / (1.0f + exp(-logits[e])) : -INFINITY;
    }
    uint taken = 0u;
    float chosen[8];
    uint ids[8];
    for (uint r = 0; r < k; ++r) {
        float best = -INFINITY;
        uint best_index = 0xFFFFFFFFu;
        for (uint j = 0; j < 16u; ++j) {
            const uint e = lane + 32u * j;
            if (e < experts && ((taken >> j) & 1u) == 0u && local[j] > best) {
                best = local[j];
                best_index = e;
            }
        }
        const float top = simd_max(best);
        uint candidate = (best == top) ? best_index : 0xFFFFFFFFu;
        uint winner = simd_min(candidate);
        if (winner == 0xFFFFFFFFu) winner = 0u;
        chosen[r] = top;
        ids[r] = winner;
        if ((winner & 31u) == lane) taken |= 1u << (winner >> 5u);
    }
    if (lane == 0u) {
        float sum = 0.0f;
        for (uint i = 0; i < k; ++i) sum += chosen[i];
        for (uint i = 0; i < k; ++i) {
            indices[i] = ids[i];
            weights[i] = half(sum > 0.0f ? chosen[i] / sum : 0.0f);
        }
    }
}

// Flint's token embedding is 8-bit affine, unlike its 4-bit LM head.
kernel void flint_embed_int8(
    device const uchar* table [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& token [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    device float* residual [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= width) return;
    const uint group = token * (width / 64u) + i / 64u;
    float value = float(table[token * width + i]) * float(scales[group]) + float(biases[group]);
    residual[i] = value;
    output[i] = half(value);
}

// Set a single shared flag if any output logit cannot be sampled safely.
kernel void flint_check_finite_logits(
    device const half* logits [[buffer(0)]],
    device atomic_uint* invalid [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count && !isfinite(float(logits[index]))) {
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
    }
}

// Prefill rows: embed a chunk of tokens into the FP32 residual.
kernel void flint_embed_int8_rows(
    device const uchar* table [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const uint* tokens [[buffer(3)]],
    device float* residual [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= width) return;
    const uint token = tokens[gid.y];
    const uint group = token * (width / 64u) + gid.x / 64u;
    residual[gid.y * width + gid.x] =
        float(table[token * width + gid.x]) * float(scales[group]) + float(biases[group]);
}

// ---------------------------------------------------------------------------
// Batched routed experts for prefill. A tile binds up to 16 experts; every
// (token, expert) pair of the tile is one independent row computation using the
// decode kernels' helpers, so a pair's numerics match one decode token exactly.
// Partials land at [token][rank][D] and are reduced in rank order afterwards,
// which is the decode kernel's accumulation order.
// ---------------------------------------------------------------------------
struct FlintPairMSL {
    uint token;
    uint expert;
    uint rank;
    uint weight_bits;
};

struct FlintTileBlobs {
    device const uint8_t* blob[16];
};

struct FlintTileParams {
    uint pair_start;
    uint pair_count;
    uint D;
    uint F;
    uint top_k;
    uint live;
    uint local_expert[16];
};

static inline uint flint_tile_slot(constant FlintTileParams& p, uint expert) {
    for (uint i = 0; i < p.live; ++i) {
        if (p.local_expert[i] == expert) return i;
    }
    return 0xFFFFFFFFu;
}

kernel void flint_moe_pairs_phase1(
    device const FlintTileBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device float* acts [[buffer(3)]],
    device const FlintPairMSL* pairs [[buffer(4)]],
    constant FlintTileParams& p [[buffer(5)]],
    constant ExpertBitWidths& widths [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= p.pair_count * p.F) return;
    const uint pair_index = row / p.F;
    const uint f = row % p.F;
    const FlintPairMSL pair = pairs[p.pair_start + pair_index];
    const uint slot = flint_tile_slot(p, pair.expert);
    if (slot == 0xFFFFFFFFu) return;
    device const uint8_t* base = routed.blob[slot];
    const float2 gu = moe_gate_up_rows(
        base + offsets.gate_W_off, (device const bfloat*)(base + offsets.gate_s_off),
        (device const bfloat*)(base + offsets.gate_b_off),
        base + offsets.up_W_off, (device const bfloat*)(base + offsets.up_s_off),
        (device const bfloat*)(base + offsets.up_b_off),
        x + pair.token * p.D, f, p.D, lane, widths);
    if (lane == 0u) acts[pair_index * p.F + f] = moe_hidden_activation(gu.x) * gu.y;
}

kernel void flint_moe_pairs_phase2(
    device const FlintTileBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& offsets [[buffer(1)]],
    device const float* acts [[buffer(2)]],
    device const FlintPairMSL* pairs [[buffer(3)]],
    constant FlintTileParams& p [[buffer(4)]],
    device float* partials [[buffer(5)]],
    constant ExpertBitWidths& widths [[buffer(6)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg * 8u + sg;
    if (row >= p.pair_count * p.D) return;
    const uint pair_index = row / p.D;
    const uint d = row % p.D;
    const FlintPairMSL pair = pairs[p.pair_start + pair_index];
    const uint slot = flint_tile_slot(p, pair.expert);
    if (slot == 0xFFFFFFFFu) return;
    device const uint8_t* base = routed.blob[slot];
    device const bfloat* scales = (device const bfloat*)(base + offsets.down_s_off);
    device const bfloat* biases = (device const bfloat*)(base + offsets.down_b_off);
    device const float* act = acts + pair_index * p.F;
    const float value = widths.down == 4u
        ? flint_down_row(base + offsets.down_W_off, scales, biases, act, d, p.F, lane)
        : moe_affine_row(base + offsets.down_W_off, scales, biases, act, d, p.F, lane, widths.down);
    if (lane == 0u) {
        const half weight = as_type<half>(ushort(pair.weight_bits & 0xFFFFu));
        partials[(pair.token * p.top_k + pair.rank) * p.D + d] = float(weight) * value;
    }
}

// y[row] = half(shared[row] + sum over ranks of partials), in rank order.
kernel void flint_moe_reduce_rows(
    device const float* partials [[buffer(0)]],
    device const half* shared_out [[buffer(1)]],
    device half* y [[buffer(2)]],
    constant uint& D [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= D) return;
    float acc = float(shared_out[gid.y * D + gid.x]);
    for (uint k = 0; k < top_k; ++k) acc += partials[(gid.y * top_k + k) * D + gid.x];
    y[gid.y * D + gid.x] = half(acc);
}
