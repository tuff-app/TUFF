#include <metal_stdlib>
using namespace metal;

// ============================================================================
// inkling — kernels for Inkling 2's looped decoder.
//
// Activations are FP32 rows of `d` values. A kernel that takes `rows` works on
// that many token rows at once, so decode (one row) and chunked prefill share
// one implementation. Reductions over a row use one threadgroup of
// kInkRowThreads threads per row.
// ============================================================================

constant constexpr uint kInkRowThreads = 256;
constant constexpr uint kInkRowSimdGroups = kInkRowThreads / 32;
constant constexpr uint kInkMaxDepth = 8;
constant constexpr uint kInkMaxQueries = 4;

// E2M1 (FP4) codes, sign in bit 3.
static inline float ink_e2m1(uint code) {
    const float magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float value = magnitude[code & 7u];
    return (code & 8u) != 0u ? -value : value;
}

// E4M3 (FP8): sign, four exponent bits biased by 7, three mantissa bits. No
// infinities; 0x7f and 0xff are NaN and never stored. Built from bits so a
// row of codes costs no exp2 calls.
static inline float ink_e4m3(uint code) {
    const uint exponent = (code >> 3) & 15u;
    const uint mantissa = code & 7u;
    const uint sign = (code & 128u) << 24;
    if (exponent == 0u) {
        const float magnitude = float(mantissa) * 0.001953125f;
        return sign != 0u ? -magnitude : magnitude;
    }
    return as_type<float>(sign | ((exponent + 120u) << 23) | (mantissa << 20));
}

// tanh for any input. The fast-math tanh divides two exponentials and
// returns NaN once they overflow, past about 44; it is 1 to the last bit
// long before that.
static inline float ink_tanh(float x) {
    return tanh(clamp(x, -16.0f, 16.0f));
}

// One expert unit: softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap).
static inline float ink_expert_unit(float gate, float up, float gate_cap, float up_cap) {
    return ink_tanh(gate / gate_cap) * gate_cap / (1.0f + exp(-gate)) * ink_tanh(up / up_cap) * up_cap;
}

// 2^(code - 127): the power-of-two block scale of MXFP8.
static inline float ink_e8m0(uint code) {
    return as_type<float>(max(code, 1u) << 23);
}

// ---------------------------------------------------------------------------
// RMS norm of each row, with an optional BF16 gain: y = x * rsqrt(mean(x^2) +
// eps) * weight. Input rows are `x_stride` floats apart and output rows `d`.
// `x` and `out` may be the same buffer when the strides agree.
// ---------------------------------------------------------------------------
kernel void ink_rms_norm(
    device const float* x [[buffer(0)]],
    device const bfloat* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    constant uint& has_weight [[buffer(5)]],
    constant uint& x_stride [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[kInkRowSimdGroups];
    device const float* source = x + size_t(row) * x_stride;
    float acc = 0.0f;
    for (uint i = tid; i < d; i += kInkRowThreads) acc = fma(source[i], source[i], acc);
    acc = simd_sum(acc);
    if (lane == 0u) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0u; s < kInkRowSimdGroups; ++s) total += partial[s];
    const float inverse = rsqrt(total / float(d) + eps);
    device float* target = out + size_t(row) * d;
    for (uint i = tid; i < d; i += kInkRowThreads) {
        float value = source[i] * inverse;
        if (has_weight != 0u) value *= float(weight[i]);
        target[i] = value;
    }
}

// ---------------------------------------------------------------------------
// Depth attention: learned queries attend over the block states of a token
// and its current state. `blocks` holds `depth` slabs of `rows` rows, each row
// already unit RMS, and `partial` holds the current rows, also unit RMS.
// out[(row * queries + q) * d + i] = sum_n softmax_n(q . s_n * scale) s_n[i].
// ---------------------------------------------------------------------------
kernel void ink_depth_attention(
    device const float* blocks [[buffer(0)]],
    device const float* partial_rows [[buffer(1)]],
    device const float* query [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& d [[buffer(4)]],
    constant uint& depth [[buffer(5)]],
    constant uint& queries [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[kInkMaxDepth * kInkMaxQueries][kInkRowSimdGroups];
    const uint sources = depth + 1u;
    float acc[kInkMaxDepth * kInkMaxQueries];
    for (uint i = 0u; i < sources * queries; ++i) acc[i] = 0.0f;
    device const float* current = partial_rows + size_t(row) * d;
    for (uint i = tid; i < d; i += kInkRowThreads) {
        for (uint n = 0u; n < sources; ++n) {
            const float value = n < depth
                ? blocks[(size_t(n) * rows + row) * d + i] : current[i];
            for (uint q = 0u; q < queries; ++q)
                acc[n * queries + q] = fma(value, query[q * d + i], acc[n * queries + q]);
        }
    }
    for (uint i = 0u; i < sources * queries; ++i) {
        const float sum = simd_sum(acc[i]);
        if (lane == 0u) partial[i][sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float scale = rsqrt(float(d));
    float weight[kInkMaxDepth * kInkMaxQueries];
    for (uint i = 0u; i < sources * queries; ++i) {
        float total = 0.0f;
        for (uint s = 0u; s < kInkRowSimdGroups; ++s) total += partial[i][s];
        weight[i] = total * scale;
    }
    for (uint q = 0u; q < queries; ++q) {
        float peak = -INFINITY;
        for (uint n = 0u; n < sources; ++n) peak = max(peak, weight[n * queries + q]);
        float total = 0.0f;
        for (uint n = 0u; n < sources; ++n) {
            weight[n * queries + q] = exp(weight[n * queries + q] - peak);
            total += weight[n * queries + q];
        }
        for (uint n = 0u; n < sources; ++n) weight[n * queries + q] /= total;
    }
    for (uint i = tid; i < d; i += kInkRowThreads) {
        float mixed[kInkMaxQueries];
        for (uint q = 0u; q < queries; ++q) mixed[q] = 0.0f;
        for (uint n = 0u; n < sources; ++n) {
            const float value = n < depth
                ? blocks[(size_t(n) * rows + row) * d + i] : current[i];
            for (uint q = 0u; q < queries; ++q)
                mixed[q] = fma(weight[n * queries + q], value, mixed[q]);
        }
        for (uint q = 0u; q < queries; ++q)
            out[(size_t(row) * queries + q) * d + i] = mixed[q];
    }
}

// ---------------------------------------------------------------------------
// Message gate, in three steps.
//   1. ink_gate_scales: 1 / rms of the state row and of its message row.
//   2. ink_gate_down:   g = down . [state / rms, message / rms]   (rank values)
//   3. ink_gate_mix:    out = (1 - tanh(F g)) state + tanh(G g) message
// `state` rows are `state_stride` floats apart and `message` rows
// `message_stride`, so a row inside a wider buffer needs no copy.
// ---------------------------------------------------------------------------
kernel void ink_gate_scales(
    device const float* state [[buffer(0)]],
    device const float* message [[buffer(1)]],
    device float* scales [[buffer(2)]],
    constant uint& d [[buffer(3)]],
    constant uint& state_stride [[buffer(4)]],
    constant uint& message_stride [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[2][kInkRowSimdGroups];
    device const float* s = state + size_t(row) * state_stride;
    device const float* m = message + size_t(row) * message_stride;
    float a = 0.0f, b = 0.0f;
    for (uint i = tid; i < d; i += kInkRowThreads) {
        a = fma(s[i], s[i], a);
        b = fma(m[i], m[i], b);
    }
    a = simd_sum(a);
    b = simd_sum(b);
    if (lane == 0u) { partial[0][sg] = a; partial[1][sg] = b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float ta = 0.0f, tb = 0.0f;
        for (uint i = 0u; i < kInkRowSimdGroups; ++i) { ta += partial[0][i]; tb += partial[1][i]; }
        scales[row * 2u] = rsqrt(ta / float(d) + eps);
        scales[row * 2u + 1u] = rsqrt(tb / float(d) + eps);
    }
}

kernel void ink_gate_down(
    device const bfloat* down [[buffer(0)]],
    device const float* state [[buffer(1)]],
    device const float* message [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device float* low [[buffer(4)]],
    constant uint& d [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    constant uint& state_stride [[buffer(7)]],
    constant uint& message_stride [[buffer(8)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x, row = tg.y;
    device const float* s = state + size_t(row) * state_stride;
    device const float* m = message + size_t(row) * message_stride;
    device const bfloat* w = down + size_t(r) * 2u * d;
    float a = 0.0f, b = 0.0f;
    for (uint i = lane; i < d; i += 32u) {
        a = fma(float(w[i]), s[i], a);
        b = fma(float(w[d + i]), m[i], b);
    }
    const float total = simd_sum(a) * scales[row * 2u] + simd_sum(b) * scales[row * 2u + 1u];
    if (lane == 0u) low[row * rank + r] = total;
}

kernel void ink_gate_mix(
    device const float* forget [[buffer(0)]],
    device const float* gain [[buffer(1)]],
    device const float* low [[buffer(2)]],
    device const float* state [[buffer(3)]],
    device const float* message [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& d [[buffer(6)]],
    constant uint& rank [[buffer(7)]],
    constant uint& state_stride [[buffer(8)]],
    constant uint& message_stride [[buffer(9)]],
    constant uint& out_stride [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, row = gid.y;
    if (i >= d) return;
    device const float* g = low + size_t(row) * rank;
    float f = 0.0f, m = 0.0f;
    for (uint r = 0u; r < rank; ++r) {
        f = fma(forget[size_t(i) * rank + r], g[r], f);
        m = fma(gain[size_t(i) * rank + r], g[r], m);
    }
    out[size_t(row) * out_stride + i] =
        (1.0f - ink_tanh(f)) * state[size_t(row) * state_stride + i]
        + ink_tanh(m) * message[size_t(row) * message_stride + i];
}

// ---------------------------------------------------------------------------
// Affine 4-bit GEMV, group 64: y[row] = sum_g scale_g (codes_g . x_g) +
// bias_g sum(x_g). Codes are packed eight to a word, element j at bits 4j.
// One SIMD group per output row; grid.y is the token row.
// ---------------------------------------------------------------------------
static inline float ink_affine4_row(device const uint* codes,
                                    device const bfloat* scales,
                                    device const bfloat* biases,
                                    device const float* x,
                                    uint columns, uint lane) {
    float acc = 0.0f;
    const uint words = columns / 8u;
    for (uint w = lane; w < words; w += 32u) {
        const uint packed = codes[w];
        device const float* v = x + w * 8u;
        float dot = 0.0f, sum = 0.0f;
        for (uint j = 0u; j < 8u; ++j) {
            const float value = v[j];
            dot = fma(float((packed >> (4u * j)) & 15u), value, dot);
            sum += value;
        }
        const uint group = w / 8u;
        acc = fma(float(scales[group]), dot, acc);
        acc = fma(float(biases[group]), sum, acc);
    }
    return simd_sum(acc);
}

kernel void ink_affine4_gemv(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& out_rows [[buffer(6)]],
    constant uint& x_stride [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x, row = tg.y;
    const float total = ink_affine4_row(codes + size_t(r) * (columns / 8u),
                                        scales + size_t(r) * (columns / 64u),
                                        biases + size_t(r) * (columns / 64u),
                                        x + size_t(row) * x_stride, columns, lane);
    if (lane == 0u) y[size_t(row) * out_rows + r] = total;
}

// The vocabulary head: the same product, written as the FP16 logits the
// sampler reads.
kernel void ink_affine4_gemv_half(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device half* y [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& out_rows [[buffer(6)]],
    constant uint& x_stride [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x, row = tg.y;
    const float total = ink_affine4_row(codes + size_t(r) * (columns / 8u),
                                        scales + size_t(r) * (columns / 64u),
                                        biases + size_t(r) * (columns / 64u),
                                        x + size_t(row) * x_stride, columns, lane);
    if (lane == 0u) y[size_t(row) * out_rows + r] = half(clamp(total, -65504.0f, 65504.0f));
}

// One row of an affine 4-bit table, as FP32: the token embedding.
kernel void ink_affine4_lookup(
    device const uint* codes [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const uint* tokens [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, row = gid.y;
    if (i >= columns) return;
    const uint token = tokens[row];
    const uint packed = codes[size_t(token) * (columns / 8u) + i / 8u];
    const uint group = i / 64u;
    out[size_t(row) * columns + i] =
        float((packed >> (4u * (i & 7u))) & 15u) * float(scales[size_t(token) * (columns / 64u) + group])
        + float(biases[size_t(token) * (columns / 64u) + group]);
}

// ---------------------------------------------------------------------------
// Short causal convolutions over the token axis, kernel size 4.
// `state` holds the three inputs before row 0, oldest first. Row r reads its
// earlier rows from `x` and from `state` once it runs off the front.
//   static:  y = (sum_t taps[c][t] w_t + x) / sqrt(5),  taps[c][3] on the token
//   dynamic: y = (sum_lag (fixed[lag][c] + moving[head][lag]) x_{-lag} + x) / 3
// ink_conv_advance then leaves the last three inputs in `state`.
// ---------------------------------------------------------------------------
static inline float ink_conv_input(device const float* x, device const float* state,
                                   uint channels, uint c, int row) {
    return row >= 0 ? x[size_t(row) * channels + c] : state[size_t(3 + row) * channels + c];
}

kernel void ink_conv_static(
    device const float* x [[buffer(0)]],
    device const float* state [[buffer(1)]],
    device const bfloat* taps [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const int row = int(gid.y);
    if (c >= channels) return;
    const float now = x[size_t(row) * channels + c];
    float acc = float(taps[c * 4u + 3u]) * now;
    for (int lag = 1; lag <= 3; ++lag)
        acc = fma(float(taps[c * 4u + uint(3 - lag)]),
                  ink_conv_input(x, state, channels, c, row - lag), acc);
    out[size_t(row) * channels + c] = (acc + now) * 0.4472135954999579f;
}

kernel void ink_conv_dynamic(
    device const float* x [[buffer(0)]],
    device const float* state [[buffer(1)]],
    device const bfloat* fixed [[buffer(2)]],
    device const float* moving [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    const int row = int(gid.y);
    if (c >= channels) return;
    const uint head = c / (channels / heads);
    device const float* dynamic_taps = moving + (size_t(row) * heads + head) * 4u;
    const float now = x[size_t(row) * channels + c];
    float acc = (float(fixed[c]) + dynamic_taps[0]) * now;
    for (int lag = 1; lag <= 3; ++lag)
        acc = fma(float(fixed[uint(lag) * channels + c]) + dynamic_taps[lag],
                  ink_conv_input(x, state, channels, c, row - lag), acc);
    out[size_t(row) * channels + c] = (acc + now) * (1.0f / 3.0f);
}

kernel void ink_conv_advance(
    device const float* x [[buffer(0)]],
    device float* state [[buffer(1)]],
    constant uint& channels [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= channels) return;
    float next[3];
    for (int j = 0; j < 3; ++j)
        next[j] = ink_conv_input(x, state, channels, c, int(rows) - 3 + j);
    for (uint j = 0u; j < 3u; ++j) state[size_t(j) * channels + c] = next[j];
}

// ---------------------------------------------------------------------------
// Per-head L2 clip, in place: v *= gain (optional), then scaled down to
// `threshold` when longer, then multiplied by the row's log scale.
// ---------------------------------------------------------------------------
kernel void ink_head_clip(
    device float* values [[buffer(0)]],
    device const bfloat* gain [[buffer(1)]],
    device const float* row_scale [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant float& threshold [[buffer(5)]],
    constant uint& has_gain [[buffer(6)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint head = tg.x, row = tg.y;
    device float* v = values + (size_t(row) * heads + head) * head_dim;
    device const bfloat* g = gain + size_t(head) * head_dim;
    float acc = 0.0f;
    for (uint i = lane; i < head_dim; i += 32u) {
        const float value = has_gain != 0u ? v[i] * float(g[i]) : v[i];
        acc = fma(value, value, acc);
    }
    const float length = sqrt(simd_sum(acc));
    const float factor = min(threshold / (length + 1e-6f), 1.0f) * row_scale[row];
    for (uint i = lane; i < head_dim; i += 32u) {
        const float value = has_gain != 0u ? v[i] * float(g[i]) : v[i];
        v[i] = value * factor;
    }
}

// Relative position bias: bias[row][head][distance] = rel[row][head] . proj[:, distance].
kernel void ink_relative_bias(
    device const float* rel [[buffer(0)]],
    device const bfloat* proj [[buffer(1)]],
    device float* bias [[buffer(2)]],
    constant uint& d_rel [[buffer(3)]],
    constant uint& extent [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint distance = gid.x, head_row = gid.y;
    if (distance >= extent) return;
    device const float* r = rel + size_t(head_row) * d_rel;
    float acc = 0.0f;
    for (uint i = 0u; i < d_rel; ++i) acc = fma(r[i], float(proj[size_t(i) * extent + distance]), acc);
    bias[size_t(head_row) * extent + distance] = acc;
}

// Stores FP32 key or value rows into the FP16 cache. Row r goes to slot
// (first_slot + r) mod capacity.
kernel void ink_cache_store(
    device const float* rows_in [[buffer(0)]],
    device half* cache [[buffer(1)]],
    constant uint& width [[buffer(2)]],
    constant uint& first_slot [[buffer(3)]],
    constant uint& capacity [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, row = gid.y;
    if (i >= width) return;
    const uint slot = (first_slot + row) % capacity;
    cache[size_t(slot) * width + i] = half(rows_in[size_t(row) * width + i]);
}

// ---------------------------------------------------------------------------
// Attention scores. Query row r sits at position first_position + r. A cache
// slot s holds the newest position at or before `last_position` that is
// congruent to s modulo `capacity` (a sliding layer's ring), or position s
// when `ring` is 0. A key is visible when 0 <= distance, and distance <
// `window` when `window` is nonzero. The relative bias applies below `extent`.
// scores[(row * q_heads + head) * slots + slot]
// ---------------------------------------------------------------------------
kernel void ink_attention_scores(
    device const float* q [[buffer(0)]],
    device const half* keys [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* scores [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& q_heads [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& slots [[buffer(7)]],
    constant uint& capacity [[buffer(8)]],
    constant uint& ring [[buffer(9)]],
    constant uint& first_position [[buffer(10)]],
    constant uint& last_position [[buffer(11)]],
    constant uint& window [[buffer(12)]],
    constant uint& extent [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint slot = gid.x, head = gid.y, row = gid.z;
    if (slot >= slots) return;
    const size_t index = (size_t(row) * q_heads + head) * slots + slot;
    long key_position = long(slot);
    if (ring != 0u) {
        const uint behind = (last_position % capacity + capacity - slot) % capacity;
        key_position = long(last_position) - long(behind);
    }
    const long distance = long(first_position) + long(row) - key_position;
    if (key_position < 0 || distance < 0 || (window != 0u && distance >= long(window))) {
        scores[index] = -INFINITY;
        return;
    }
    device const float* query = q + (size_t(row) * q_heads + head) * head_dim;
    device const half* key = keys + (size_t(slot) * kv_heads + head / (q_heads / kv_heads)) * head_dim;
    float acc = 0.0f;
    for (uint i = 0u; i < head_dim; ++i) acc = fma(query[i], float(key[i]), acc);
    acc *= scale;
    if (distance < long(extent))
        acc += bias[(size_t(row) * q_heads + head) * extent + uint(distance)];
    scores[index] = acc;
}

// Softmax over the visible keys and the sink logit, then the mix of values.
// One threadgroup per (head, row). `scores` is overwritten with probabilities.
kernel void ink_attention_mix(
    device float* scores [[buffer(0)]],
    device const half* values [[buffer(1)]],
    device const float* sinks [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& q_heads [[buffer(5)]],
    constant uint& kv_heads [[buffer(6)]],
    constant uint& slots [[buffer(7)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[kInkRowSimdGroups];
    const uint head = tg.x, row = tg.y;
    device float* s = scores + (size_t(row) * q_heads + head) * slots;
    const float sink = sinks[size_t(row) * q_heads + head];

    float peak = -INFINITY;
    for (uint i = tid; i < slots; i += kInkRowThreads) peak = max(peak, s[i]);
    peak = simd_max(peak);
    if (lane == 0u) partial[sg] = peak;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    peak = sink;
    for (uint i = 0u; i < kInkRowSimdGroups; ++i) peak = max(peak, partial[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    for (uint i = tid; i < slots; i += kInkRowThreads) {
        const float p = exp(s[i] - peak);
        s[i] = p;
        total += p;
    }
    total = simd_sum(total);
    if (lane == 0u) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    total = exp(sink - peak);
    for (uint i = 0u; i < kInkRowSimdGroups; ++i) total += partial[i];

    if (tid >= head_dim) return;
    device const half* v = values + size_t(head / (q_heads / kv_heads)) * head_dim + tid;
    const size_t stride = size_t(kv_heads) * head_dim;
    float acc = 0.0f;
    for (uint i = 0u; i < slots; ++i) acc = fma(s[i], float(v[size_t(i) * stride]), acc);
    out[(size_t(row) * q_heads + head) * head_dim + tid] = acc / total;
}

// out = a * sigmoid(gate)
kernel void ink_sigmoid_gate(
    device const float* a [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    out[i] = a[i] / (1.0f + exp(-gate[i]));
}

// out = a + b, or a copy of `a` when `add` is 0.
kernel void ink_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant uint& add [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    out[i] = add != 0u ? a[i] + b[i] : a[i];
}

kernel void ink_zero(
    device float* out [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    out[i] = 0.0f;
}

// ---------------------------------------------------------------------------
// Routed experts. Row `rows_index[n]` of `x` goes through one expert.
//
// NVFP4: E2M1 codes eight to a word, one E4M3 scale per 16 values, and one
// factor per expert. MXFP8: E4M3 bytes with one power-of-two exponent per 32.
// The gate/up matrix has gate and up rows alternating; ink_expert_activation
// folds each pair into one hidden value.
// ---------------------------------------------------------------------------
static inline float ink_nvfp4_row(device const uint* codes, device const uchar* scales,
                                  device const float* x, uint columns, uint lane) {
    float acc = 0.0f;
    const uint words = columns / 8u;
    for (uint w = lane; w < words; w += 32u) {
        const uint packed = codes[w];
        device const float* v = x + w * 8u;
        float dot = 0.0f;
        for (uint j = 0u; j < 8u; ++j)
            dot = fma(ink_e2m1((packed >> (4u * j)) & 15u), v[j], dot);
        acc = fma(ink_e4m3(uint(scales[w / 2u])), dot, acc);
    }
    return simd_sum(acc);
}

static inline float ink_mxfp8_row(device const uchar* codes, device const uchar* scales,
                                  device const float* x, uint columns, uint lane) {
    float acc = 0.0f;
    const uint blocks = columns / 32u;
    for (uint b = lane; b < blocks; b += 32u) {
        device const uchar* c = codes + b * 32u;
        device const float* v = x + b * 32u;
        float dot = 0.0f;
        for (uint j = 0u; j < 32u; ++j) dot = fma(ink_e4m3(uint(c[j])), v[j], dot);
        acc = fma(ink_e8m0(uint(scales[b])), dot, acc);
    }
    return simd_sum(acc);
}

kernel void ink_expert_gemv_nvfp4(
    device const uint* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const uint* rows_index [[buffer(4)]],
    device const float* weights [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    constant uint& out_rows [[buffer(7)]],
    constant float& factor [[buffer(8)]],
    constant uint& accumulate [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x, n = tg.y;
    const uint row = rows_index[n];
    const float total = ink_nvfp4_row(codes + size_t(r) * (columns / 8u),
                                      scales + size_t(r) * (columns / 16u),
                                      x + size_t(accumulate != 0u ? n : row) * columns,
                                      columns, lane) * factor;
    if (lane != 0u) return;
    if (accumulate != 0u) y[size_t(row) * out_rows + r] += total * weights[n];
    else y[size_t(n) * out_rows + r] = total;
}

kernel void ink_expert_gemv_mxfp8(
    device const uchar* codes [[buffer(0)]],
    device const uchar* scales [[buffer(1)]],
    device const float* x [[buffer(2)]],
    device float* y [[buffer(3)]],
    device const uint* rows_index [[buffer(4)]],
    device const float* weights [[buffer(5)]],
    constant uint& columns [[buffer(6)]],
    constant uint& out_rows [[buffer(7)]],
    constant float& factor [[buffer(8)]],
    constant uint& accumulate [[buffer(9)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint r = tg.x, n = tg.y;
    const uint row = rows_index[n];
    const float total = ink_mxfp8_row(codes + size_t(r) * columns,
                                      scales + size_t(r) * (columns / 32u),
                                      x + size_t(accumulate != 0u ? n : row) * columns,
                                      columns, lane) * factor;
    if (lane != 0u) return;
    if (accumulate != 0u) y[size_t(row) * out_rows + r] += total * weights[n];
    else y[size_t(n) * out_rows + r] = total;
}

// hidden[i] = softcap(gate, 4) * sigmoid(gate) * softcap(up, 25), from the
// alternating gate and up rows.
kernel void ink_expert_activation(
    device const float* gate_up [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    constant uint& intermediate [[buffer(2)]],
    constant float& gate_cap [[buffer(3)]],
    constant float& up_cap [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, n = gid.y;
    if (i >= intermediate) return;
    const float gate = gate_up[(size_t(n) * intermediate + i) * 2u];
    const float up = gate_up[(size_t(n) * intermediate + i) * 2u + 1u];
    hidden[size_t(n) * intermediate + i] =
        ink_expert_unit(gate, up, gate_cap, up_cap);
}

// ---------------------------------------------------------------------------
// A decoded token's eight experts in two dispatches, so the GPU works on all
// of them at once. Phase 1 folds each gate and up row into a hidden value;
// phase 2 is the weighted sum of the eight down products.
//
// Buffers 0-7 are the experts' blobs; `codes_offset` and `scales_offset`
// find a matrix inside one. `factors` holds a factor per expert, which in
// phase 2 already includes the expert's routing weight.
// ---------------------------------------------------------------------------
#define INK_EXPERT_BLOBS                                                        \
    device const uchar* e0 [[buffer(0)]], device const uchar* e1 [[buffer(1)]], \
    device const uchar* e2 [[buffer(2)]], device const uchar* e3 [[buffer(3)]], \
    device const uchar* e4 [[buffer(4)]], device const uchar* e5 [[buffer(5)]], \
    device const uchar* e6 [[buffer(6)]], device const uchar* e7 [[buffer(7)]]

static inline float ink_expert_row(device const uchar* blob, uint codes_offset, uint scales_offset,
                                   device const float* x, uint columns, uint row, uint lane,
                                   bool nvfp4) {
    if (nvfp4)
        return ink_nvfp4_row((device const uint*)(blob + codes_offset) + size_t(row) * (columns / 8u),
                             blob + scales_offset + size_t(row) * (columns / 16u), x, columns, lane);
    return ink_mxfp8_row(blob + codes_offset + size_t(row) * columns,
                         blob + scales_offset + size_t(row) * (columns / 32u), x, columns, lane);
}

// Each kernel is written out so the kernel index, which reads names from
// this source, finds it.
#define INK_EXPERT_PHASE1_ARGS                                                  \
    INK_EXPERT_BLOBS,                                                           \
    device const float* x [[buffer(8)]],                                        \
    device float* hidden [[buffer(9)]],                                         \
    constant float* factors [[buffer(10)]],                                     \
    constant uint& codes_offset [[buffer(11)]],                                 \
    constant uint& scales_offset [[buffer(12)]],                                \
    constant uint& columns [[buffer(13)]],                                      \
    constant uint& intermediate [[buffer(14)]],                                 \
    constant float& gate_cap [[buffer(15)]],                                    \
    constant float& up_cap [[buffer(16)]],                                      \
    uint2 tg [[threadgroup_position_in_grid]],                                  \
    uint lane [[thread_index_in_simdgroup]]
#define INK_EXPERT_PHASE1_BODY(NVFP4)                                           \
    device const uchar* blobs[8] = {e0, e1, e2, e3, e4, e5, e6, e7};            \
    const uint i = tg.x, e = tg.y;                                              \
    const float gate = ink_expert_row(blobs[e], codes_offset, scales_offset, x, columns,    \
                                      2u * i, lane, NVFP4) * factors[e];        \
    const float up = ink_expert_row(blobs[e], codes_offset, scales_offset, x, columns,      \
                                    2u * i + 1u, lane, NVFP4) * factors[e];     \
    if (lane != 0u) return;                                                     \
    hidden[size_t(e) * intermediate + i] = ink_expert_unit(gate, up, gate_cap, up_cap);

kernel void ink_expert_phase1_nvfp4(INK_EXPERT_PHASE1_ARGS) { INK_EXPERT_PHASE1_BODY(true) }
kernel void ink_expert_phase1_mxfp8(INK_EXPERT_PHASE1_ARGS) { INK_EXPERT_PHASE1_BODY(false) }

#define INK_EXPERT_PHASE2_ARGS                                                  \
    INK_EXPERT_BLOBS,                                                           \
    device const float* hidden [[buffer(8)]],                                   \
    device float* y [[buffer(9)]],                                              \
    constant float* factors [[buffer(10)]],                                     \
    constant uint& codes_offset [[buffer(11)]],                                 \
    constant uint& scales_offset [[buffer(12)]],                                \
    constant uint& intermediate [[buffer(13)]],                                 \
    uint2 tg [[threadgroup_position_in_grid]],                                  \
    uint lane [[thread_index_in_simdgroup]]
#define INK_EXPERT_PHASE2_BODY(NVFP4)                                           \
    device const uchar* blobs[8] = {e0, e1, e2, e3, e4, e5, e6, e7};            \
    const uint row = tg.x;                                                      \
    float total = 0.0f;                                                         \
    for (uint e = 0u; e < 8u; ++e)                                              \
        total = fma(ink_expert_row(blobs[e], codes_offset, scales_offset,       \
                                   hidden + size_t(e) * intermediate, intermediate, row, lane, NVFP4), \
                    factors[e], total);                                         \
    if (lane == 0u) y[row] = total;

kernel void ink_expert_phase2_nvfp4(INK_EXPERT_PHASE2_ARGS) { INK_EXPERT_PHASE2_BODY(true) }
kernel void ink_expert_phase2_mxfp8(INK_EXPERT_PHASE2_ARGS) { INK_EXPERT_PHASE2_BODY(false) }

#undef INK_EXPERT_PHASE1_ARGS
#undef INK_EXPERT_PHASE1_BODY
#undef INK_EXPERT_PHASE2_ARGS
#undef INK_EXPERT_PHASE2_BODY
#undef INK_EXPERT_BLOBS

// ---------------------------------------------------------------------------
// Chunked prefill glue. The batched projections and expert products run on
// FP16 rows; everything between them stays FP32.
// ---------------------------------------------------------------------------
kernel void ink_to_half(
    device const float* source [[buffer(0)]],
    device half* target [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    target[i] = half(clamp(source[i], -65504.0f, 65504.0f));
}

kernel void ink_to_float(
    device const half* source [[buffer(0)]],
    device float* target [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    target[i] = float(source[i]);
}

// rows_out[n] = half(rows_in[index[n]]): the rows routed to each expert,
// gathered into the expert-sorted order the batched products read.
kernel void ink_gather_rows_half(
    device const float* rows_in [[buffer(0)]],
    device const uint* index [[buffer(1)]],
    device half* rows_out [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, n = gid.y;
    if (i >= width) return;
    rows_out[size_t(n) * width + i] =
        half(clamp(rows_in[size_t(index[n]) * width + i], -65504.0f, 65504.0f));
}

// The FP16 form of ink_expert_activation, for the batched down projection.
kernel void ink_expert_activation_half(
    device const float* gate_up [[buffer(0)]],
    device half* hidden [[buffer(1)]],
    constant uint& intermediate [[buffer(2)]],
    constant float& gate_cap [[buffer(3)]],
    constant float& up_cap [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, n = gid.y;
    if (i >= intermediate) return;
    const float gate = gate_up[(size_t(n) * intermediate + i) * 2u];
    const float up = gate_up[(size_t(n) * intermediate + i) * 2u + 1u];
    hidden[size_t(n) * intermediate + i] = half(
        ink_expert_unit(gate, up, gate_cap, up_cap));
}

// out[row] = sum_j weight[row][j] * down[pair[row][j]]: each token's routed
// experts, weighted and summed in FP32.
kernel void ink_expert_reduce(
    device const float* down [[buffer(0)]],
    device const uint* pair [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, row = gid.y;
    if (i >= width) return;
    float acc = 0.0f;
    for (uint j = 0u; j < top_k; ++j) {
        acc = fma(weight[size_t(row) * top_k + j],
                  down[size_t(pair[size_t(row) * top_k + j]) * width + i], acc);
    }
    out[size_t(row) * width + i] = acc;
}

// ---------------------------------------------------------------------------
// Decode products, several rows per SIMD group.
//
// The one-row expert kernels above read the input row again for every
// output row, eight floats for every 4-bit word. These give each SIMD group
// kInkRows consecutive rows: every lane loads its slice of the input once and
// runs it against all of them. On a 16 GB M2 a decoded token's eight NVFP4
// experts take 454 us instead of 741, MXFP8 1.27 ms instead of 1.36. (The
// same change made the affine projections slower, so they keep one row.)
//
// Every lane still walks the same words in the same order with the same
// fused multiply-adds, and each row ends in the same simd_sum, so a row's
// result is the one-row kernel's to the bit. The runner checks that once per
// GPU when the model loads and keeps the one-row kernels if it differs.
// ---------------------------------------------------------------------------
constant constexpr uint kInkRows = 4;
constant constexpr uint kInkRowGroups = 2;
// ink_e2m1 as a table, sign in bit 3; the negative zero matches it too.
constant float kInkE2M1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

static inline void ink_load8(device const float* x, thread float v[8]) {
    const packed_float4 a = ((device const packed_float4*)x)[0];
    const packed_float4 b = ((device const packed_float4*)x)[1];
    v[0] = a[0]; v[1] = a[1]; v[2] = a[2]; v[3] = a[3];
    v[4] = b[0]; v[5] = b[1]; v[6] = b[2]; v[7] = b[3];
}

// Rows first .. first + R - 1 of one expert matrix, reduced: what
// ink_expert_row returns for each.
template <uint R>
static inline void ink_expert_rows(device const uchar* blob, uint codes_offset, uint scales_offset,
                                   device const float* x, uint columns, uint first, uint lane,
                                   bool nvfp4, thread float out[R]) {
    float acc[R];
    for (uint r = 0u; r < R; ++r) acc[r] = 0.0f;
    if (nvfp4) {
        device const uint* codes = (device const uint*)(blob + codes_offset);
        device const uchar* scales = blob + scales_offset;
        const uint words = columns / 8u;
        for (uint w = lane; w < words; w += 32u) {
            float v[8];
            ink_load8(x + w * 8u, v);
            for (uint r = 0u; r < R; ++r) {
                const uint packed = codes[size_t(first + r) * words + w];
                float dot = 0.0f;
                for (uint j = 0u; j < 8u; ++j)
                    dot = fma(kInkE2M1[(packed >> (4u * j)) & 15u], v[j], dot);
                acc[r] = fma(ink_e4m3(uint(scales[size_t(first + r) * (columns / 16u) + w / 2u])),
                             dot, acc[r]);
            }
        }
    } else {
        device const uchar* codes = blob + codes_offset;
        device const uchar* scales = blob + scales_offset;
        const uint blocks = columns / 32u;
        for (uint b = lane; b < blocks; b += 32u) {
            float v[32];
            for (uint q = 0u; q < 4u; ++q) ink_load8(x + b * 32u + q * 8u, v + q * 8u);
            for (uint r = 0u; r < R; ++r) {
                device const uchar* c = codes + size_t(first + r) * columns + b * 32u;
                float dot = 0.0f;
                for (uint j = 0u; j < 32u; ++j) dot = fma(ink_e4m3(uint(c[j])), v[j], dot);
                acc[r] = fma(ink_e8m0(uint(scales[size_t(first + r) * blocks + b])), dot, acc[r]);
            }
        }
    }
    for (uint r = 0u; r < R; ++r) out[r] = simd_sum(acc[r]);
}

// Phase 1 for kInkRows hidden units of one expert per SIMD group: their gate
// and up rows are 2 * kInkRows consecutive rows. Grid: (units / (kInkRows *
// kInkRowGroups), 8), with `intermediate` a multiple of that.
#define INK_EXPERT_PHASE1_ROWS_BODY(NVFP4)                                      \
    device const uchar* blobs[8] = {e0, e1, e2, e3, e4, e5, e6, e7};            \
    const uint unit = (tg.x * kInkRowGroups + sg) * kInkRows, e = tg.y;         \
    float rows[2u * kInkRows];                                                  \
    ink_expert_rows<2u * kInkRows>(blobs[e], codes_offset, scales_offset, x, columns,      \
                                   2u * unit, lane, NVFP4, rows);               \
    if (lane != 0u) return;                                                     \
    for (uint r = 0u; r < kInkRows; ++r)                                        \
        hidden[size_t(e) * intermediate + unit + r] = ink_expert_unit(          \
            rows[2u * r] * factors[e], rows[2u * r + 1u] * factors[e], gate_cap, up_cap);

#define INK_EXPERT_PHASE1_ROWS_ARGS                                             \
    device const uchar* e0 [[buffer(0)]], device const uchar* e1 [[buffer(1)]], \
    device const uchar* e2 [[buffer(2)]], device const uchar* e3 [[buffer(3)]], \
    device const uchar* e4 [[buffer(4)]], device const uchar* e5 [[buffer(5)]], \
    device const uchar* e6 [[buffer(6)]], device const uchar* e7 [[buffer(7)]], \
    device const float* x [[buffer(8)]],                                        \
    device float* hidden [[buffer(9)]],                                         \
    constant float* factors [[buffer(10)]],                                     \
    constant uint& codes_offset [[buffer(11)]],                                 \
    constant uint& scales_offset [[buffer(12)]],                                \
    constant uint& columns [[buffer(13)]],                                      \
    constant uint& intermediate [[buffer(14)]],                                 \
    constant float& gate_cap [[buffer(15)]],                                    \
    constant float& up_cap [[buffer(16)]],                                      \
    uint2 tg [[threadgroup_position_in_grid]],                                  \
    uint sg [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]]

kernel void ink_expert_phase1_rows_nvfp4(INK_EXPERT_PHASE1_ROWS_ARGS) { INK_EXPERT_PHASE1_ROWS_BODY(true) }
kernel void ink_expert_phase1_rows_mxfp8(INK_EXPERT_PHASE1_ROWS_ARGS) { INK_EXPERT_PHASE1_ROWS_BODY(false) }

// Phase 2 for kInkRows output rows per SIMD group. Grid: (width / (kInkRows *
// kInkRowGroups), 1), with `width` a multiple of that.
#define INK_EXPERT_PHASE2_ROWS_BODY(NVFP4)                                      \
    device const uchar* blobs[8] = {e0, e1, e2, e3, e4, e5, e6, e7};            \
    const uint first = (tg.x * kInkRowGroups + sg) * kInkRows;                  \
    float total[kInkRows];                                                      \
    for (uint r = 0u; r < kInkRows; ++r) total[r] = 0.0f;                       \
    for (uint e = 0u; e < 8u; ++e) {                                            \
        float rows[kInkRows];                                                   \
        ink_expert_rows<kInkRows>(blobs[e], codes_offset, scales_offset,        \
                                  hidden + size_t(e) * intermediate, intermediate, first, \
                                  lane, NVFP4, rows);                           \
        for (uint r = 0u; r < kInkRows; ++r) total[r] = fma(rows[r], factors[e], total[r]); \
    }                                                                           \
    if (lane != 0u) return;                                                     \
    for (uint r = 0u; r < kInkRows; ++r) y[first + r] = total[r];

#define INK_EXPERT_PHASE2_ROWS_ARGS                                             \
    device const uchar* e0 [[buffer(0)]], device const uchar* e1 [[buffer(1)]], \
    device const uchar* e2 [[buffer(2)]], device const uchar* e3 [[buffer(3)]], \
    device const uchar* e4 [[buffer(4)]], device const uchar* e5 [[buffer(5)]], \
    device const uchar* e6 [[buffer(6)]], device const uchar* e7 [[buffer(7)]], \
    device const float* hidden [[buffer(8)]],                                   \
    device float* y [[buffer(9)]],                                              \
    constant float* factors [[buffer(10)]],                                     \
    constant uint& codes_offset [[buffer(11)]],                                 \
    constant uint& scales_offset [[buffer(12)]],                                \
    constant uint& intermediate [[buffer(13)]],                                 \
    uint2 tg [[threadgroup_position_in_grid]],                                  \
    uint sg [[simdgroup_index_in_threadgroup]],                                 \
    uint lane [[thread_index_in_simdgroup]]

kernel void ink_expert_phase2_rows_nvfp4(INK_EXPERT_PHASE2_ROWS_ARGS) { INK_EXPERT_PHASE2_ROWS_BODY(true) }
kernel void ink_expert_phase2_rows_mxfp8(INK_EXPERT_PHASE2_ROWS_ARGS) { INK_EXPERT_PHASE2_ROWS_BODY(false) }

#undef INK_EXPERT_PHASE1_ROWS_ARGS
#undef INK_EXPERT_PHASE1_ROWS_BODY
#undef INK_EXPERT_PHASE2_ROWS_ARGS
#undef INK_EXPERT_PHASE2_ROWS_BODY
