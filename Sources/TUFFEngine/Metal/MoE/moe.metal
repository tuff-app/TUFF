#include <metal_stdlib>
using namespace metal;

// Group size and streamed-expert count live in quant_group.metal.

constant uint FC_ROUTER_NUM_EXPERTS [[function_constant(40)]];
constant uint FC_ROUTER_D [[function_constant(41)]];
constant uint FC_ROUTER_TOP_K [[function_constant(42)]];
constant bool FC_ROUTER_USE_FC [[function_constant(43)]];

constant uint FC_MOE_D [[function_constant(0)]];
constant uint FC_MOE_F [[function_constant(1)]];
constant uint FC_MOE_TOP_K [[function_constant(2)]];
constant bool FC_MOE_USE_FC [[function_constant(3)]];
// Hidden activation for expert FFNs: unset/false = Gemma's gelu_pytorch_tanh,
// true = silu (Qwen 3.6 SwiGLU). Applies to routed decode, routed prefill,
// and the fused INT8 shared-expert kernel.
constant bool FC_MOE_ACT_SILU [[function_constant(4)]];

static inline uint router_fc_num_experts(constant uint& num_experts) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_NUM_EXPERTS))
        ? FC_ROUTER_NUM_EXPERTS
        : num_experts;
}

static inline uint router_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_ROUTER_USE_FC) &&
            FC_ROUTER_USE_FC &&
            is_function_constant_defined(FC_ROUTER_D))
        ? FC_ROUTER_D
        : D;
}

static inline uint moe_fc_d(constant uint& D) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_D)) ? FC_MOE_D : D;
}

static inline uint moe_fc_f(constant uint& F) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_F)) ? FC_MOE_F : F;
}

static inline uint moe_fc_top_k(constant uint& top_k) {
    return (is_function_constant_defined(FC_MOE_USE_FC) &&
            FC_MOE_USE_FC &&
            is_function_constant_defined(FC_MOE_TOP_K)) ? FC_MOE_TOP_K : top_k;
}

// gelu_pytorch_tanh is in activation.metal, compiled ahead of every module.

static inline float moe_hidden_activation(float x) {
    if (is_function_constant_defined(FC_MOE_ACT_SILU) && FC_MOE_ACT_SILU) {
        return x / (1.0f + exp(-x));
    }
    return gelu_pytorch_tanh(x);
}

// GPT-OSS caps only the positive side of the gate, clamps the linear branch
// symmetrically, and offsets that branch by one before multiplication.
kernel void gptoss_capped_swiglu(
    device const half* gate [[buffer(0)]],
    device const half* linear [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant float& limit [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float g = min(float(gate[gid]), limit);
    const float l = clamp(float(linear[gid]), -limit, limit);
    const float silu = g / (1.0f + exp(-1.702f * g));
    output[gid] = half(silu * (l + 1.0f));
}

kernel void gptoss_capped_swiglu_interleaved(
    device const half* gate_linear [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& limit [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    const float g = min(float(gate_linear[gid * 2u]), limit);
    const float l = clamp(float(gate_linear[gid * 2u + 1u]), -limit, limit);
    const float silu = g / (1.0f + exp(-1.702f * g));
    output[gid] = half(silu * (l + 1.0f));
}

kernel void gptoss_route_reduce(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& query_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = query_count * hidden_size;
    if (gid >= total) return;
    const uint token = gid / hidden_size;
    const uint dimension = gid - token * hidden_size;
    float value = float(residual[gid]);
    const uint route_base = token * top_k;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const uint route = route_base + slot;
        value = fma(float(route_weights[route]),
                    float(route_partials[route * hidden_size + dimension]),
                    value);
    }
    output[gid] = half(value);
}

kernel void gptoss_route_reduce_float_residual(
    device const half* route_partials [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& query_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = query_count * hidden_size;
    if (gid >= total) return;
    const uint token = gid / hidden_size;
    const uint dimension = gid - token * hidden_size;
    float value = residual[gid];
    const uint route_base = token * top_k;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const uint route = route_base + slot;
        value = fma(float(route_weights[route]),
                    float(route_partials[route * hidden_size + dimension]),
                    value);
    }
    output[gid] = value;
}

kernel void gptoss_route_reduce_float_partials_half_residual(
    device const float* route_partials [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& query_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = query_count * hidden_size;
    if (gid >= total) return;
    const uint token = gid / hidden_size;
    const uint dimension = gid - token * hidden_size;
    float value = float(residual[gid]);
    const uint route_base = token * top_k;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const uint route = route_base + slot;
        value = fma(float(route_weights[route]),
                    float(route_partials[route * hidden_size + dimension]),
                    value);
    }
    output[gid] = half(value);
}

kernel void gptoss_route_reduce_float_partials(
    device const float* route_partials [[buffer(0)]],
    device const half* route_weights [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& query_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = query_count * hidden_size;
    if (gid >= total) return;
    const uint token = gid / hidden_size;
    const uint dimension = gid - token * hidden_size;
    float value = residual[gid];
    const uint route_base = token * top_k;
    for (uint slot = 0u; slot < top_k; ++slot) {
        const uint route = route_base + slot;
        value = fma(float(route_weights[route]),
                    float(route_partials[route * hidden_size + dimension]),
                    value);
    }
    output[gid] = value;
}

// GPT-OSS router selection: stable top-4 followed by a softmax over only the
// selected logits. Ties prefer the lower expert index, matching the CPU and
// official reference implementations.
kernel void gptoss_router_top4(
    device const float* logits [[buffer(0)]],
    device uint* output_indices [[buffer(1)]],
    device half* output_weights [[buffer(2)]],
    constant uint& num_experts [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]
) {
    if (tid != 0u) return;
    // One threadgroup per token row; decode dispatches one.
    logits += row * num_experts;
    output_indices += row * 4u;
    output_weights += row * 4u;
    uint top_indices[4] = { 0u, 0u, 0u, 0u };
    float top_scores[4] = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    for (uint expert = 0; expert < num_experts; ++expert) {
        const float score = logits[expert];
        uint position = 4u;
        for (uint slot = 0; slot < 4u; ++slot) {
            if (score > top_scores[slot]
                || (score == top_scores[slot] && expert < top_indices[slot])) {
                position = slot;
                break;
            }
        }
        if (position == 4u) continue;
        for (uint slot = 3u; slot > position; --slot) {
            top_indices[slot] = top_indices[slot - 1u];
            top_scores[slot] = top_scores[slot - 1u];
        }
        top_indices[position] = expert;
        top_scores[position] = score;
    }

    const float maximum = top_scores[0];
    float exponentials[4];
    float denominator = 0.0f;
    for (uint slot = 0; slot < 4u; ++slot) {
        exponentials[slot] = exp(top_scores[slot] - maximum);
        denominator += exponentials[slot];
    }
    for (uint slot = 0; slot < 4u; ++slot) {
        output_indices[slot] = top_indices[slot];
        output_weights[slot] = half(exponentials[slot] / denominator);
    }
}

struct ExpertOffsets {
    uint gate_W_off;
    uint gate_s_off;
    uint gate_b_off;
    uint up_W_off;
    uint up_s_off;
    uint up_b_off;
    uint down_W_off;
    uint down_s_off;
    uint down_b_off;
};

struct RoutedBlobs {
    device const uint8_t* blob[kMaxStreamedExperts];
};

static inline void router_gemv_gemma4_body(
    device const uint8_t* W,
    device const bfloat* scales,
    device const bfloat* biases,
    device const half* hidden,
    device const bfloat* effective_scale,
    device float* out_logits,
    constant uint& num_experts,
    constant uint& D,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane
) {
    const uint NE = router_fc_num_experts(num_experts);
    const uint DD = router_fc_d(D);
    const uint e = tg_idx * rows_per_tg + sg_idx;
    if (e >= NE) return;

    const uint G = quant_group_size();
    const uint n_groups = DD / G;
    device const uint8_t* W_row = W + uint(e) * DD;
    device const bfloat* s_row = scales + uint(e) * n_groups;
    device const bfloat* b_row = biases + uint(e) * n_groups;

    float acc = 0.0f;
    for (uint g = 0; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        if (lane >= G / 2u) continue;
        const uint idx = g * G + lane * 2u;
        const float q0 = float(uint(W_row[idx]));
        const float q1 = float(uint(W_row[idx + 1u]));
        const float x0 = float(hidden[idx]) * float(effective_scale[idx]);
        const float x1 = float(hidden[idx + 1u]) * float(effective_scale[idx + 1u]);
        acc = fma(s, q0 * x0 + q1 * x1, acc);
        acc = fma(b, x0 + x1, acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) out_logits[e] = acc;
}

kernel void router_gemv_gemma4_r4(
    device const uint8_t* W [[buffer(0)]],
    device const bfloat* scales [[buffer(1)]],
    device const bfloat* biases [[buffer(2)]],
    device const half* hidden [[buffer(3)]],
    device const bfloat* effective_scale [[buffer(4)]],
    device float* out_logits [[buffer(5)]],
    constant uint& num_experts [[buffer(6)]],
    constant uint& D [[buffer(7)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    router_gemv_gemma4_body(W, scales, biases, hidden, effective_scale,
                            out_logits, num_experts, D, 4, tg_idx, sg_idx, lane);
}

kernel void router_topk_select_k8(
    device const float* logits [[buffer(0)]],
    device const bfloat* per_expert_scale [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    // K is specialized, so the eight-expert path keeps the constant-folded
    // bounds it had when eight was the only possibility.
    const uint K = streamed_expert_count();
    uint top_idx[kMaxStreamedExperts];
    float top_score[kMaxStreamedExperts];
    for (uint i = 0; i < K; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }

    for (uint e = 0; e < NE; ++e) {
        const float s = logits[e];
        if (s <= top_score[K - 1u]) continue;
        uint pos = K;
        for (uint i = 0; i < K; ++i) {
            if (s > top_score[i] || (s == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= K) continue;
        for (uint i = K - 1u; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = s;
    }

    const float max_s = top_score[0];
    float sum_exp = 0.0f;
    float exps[kMaxStreamedExperts];
    for (uint i = 0; i < K; ++i) {
        const float ex = fast::exp(top_score[i] - max_s);
        exps[i] = ex;
        sum_exp += ex;
    }
    for (uint i = 0; i < K; ++i) {
        const uint expert_idx = top_idx[i];
        const float weight = exps[i] / sum_exp;
        out_indices[i] = expert_idx;
        out_weights[i] = half(weight * float(per_expert_scale[expert_idx]));
    }
}

// MiniMax M2.7 chooses experts by sigmoid(logit) plus a learned correction,
// but combines the selected experts with the uncorrected sigmoid probabilities
// renormalized over top-k.
kernel void router_topk_select_minimax_k8(
    device const float* logits [[buffer(0)]],
    device const float* correction_bias [[buffer(1)]],
    device uint* out_indices [[buffer(2)]],
    device half* out_weights [[buffer(3)]],
    constant uint& num_experts [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    const uint NE = router_fc_num_experts(num_experts);
    uint top_idx[8];
    float top_score[8];
    for (uint i = 0; i < 8; ++i) {
        top_idx[i] = 0u;
        top_score[i] = -INFINITY;
    }
    for (uint e = 0; e < NE; ++e) {
        const float probability = 1.0f / (1.0f + fast::exp(-logits[e]));
        const float score = probability + correction_bias[e];
        if (score <= top_score[7]) continue;
        uint pos = 8u;
        for (uint i = 0; i < 8; ++i) {
            if (score > top_score[i] || (score == top_score[i] && e < top_idx[i])) {
                pos = i;
                break;
            }
        }
        if (pos >= 8u) continue;
        for (uint i = 7; i > pos; --i) {
            top_idx[i] = top_idx[i - 1];
            top_score[i] = top_score[i - 1];
        }
        top_idx[pos] = e;
        top_score[pos] = score;
    }
    float sum = 0.0f;
    float probabilities[8];
    for (uint i = 0; i < 8; ++i) {
        probabilities[i] = 1.0f / (1.0f + fast::exp(-logits[top_idx[i]]));
        sum += probabilities[i];
    }
    for (uint i = 0; i < 8; ++i) {
        out_indices[i] = top_idx[i];
        out_weights[i] = half(probabilities[i] / sum);
    }
}

// Each SIMD computes one affine INT4 row. Four adjacent groups are loaded as
// aligned 32-bit chunks; remaining groups use one byte per lane.
static inline float moe_int4_gemv_row_simd_dev_vec(
    device const uint8_t* W,
    device const bfloat* S,
    device const bfloat* B,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint G = quant_group_size();
    const uint n_groups = N / G;
    const uint row_bytes = N / 2;
    device const uint8_t* W_row = W + uint(row) * row_bytes;
    device const bfloat* s_row = S + uint(row) * n_groups;
    device const bfloat* b_row = B + uint(row) * n_groups;

    float acc = 0.0f;
    // A 128-byte block spans 256/G affine groups and G/8 lanes cover one of
    // them; see dequant_int4.metal. At G = 64 this is the original 4 and 8.
    const uint groups_per_block = 256u / G;
    const uint lane_shift = (G == 64u) ? 3u : ((G == 32u) ? 2u : 1u);
    const uint full_blocks = n_groups / groups_per_block;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        const uint w4 = *((device const uint*)(W_row + byte_base));
        const uint g = blk * groups_per_block + (lane >> lane_shift);
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const uint b0 = w4 & 0xFFu;
        const uint b1 = (w4 >> 8) & 0xFFu;
        const uint b2 = (w4 >> 16) & 0xFFu;
        const uint b3 = (w4 >> 24) & 0xFFu;
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        float dot = 0.0f;
        dot = fma(float(b0 & 0x0Fu), e0, dot); dot = fma(float(b0 >> 4), e1, dot);
        dot = fma(float(b1 & 0x0Fu), e2, dot); dot = fma(float(b1 >> 4), e3, dot);
        dot = fma(float(b2 & 0x0Fu), e4, dot); dot = fma(float(b2 >> 4), e5, dot);
        dot = fma(float(b3 & 0x0Fu), e6, dot); dot = fma(float(b3 >> 4), e7, dot);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;
        acc = fma(s, dot, acc);
        acc = fma(b, sum, acc);
    }
    for (uint g = full_blocks * groups_per_block; g < n_groups; ++g) {
        const float s = float(s_row[g]);
        const float b = float(b_row[g]);
        if (lane >= G / 2u) continue;
        const uint8_t byte = W_row[g * (G / 2u) + lane];
        const float x0 = float(x[g * G + lane * 2u]);
        const float x1 = float(x[g * G + lane * 2u + 1u]);
        float dot = fma(float(uint(byte & 0x0Fu)), x0, 0.0f);
        dot = fma(float(uint(byte >> 4)), x1, dot);
        acc = fma(s, dot, acc);
        acc = fma(b, x0 + x1, acc);
    }
    return simd_sum(acc);
}

// Gate and up rows share activation loads. Two 16-bit loads assemble each
// 4-byte weight chunk because packed sub-tensor offsets need only be 2-byte aligned.
static inline float2 moe_int4_gate_up_rows_simd_dev_vec_u16load(
    device const uint8_t* gateW,
    device const bfloat* gateS,
    device const bfloat* gateB,
    device const uint8_t* upW,
    device const bfloat* upS,
    device const bfloat* upB,
    device const half* x,
    uint row,
    uint N,
    uint lane
) {
    const uint G = quant_group_size();
    const uint n_groups = N / G;
    const uint row_bytes = N / 2;
    device const uint8_t* gW_row = gateW + uint(row) * row_bytes;
    device const uint8_t* uW_row = upW + uint(row) * row_bytes;
    device const bfloat* gS_row = gateS + uint(row) * n_groups;
    device const bfloat* gB_row = gateB + uint(row) * n_groups;
    device const bfloat* uS_row = upS + uint(row) * n_groups;
    device const bfloat* uB_row = upB + uint(row) * n_groups;

    float g_acc = 0.0f;
    float u_acc = 0.0f;
    // A 128-byte block spans 256/G affine groups and G/8 lanes cover one of
    // them; see dequant_int4.metal. At G = 64 this is the original 4 and 8.
    const uint groups_per_block = 256u / G;
    const uint lane_shift = (G == 64u) ? 3u : ((G == 32u) ? 2u : 1u);
    const uint full_blocks = n_groups / groups_per_block;
    for (uint blk = 0; blk < full_blocks; ++blk) {
        const uint byte_base = blk * 128u + lane * 4u;
        device const ushort* gp = (device const ushort*)(gW_row + byte_base);
        device const ushort* up = (device const ushort*)(uW_row + byte_base);
        const uint gw4 = uint(gp[0]) | (uint(gp[1]) << 16);
        const uint uw4 = uint(up[0]) | (uint(up[1]) << 16);
        const uint g = blk * groups_per_block + (lane >> lane_shift);
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        const uint elem = byte_base * 2u;
        const half4 xa = *((device const half4*)(x + elem));
        const half4 xb = *((device const half4*)(x + elem + 4u));
        const float e0 = float(xa.x), e1 = float(xa.y);
        const float e2 = float(xa.z), e3 = float(xa.w);
        const float e4 = float(xb.x), e5 = float(xb.y);
        const float e6 = float(xb.z), e7 = float(xb.w);
        const float sum = e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7;

        const uint gb0 = gw4 & 0xFFu;
        const uint gb1 = (gw4 >> 8) & 0xFFu;
        const uint gb2 = (gw4 >> 16) & 0xFFu;
        const uint gb3 = (gw4 >> 24) & 0xFFu;
        float g_dot = 0.0f;
        g_dot = fma(float(gb0 & 0x0Fu), e0, g_dot); g_dot = fma(float(gb0 >> 4), e1, g_dot);
        g_dot = fma(float(gb1 & 0x0Fu), e2, g_dot); g_dot = fma(float(gb1 >> 4), e3, g_dot);
        g_dot = fma(float(gb2 & 0x0Fu), e4, g_dot); g_dot = fma(float(gb2 >> 4), e5, g_dot);
        g_dot = fma(float(gb3 & 0x0Fu), e6, g_dot); g_dot = fma(float(gb3 >> 4), e7, g_dot);

        const uint ub0 = uw4 & 0xFFu;
        const uint ub1 = (uw4 >> 8) & 0xFFu;
        const uint ub2 = (uw4 >> 16) & 0xFFu;
        const uint ub3 = (uw4 >> 24) & 0xFFu;
        float u_dot = 0.0f;
        u_dot = fma(float(ub0 & 0x0Fu), e0, u_dot); u_dot = fma(float(ub0 >> 4), e1, u_dot);
        u_dot = fma(float(ub1 & 0x0Fu), e2, u_dot); u_dot = fma(float(ub1 >> 4), e3, u_dot);
        u_dot = fma(float(ub2 & 0x0Fu), e4, u_dot); u_dot = fma(float(ub2 >> 4), e5, u_dot);
        u_dot = fma(float(ub3 & 0x0Fu), e6, u_dot); u_dot = fma(float(ub3 >> 4), e7, u_dot);

        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    for (uint g = full_blocks * groups_per_block; g < n_groups; ++g) {
        const float gs = float(gS_row[g]);
        const float gb = float(gB_row[g]);
        const float us = float(uS_row[g]);
        const float ub = float(uB_row[g]);
        if (lane >= G / 2u) continue;
        const uint8_t gbv = gW_row[g * (G / 2u) + lane];
        const uint8_t ubv = uW_row[g * (G / 2u) + lane];
        const float x0 = float(x[g * G + lane * 2u]);
        const float x1 = float(x[g * G + lane * 2u + 1u]);
        const float sum = x0 + x1;
        float g_dot = fma(float(uint(gbv & 0x0Fu)), x0, 0.0f);
        g_dot = fma(float(uint(gbv >> 4)), x1, g_dot);
        float u_dot = fma(float(uint(ubv & 0x0Fu)), x0, 0.0f);
        u_dot = fma(float(uint(ubv >> 4)), x1, u_dot);
        g_acc = fma(gs, g_dot, g_acc);
        g_acc = fma(gb, sum, g_acc);
        u_acc = fma(us, u_dot, u_acc);
        u_acc = fma(ub, sum, u_acc);
    }
    return float2(simd_sum(g_acc), simd_sum(u_acc));
}

struct ExpertBitWidths { uint gate; uint up; uint down; };

// MLX U32 packing is little-endian: its bytes are the same consecutive
// nibbles/bytes consumed here, independent of the stored integer dtype.
template <typename Activation>
static inline float moe_affine_row(device const uint8_t* W, device const bfloat* S,
    device const bfloat* B, device const Activation* x, uint row, uint N,
    uint lane, uint bits) {
    const uint G = quant_group_size();
    const uint groups = N / G;
    device const uint8_t* weights = W + row * (N * bits / 8u);
    device const bfloat* scales = S + row * groups;
    device const bfloat* biases = B + row * groups;
    float sum = 0.0f;
    if (bits == 8u && (N & 127u) == 0u && (G & 3u) == 0u) {
        // Each lane reads four consecutive weight bytes per 128 elements: one
        // 128-byte line per simdgroup load. A lane's four elements share a
        // group, so its scale and bias apply once to the lane's dot product.
        for (uint base = 0; base < N; base += 128u) {
            const uint element = base + lane * 4u;
            const uint group = element / G;
            const uchar4 q = *reinterpret_cast<device const uchar4*>(weights + element);
            const float x0 = float(x[element]), x1 = float(x[element + 1u]);
            const float x2 = float(x[element + 2u]), x3 = float(x[element + 3u]);
            const float dot = fma(float(q.x), x0, fma(float(q.y), x1, fma(float(q.z), x2, float(q.w) * x3)));
            sum = fma(float(scales[group]), dot, sum);
            sum = fma(float(biases[group]), (x0 + x1) + (x2 + x3), sum);
        }
        return simd_sum(sum);
    }
    for (uint group = 0; group < groups; ++group) {
        for (uint i = lane; i < G; i += 32u) {
            const uint element = group * G + i;
            const uint q = bits == 8u ? uint(weights[element])
                : ((uint(weights[element / 2u]) >> (4u * (element & 1u))) & 15u);
            const float v = float(x[element]);
            sum = fma(fma(float(scales[group]), float(q), float(biases[group])), v, sum);
        }
    }
    return simd_sum(sum);
}

static inline float2 moe_gate_up_rows(device const uint8_t* gW, device const bfloat* gS,
    device const bfloat* gB, device const uint8_t* uW, device const bfloat* uS,
    device const bfloat* uB, device const half* x, uint row, uint N, uint lane,
    constant ExpertBitWidths& widths) {
    if (widths.gate == 4u && widths.up == 4u) {
        return moe_int4_gate_up_rows_simd_dev_vec_u16load(gW, gS, gB, uW, uS, uB, x, row, N, lane);
    }
    return float2(moe_affine_row(gW, gS, gB, x, row, N, lane, widths.gate),
                  moe_affine_row(uW, uS, uB, x, row, N, lane, widths.up));
}


template <typename Activation>
static inline void moe_phase1_gate_up_act_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device Activation* acts,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane,
    constant ExpertBitWidths& widths
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= top_k * F) return;
    const uint slot = rowg / F;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_gate_up_rows(gW, gS, gB, uW, uS, uB, x, f, D, lane, widths);
    if (lane == 0) acts[slot * F + f] = Activation(moe_hidden_activation(gu.x) * gu.y);
}

template <typename Activation>
static inline void moe_phase1_gate_up_act_subset_u16load_body(
    device const RoutedBlobs& routed,
    constant ExpertOffsets& routed_offsets,
    device const half* x,
    device Activation* acts,
    device const uint* active_slots,
    uint active_count,
    uint D,
    uint F,
    uint top_k,
    uint rows_per_tg,
    uint tg_idx,
    uint sg_idx,
    uint lane,
    constant ExpertBitWidths& widths
) {
    const uint rowg = tg_idx * rows_per_tg + sg_idx;
    if (rowg >= active_count * F) return;
    const uint active_idx = rowg / F;
    const uint slot = active_slots[active_idx];
    if (slot >= top_k) return;
    const uint f = rowg % F;

    device const uint8_t* base = routed.blob[slot];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* gW = base + re.gate_W_off;
    device const uint8_t* uW = base + re.up_W_off;
    device const bfloat* gS = (device const bfloat*)(base + re.gate_s_off);
    device const bfloat* uS = (device const bfloat*)(base + re.up_s_off);
    device const bfloat* gB = (device const bfloat*)(base + re.gate_b_off);
    device const bfloat* uB = (device const bfloat*)(base + re.up_b_off);

    const float2 gu = moe_gate_up_rows(gW, gS, gB, uW, uS, uB, x, f, D, lane, widths);
    if (lane == 0) acts[slot * F + f] = Activation(moe_hidden_activation(gu.x) * gu.y);
}

kernel void moe_phase1_gate_up_act_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_u16load_body(
        routed, routed_offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), rows_per_tg, tg_idx, sg_idx, lane, widths);
}

kernel void moe_phase1_gate_up_act_subset_u16load(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* x [[buffer(2)]],
    device half* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]],
    constant uint& F [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    device const uint* active_slots [[buffer(7)]],
    constant uint& active_count [[buffer(8)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg_idx [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint rows_per_tg = 8;
    moe_phase1_gate_up_act_subset_u16load_body(
        routed, routed_offsets, x, acts, active_slots, active_count,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), rows_per_tg,
        tg_idx, sg_idx, lane, widths);
}

// Flint's publisher uses BF16 expert activations. The SwiGLU product can
// exceed FP16 range even when the down-projected residual is representable.
kernel void flint_moe_phase1_bf16(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device bfloat* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]], constant uint& top_k [[buffer(6)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    moe_phase1_gate_up_act_u16load_body(routed, offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), 8u, tg, sg, lane, widths);
}

kernel void flint_moe_phase1_subset_bf16(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device bfloat* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]], constant uint& top_k [[buffer(6)]],
    device const uint* slots [[buffer(7)]], constant uint& active [[buffer(8)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    moe_phase1_gate_up_act_subset_u16load_body(routed, offsets, x, acts, slots, active,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), 8u, tg, sg, lane, widths);
}

kernel void flint_moe_phase1_float(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device float* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]], constant uint& top_k [[buffer(6)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    moe_phase1_gate_up_act_u16load_body(routed, offsets, x, acts, moe_fc_d(D), moe_fc_f(F),
        moe_fc_top_k(top_k), 8u, tg, sg, lane, widths);
}

kernel void flint_moe_phase1_subset_float(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const half* x [[buffer(2)]], device float* acts [[buffer(3)]],
    constant uint& D [[buffer(4)]], constant uint& F [[buffer(5)]], constant uint& top_k [[buffer(6)]],
    device const uint* slots [[buffer(7)]], constant uint& active [[buffer(8)]],
    constant ExpertBitWidths& widths [[buffer(9)]],
    uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    moe_phase1_gate_up_act_subset_u16load_body(routed, offsets, x, acts, slots, active,
        moe_fc_d(D), moe_fc_f(F), moe_fc_top_k(top_k), 8u, tg, sg, lane, widths);
}

template <typename Activation>
static inline float flint_down_row(device const uint8_t* W, device const bfloat* S,
    device const bfloat* B, device const Activation* x, uint row, uint N, uint lane) {
    const uint G = quant_group_size();
    const uint groups = N / G;
    device const uint8_t* weights = W + row * (N / 2u);
    device const bfloat* scales = S + row * groups;
    device const bfloat* biases = B + row * groups;
    float sum = 0.0f;
    for (uint group = 0; group < groups; ++group) {
        if (lane >= G / 2u) continue;
        uint byte = weights[group * (G / 2u) + lane];
        float x0 = float(x[group * G + 2u * lane]);
        float x1 = float(x[group * G + 2u * lane + 1u]);
        float dot = fma(float(byte & 15u), x0, float(byte >> 4u) * x1);
        sum = fma(float(scales[group]), dot, sum);
        sum = fma(float(biases[group]), x0 + x1, sum);
    }
    return simd_sum(sum);
}

kernel void flint_moe_phase2_bf16(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const bfloat* acts [[buffer(2)]], device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]], device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& F [[buffer(7)]],
    constant ExpertBitWidths& widths [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[kMaxStreamedExperts];
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    device const uint8_t* base = routed.blob[sg];
    device const bfloat* scales = (device const bfloat*)(base + offsets.down_s_off);
    device const bfloat* biases = (device const bfloat*)(base + offsets.down_b_off);
    const float value = widths.down == 4u
        ? flint_down_row(base + offsets.down_W_off, scales, biases, acts + sg * FF, d, FF, lane)
        : moe_affine_row(base + offsets.down_W_off, scales, biases, acts + sg * FF, d, FF, lane, widths.down);
    if (lane == 0) partial[sg] = float(routing_w[sg]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint e = 0; e < streamed_expert_count(); ++e) acc += partial[e];
        y[d] = half(acc);
    }
}

kernel void flint_moe_phase2_float(
    device const RoutedBlobs& routed [[buffer(0)]], constant ExpertOffsets& offsets [[buffer(1)]],
    device const float* acts [[buffer(2)]], device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]], device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]], constant uint& F [[buffer(7)]],
    constant ExpertBitWidths& widths [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[kMaxStreamedExperts];
    const uint DD = moe_fc_d(D), FF = moe_fc_f(F);
    if (d >= DD) return;
    device const uint8_t* base = routed.blob[sg];
    device const bfloat* scales = (device const bfloat*)(base + offsets.down_s_off);
    device const bfloat* biases = (device const bfloat*)(base + offsets.down_b_off);
    const float value = widths.down == 4u
        ? flint_down_row(base + offsets.down_W_off, scales, biases, acts + sg * FF, d, FF, lane)
        : moe_affine_row(base + offsets.down_W_off, scales, biases, acts + sg * FF, d, FF, lane, widths.down);
    if (lane == 0) partial[sg] = float(routing_w[sg]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float acc = float(residual[d]);
        for (uint e = 0; e < streamed_expert_count(); ++e) acc += partial[e];
        y[d] = half(acc);
    }
}

kernel void moe_phase2_down_reduce_k8(
    device const RoutedBlobs& routed [[buffer(0)]],
    constant ExpertOffsets& routed_offsets [[buffer(1)]],
    device const half* acts [[buffer(2)]],
    device const half* routing_w [[buffer(3)]],
    device const half* residual [[buffer(4)]],
    device half* y [[buffer(5)]],
    constant uint& D [[buffer(6)]],
    constant uint& F [[buffer(7)]],
    constant ExpertBitWidths& widths [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial[kMaxStreamedExperts];
    const uint DD = moe_fc_d(D);
    const uint FF = moe_fc_f(F);
    if (d >= DD) return;

    device const uint8_t* base = routed.blob[sg_idx];
    const ExpertOffsets re = routed_offsets;
    device const uint8_t* dW = base + re.down_W_off;
    device const bfloat* dS = (device const bfloat*)(base + re.down_s_off);
    device const bfloat* dB = (device const bfloat*)(base + re.down_b_off);
    device const half* act_slot = acts + sg_idx * FF;

    const float value = widths.down == 4u
        ? moe_int4_gemv_row_simd_dev_vec(dW, dS, dB, act_slot, d, FF, lane)
        : moe_affine_row(dW, dS, dB, act_slot, d, FF, lane, widths.down);
    if (lane == 0) partial[sg_idx] = float(routing_w[sg_idx]) * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_idx == 0 && lane == 0) {
        // One partial per routed expert. Unrolled to eight while eight was the
        // only count; Qwen3.8 Flash Next routes ten, and the missing two would
        // have been dropped silently rather than caught.
        float acc = float(residual[d]);
        const uint experts = streamed_expert_count();
        for (uint e = 0; e < experts; ++e) { acc += partial[e]; }
        y[d] = half(acc);
    }
}
