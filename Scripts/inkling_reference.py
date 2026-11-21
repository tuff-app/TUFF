"""NumPy reference for Inkling 2 (looped) text decoding, one token at a time.

Reads either the vendor checkpoint layout (model.llm.*, used by the tiny test
checkpoint) or the MLX 4-bit layout (model.language_model.*). Only the tensors
a token touches are read, so the 26B model runs on a small machine.
"""
import json, math, mmap, struct, sys
import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def _e4m3_table():
    t = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = float("nan")
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[b] = s * v
    return t


E4M3 = _e4m3_table()


class SafeTensors:
    def __init__(self, paths):
        self.t = {}
        self.maps = []
        for p in paths:
            f = open(p, "rb")
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.maps.append(mm)
            for k, v in header.items():
                self.t[k] = (v["dtype"], tuple(v["shape"]), mm, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1])

    def has(self, name):
        return name in self.t

    def shape(self, name):
        return self.t[name][1]

    def dtype(self, name):
        return self.t[name][0]

    def raw(self, name, index=None):
        """The tensor (or tensor[index] along axis 0) in its stored integer or float layout."""
        dt, shape, mm, a, b = self.t[name]
        np_dt = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16, "U8": np.uint8,
                 "U32": np.uint32, "F8_E4M3": np.uint8, "I64": np.int64}[dt]
        if index is None:
            return np.frombuffer(mm, dtype=np_dt, count=int(np.prod(shape)), offset=a).reshape(shape)
        inner = shape[1:]
        n = int(np.prod(inner))
        return np.frombuffer(mm, dtype=np_dt, count=n, offset=a + index * n * np.dtype(np_dt).itemsize).reshape(inner)

    def f32(self, name, index=None):
        dt = self.t[name][0]
        r = self.raw(name, index)
        if dt == "BF16":
            return (r.astype(np.uint32) << 16).view(np.float32)
        if dt == "F8_E4M3":
            return E4M3[r]
        return r.astype(np.float32)


def unpack_nibbles(u8):
    """[..., n] bytes -> [..., 2n] codes, even element in the low nibble."""
    out = np.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = u8 & 0xF
    out[..., 1::2] = u8 >> 4
    return out


def dequant_nvfp4(codes_u8, scale_u8, factor):
    """codes [rows, cols/2] bytes, scale [rows, cols/16] E4M3 bytes, factor scalar."""
    vals = E2M1[unpack_nibbles(codes_u8)]
    rows, cols = vals.shape
    vals = vals.reshape(rows, cols // 16, 16) * E4M3[scale_u8][..., None]
    return (vals * np.float32(factor)).reshape(rows, cols)


def dequant_mxfp8(w_u8, scale_u8):
    """w [rows, cols] E4M3 bytes, scale [rows, cols/32] exponent bytes."""
    vals = E4M3[w_u8]
    rows, cols = vals.shape
    vals = vals.reshape(rows, cols // 32, 32) * np.exp2(scale_u8.astype(np.float32) - 127.0)[..., None]
    return vals.reshape(rows, cols)


def dequant_affine4(w_u32, scales, biases, group=64):
    """MLX affine 4-bit: element j of a word sits at bits 4j."""
    u8 = w_u32.view(np.uint8)
    codes = unpack_nibbles(u8).astype(np.float32)
    rows, cols = codes.shape
    codes = codes.reshape(rows, cols // group, group)
    return (codes * scales[..., None] + biases[..., None]).reshape(rows, cols)


def urms(x, eps=1e-6):
    x = x.astype(np.float32)
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(eps)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def l2clip(x, threshold):
    n = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x * np.minimum(threshold / (n + 1e-6), 1.0)


class Params:
    """Weights by role, whichever layout the checkpoint uses."""

    def __init__(self, st, cfg):
        self.st = st
        self.cfg = cfg
        self.mlx = st.has("model.language_model.norm.weight")
        self.cache = {}

    def _memo(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def _linear(self, mlx_name, vendor_name):
        st = self.st
        if self.mlx:
            if st.has(mlx_name + ".scales"):
                return dequant_affine4(st.raw(mlx_name + ".weight"), st.f32(mlx_name + ".scales"),
                                       st.f32(mlx_name + ".biases"))
            return st.f32(mlx_name + ".weight")
        return st.f32(vendor_name)

    def vec(self, mlx_name, vendor_name):
        return self._memo(("v", mlx_name), lambda: self.st.f32(mlx_name if self.mlx else vendor_name))

    # --- global tables
    def embedding(self, token):
        st = self.st
        if self.mlx:
            n = "model.language_model.embed_tokens"
            if st.has(n + ".scales"):
                return dequant_affine4(st.raw(n + ".weight", token)[None], st.f32(n + ".scales", token)[None],
                                       st.f32(n + ".biases", token)[None])[0]
            return st.f32(n + ".weight", token)
        return st.f32("model.llm.embed.weight", token)

    def embed_norm(self):
        return self.vec("model.language_model.embed_norm.weight", "model.llm.embed_norm.weight")

    def final_norm(self):
        return self.vec("model.language_model.norm.weight", "model.llm.norm.weight")

    def lm_head(self):
        return self._memo("lm_head", lambda: self._linear("lm_head", "model.llm.unembed.weight"))

    def final_query(self):
        return self.vec("model.language_model.final_input.query", "model.llm.attn_res.output_query")

    def attn_query(self, layer):
        return self.vec(f"model.language_model.layers.{layer}.attention_input.query",
                        f"model.llm.attn_res.attn_queries.{layer - 1}").reshape(-1, self.cfg["hidden_size"])

    def mlp_query(self, layer):
        return self.vec(f"model.language_model.layers.{layer}.mlp_input.query",
                        f"model.llm.attn_res.mlp_queries.{layer}").reshape(1, -1)

    def message_query(self, loop_layer):
        """[4, D]: query, key, value, mlp."""
        if self.mlx:
            return self.vec("model.language_model.message_input.query", "")[loop_layer]
        q = self.st.f32("model.llm.middle_cycle_cross_pass_queries").reshape(-1, 2, self.cfg["hidden_size"])
        kv = self.st.f32("model.llm.middle_cycle_cross_pass_kv_queries").reshape(-1, 2, self.cfg["hidden_size"])
        return np.concatenate([q[loop_layer, :1], kv[loop_layer], q[loop_layer, 1:]], axis=0)

    # --- per layer
    def norm(self, layer, which):
        mlx = {"input": "input_layernorm", "q": "q_input_layernorm", "k": "k_input_layernorm",
               "v": "v_input_layernorm", "mlp": "post_attention_layernorm"}[which]
        ven = {"input": "attn_norm", "q": "q_input_norm", "k": "k_input_norm", "v": "v_input_norm",
               "mlp": "mlp_norm"}[which]
        return self.vec(f"model.language_model.layers.{layer}.{mlx}.weight", f"model.llm.layers.{layer}.{ven}.weight")

    def proj(self, layer, which):
        mlx = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "r": "r_proj", "o": "o_proj", "gate": "gate_proj",
               "sink": "sink_proj", "k_dyn": "k_sconv.weight_proj", "v_dyn": "v_sconv.weight_proj"}[which]
        ven = {"q": "wq_du.weight", "k": "wk_dv.weight", "v": "wv_dv.weight", "r": "wr_du.weight",
               "o": "wo_ud.weight", "gate": "gate_attn.weight", "sink": "attn_sink_proj.weight",
               "k_dyn": "k_sconv.weight_proj", "v_dyn": "v_sconv.weight_proj"}[which]
        return self._memo(("p", layer, which), lambda: self._linear(
            f"model.language_model.layers.{layer}.self_attn.{mlx}", f"model.llm.layers.{layer}.attn.{ven}"))

    def attn_vec(self, layer, which):
        mlx = {"q_scale": "q_l2_clip.scale", "rel_scale": "rel_l2_clip.scale", "rel_proj": "rel_logits_proj.proj",
               "k_static": "k_sconv.static_weight", "v_static": "v_sconv.static_weight"}[which]
        ven = {"q_scale": "q_l2_clip.scale", "rel_scale": "rel_l2_clip.scale", "rel_proj": "rel_logits_proj.proj",
               "k_static": "k_sconv.static_w", "v_static": "v_sconv.static_w"}[which]
        return self.vec(f"model.language_model.layers.{layer}.self_attn.{mlx}", f"model.llm.layers.{layer}.attn.{ven}")

    def conv(self, layer, which):
        """[C, K] taps, index K-1 on the current token."""
        mlx = {"k": "self_attn.k_sconv.conv1d.weight", "v": "self_attn.v_sconv.conv1d.weight",
               "attn": "post_attention_convolution.conv1d.weight", "mlp": "post_mlp_convolution.conv1d.weight"}[which]
        ven = {"k": "attn.k_sconv.weight", "v": "attn.v_sconv.weight", "attn": "attn_sconv.weight",
               "mlp": "mlp_sconv.weight"}[which]
        w = self.vec(f"model.language_model.layers.{layer}.{mlx}", f"model.llm.layers.{layer}.{ven}")
        return w.reshape(w.shape[0], -1)

    def message_gate(self, layer, which):
        mlx = {"q": "q_message_gate", "k": "k_message_gate", "v": "v_message_gate", "mlp": "mlp_message_gate"}[which]
        ven = {"q": "controllers.attn", "k": "controllers.k", "v": "controllers.v", "mlp": "controllers.mlp"}[which]
        a = f"model.language_model.layers.{layer}.{mlx}."
        b = f"model.llm.layers.{layer}.{ven}."
        return (self.vec(a + "down_proj.weight", b + "down_weight"),
                self.vec(a + "forget_gate_proj.weight", b + "forget_weight"),
                self.vec(a + "message_gate_proj.weight", b + "gain_weight"))

    def router(self, layer):
        return (self.vec(f"model.language_model.layers.{layer}.mlp.gate.weight", f"model.llm.layers.{layer}.mlp.gate.weight"),
                self.vec(f"model.language_model.layers.{layer}.mlp.gate.e_score_correction_bias",
                         f"model.llm.layers.{layer}.mlp.gate.bias"))

    def expert(self, layer, e):
        """(gate_up [2I, H] with gate and up rows alternating, down [H, I])."""
        st = self.st
        out = []
        for mlx_leaf, ven_leaf in (("gate_up_proj", "w13_weight"), ("down_proj", "w2_weight")):
            if self.mlx:
                base = f"model.language_model.layers.{layer}.mlp.experts.{mlx_leaf}"
                w = st.raw(base + ".weight", e)
                s = st.raw(base + ".scales", e)
                if st.has(base + ".global_scale"):
                    g = st.raw(base + ".global_scale")[e]
                    out.append(dequant_nvfp4(w.view(np.uint8), s, g / (6.0 * 448.0)))
                else:
                    out.append(dequant_mxfp8(w.view(np.uint8), s))
            else:
                base = f"model.llm.layers.{layer}.mlp.experts.{ven_leaf}"
                w = st.raw(base, e)
                s = st.raw(base + "_scale", e)
                if st.has(base + "_amax"):
                    f = st.raw(base + "_amax")[e] / st.raw(base + "_outer_scale_denominator")[0]
                    out.append(dequant_nvfp4(w, s, f))
                else:
                    out.append(dequant_mxfp8(w, s))
        return out[0], out[1]


class Inkling2Ref:
    def __init__(self, paths, config_path, trace=None):
        cfg = json.load(open(config_path))
        self.cfg = c = cfg["text_config"]
        loop = c.get("loop_config") or {}
        self.loop_start = c.get("loop_layer_index_start", loop.get("core_start", 0))
        self.loop_end = c.get("loop_layer_index_end", loop.get("core_end", 0))
        self.passes = c.get("loop_num_passes", loop.get("num_passes", 1))
        self.gate_rank = c.get("loop_gate_rank", loop.get("gate_rank", 0))
        self.L = c["num_hidden_layers"]
        self.D = c["hidden_size"]
        self.local = set(c["local_layer_ids"]) & set(range(self.L)) if "layer_types" not in c else \
            {i for i, t in enumerate(c["layer_types"]) if t == "hybrid_sliding"}
        self.block = c["n_layers_per_attn_res_block"]
        self.K = c.get("sconv_kernel_size", c.get("conv_kernel_size", 4))
        self.eps = c["rms_norm_eps"]
        self.p = Params(SafeTensors(paths), c)
        self.trace = trace
        self.reset()

    def reset(self):
        visits = self.L + (self.passes - 1) * (self.loop_end - self.loop_start)
        self.kv = [([], []) for _ in range(visits)]
        self.conv = [[None, None, None, None] for _ in range(visits)]
        self.pos = 0

    def note(self, name, value):
        if self.trace is not None:
            self.trace.setdefault(self.pos, {})[name] = np.array(value, dtype=np.float32, copy=True)

    # --- pieces
    def wnorm(self, x, weight):
        return urms(x, self.eps) * weight

    def depth_attention(self, states, partial, queries):
        stacked = urms(np.stack(list(states) + [partial]).astype(np.float32))  # [N, D]
        logits = (queries @ stacked.T) * (self.D ** -0.5)  # [Q, N]
        return softmax(logits) @ stacked  # [Q, D]

    def message_gate(self, layer, which, h, m):
        down, forget, gain = self.p.message_gate(layer, which)
        g = down @ np.concatenate([urms(h, self.eps), urms(m, self.eps)])
        return (1.0 - np.tanh(forget @ g)) * h + np.tanh(gain @ g) * m

    def sconv_static(self, visit, idx, x, taps):
        K = self.K
        state = self.conv[visit][idx]
        if state is None:
            state = np.zeros((K - 1, x.shape[0]), dtype=np.float32)
        window = np.concatenate([state, x[None]], axis=0)  # oldest first
        self.conv[visit][idx] = window[1:]
        out = np.einsum("tc,ct->c", window, taps)
        return (out + x) / math.sqrt(K + 1)

    def sconv_dynamic(self, visit, idx, x, static, dyn, heads):
        K = self.K
        state = self.conv[visit][idx]
        if state is None:
            state = np.zeros((K - 1, x.shape[0]), dtype=np.float32)
        window = np.concatenate([state, x[None]], axis=0)
        self.conv[visit][idx] = window[1:]
        lagged = window[::-1]  # lag 0 is the current token
        per_head = x.shape[0] // heads
        w = static.reshape(K, heads, per_head) + dyn.reshape(heads, K).T[:, :, None]
        out = np.sum(lagged.reshape(K, heads, per_head) * w, axis=0).reshape(-1)
        return (out + x) / math.sqrt(2 * K + 1)

    def attention(self, layer, visit, qh, kh, vh):
        c = self.cfg
        local = layer in self.local
        if local:
            Hq, Hkv, hd = c["swa_num_attention_heads"], c["swa_num_key_value_heads"], c["swa_head_dim"]
            # `window_override` shortens the window (not the bias table) so a
            # short run can cover keys leaving it.
            window, R = c.get("window_override", c["sliding_window_size"]), c["sliding_window_size"]
        else:
            Hq, Hkv, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
            window, R = 0, c["rel_extent"]
        p = self.p
        tau = 1.0
        if not local:
            tau = 1.0 + c["log_scaling_alpha"] * math.log(max((self.pos + 1) / c["log_scaling_n_floor"], 1.0))
        q = (p.proj(layer, "q") @ qh).reshape(Hq, hd)
        q = l2clip(q * p.attn_vec(layer, "q_scale").reshape(Hq, hd), c["qk_l2_clip_threshold"]) * tau
        k = p.proj(layer, "k") @ kh
        v = p.proj(layer, "v") @ vh
        if local:
            k = self.sconv_static(visit, 0, k, p.conv(layer, "k"))
            v = self.sconv_static(visit, 1, v, p.conv(layer, "v"))
        else:
            k = self.sconv_dynamic(visit, 0, k, p.attn_vec(layer, "k_static"), p.proj(layer, "k_dyn") @ kh, Hkv)
            v = self.sconv_dynamic(visit, 1, v, p.attn_vec(layer, "v_static"), p.proj(layer, "v_dyn") @ vh, Hkv)
        k = l2clip(k.reshape(Hkv, hd), c["qk_l2_clip_threshold"])
        v = urms(v.reshape(Hkv, hd), self.eps)
        self.note(f"v{visit}.q", q)
        self.note(f"v{visit}.k", k)
        self.note(f"v{visit}.v", v)
        keys, values = self.kv[visit]
        keys.append(k)
        values.append(v)
        if window and len(keys) > window:
            del keys[0]
            del values[0]
        rel = (p.proj(layer, "r") @ qh).reshape(Hq, -1)
        rel = l2clip(rel * p.attn_vec(layer, "rel_scale").reshape(Hq, -1), c["rel_l2_clip_threshold"]) * tau
        bias = rel @ p.attn_vec(layer, "rel_proj").reshape(-1, R)  # [Hq, R]
        sink = p.proj(layer, "sink") @ qh  # [Hq]
        Kmat = np.stack(keys)  # [S, Hkv, hd]
        Vmat = np.stack(values)
        S = Kmat.shape[0]
        dist = (S - 1) - np.arange(S)  # the newest key is the query's own
        group = Hq // Hkv
        out = np.zeros((Hq, hd), dtype=np.float32)
        for h in range(Hq):
            logits = (Kmat[:, h // group] @ q[h]) * (hd ** -0.5)
            logits = logits + np.where(dist < R, bias[h][np.minimum(dist, R - 1)], 0.0)
            probs = softmax(np.concatenate([logits, sink[h:h + 1]]).astype(np.float32))[:-1]
            out[h] = probs @ Vmat[:, h // group]
        self.note(f"v{visit}.attn", out)
        gated = out.reshape(-1) * sigmoid(p.proj(layer, "gate") @ qh)
        return p.proj(layer, "o") @ gated

    def mlp(self, layer, x):
        c = self.cfg
        w, bias = self.p.router(layer)
        scores = sigmoid(w @ x)
        k = c["num_experts_per_tok"]
        order = np.argsort(-(scores + bias), kind="stable")[:k]
        weights = scores[order] / (np.sum(scores[order]) + 1e-20) * c["route_scale"]
        self.note(f"l{layer}.route", np.sort(order))
        out = np.zeros(self.D, dtype=np.float32)
        gs, us = c["mlp_gate_softcap"], c["mlp_up_softcap"]
        for e, we in zip(order, weights):
            gate_up, down = self.p.expert(layer, int(e))
            gu = gate_up @ x
            gate, up = self.split_gate_up(gu)
            act = (np.tanh(gate / gs) * gs) * sigmoid(gate) * (np.tanh(up / us) * us)
            out += we * (down @ act)
        return out

    def split_gate_up(self, gu):
        return gu[0::2], gu[1::2]

    def layer(self, layer, visit, h, blocks, messages):
        p = self.p
        local = layer in self.local
        residual_in = h
        if layer == 0:
            xs = [h, h, h]
        else:
            got = self.depth_attention(blocks, h, p.attn_query(layer))
            xs = [got[0]] * 3 if got.shape[0] == 1 else [got[0], got[1], got[2]]
        if messages is not None:
            xs = [self.message_gate(layer, n, x, messages[i]) for i, (n, x) in enumerate(zip("qkv", xs))]
        if local:
            w = p.norm(layer, "input")
            qh, kh, vh = (self.wnorm(x, w) for x in xs)
        else:
            qh, kh, vh = (self.wnorm(x, p.norm(layer, n)) for n, x in zip("qkv", xs))
        self.note(f"v{visit}.qh", qh)
        a = self.attention(layer, visit, qh, kh, vh)
        a = self.sconv_static(visit, 2, a, p.conv(layer, "attn"))
        self.note(f"v{visit}.attn_out", a)
        if layer % self.block == 0:
            blocks = blocks + [residual_in]
            residual = a
        else:
            residual = residual_in + a
        y = self.depth_attention(blocks, residual, p.mlp_query(layer))[0]
        if messages is not None:
            y = self.message_gate(layer, "mlp", y, messages[3])
        y = self.mlp(layer, self.wnorm(y, p.norm(layer, "mlp")))
        y = self.sconv_static(visit, 3, y, p.conv(layer, "mlp"))
        out = residual + y
        self.note(f"v{visit}.out", out)
        return out, blocks

    def hidden_for(self, embedding):
        """Runs one position from its (already normalized) input embedding."""
        h = embedding.astype(np.float32)
        blocks = []
        ls, le = self.loop_start, self.loop_end
        for layer in range(ls):
            h, blocks = self.layer(layer, layer, h, blocks, None)
        pre_h, pre_blocks = h, blocks
        for layer in range(ls, le):
            h, blocks = self.layer(layer, layer, h, blocks, None)
        for loop_pass in range(1, self.passes):
            bank_states = blocks[len(pre_blocks) + 1:]
            bank_partial = h
            h, blocks = pre_h, pre_blocks
            base = self.L + (loop_pass - 1) * (le - ls) - ls
            for i, layer in enumerate(range(ls, le)):
                messages = self.depth_attention(bank_states, bank_partial, self.p.message_query(i))
                h, blocks = self.layer(layer, layer + base, h, blocks, messages)
        for layer in range(le, self.L):
            h, blocks = self.layer(layer, layer, h, blocks, None)
        final = self.depth_attention(blocks, h, self.p.final_query().reshape(1, -1))[0]
        self.pos += 1
        return self.wnorm(final, self.p.final_norm())

    def embed(self, token):
        return self.wnorm(self.p.embedding(token), self.p.embed_norm())

    def logits(self, hidden):
        c = self.cfg
        cap = c["final_logit_softcapping"]
        st = self.p.st
        n = "lm_head"
        if self.p.mlx and st.has(n + ".scales"):
            # The head is too large to dequantize whole on a small machine.
            rows = c["unpadded_vocab_size"]
            raw = np.empty(rows, dtype=np.float32)
            w, sc, bi = st.raw(n + ".weight"), st.raw(n + ".scales"), st.raw(n + ".biases")
            for a in range(0, rows, 8192):
                b = min(rows, a + 8192)
                raw[a:b] = dequant_affine4(w[a:b], (sc[a:b].astype(np.uint32) << 16).view(np.float32),
                                           (bi[a:b].astype(np.uint32) << 16).view(np.float32)) @ hidden
        else:
            raw = self.p.lm_head() @ hidden
        return (np.tanh(raw / cap) * cap)[: c["unpadded_vocab_size"]]

    def step(self, token, want_logits=True):
        hidden = self.hidden_for(self.embed(token))
        self.note("final", hidden)
        return self.logits(hidden) if want_logits else None
