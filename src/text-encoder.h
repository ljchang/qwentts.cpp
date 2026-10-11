#pragma once
// text-encoder.h: Breeze TTS 2 text encoder (T5Gemma2TextEncoder, a
// Gemma-3-1B shaped bidirectional encoder) plus its text_encoder_proj.
//
// Mirrors models/t5gemma2_compat.py :: T5Gemma2TextEncoder.forward and
// models/breeze.py :: _project_segments for text_encoder_proj_type
// "linear":
//
//   x   = embed(ids) * embed_scale            (eoi id -> eoi_embedding, unscaled)
//   per layer (26):
//     h = attn(rms(x) * (1+w_pre_attn))       bidirectional, GQA 4:1, hd 256,
//                                             QK-norm (1+w), NEOX RoPE,
//                                             sliding window 512 on "sliding" layers
//     x = x + rms(h) * (1+w_post_attn)
//     h = down(gelu_tanh(gate(rms(x)*(1+w_pre_ffn))) * up(...))
//     x = x + rms(h) * (1+w_post_ffn)
//   x   = rms(x) * (1+w_final)
//   out = text_encoder_proj(x)                1152 -> backbone hidden
//
// The (1+w) Gemma norm gains are folded into the GGUF weights by
// convert_breeze.py, so every norm here is a plain ggml_rms_norm * w.
// Scale is query_pre_attn_scalar^-0.5. RoPE: theta local_base on
// sliding layers, global_base with linear freq_scale on full layers.
//
// One graph per text segment through the pipeline sched (segments are
// encoded independently in Breeze, positions from 0). Embedding rows are
// gathered on the host from the GGUF mmap, like the Qwen prompt path.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf-weights.h"
#include "qt-error.h"
#include "weight-ctx.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct TextEncLayer {
    struct ggml_tensor * attn_pre_norm;
    struct ggml_tensor * attn_post_norm;
    struct ggml_tensor * ffn_pre_norm;
    struct ggml_tensor * ffn_post_norm;
    struct ggml_tensor * q;
    struct ggml_tensor * k;
    struct ggml_tensor * v;
    struct ggml_tensor * o;
    struct ggml_tensor * q_norm;
    struct ggml_tensor * k_norm;
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    struct ggml_tensor * down;
    bool                 full;
};

struct TextEncoderWeights {
    int   hidden      = 0;
    int   ffn         = 0;
    int   n_layers    = 0;
    int   n_heads     = 0;
    int   n_kv        = 0;
    int   head_dim    = 0;
    int   out_dim     = 0;
    int   window      = 0;
    int   eoi_id      = -1;
    float eps         = 1e-6f;
    float attn_scale  = 1.0f;
    float rope_local  = 10000.0f;
    float rope_global = 1e6f;
    float rope_gscale = 1.0f;
    float embed_scale = 1.0f;

    std::vector<TextEncLayer> layers;
    struct ggml_tensor *      out_norm = nullptr;
    struct ggml_tensor *      proj     = nullptr;
    std::vector<float>        eoi_embd;  // host copy [hidden]

    struct ggml_context * weight_ctx = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
};

static bool text_encoder_weights_load(TextEncoderWeights * te, const GGUFModel & gf, ggml_backend_t backend) {
    te->hidden      = (int) gf_get_u32(gf, "breeze.text_enc.embedding_length");
    te->ffn         = (int) gf_get_u32(gf, "breeze.text_enc.feed_forward_length");
    te->n_layers    = (int) gf_get_u32(gf, "breeze.text_enc.block_count");
    te->n_heads     = (int) gf_get_u32(gf, "breeze.text_enc.attention.head_count");
    te->n_kv        = (int) gf_get_u32(gf, "breeze.text_enc.attention.head_count_kv");
    te->head_dim    = (int) gf_get_u32(gf, "breeze.text_enc.attention.key_length");
    te->window      = (int) gf_get_u32(gf, "breeze.text_enc.sliding_window");
    te->eps         = gf_get_f32(gf, "breeze.text_enc.attention.layer_norm_rms_epsilon");
    te->attn_scale  = 1.0f / sqrtf(gf_get_f32(gf, "breeze.text_enc.query_pre_attn_scalar"));
    te->rope_local  = gf_get_f32(gf, "breeze.text_enc.rope.local_freq_base");
    te->rope_global = gf_get_f32(gf, "breeze.text_enc.rope.global_freq_base");
    te->rope_gscale = gf_get_f32(gf, "breeze.text_enc.rope.global_freq_scale");
    te->embed_scale = gf_get_f32(gf, "breeze.text_enc.embed_scale");
    te->eoi_id      = (int) gf_get_u32(gf, "breeze.text_enc.eoi_token_id");
    std::vector<uint32_t> full = gf_get_array_u32(gf, "breeze.text_enc.layer_is_full");
    if (te->n_layers <= 0 || te->hidden <= 0 || (int) full.size() != te->n_layers) {
        qt_log(QT_LOG_ERROR, "[TextEnc] FATAL: invalid hyperparameters (layers=%d hidden=%d types=%zu)", te->n_layers,
               te->hidden, full.size());
        return false;
    }

    WeightCtx wctx;
    wctx_init(&wctx, te->n_layers * 13 + 8);
    te->layers.resize((size_t) te->n_layers);
    for (int l = 0; l < te->n_layers; l++) {
        TextEncLayer & L = te->layers[(size_t) l];
        char           n[160];
#define TE_LOAD(field, suffix)                                   \
    snprintf(n, sizeof(n), "text_enc.blk.%d." suffix, l);       \
    L.field = gf_load_tensor(&wctx, gf, n);
        TE_LOAD(attn_pre_norm, "attn_pre_norm.weight")
        TE_LOAD(attn_post_norm, "attn_post_norm.weight")
        TE_LOAD(ffn_pre_norm, "ffn_pre_norm.weight")
        TE_LOAD(ffn_post_norm, "ffn_post_norm.weight")
        TE_LOAD(q, "attn_q.weight")
        TE_LOAD(k, "attn_k.weight")
        TE_LOAD(v, "attn_v.weight")
        TE_LOAD(o, "attn_output.weight")
        TE_LOAD(q_norm, "attn_q_norm.weight")
        TE_LOAD(k_norm, "attn_k_norm.weight")
        TE_LOAD(gate, "ffn_gate.weight")
        TE_LOAD(up, "ffn_up.weight")
        TE_LOAD(down, "ffn_down.weight")
#undef TE_LOAD
        L.full = full[(size_t) l] != 0;
    }
    te->out_norm = gf_load_tensor(&wctx, gf, "text_enc.output_norm.weight");
    te->proj     = gf_load_tensor(&wctx, gf, "text_enc.proj.weight");
    te->out_dim  = (int) te->proj->ne[1];

    if (!wctx_alloc(&wctx, backend)) {
        qt_log(QT_LOG_ERROR, "[TextEnc] FATAL: backend allocation failed");
        return false;
    }
    te->weight_ctx = wctx.ctx;
    te->weight_buf = wctx.buffer;

    te->eoi_embd.assign((size_t) te->hidden, 0.0f);
    {
        const float * eoi = (const float *) gf_get_data(gf, "text_enc.eoi_embd.weight");
        if (eoi) {
            std::memcpy(te->eoi_embd.data(), eoi, (size_t) te->hidden * sizeof(float));
        }
    }
    qt_log(QT_LOG_INFO,
           "[TextEnc] Loaded: %d layers, hidden %d, FFN %d, heads %d/%d, head_dim %d, window %d, "
           "rope %.0f / %.0f x%.3f, embed_scale %.4f, proj -> %d",
           te->n_layers, te->hidden, te->ffn, te->n_heads, te->n_kv, te->head_dim, te->window, (double) te->rope_local,
           (double) te->rope_global, (double) te->rope_gscale, (double) te->embed_scale, te->out_dim);
    return true;
}

static void text_encoder_weights_free(TextEncoderWeights * te) {
    if (te->weight_buf) {
        ggml_backend_buffer_free(te->weight_buf);
        te->weight_buf = nullptr;
    }
    if (te->weight_ctx) {
        ggml_free(te->weight_ctx);
        te->weight_ctx = nullptr;
    }
    te->layers.clear();
}

// Plain F32 attention over the whole segment: q [hd, T, nh], k/v
// [hd, T, nkv] (mul_mat broadcasts the kv heads over the query heads),
// mask [T, T] or NULL. Returns [hd, nh, T].
static struct ggml_tensor * text_encoder_attn(struct ggml_context * ctx,
                                              struct ggml_tensor *  q,
                                              struct ggml_tensor *  k,
                                              struct ggml_tensor *  v,
                                              struct ggml_tensor *  mask,
                                              float                 scale) {
    struct ggml_tensor * scores = ggml_mul_mat(ctx, k, q);  // [T_k, T_q, nh]
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    scores                   = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
    struct ggml_tensor * vt  = ggml_cont(ctx, ggml_transpose(ctx, v));  // [T_k, hd, nkv]
    struct ggml_tensor * out = ggml_mul_mat(ctx, vt, scores);           // [hd, T_q, nh]
    return ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));          // [hd, nh, T_q]
}

// Encode one segment of token ids. token_embd is read row by row from
// the GGUF mmap (any dtype). dst receives [T, out_dim] f32 row major.
static bool text_encoder_forward(const TextEncoderWeights * te,
                                 const GGUFModel &          gf,
                                 ggml_backend_sched_t       sched,
                                 const int32_t *            ids,
                                 int                        T,
                                 float *                    dst) {
    if (T <= 0) {
        return true;
    }
    const int H   = te->hidden;
    const int hd  = te->head_dim;
    const int nh  = te->n_heads;
    const int nkv = te->n_kv;

    // Host embedding rows.
    std::vector<float> x0((size_t) T * (size_t) H);
    {
        struct ggml_tensor * src = ggml_get_tensor(gf.meta, "text_enc.token_embd.weight");
        const uint8_t *      base = (const uint8_t *) gf_get_data(gf, "text_enc.token_embd.weight");
        if (!src || !base) {
            qt_log(QT_LOG_ERROR, "[TextEnc] token_embd missing");
            return false;
        }
        const size_t                    row_bytes = ggml_row_size(src->type, H);
        const struct ggml_type_traits * tt        = ggml_get_type_traits(src->type);
        for (int t = 0; t < T; t++) {
            float * row = x0.data() + (size_t) t * (size_t) H;
            if (ids[t] == te->eoi_id) {
                std::memcpy(row, te->eoi_embd.data(), (size_t) H * sizeof(float));
                continue;
            }
            if (ids[t] < 0 || ids[t] >= (int) src->ne[1]) {
                qt_log(QT_LOG_ERROR, "[TextEnc] token id %d out of range", ids[t]);
                return false;
            }
            const void * r = base + (size_t) ids[t] * row_bytes;
            if (src->type == GGML_TYPE_F32) {
                std::memcpy(row, r, (size_t) H * sizeof(float));
            } else {
                tt->to_float(r, row, H);
            }
            // The bf16 model computes embed(ids) * embed_scale in bf16; the
            // product rounding is below the run to run noise, kept in f32.
            for (int i = 0; i < H; i++) {
                row[i] *= te->embed_scale;
            }
        }
    }

    // Sliding window mask (only needed past half the window).
    const int  left        = (te->window + 1) / 2;
    const int  right       = te->window / 2 + 1;
    const bool need_window = te->window > 0 && T > left;

    const int    max_nodes = 40 * te->n_layers + 64;
    const size_t arena =
        ggml_tensor_overhead() * (size_t) max_nodes + ggml_graph_overhead_custom((size_t) max_nodes, false);
    struct ggml_init_params gp   = { arena, NULL, true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return false;
    }

    struct ggml_tensor * x_in = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, H, T);
    struct ggml_tensor * pos  = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
    struct ggml_tensor * mask = need_window ? ggml_new_tensor_2d(gctx, GGML_TYPE_F32, T, T) : NULL;
    ggml_set_name(x_in, "te_input");
    ggml_set_input(x_in);
    ggml_set_name(pos, "te_pos");
    ggml_set_input(pos);
    if (mask) {
        ggml_set_name(mask, "te_window_mask");
        ggml_set_input(mask);
    }

    struct ggml_tensor * x = x_in;
    for (int l = 0; l < te->n_layers; l++) {
        const TextEncLayer & L = te->layers[(size_t) l];

        struct ggml_tensor * h = ggml_mul(gctx, ggml_rms_norm(gctx, x, te->eps), L.attn_pre_norm);
        struct ggml_tensor * q = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, L.q, h), hd, nh, T);
        struct ggml_tensor * k = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, L.k, h), hd, nkv, T);
        struct ggml_tensor * v = ggml_reshape_3d(gctx, ggml_mul_mat(gctx, L.v, h), hd, nkv, T);
        q                      = ggml_mul(gctx, ggml_rms_norm(gctx, q, te->eps), L.q_norm);
        k                      = ggml_mul(gctx, ggml_rms_norm(gctx, k, te->eps), L.k_norm);

        const float base   = L.full ? te->rope_global : te->rope_local;
        const float fscale = L.full ? te->rope_gscale : 1.0f;
        q = ggml_rope_ext(gctx, q, pos, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, base, fscale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(gctx, k, pos, NULL, hd, GGML_ROPE_TYPE_NEOX, 0, base, fscale, 0.0f, 1.0f, 0.0f, 0.0f);

        struct ggml_tensor * qp = ggml_cont(gctx, ggml_permute(gctx, q, 0, 2, 1, 3));  // [hd, T, nh]
        struct ggml_tensor * kp = ggml_cont(gctx, ggml_permute(gctx, k, 0, 2, 1, 3));  // [hd, T, nkv]
        struct ggml_tensor * vp = ggml_cont(gctx, ggml_permute(gctx, v, 0, 2, 1, 3));  // [hd, T, nkv]
        struct ggml_tensor * a =
            text_encoder_attn(gctx, qp, kp, vp, L.full ? NULL : mask, te->attn_scale);  // [hd, nh, T]
        a                      = ggml_reshape_2d(gctx, a, hd * nh, T);
        struct ggml_tensor * o = ggml_mul_mat(gctx, L.o, a);
        o                      = ggml_mul(gctx, ggml_rms_norm(gctx, o, te->eps), L.attn_post_norm);
        x                      = ggml_add(gctx, x, o);

        struct ggml_tensor * h2   = ggml_mul(gctx, ggml_rms_norm(gctx, x, te->eps), L.ffn_pre_norm);
        struct ggml_tensor * gate = ggml_mul_mat(gctx, L.gate, h2);
        struct ggml_tensor * up   = ggml_mul_mat(gctx, L.up, h2);
        struct ggml_tensor * gu   = ggml_geglu_split(gctx, gate, up);
        struct ggml_tensor * m    = ggml_mul_mat(gctx, L.down, gu);
        m                         = ggml_mul(gctx, ggml_rms_norm(gctx, m, te->eps), L.ffn_post_norm);
        x                         = ggml_add(gctx, x, m);
    }
    x                      = ggml_mul(gctx, ggml_rms_norm(gctx, x, te->eps), te->out_norm);
    struct ggml_tensor * y = ggml_mul_mat(gctx, te->proj, x);  // [out_dim, T]
    ggml_set_name(y, "te_proj");
    ggml_set_output(y);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, max_nodes, false);
    ggml_build_forward_expand(graph, y);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, graph)) {
        qt_log(QT_LOG_ERROR, "[TextEnc] graph alloc failed (T=%d)", T);
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_set(x_in, x0.data(), 0, x0.size() * sizeof(float));
    {
        std::vector<int32_t> p((size_t) T);
        for (int i = 0; i < T; i++) {
            p[(size_t) i] = i;
        }
        ggml_backend_tensor_set(pos, p.data(), 0, p.size() * sizeof(int32_t));
    }
    if (mask) {
        std::vector<float> mk((size_t) T * (size_t) T, -INFINITY);
        for (int qi = 0; qi < T; qi++) {
            for (int ki = 0; ki < T; ki++) {
                const int  d  = qi - ki;
                const bool ok = (d >= 0 && d < left) || (d < 0 && -d < right);
                if (ok) {
                    mk[(size_t) qi * (size_t) T + (size_t) ki] = 0.0f;
                }
            }
        }
        ggml_backend_tensor_set(mask, mk.data(), 0, mk.size() * sizeof(float));
    }
    if (ggml_backend_sched_graph_compute(sched, graph) != GGML_STATUS_SUCCESS) {
        qt_log(QT_LOG_ERROR, "[TextEnc] graph compute failed");
        ggml_backend_sched_reset(sched);
        ggml_free(gctx);
        return false;
    }
    ggml_backend_tensor_get(y, dst, 0, (size_t) T * (size_t) te->out_dim * sizeof(float));
    ggml_backend_sched_reset(sched);
    ggml_free(gctx);
    return true;
}
