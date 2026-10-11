#!/usr/bin/env python3
# convert_breeze.py: Breeze TTS 2 (BreezeBlue/Breeze-TTS-2) HF checkpoint -> GGUF.
#
# Writes one "talker" GGUF in the qwen3-tts layout so the qwentts.cpp
# talker / code predictor loaders read it unchanged, plus the Breeze-only
# parts (T5Gemma2 text encoder, Gemma tokenizer payload, llama3 RoPE
# factors for the depth decoder). 2D weights are stored BF16 (the
# checkpoint is BF16, F32 would add nothing), 1D tensors F32. The audio
# tokenizer is byte-identical to Qwen3-TTS-Tokenizer-12Hz, so the codec
# GGUF from convert.py is reused as is.
#
# Mapping (Breeze -> qwentts names):
#   backbone_model.layers.N.*                -> talker.blk.N.*          (Qwen3, QK-norm)
#   backbone_model.norm                      -> talker.output_norm
#   lm_head [2052, 2048]                     -> talker.codec_head        (2051 = EOS, 2048..2050 masked)
#   depth_decoder.model.embed_tokens [16*2051, 2048], tied with the backbone audio table:
#       rows [0, 2051)                       -> talker.codec_embd        (codebook 0)
#       rows [(g+1)*2051, (g+2)*2051)        -> code_pred.codec_embd.g   (codebook g+1), g = 0..14
#   depth_decoder.codebooks_head [15, 1024, 2051] -> code_pred.lm_head.g = W[g].T[:2048]
#       (rows 2048..2050 are masked to -inf by Breeze, dropping them is exact)
#   depth_decoder.model.inputs_embeds_projector -> code_pred.mtp_proj   (no bias)
#   depth_decoder.model.layers.N.*           -> code_pred.blk.N.*       (Llama, no QK-norm)
#   llama3 rope factors                      -> code_pred.rope_freqs    [64] f32
#   text_encoder.*                           -> text_enc.*              (Gemma (1+w) folded into norms)
#   text_encoder_proj                        -> text_enc.proj
#
# Dropped: embed_text_tokens (unused when the text encoder is present),
# codec_model.* (Mimi, unused at inference).
#
# Usage: convert_breeze.py <breeze snapshot dir> <out.gguf>

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import gguf

ARCH = "qwen3-tts"


def bf16_np(t: torch.Tensor) -> np.ndarray:
    t = t.to(torch.bfloat16).contiguous()
    return t.view(torch.int16).numpy().view(np.uint16)


def f32_np(t: torch.Tensor) -> np.ndarray:
    return t.float().contiguous().numpy()


def llama3_freq_factors(cfg: dict) -> np.ndarray:
    # Same arithmetic as transformers _compute_llama3_parameters, in float32.
    # ggml divides theta by the factor, so factor = inv_freq / inv_freq_llama3.
    base = float(cfg["rope_theta"])
    dim = int(cfg["head_dim"])
    rs = cfg["rope_scaling"]
    factor = float(rs["factor"])
    low = float(rs["low_freq_factor"])
    high = float(rs["high_freq_factor"])
    old_ctx = float(rs["original_max_position_embeddings"])
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(torch.float32) / dim))
    low_wl = old_ctx / low
    high_wl = old_ctx / high
    wavelen = 2 * math.pi / inv_freq
    inv_l = torch.where(wavelen > low_wl, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * inv_l / factor + smooth * inv_l
    is_medium = ~(wavelen < high_wl) * ~(wavelen > low_wl)
    inv_l = torch.where(is_medium, smoothed, inv_l)
    return (inv_freq / inv_l).float().numpy()


LM_SUFFIX = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}

TE_SUFFIX = {
    "pre_self_attn_layernorm.weight": ("attn_pre_norm.weight", True),
    "post_self_attn_layernorm.weight": ("attn_post_norm.weight", True),
    "pre_feedforward_layernorm.weight": ("ffn_pre_norm.weight", True),
    "post_feedforward_layernorm.weight": ("ffn_post_norm.weight", True),
    "self_attn.q_norm.weight": ("attn_q_norm.weight", True),
    "self_attn.k_norm.weight": ("attn_k_norm.weight", True),
    "self_attn.q_proj.weight": ("attn_q.weight", False),
    "self_attn.k_proj.weight": ("attn_k.weight", False),
    "self_attn.v_proj.weight": ("attn_v.weight", False),
    "self_attn.o_proj.weight": ("attn_output.weight", False),
    "mlp.gate_proj.weight": ("ffn_gate.weight", False),
    "mlp.up_proj.weight": ("ffn_up.weight", False),
    "mlp.down_proj.weight": ("ffn_down.weight", False),
}


def load_tokenizer(snap: Path):
    t = json.loads((snap / "tokenizer.json").read_text())
    model = t["model"]
    assert model["type"] == "BPE" and model.get("byte_fallback"), "expected a byte-fallback BPE"
    norm = t["normalizer"]
    assert norm["type"] == "Replace" and norm["pattern"]["String"] == " " and norm["content"] == "▁", norm
    vocab = model["vocab"]
    added = t["added_tokens"]
    n = max(max(vocab.values()), max(a["id"] for a in added)) + 1
    tokens = [""] * n
    types = [5] * n  # 5 = unused
    for s, i in vocab.items():
        tokens[i] = s
        types[i] = 1  # normal
    added_ids = []
    for a in added:
        tokens[a["id"]] = a["content"]
        types[a["id"]] = 3 if a["special"] else 4  # 3 = control, 4 = user defined
        added_ids.append(a["id"])
    for i in range(n):
        if tokens[i] == "":
            tokens[i] = f"<unused-slot-{i}>"
    merges = model["merges"]
    left, right = [], []
    for m in merges:
        if isinstance(m, str):
            a, b = m.split(" ", 1)
        else:
            a, b = m
        left.append(a)
        right.append(b)
    return tokens, types, sorted(added_ids), left, right


def main() -> int:
    snap = Path(sys.argv[1])
    out = Path(sys.argv[2])
    cfg = json.loads((snap / "config.json").read_text())
    gen = json.loads((snap / "generation_config.json").read_text())
    bb = cfg["backbone_config"]
    dd = cfg["depth_decoder_config"]
    te = cfg["text_encoder_config"]
    V = int(cfg["vocab_size"])  # 2051 per codebook
    K = int(cfg["num_codebooks"])  # 16
    CB = int(cfg["codec_config"]["codebook_size"])  # 2048
    assert cfg["backbone_model_type"] == "qwen3"
    assert cfg.get("text_encoder_proj_type", "linear") == "linear"
    assert bb.get("rope_scaling") is None
    assert dd["rope_scaling"]["rope_type"] == "llama3"
    assert int(cfg["audio_embed_size"]) == int(cfg["hidden_size"])

    w = gguf.GGUFWriter(str(out), ARCH, use_temp_file=True)
    w.add_string("general.name", "Breeze-TTS-2")
    w.add_string("qwen3-tts.tokenizer_type", "qwen3_tts_tokenizer_12hz")
    w.add_string("qwen3-tts.model_size", "1.7b")
    w.add_string("qwen3-tts.model_type", "breeze")
    w.add_uint32("qwen3-tts.num_code_groups", K)

    # backbone = talker
    w.add_uint32("qwen3-tts.talker.embedding_length", bb["hidden_size"])
    w.add_uint32("qwen3-tts.talker.feed_forward_length", bb["intermediate_size"])
    w.add_uint32("qwen3-tts.talker.block_count", bb["num_hidden_layers"])
    w.add_uint32("qwen3-tts.talker.attention.head_count", bb["num_attention_heads"])
    w.add_uint32("qwen3-tts.talker.attention.head_count_kv", bb["num_key_value_heads"])
    w.add_uint32("qwen3-tts.talker.attention.key_length", bb["head_dim"])
    w.add_uint32("qwen3-tts.talker.vocab_size", V + 1)  # 2052, EOS = 2051
    w.add_uint32("qwen3-tts.talker.context_length", cfg["max_position_embeddings"])
    w.add_float32("qwen3-tts.talker.rope.freq_base", float(bb["rope_theta"]))
    w.add_float32("qwen3-tts.talker.attention.layer_norm_rms_epsilon", float(bb["rms_norm_eps"]))
    w.add_bool("qwen3-tts.talker.mrope_interleaved", False)

    # depth decoder = code predictor
    w.add_uint32("qwen3-tts.code_pred.embedding_length", dd["hidden_size"])
    w.add_uint32("qwen3-tts.code_pred.feed_forward_length", dd["intermediate_size"])
    w.add_uint32("qwen3-tts.code_pred.block_count", dd["num_hidden_layers"])
    w.add_uint32("qwen3-tts.code_pred.attention.head_count", dd["num_attention_heads"])
    w.add_uint32("qwen3-tts.code_pred.attention.head_count_kv", dd["num_key_value_heads"])
    w.add_uint32("qwen3-tts.code_pred.attention.key_length", dd["head_dim"])
    w.add_uint32("qwen3-tts.code_pred.vocab_size", CB)
    w.add_uint32("qwen3-tts.code_pred.context_length", dd["max_position_embeddings"])
    w.add_float32("qwen3-tts.code_pred.rope.freq_base", float(dd["rope_theta"]))
    w.add_float32("qwen3-tts.code_pred.attention.layer_norm_rms_epsilon", float(dd["rms_norm_eps"]))

    # codec specials the generic loop reads: EOS of the backbone head
    w.add_uint32("qwen3-tts.codec.eos_id", V)
    for k in ("pad_id", "bos_id", "think_id", "nothink_id", "think_bos_id", "think_eos_id"):
        w.add_uint32(f"qwen3-tts.codec.{k}", 0)
    for k in ("im_start_id", "im_end_id", "tts_pad_id", "tts_bos_id", "tts_eos_id"):
        w.add_uint32(f"qwen3-tts.text.{k}", 0)

    # Breeze specifics
    w.add_uint32("breeze.c0_suppress_lo", CB)
    w.add_uint32("breeze.c0_suppress_hi", V)
    w.add_uint32("breeze.codebook_eos_token_id", int(cfg["codebook_eos_token_id"]))
    w.add_uint32("breeze.audio_token_id", int(cfg["audio_token_id"]))
    w.add_uint32("breeze.audio_eos_token_id", int(cfg["audio_eos_token_id"]))
    w.add_uint32("breeze.text.bos_id", int(te["bos_token_id"]))
    w.add_float32("breeze.generation.temperature", float(gen.get("temperature", 0.9)))
    w.add_float32("breeze.generation.depth_temperature", float(gen.get("depth_decoder_temperature", 0.9)))
    w.add_float32("breeze.generation.repetition_penalty", 1.1)  # breeze_infer/api.py REPETITION_PENALTY
    w.add_uint32("breeze.generation.top_k", 50)

    # text encoder (T5Gemma2 / Gemma-3-1B shaped, bidirectional)
    lt = [1 if x == "full_attention" else 0 for x in te["layer_types"]]
    rp = te["rope_parameters"]
    assert rp["sliding_attention"].get("rope_type", "default") == "default"
    assert rp["full_attention"]["rope_type"] == "linear"
    assert te.get("attn_logit_softcapping") is None
    assert te["hidden_activation"] == "gelu_pytorch_tanh"
    w.add_uint32("breeze.text_enc.embedding_length", te["hidden_size"])
    w.add_uint32("breeze.text_enc.feed_forward_length", te["intermediate_size"])
    w.add_uint32("breeze.text_enc.block_count", te["num_hidden_layers"])
    w.add_uint32("breeze.text_enc.attention.head_count", te["num_attention_heads"])
    w.add_uint32("breeze.text_enc.attention.head_count_kv", te["num_key_value_heads"])
    w.add_uint32("breeze.text_enc.attention.key_length", te["head_dim"])
    w.add_uint32("breeze.text_enc.sliding_window", te["sliding_window"])
    w.add_float32("breeze.text_enc.attention.layer_norm_rms_epsilon", float(te["rms_norm_eps"]))
    w.add_float32("breeze.text_enc.query_pre_attn_scalar", float(te["query_pre_attn_scalar"]))
    w.add_float32("breeze.text_enc.rope.local_freq_base", float(rp["sliding_attention"]["rope_theta"]))
    w.add_float32("breeze.text_enc.rope.global_freq_base", float(rp["full_attention"]["rope_theta"]))
    w.add_float32("breeze.text_enc.rope.global_freq_scale", 1.0 / float(rp["full_attention"]["factor"]))
    # Embedding scale as the bf16 model applies it: embed_scale.to(bf16).
    es = float(torch.tensor(te["hidden_size"] ** 0.5).to(torch.bfloat16).float())
    w.add_float32("breeze.text_enc.embed_scale", es)
    w.add_uint32("breeze.text_enc.eoi_token_id", int(te["eoi_token_index"]))
    w.add_array("breeze.text_enc.layer_is_full", lt)

    tokens, types, added_ids, mleft, mright = load_tokenizer(snap)
    w.add_string("breeze.tokenizer.model", "gemma-bpe-bytefallback")
    w.add_array("breeze.tokenizer.tokens", tokens)
    w.add_array("breeze.tokenizer.token_type", types)
    w.add_array("breeze.tokenizer.added_ids", added_ids)
    w.add_array("breeze.tokenizer.merges_left", mleft)
    w.add_array("breeze.tokenizer.merges_right", mright)
    print(f"[Convert] tokenizer: {len(tokens)} tokens, {len(added_ids)} added, {len(mleft)} merges")

    rope_ff = llama3_freq_factors(dd)
    print(f"[Convert] llama3 freq factors: min {rope_ff.min():.4f} max {rope_ff.max():.4f}")
    w.add_tensor("code_pred.rope_freqs.weight", rope_ff.astype(np.float32))

    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted(set(idx.values()))
    handles = {f: safe_open(str(snap / f), framework="pt") for f in files}

    def get(name):
        return handles[idx[name]].get_tensor(name)

    n = 0
    skipped = []

    def put(name, t, f32=False):
        nonlocal n
        if f32 or t.dim() == 1:
            w.add_tensor(name, f32_np(t))
        else:
            w.add_tensor(name, bf16_np(t), raw_dtype=gguf.GGMLQuantizationType.BF16)
        n += 1

    for name in sorted(idx.keys()):
        if name.startswith("codec_model.") or name == "embed_text_tokens.weight":
            skipped.append(name)
            continue
        t = get(name)
        if name.startswith("backbone_model.layers."):
            rest = name[len("backbone_model.layers."):]
            li, suf = rest.split(".", 1)
            put(f"talker.blk.{li}.{LM_SUFFIX[suf]}", t)
        elif name == "backbone_model.norm.weight":
            put("talker.output_norm.weight", t)
        elif name == "lm_head.weight":
            assert tuple(t.shape) == (V + 1, bb["hidden_size"])
            put("talker.codec_head.weight", t)
        elif name == "depth_decoder.model.embed_tokens.weight":
            assert tuple(t.shape) == (K * V, cfg["audio_embed_size"])
            put("talker.codec_embd.weight", t[0:V])
            for g in range(K - 1):
                put(f"code_pred.codec_embd.{g}.weight", t[(g + 1) * V:(g + 2) * V])
        elif name == "depth_decoder.codebooks_head.weight":
            assert tuple(t.shape) == (K - 1, dd["hidden_size"], V)
            for g in range(K - 1):
                put(f"code_pred.lm_head.{g}.weight", t[g].t().contiguous()[:CB])
        elif name == "depth_decoder.model.inputs_embeds_projector.weight":
            put("code_pred.mtp_proj.weight", t)
        elif name == "depth_decoder.model.norm.weight":
            put("code_pred.output_norm.weight", t)
        elif name.startswith("depth_decoder.model.layers."):
            rest = name[len("depth_decoder.model.layers."):]
            li, suf = rest.split(".", 1)
            put(f"code_pred.blk.{li}.{LM_SUFFIX[suf]}", t)
        elif name == "text_encoder.embed_tokens.weight":
            put("text_enc.token_embd.weight", t)
        elif name == "text_encoder.embed_tokens.eoi_embedding":
            put("text_enc.eoi_embd.weight", t, f32=True)
        elif name == "text_encoder.norm.weight":
            put("text_enc.output_norm.weight", 1.0 + t.float(), f32=True)
        elif name.startswith("text_encoder.layers."):
            rest = name[len("text_encoder.layers."):]
            li, suf = rest.split(".", 1)
            new, is_norm = TE_SUFFIX[suf]
            if is_norm:
                put(f"text_enc.blk.{li}.{new}", 1.0 + t.float(), f32=True)
            else:
                put(f"text_enc.blk.{li}.{new}", t)
        elif name == "text_encoder_proj.weight":
            put("text_enc.proj.weight", t)
        else:
            raise SystemExit(f"[Convert] unhandled tensor {name} {tuple(t.shape)}")

    print(f"[Convert] tensors written: {n}; dropped: {len(skipped)} (codec_model.*, embed_text_tokens)")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[Convert] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
