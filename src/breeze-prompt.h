#pragma once
// breeze-prompt.h: Breeze TTS 2 backbone prefix, mirroring
// breeze_infer/templates.py (ref_clone_tata / ref_edit_tata / tts_plain /
// tts_instruction) and breeze.py :: _merge_input_ids_with_input_values.
//
// One embedding stream, no text/codec pad alignment:
//
//   [ref]  T5Gemma2("<bos>[S0]{ref_text}")          one row per token
//          sum_k codebook_k(ref_codes[k, t])          one row per ref frame
//          sum_k codebook_k(codebook_eos)             <|audio_eos|> row
//   [tgt]  T5Gemma2("<bos>[S0][<ins_bos>{instr}<ins_eos>]{text}")
//
// Every text segment is encoded on its own (positions from 0) and
// projected 1152 -> 2048 by text_encoder_proj. Generation starts right
// after the last target token; there is no BOS frame and no trailing
// text overlay (T_trailing = 0, the overlay row is zeros).
//
// Text segment encodings are cached by token ids (a voice's reference
// transcript repeats across requests).

#include "gemma-tokenizer.h"
#include "pipeline-tts.h"
#include "prompt-builder.h"
#include "qt-error.h"
#include "text-encoder.h"

#include <cstring>
#include <exception>
#include <string>
#include <vector>

static const std::vector<float> * breeze_segment_rows(PipelineTTS * pt, const std::vector<int32_t> & ids) {
    std::string key;
    key.reserve(ids.size() * 7);
    for (int32_t id : ids) {
        key += std::to_string(id);
        key.push_back(',');
    }
    for (size_t i = 0; i < pt->breeze_seg_cache.size(); i++) {
        if (pt->breeze_seg_cache[i].first == key) {
            return &pt->breeze_seg_cache[i].second;
        }
    }
    std::vector<float> rows((size_t) ids.size() * (size_t) pt->text_enc.out_dim);
    if (!text_encoder_forward(&pt->text_enc, pt->gguf_talker, pt->sched, ids.data(), (int) ids.size(), rows.data())) {
        return nullptr;
    }
    if (pt->breeze_seg_cache.size() >= 32) {
        pt->breeze_seg_cache.erase(pt->breeze_seg_cache.begin());
    }
    pt->breeze_seg_cache.push_back({ key, std::move(rows) });
    return &pt->breeze_seg_cache.back().second;
}

static bool prompt_builder_build_breeze(PipelineTTS *         pt,
                                        const std::string &   utterance_text,
                                        const std::string &   instruct_text,
                                        const std::string &   ref_text,
                                        const int32_t *       ref_codes,
                                        int                   ref_codes_T,
                                        PromptBuilderOutput * out) {
    try {
        const int  H   = pt->talker.hidden_size;
        const int  K   = pt->num_code_groups;
        const bool ref = ref_codes != NULL && ref_codes_T > 0 && !ref_text.empty();
        if (pt->text_enc.out_dim != H) {
            qt_log(QT_LOG_ERROR, "[BreezePrompt] text encoder proj %d != backbone hidden %d", pt->text_enc.out_dim, H);
            return false;
        }
        const std::string spk = "[S0]";

        out->input_embed.clear();
        out->prompt_ids.clear();

        auto add_text = [&](const std::string & s) -> int {
            std::vector<int32_t> ids = gemma_encode(&pt->gtok, s, /*add_bos=*/true);
            const std::vector<float> * rows = breeze_segment_rows(pt, ids);
            if (!rows) {
                return -1;
            }
            out->input_embed.insert(out->input_embed.end(), rows->begin(), rows->end());
            out->prompt_ids.insert(out->prompt_ids.end(), ids.begin(), ids.end());
            return (int) ids.size();
        };

        if (ref) {
            if (add_text(spk + ref_text) < 0) {
                return false;
            }
            std::vector<float> a((size_t) ref_codes_T * (size_t) H);
            project_ref_codes_backend(pt, ref_codes, ref_codes_T, a.data());
            out->input_embed.insert(out->input_embed.end(), a.begin(), a.end());
            out->prompt_ids.insert(out->prompt_ids.end(), (size_t) ref_codes_T, pt->breeze_audio_token_id);

            std::vector<int32_t> eos_frame((size_t) K, pt->breeze_cb_eos);
            std::vector<float>   e((size_t) H);
            project_ref_codes_backend(pt, eos_frame.data(), 1, e.data());
            out->input_embed.insert(out->input_embed.end(), e.begin(), e.end());
            out->prompt_ids.push_back(pt->breeze_audio_eos_id);
        }

        std::string tgt = spk;
        if (!instruct_text.empty()) {
            tgt += "<ins_bos>";
            tgt += instruct_text;
            tgt += "<ins_eos>";
        }
        tgt += utterance_text;
        const int n_tgt = add_text(tgt);
        if (n_tgt < 0) {
            return false;
        }

        out->hidden = H;
        out->T_ctx  = (int) (out->input_embed.size() / (size_t) H);
        out->trailing_text_hidden.clear();
        out->T_trailing = 0;
        out->tts_pad_embed.assign((size_t) H, 0.0f);
        out->N_text = n_tgt;
        qt_log(QT_LOG_INFO, "[BreezePrompt] T_ctx=%d (ref frames %d, target tokens %d, instruct %s)", out->T_ctx,
               ref ? ref_codes_T : 0, n_tgt, instruct_text.empty() ? "no" : "yes");
        return true;
    } catch (const std::exception & ex) {
        qt_set_error("prompt_builder_build_breeze: %s", ex.what());
        qt_log(QT_LOG_ERROR, "[BreezePrompt] %s", ex.what());
        return false;
    }
}
