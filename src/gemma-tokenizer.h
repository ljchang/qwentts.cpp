#pragma once
// gemma-tokenizer.h: the Gemma 3 tokenizer Breeze TTS 2 ships
// (GemmaTokenizerFast, tokenizer.json model BPE with byte_fallback),
// reimplemented to match HF `tokenizers` token for token on TTS text.
//
// Pipeline, as HF runs it for this tokenizer.json:
//   1. added tokens (<bos>, [S0..S9], <ins_bos>, <ins_eos>, <|AUDIO|>,
//      the <unusedN> block ...) are matched verbatim on the raw text,
//      leftmost then longest, and emitted as their ids
//   2. every other span: normalizer Replace(" " -> U+2581); the Split
//      pre-tokenizer on " " never fires after that, so the whole span
//      is one BPE word (no dummy prefix)
//   3. BPE: symbols are UTF-8 characters, a character missing from the
//      vocab falls back to its <0xXX> byte tokens; merges apply by rank,
//      lowest rank first, leftmost on ties, until none applies
//   4. add_special_tokens prepends <bos> (TemplateProcessing single)
//
// The payload comes from convert_breeze.py: breeze.tokenizer.tokens,
// breeze.tokenizer.added_ids, breeze.tokenizer.merges_{left,right}.

#include "gguf.h"
#include "qt-error.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

struct GemmaTokenizer {
    std::vector<std::string>             tokens;
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<std::string, int> merge_rank;  // left + '\x01' + right -> rank
    // Added tokens bucketed by first byte, each bucket sorted longest first.
    std::vector<std::vector<std::pair<std::string, int>>> added_by_byte;
    int byte_ids[256];
    int bos_id = 2;
    int unk_id = 3;
    bool loaded = false;
};

static bool gemma_tokenizer_load(GemmaTokenizer * t, const char * gguf_path) {
    struct gguf_init_params p   = { /*no_alloc=*/true, /*ctx=*/nullptr };
    struct gguf_context *   ctx = gguf_init_from_file(gguf_path, p);
    if (!ctx) {
        qt_log(QT_LOG_ERROR, "[GemmaTok] cannot open %s", gguf_path);
        return false;
    }
    const int64_t k_tok = gguf_find_key(ctx, "breeze.tokenizer.tokens");
    const int64_t k_add = gguf_find_key(ctx, "breeze.tokenizer.added_ids");
    const int64_t k_ml  = gguf_find_key(ctx, "breeze.tokenizer.merges_left");
    const int64_t k_mr  = gguf_find_key(ctx, "breeze.tokenizer.merges_right");
    if (k_tok < 0 || k_add < 0 || k_ml < 0 || k_mr < 0) {
        qt_log(QT_LOG_ERROR, "[GemmaTok] tokenizer payload missing in %s", gguf_path);
        gguf_free(ctx);
        return false;
    }
    const size_t n_tok = gguf_get_arr_n(ctx, k_tok);
    t->tokens.resize(n_tok);
    t->tok2id.reserve(n_tok * 2);
    for (size_t i = 0; i < n_tok; i++) {
        t->tokens[i] = gguf_get_arr_str(ctx, k_tok, i);
    }
    // Vocab ids win over added-token aliases with the same text only when
    // the id is lower: HF looks up the base vocab first.
    for (size_t i = 0; i < n_tok; i++) {
        auto it = t->tok2id.find(t->tokens[i]);
        if (it == t->tok2id.end()) {
            t->tok2id.emplace(t->tokens[i], (int) i);
        }
    }
    const size_t n_m = gguf_get_arr_n(ctx, k_ml);
    if (gguf_get_arr_n(ctx, k_mr) != n_m) {
        qt_log(QT_LOG_ERROR, "[GemmaTok] merges arrays size mismatch");
        gguf_free(ctx);
        return false;
    }
    t->merge_rank.reserve(n_m * 2);
    for (size_t i = 0; i < n_m; i++) {
        std::string key = gguf_get_arr_str(ctx, k_ml, i);
        key.push_back('\x01');
        key += gguf_get_arr_str(ctx, k_mr, i);
        t->merge_rank.emplace(std::move(key), (int) i);
    }
    t->added_by_byte.assign(256, {});
    const size_t     n_add = gguf_get_arr_n(ctx, k_add);
    const uint32_t * add   = (const uint32_t *) gguf_get_arr_data(ctx, k_add);
    for (size_t i = 0; i < n_add; i++) {
        const int           id = (int) add[i];
        const std::string & s  = t->tokens[(size_t) id];
        if (s.empty()) {
            continue;
        }
        t->added_by_byte[(unsigned char) s[0]].push_back({ s, id });
    }
    for (auto & b : t->added_by_byte) {
        std::stable_sort(b.begin(), b.end(), [](const std::pair<std::string, int> & a,
                                                const std::pair<std::string, int> & c) {
            return a.first.size() > c.first.size();
        });
    }
    for (int b = 0; b < 256; b++) {
        char name[16];
        snprintf(name, sizeof(name), "<0x%02X>", b);
        auto it        = t->tok2id.find(name);
        t->byte_ids[b] = it == t->tok2id.end() ? -1 : it->second;
    }
    {
        auto it   = t->tok2id.find("<bos>");
        t->bos_id = it == t->tok2id.end() ? 2 : it->second;
        it        = t->tok2id.find("<unk>");
        t->unk_id = it == t->tok2id.end() ? 3 : it->second;
    }
    gguf_free(ctx);
    t->loaded = true;
    qt_log(QT_LOG_INFO, "[GemmaTok] Loaded %zu tokens, %zu added, %zu merges", n_tok, n_add, n_m);
    return true;
}

static int gemma_utf8_len(unsigned char c) {
    if (c < 0x80) {
        return 1;
    }
    if ((c >> 5) == 0x6) {
        return 2;
    }
    if ((c >> 4) == 0xE) {
        return 3;
    }
    if ((c >> 3) == 0x1E) {
        return 4;
    }
    return 1;  // invalid lead byte: treat as one byte, byte fallback handles it
}

// BPE over one normalized span, appending ids.
static void gemma_bpe_word(const GemmaTokenizer * t, const std::string & word, std::vector<int32_t> & out) {
    // Initial symbols: UTF-8 characters present in the vocab, else their bytes.
    std::vector<std::string> sym;
    sym.reserve(word.size());
    for (size_t i = 0; i < word.size();) {
        int n = gemma_utf8_len((unsigned char) word[i]);
        if (i + (size_t) n > word.size()) {
            n = (int) (word.size() - i);
        }
        std::string ch = word.substr(i, (size_t) n);
        if (t->tok2id.count(ch)) {
            sym.push_back(ch);
        } else {
            for (int b = 0; b < n; b++) {
                char name[16];
                snprintf(name, sizeof(name), "<0x%02X>", (unsigned char) word[i + (size_t) b]);
                sym.push_back(name);
            }
        }
        i += (size_t) n;
    }
    // Greedy merges: lowest rank, leftmost on ties.
    std::string key;
    for (;;) {
        int best_rank = INT32_MAX;
        int best_pos  = -1;
        for (size_t i = 0; i + 1 < sym.size(); i++) {
            key.assign(sym[i]);
            key.push_back('\x01');
            key += sym[i + 1];
            auto it = t->merge_rank.find(key);
            if (it != t->merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos  = (int) i;
            }
        }
        if (best_pos < 0) {
            break;
        }
        sym[(size_t) best_pos] += sym[(size_t) best_pos + 1];
        sym.erase(sym.begin() + best_pos + 1);
    }
    for (const std::string & s : sym) {
        auto it = t->tok2id.find(s);
        out.push_back(it == t->tok2id.end() ? t->unk_id : it->second);
    }
}

// Encode text; add_bos mirrors add_special_tokens=True.
static std::vector<int32_t> gemma_encode(const GemmaTokenizer * t, const std::string & text, bool add_bos) {
    std::vector<int32_t> ids;
    if (add_bos) {
        ids.push_back(t->bos_id);
    }
    std::string span;
    auto        flush = [&]() {
        if (span.empty()) {
            return;
        }
        std::string norm;
        norm.reserve(span.size() + 16);
        for (char c : span) {
            if (c == ' ') {
                norm += "\xE2\x96\x81";  // U+2581
            } else {
                norm.push_back(c);
            }
        }
        gemma_bpe_word(t, norm, ids);
        span.clear();
    };
    for (size_t i = 0; i < text.size();) {
        const auto & bucket  = t->added_by_byte[(unsigned char) text[i]];
        bool         matched = false;
        for (const auto & a : bucket) {
            if (text.compare(i, a.first.size(), a.first) == 0) {
                flush();
                ids.push_back(a.second);
                i += a.first.size();
                matched = true;
                break;
            }
        }
        if (!matched) {
            span.push_back(text[i]);
            i++;
        }
    }
    flush();
    return ids;
}
