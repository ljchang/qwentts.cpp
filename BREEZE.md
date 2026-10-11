# Breeze TTS 2 on qwentts.cpp

This branch runs [Breeze TTS 2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(BreezeBlue) on the qwentts.cpp engine. Its per-frame loop has the same shape
as Qwen3-TTS: a backbone step, a 15-pass depth model, then the 12 Hz codec.
The codec is byte-identical to Qwen3-TTS's, and the backbone has the 1.7B
talker's geometry. So the port is mostly parameters. Two pieces are new, and
both sit off the per-frame path:

- `src/text-encoder.h` — the T5Gemma2 (Gemma-3-1B-shaped) bidirectional text
  encoder. It runs once per text segment at prompt time.
- `src/gemma-tokenizer.h` — Gemma's SentencePiece-BPE tokenizer, its vocabulary
  and merges read from the GGUF.

`src/breeze-prompt.h` builds Breeze's single-stream prompt (reference audio
rows, text-encoder rows, an optional instruction). The rest is
parameterisation, all read from GGUF metadata:

- optional QK-norm and llama3 RoPE factors in the code predictor;
- the codebook suppress range and repetition penalty;
- a `breeze` model type in `qwen.cpp` and the server.

`tools/quantize.cpp` takes an optional fourth argument: a list of
`substring=type` rules, first match wins, for mixed quants.

## Converting the weights

`convert_breeze.py` turns a Breeze TTS 2 snapshot into one BF16 GGUF in the
qwen3-tts layout. It runs on the CPU, needs `torch`, `safetensors` and
`gguf`, and takes about five minutes:

```
python convert_breeze.py <Breeze-TTS-2 snapshot dir> breeze-tts2-BF16.gguf
build/quantize breeze-tts2-BF16.gguf breeze-tts2-Q5_K_M.gguf Q5_K_M
```

The codec is the stock `qwen-tokenizer-12hz-*.gguf` that qwentts.cpp already
uses.

Converted files are published at
[mecha-factory/mecha-speech-gguf](https://huggingface.co/mecha-factory/mecha-speech-gguf):
Q5_K_M and Q6_K, the codec, and a default voice. They're under the same
non-commercial licence, with a NOTICE and every file's sha256.

## Serving

```
build/tts-server --model breeze-tts2-Q5_K_M.gguf --codec qwen-tokenizer-12hz-BF16.gguf \
  --host 127.0.0.1 --port 8886 --alias breeze-tts2
```

## Checking it

The port was checked against the PyTorch reference (bf16):

- token ids are identical (`gemma-tok`);
- the prefill logits match at cosine 0.99999, with the same argmax;
- the depth decoder's teacher-forced KL against bf16 is 0.011 for Q5_K_M and
  0.0042 for Q6_K.

**Not yet ported:**

- CFG (`cfg_scale` other than 1);
- the dual-CFG `ref_edit` mode.

**Known issue:** one long-passage render in five stopped a sentence early,
so send text a sentence at a time.

## Credits and licences

- **The C++ port** (this branch) is MIT, like the rest of qwentts.cpp.
- **The Python inference code it was ported from** is BreezeBlue's
  [breeze-tts](https://github.com/breezeblue-ai/breeze-tts), under the Apache
  License 2.0. This port reimplements its model and prompt logic in C++.
- **The Breeze TTS 2 weights are not covered by either licence.** They are
  under the BreezeBlue Research and Non-Commercial License: research and
  non-commercial use only. A GGUF converted with `convert_breeze.py` is a
  derivative model under that licence, and redistributing one carries its
  conditions (section 4 of the licence).
