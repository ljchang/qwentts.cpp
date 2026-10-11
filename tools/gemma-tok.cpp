// gemma-tok: print the Breeze Gemma tokenizer ids of every stdin line,
// one JSON array per line (parity test against HF tokenizers).
// Usage: gemma-tok <breeze.gguf> [--bos] < lines.txt
#include "gemma-tokenizer.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <breeze.gguf> [--bos] < lines\n", argv[0]);
        return 1;
    }
    const bool     bos = argc > 2 && std::strcmp(argv[2], "--bos") == 0;
    GemmaTokenizer t;
    if (!gemma_tokenizer_load(&t, argv[1])) {
        return 1;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::vector<int32_t> ids = gemma_encode(&t, line, bos);
        printf("[");
        for (size_t i = 0; i < ids.size(); i++) {
            printf("%s%d", i ? ", " : "", ids[i]);
        }
        printf("]\n");
    }
    return 0;
}
