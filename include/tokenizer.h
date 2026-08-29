#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "model.h"
#include "gguf.h"

typedef struct {
    char **tokens;
    int vocab_size;
    int bos_id;
    int eos_id;
} Tokenizer;

int tokenizer_init(const Model *model, const GGUFHeader *header, Tokenizer *tok);
const char* tokenizer_decode(const Tokenizer *tok, int token_id);
void tokenizer_free(Tokenizer *tok);

#endif
