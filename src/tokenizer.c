#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tokenizer.h"

static uint32_t read_u32(const uint8_t *p) { return *(const uint32_t *)p; }
static uint64_t read_u64(const uint8_t *p) { return *(const uint64_t *)p; }

static const uint8_t* skip_val(const uint8_t *p, uint32_t type)
{
    switch (type) {
        case 0: case 1: case 7: return p + 1;
        case 2: case 3: return p + 2;
        case 4: case 5: case 6: return p + 4;
        case 10: case 11: case 12: return p + 8;
        case 8: {
            uint64_t len = read_u64(p);
            return p + 8 + len;
        }
        case 9: {
            uint32_t elem = read_u32(p); p += 4;
            uint64_t count = read_u64(p); p += 8;
            for (uint64_t i = 0; i < count; i++) {
                p = skip_val(p, elem);
            }
            return p;
        }
        default: return p;
    }
}

int tokenizer_init(const Model *model, const GGUFHeader *header, Tokenizer *tok)
{
    if (!model || !header || !tok) return 0;

    tok->tokens = NULL;
    tok->vocab_size = 0;
    tok->bos_id = 1;
    tok->eos_id = 2;

    const uint8_t *p = model->data + 24;

    for (uint64_t i = 0; i < header->metadata_count; i++) {
        uint64_t key_len = read_u64(p);
        p += 8;

        char key[128];
        size_t copy_len = (key_len < sizeof(key) - 1) ? key_len : sizeof(key) - 1;
        memcpy(key, p, copy_len);
        key[copy_len] = '\0';
        p += key_len;

        uint32_t val_type = read_u32(p);
        p += 4;

        if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0 && val_type == 4) {
            tok->bos_id = (int)read_u32(p);
            p += 4;
        } else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0 && val_type == 4) {
            tok->eos_id = (int)read_u32(p);
            p += 4;
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && val_type == 9) {
            uint32_t elem_type = read_u32(p); p += 4;
            uint64_t count = read_u64(p); p += 8;

            if (elem_type == 8) { /* Array of strings */
                tok->vocab_size = (int)count;
                tok->tokens = (char **)calloc(count, sizeof(char *));

                for (uint64_t t = 0; t < count; t++) {
                    uint64_t slen = read_u64(p);
                    p += 8;

                    char *s = (char *)malloc(slen + 1);
                    if (s) {
                        memcpy(s, p, slen);
                        s[slen] = '\0';
                        tok->tokens[t] = s;
                    }
                    p += slen;
                }
            } else {
                p = skip_val(p - 12, 9);
            }
        } else {
            p = skip_val(p, val_type);
        }
    }

    return (tok->tokens != NULL && tok->vocab_size > 0);
}

const char* tokenizer_decode(const Tokenizer *tok, int token_id)
{
    if (!tok || !tok->tokens || token_id < 0 || token_id >= tok->vocab_size) {
        return "";
    }
    return tok->tokens[token_id] ? tok->tokens[token_id] : "";
}

void tokenizer_free(Tokenizer *tok)
{
    if (!tok || !tok->tokens) return;

    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->tokens[i]) free(tok->tokens[i]);
    }
    free(tok->tokens);
    tok->tokens = NULL;
    tok->vocab_size = 0;
}
