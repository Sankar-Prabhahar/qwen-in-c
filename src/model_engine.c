#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "model_engine.h"
#include "mmap.h"
#include "rmsnorm.h"
#include "gemv_q6k.h"
#include "embedding.h"
#include "q6k.h"

static void print_token_clean(const char *tok_str)
{
    if (!tok_str) return;

    /* Skip BOS tag */
    if (strcmp(tok_str, "<s>") == 0) return;

    /* Handle SentencePiece space prefix (\xe2\x96\x81) */
    const unsigned char *s = (const unsigned char *)tok_str;
    while (*s) {
        if (s[0] == 0xE2 && s[1] == 0x96 && s[2] == 0x81) {
            putchar(' ');
            s += 3;
        } else if (s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            /* Raw byte tokens like <0x0A> for newline */
            unsigned int byte_val = 0;
            if (sscanf((const char *)s, "<0x%02X>", &byte_val) == 1) {
                putchar((char)byte_val);
                s += 6;
            } else {
                putchar(*s++);
            }
        } else {
            putchar(*s++);
        }
    }
    fflush(stdout);
}

int engine_init(Engine *engine, const char *model_path)
{
    if (!engine || !model_path) return 0;
    memset(engine, 0, sizeof(Engine));

    /* Default hyperparameters for TinyLlama */
    engine->hidden_dim = 2048;
    engine->n_layers = 22;
    engine->n_heads = 32;
    engine->n_kv_heads = 4;
    engine->head_dim = 64;
    engine->intermediate_dim = 5632;
    engine->eps = 1e-5f;
    engine->theta = 10000.0f;

    if (!model_map(model_path, &engine->model)) {
        printf("engine_init: failed to mmap %s\n", model_path);
        return 0;
    }

    gguf_read_header(&engine->model, &engine->header);
    build_tensor_index(&engine->model, &engine->header, &engine->index);
    tokenizer_init(&engine->model, &engine->header, &engine->tokenizer);

    engine->token_embd  = find_tensor(&engine->index, "token_embd.weight");
    engine->output_norm = find_tensor(&engine->index, "output_norm.weight");
    engine->output      = find_tensor(&engine->index, "output.weight");

    if (!engine->token_embd) printf("engine_init: missing token_embd.weight\n");
    if (!engine->output_norm) printf("engine_init: missing output_norm.weight\n");
    if (!engine->output) printf("engine_init: missing output.weight\n");

    if (!engine->token_embd || !engine->output_norm || !engine->output) {
        printf("engine_init: missing essential base tensors\n");
        return 0;
    }

    /* Assign real dimensions from tensor metadata */
    engine->hidden_dim  = (int)engine->token_embd->dims[0];
    engine->vocab_size  = (int)engine->token_embd->dims[1];
    engine->head_dim    = engine->hidden_dim / engine->n_heads;

    /* Sanity check: ensure output_norm fits in the mmap region */
    uint64_t norm_abs_offset = engine->model.data_start + engine->output_norm->offset;
    uint64_t norm_size       = (uint64_t)engine->hidden_dim * sizeof(float);
    if (norm_abs_offset + norm_size > engine->model.file_size) {
        printf("engine_init: model file is too small (truncated?). "
               "output_norm at %llu, file_size=%llu\n",
               (unsigned long long)norm_abs_offset,
               (unsigned long long)engine->model.file_size);
        return 0;
    }

    /* Load output_norm weights into float buffer */
    engine->output_norm_weights = (float *)malloc(norm_size);
    const uint8_t *norm_src = engine->model.data + norm_abs_offset;

    if (engine->output_norm->type == 0) {
        memcpy(engine->output_norm_weights, norm_src, norm_size);
    } else {
        const uint16_t *src16 = (const uint16_t *)norm_src;
        for (int i = 0; i < engine->hidden_dim; i++) {
            engine->output_norm_weights[i] = fp16_to_float(src16[i]);
        }
    }

    /* Load all transformer layer weights */
    for (int l = 0; l < engine->n_layers; l++) {
        if (!load_block_weights(&engine->index, l, &engine->layers[l])) {
            printf("engine_init: failed to load weights for layer %d\n", l);
            return 0;
        }
    }

    printf("engine_init: loaded %d layers | hidden=%d | vocab=%d\n",
           engine->n_layers, engine->hidden_dim, engine->vocab_size);
    fflush(stdout);
    return 1;
}

int engine_forward(Engine *engine,
                   ModelKVCache *cache,
                   int token_id,
                   int pos,
                   float *logits)
{
    if (!engine || !cache || !logits) return 0;

    int hidden_dim = engine->hidden_dim;
    float *x = (float *)malloc(hidden_dim * sizeof(float));
    float *norm_x = (float *)malloc(hidden_dim * sizeof(float));

    if (!x || !norm_x) {
        if (x) free(x);
        if (norm_x) free(norm_x);
        return 0;
    }

    /* 1. Token Embedding Lookup */
    if (!embedding_lookup(&engine->model, engine->token_embd, token_id, x)) {
        free(x); free(norm_x);
        return 0;
    }

    /* 2. Loop through all 22 Transformer Layers */
    for (int l = 0; l < engine->n_layers; l++) {
        transformer_block_forward(&engine->model,
                                  &engine->layers[l],
                                  cache,
                                  l,
                                  pos,
                                  x,
                                  engine->hidden_dim,
                                  engine->n_heads,
                                  engine->n_kv_heads,
                                  engine->head_dim,
                                  engine->intermediate_dim,
                                  engine->eps,
                                  engine->theta);
    }

    /* 3. Final RMSNorm */
    rmsnorm(norm_x, x, engine->output_norm_weights, hidden_dim, engine->eps);

    /* 4. LM Head Projection -> Logits */
    gemv_q6k(&engine->model, engine->output, norm_x, logits, engine->vocab_size, hidden_dim);

    free(x);
    free(norm_x);
    return 1;
}

void engine_generate(Engine *engine,
                     const int *prompt_tokens,
                     int prompt_len,
                     int max_new_tokens,
                     SamplerConfig *config)
{
    if (!engine || !prompt_tokens || prompt_len <= 0) return;

    ModelKVCache cache;
    int max_seq = prompt_len + max_new_tokens + 64;
    kv_cache_init(&cache, engine->n_layers, max_seq, engine->n_kv_heads, engine->head_dim);

    float *logits = (float *)malloc(engine->vocab_size * sizeof(float));
    if (!logits) {
        kv_cache_free(&cache);
        return;
    }

    /* 1. Ingest Prompt Tokens */
    for (int i = 0; i < prompt_len; i++) {
        const char *tok_str = tokenizer_decode(&engine->tokenizer, prompt_tokens[i]);
        print_token_clean(tok_str);

        engine_forward(engine, &cache, prompt_tokens[i], i, logits);
    }

    /* 2. Sample first generation token */
    int next_token = sample_token(logits, engine->vocab_size, config);

    /* 3. Autoregressive Generation Loop */
    for (int step = 0; step < max_new_tokens; step++) {
        int pos = prompt_len + step;

        if (next_token == engine->tokenizer.eos_id) {
            break;
        }

        const char *tok_str = tokenizer_decode(&engine->tokenizer, next_token);
        print_token_clean(tok_str);

        if (!engine_forward(engine, &cache, next_token, pos, logits)) {
            break;
        }

        next_token = sample_token(logits, engine->vocab_size, config);
    }

    printf("\n");
    free(logits);
    kv_cache_free(&cache);
}

void engine_free(Engine *engine)
{
    if (!engine) return;

    if (engine->output_norm_weights) {
        free(engine->output_norm_weights);
        engine->output_norm_weights = NULL;
    }

    tokenizer_free(&engine->tokenizer);
    model_unmap(&engine->model);
}
