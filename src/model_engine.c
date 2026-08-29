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

    if (!model_map(model_path, &engine->model)) {
        printf("engine_init: failed to mmap %s\n", model_path);
        return 0;
    }

    gguf_read_header(&engine->model, &engine->header);
    build_tensor_index(&engine->model, &engine->header, &engine->index);
    tokenizer_init(&engine->model, &engine->header, &engine->tokenizer);

    /* 1. Architecture Detection (qwen2, qwen3, qwen2moe, llama, etc.) */
    char arch[64] = "llama";
    if (gguf_get_meta_str(&engine->model, &engine->header, "general.architecture", arch, sizeof(arch))) {
        snprintf(engine->arch, sizeof(engine->arch), "%s", arch);
    } else {
        snprintf(engine->arch, sizeof(engine->arch), "llama");
    }

    /* 2. Hyperparameter Extraction */
    char key[128];
    uint32_t val_u32 = 0;
    float val_f32 = 0.0f;

    snprintf(key, sizeof(key), "%s.block_count", engine->arch);
    engine->n_layers = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 22;

    snprintf(key, sizeof(key), "%s.embedding_length", engine->arch);
    engine->hidden_dim = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 2048;

    snprintf(key, sizeof(key), "%s.attention.head_count", engine->arch);
    engine->n_heads = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 32;

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", engine->arch);
    engine->n_kv_heads = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 4;

    snprintf(key, sizeof(key), "%s.feed_forward_length", engine->arch);
    engine->intermediate_dim = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 5632;

    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", engine->arch);
    engine->eps = gguf_get_meta_f32(&engine->model, &engine->header, key, &val_f32) ? val_f32 : 1e-5f;

    snprintf(key, sizeof(key), "%s.rope.freq_base", engine->arch);
    engine->theta = gguf_get_meta_f32(&engine->model, &engine->header, key, &val_f32) ? val_f32 : 10000.0f;

    /* 3. Mixture-of-Experts Parameters (Qwen3-30B-A3B / Qwen-MoE) */
    snprintf(key, sizeof(key), "%s.expert_count", engine->arch);
    if (gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) && val_u32 > 0) {
        engine->is_moe = 1;
        engine->num_experts = (int)val_u32;

        snprintf(key, sizeof(key), "%s.expert_used_count", engine->arch);
        engine->num_active_experts = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 4;

        snprintf(key, sizeof(key), "%s.expert_feed_forward_length", engine->arch);
        engine->expert_intermediate_dim = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : engine->intermediate_dim;

        snprintf(key, sizeof(key), "%s.expert_shared_feed_forward_length", engine->arch);
        engine->shared_intermediate_dim = gguf_get_meta_u32(&engine->model, &engine->header, key, &val_u32) ? (int)val_u32 : 0;
    } else {
        engine->is_moe = 0;
        engine->num_experts = 0;
        engine->num_active_experts = 0;
        engine->expert_intermediate_dim = 0;
        engine->shared_intermediate_dim = 0;
    }

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

    printf("engine_init: loaded architecture '%s' | %d layers | hidden=%d | vocab=%d%s\n",
           engine->arch, engine->n_layers, engine->hidden_dim, engine->vocab_size,
           engine->is_moe ? " [MoE Enabled]" : "");
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

    /* 2. Loop through all Transformer Layers (Dense or MoE) */
    for (int l = 0; l < engine->n_layers; l++) {
        int intermediate = engine->layers[l].is_moe ? engine->expert_intermediate_dim : engine->intermediate_dim;
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
                                  intermediate,
                                  engine->num_experts,
                                  engine->num_active_experts,
                                  engine->shared_intermediate_dim,
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

typedef struct {
    int id;
    float logit;
} LogitPair;

static int compare_logit_pairs(const void *a, const void *b)
{
    float la = ((const LogitPair *)a)->logit;
    float lb = ((const LogitPair *)b)->logit;
    if (lb > la) return 1;
    if (lb < la) return -1;
    return 0;
}

void engine_print_top_logits(const Engine *engine, const float *logits, int k)
{
    if (!engine || !logits || k <= 0) return;

    int n = engine->vocab_size;
    LogitPair *pairs = (LogitPair *)malloc(n * sizeof(LogitPair));
    if (!pairs) return;

    for (int i = 0; i < n; i++) {
        pairs[i].id = i;
        pairs[i].logit = logits[i];
    }

    qsort(pairs, n, sizeof(LogitPair), compare_logit_pairs);

    if (k > n) k = n;
    printf("    Top-%d Logits:\n", k);
    for (int i = 0; i < k; i++) {
        const char *tok = tokenizer_decode(&engine->tokenizer, pairs[i].id);
        printf("      [%2d] ID: %5d | Logit: %+8.4f | Token: \"", i + 1, pairs[i].id, pairs[i].logit);
        print_token_clean(tok);
        printf("\"\n");
    }
    free(pairs);
    fflush(stdout);
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

