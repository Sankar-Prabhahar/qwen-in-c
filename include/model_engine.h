#ifndef MODEL_ENGINE_H
#define MODEL_ENGINE_H

#include "model.h"
#include "gguf.h"
#include "tensor_index.h"
#include "block.h"
#include "tokenizer.h"
#include "kv_cache.h"
#include "sampler.h"

#define MAX_LAYERS 64

typedef struct {
    Model model;
    GGUFHeader header;
    TensorIndex index;
    Tokenizer tokenizer;

    Tensor *token_embd;
    Tensor *output_norm;
    Tensor *output;
    float *output_norm_weights;

    TransformerBlockWeights layers[MAX_LAYERS];

    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int intermediate_dim;
    int vocab_size;
    float eps;
    float theta;
} Engine;

int engine_init(Engine *engine, const char *model_path);

int engine_forward(Engine *engine,
                   ModelKVCache *cache,
                   int token_id,
                   int pos,
                   float *logits);

void engine_generate(Engine *engine,
                     const int *prompt_tokens,
                     int prompt_len,
                     int max_new_tokens,
                     SamplerConfig *config);

void engine_print_top_logits(const Engine *engine,
                             const float *logits,
                             int k);

void engine_free(Engine *engine);

#endif
