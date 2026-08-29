#ifndef BLOCK_H
#define BLOCK_H

#include "model.h"
#include "tensor_index.h"
#include "kv_cache.h"
#include "moe.h"

/*
 * TransformerBlockWeights: holds cached tensor pointers for a layer.
 */
typedef struct {
    Tensor *attn_norm;
    Tensor *attn_q;
    Tensor *attn_k;
    Tensor *attn_v;
    Tensor *attn_output;
    Tensor *ffn_norm;

    /* Dense FFN */
    Tensor *ffn_gate;
    Tensor *ffn_up;
    Tensor *ffn_down;

    /* Mixture-of-Experts FFN */
    int is_moe;
    MoEWeights moe;
} TransformerBlockWeights;

int load_block_weights(TensorIndex *index, int layer, TransformerBlockWeights *w);

/*
 * transformer_block_forward:
 * Executes one complete transformer layer block for a single token at position `pos`.
 * Supports both Dense SwiGLU and Mixture-of-Experts (Qwen3-30B-A3B).
 * Modifies x in-place.
 */
int transformer_block_forward(const Model *model,
                              const TransformerBlockWeights *w,
                              ModelKVCache *cache,
                              int layer,
                              int pos,
                              float *x,
                              int hidden_dim,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              int intermediate_dim,
                              int num_experts,
                              int num_active_experts,
                              int shared_intermediate_dim,
                              float eps,
                              float theta);

#endif
