#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "block.h"
#include "rmsnorm.h"
#include "gemv_q6k.h"
#include "rope.h"
#include "attention.h"
#include "ffn.h"
#include "moe.h"
#include "q6k.h"

static void read_float_tensor(const Model *model, const Tensor *t, float *dest, int n)
{
    const uint8_t *src = model->data + model->data_start + t->offset;
    if (t->type == 0) { /* F32 */
        memcpy(dest, src, n * sizeof(float));
    } else if (t->type == 1) { /* F16 */
        const uint16_t *src16 = (const uint16_t *)src;
        for (int i = 0; i < n; i++) {
            dest[i] = fp16_to_float(src16[i]);
        }
    } else {
        printf("read_float_tensor: unsupported type %u for 1D norm tensor\n", t->type);
    }
}

int load_block_weights(TensorIndex *index, int layer, TransformerBlockWeights *w)
{
    char name[128];
    memset(w, 0, sizeof(TransformerBlockWeights));

    /* 1. Attention Base Tensors */
    snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
    w->attn_norm = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
    w->attn_q = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
    w->attn_k = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
    w->attn_v = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
    w->attn_output = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", layer);
    w->ffn_norm = find_tensor(index, name);

    if (!w->attn_norm || !w->attn_q || !w->attn_k || !w->attn_v || !w->attn_output || !w->ffn_norm) {
        return 0;
    }

    /* 2. Check for Mixture-of-Experts Tensors (Qwen-MoE / Mixtral) */
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_inp.weight", layer);
    Tensor *router = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    Tensor *gate_exps = find_tensor(index, name);

    if (router && gate_exps) {
        w->is_moe = 1;
        w->moe.router_weight = router;
        w->moe.gate_exps     = gate_exps;

        snprintf(name, sizeof(name), "blk.%d.ffn_up_exps.weight", layer);
        w->moe.up_exps = find_tensor(index, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down_exps.weight", layer);
        w->moe.down_exps = find_tensor(index, name);

        /* Optional shared expert */
        snprintf(name, sizeof(name), "blk.%d.ffn_gate_shexp.weight", layer);
        w->moe.gate_shexp = find_tensor(index, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_up_shexp.weight", layer);
        w->moe.up_shexp = find_tensor(index, name);

        snprintf(name, sizeof(name), "blk.%d.ffn_down_shexp.weight", layer);
        w->moe.down_shexp = find_tensor(index, name);

        return (w->moe.router_weight && w->moe.gate_exps && w->moe.up_exps && w->moe.down_exps);
    }

    /* 3. Dense FFN Tensors (TinyLlama / LLaMA / Qwen-Dense) */
    w->is_moe = 0;
    snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
    w->ffn_gate = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
    w->ffn_up = find_tensor(index, name);

    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
    w->ffn_down = find_tensor(index, name);

    return (w->ffn_gate && w->ffn_up && w->ffn_down);
}

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
                              float theta)
{
    int q_dim  = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Temporary scratch buffers */
    float *norm_x      = (float *)malloc(hidden_dim * sizeof(float));
    float *norm_w      = (float *)malloc(hidden_dim * sizeof(float));
    float *Q           = (float *)malloc(q_dim * sizeof(float));
    float *K           = (float *)malloc(kv_dim * sizeof(float));
    float *V           = (float *)malloc(kv_dim * sizeof(float));
    float *attn_heads  = (float *)malloc(q_dim * sizeof(float));
    float *attn_proj   = (float *)malloc(hidden_dim * sizeof(float));
    float *ffn_out     = (float *)malloc(hidden_dim * sizeof(float));

    if (!norm_x || !norm_w || !Q || !K || !V || !attn_heads || !attn_proj || !ffn_out) {
        if (norm_x) free(norm_x);
        if (norm_w) free(norm_w);
        if (Q) free(Q);
        if (K) free(K);
        if (V) free(V);
        if (attn_heads) free(attn_heads);
        if (attn_proj) free(attn_proj);
        if (ffn_out) free(ffn_out);
        return 0;
    }

    /* 1. Attention Pre-RMSNorm */
    read_float_tensor(model, w->attn_norm, norm_w, hidden_dim);
    rmsnorm(norm_x, x, norm_w, hidden_dim, eps);

    /* 2. Q, K, V Projections */
    gemv_q6k(model, w->attn_q, norm_x, Q, q_dim, hidden_dim);
    gemv_q6k(model, w->attn_k, norm_x, K, kv_dim, hidden_dim);
    gemv_q6k(model, w->attn_v, norm_x, V, kv_dim, hidden_dim);

    /* 3. Rotary Positional Embedding (RoPE) */
    for (int h = 0; h < n_heads; h++) {
        rope_apply(Q + h * head_dim, head_dim, pos, theta);
    }
    for (int k = 0; k < n_kv_heads; k++) {
        rope_apply(K + k * head_dim, head_dim, pos, theta);
    }

    /* 4. Store K, V in KV Cache */
    kv_cache_put(cache, layer, pos, K, V);

    /* 5. Multi-Head / Grouped-Query Attention */
    attention_gqa(Q, cache, layer, pos, n_heads, n_kv_heads, head_dim, attn_heads);

    /* 6. Output Projection */
    gemv_q6k(model, w->attn_output, attn_heads, attn_proj, hidden_dim, q_dim);

    /* 7. Residual Connection 1: x = x + attn_proj */
    for (int i = 0; i < hidden_dim; i++) {
        x[i] += attn_proj[i];
    }

    /* 8. FFN Pre-RMSNorm */
    read_float_tensor(model, w->ffn_norm, norm_w, hidden_dim);
    rmsnorm(norm_x, x, norm_w, hidden_dim, eps);

    /* 9. Feed-Forward Network: MoE or Dense SwiGLU */
    if (w->is_moe) {
        moe_forward(model, &w->moe, norm_x, hidden_dim, intermediate_dim,
                    num_experts, num_active_experts, shared_intermediate_dim, ffn_out);
    } else {
        ffn_swiglu(model, w->ffn_gate, w->ffn_up, w->ffn_down, norm_x,
                   hidden_dim, intermediate_dim, ffn_out);
    }

    /* 10. Residual Connection 2: x = x + ffn_out */
    for (int i = 0; i < hidden_dim; i++) {
        x[i] += ffn_out[i];
    }

    free(norm_x);
    free(norm_w);
    free(Q);
    free(K);
    free(V);
    free(attn_heads);
    free(attn_proj);
    free(ffn_out);

    return 1;
}
