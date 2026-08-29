#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include "moe.h"
#include "softmax.h"
#include "gemv_q6k.h"
#include "ffn.h"
#include "simd.h"

typedef struct {
    int id;
    float logit;
} ExpertLogit;

static int compare_logits(const void *a, const void *b)
{
    float la = ((const ExpertLogit *)a)->logit;
    float lb = ((const ExpertLogit *)b)->logit;
    if (lb > la) return 1;
    if (lb < la) return -1;
    return 0;
}

static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

void moe_route(const float *router_logits,
               int num_experts,
               int num_active,
               ActiveExpert *selected_experts)
{
    if (!router_logits || !selected_experts || num_experts <= 0 || num_active <= 0) return;

    if (num_active > num_experts) num_active = num_experts;

    ExpertLogit stack_buf[128];
    ExpertLogit *all_experts = (num_experts <= 128) ? stack_buf : (ExpertLogit *)malloc(num_experts * sizeof(ExpertLogit));

    for (int e = 0; e < num_experts; e++) {
        all_experts[e].id = e;
        all_experts[e].logit = router_logits[e];
    }

    qsort(all_experts, num_experts, sizeof(ExpertLogit), compare_logits);

    float active_logits[64];
    for (int i = 0; i < num_active; i++) {
        selected_experts[i].expert_id = all_experts[i].id;
        active_logits[i] = all_experts[i].logit;
    }

    softmax(active_logits, num_active);

    for (int i = 0; i < num_active; i++) {
        selected_experts[i].weight = active_logits[i];
    }

    if (all_experts != stack_buf) {
        free(all_experts);
    }
}

int moe_forward(const Model *model,
                const MoEWeights *w,
                const float *x,
                int hidden_dim,
                int expert_intermediate_dim,
                int num_experts,
                int num_active,
                int shared_intermediate_dim,
                float *out)
{
    if (!model || !w || !x || !out) return 0;

    /* 1. Compute Router Logits */
    float *router_logits = (float *)malloc(num_experts * sizeof(float));
    if (!router_logits) return 0;

    if (w->router_weight->type == 14) { /* Q6_K */
        gemv_q6k(model, w->router_weight, x, router_logits, num_experts, hidden_dim);
    } else if (w->router_weight->type == 0) { /* F32 */
        const float *rw = (const float *)(model->data + model->data_start + w->router_weight->offset);
        for (int e = 0; e < num_experts; e++) {
            router_logits[e] = dot_product_avx2(x, rw + e * hidden_dim, hidden_dim);
        }
    } else {
        /* Fallback for other router types */
        gemv_q6k(model, w->router_weight, x, router_logits, num_experts, hidden_dim);
    }

    /* 2. Top-k Routing */
    ActiveExpert selected[64];
    moe_route(router_logits, num_experts, num_active, selected);
    free(router_logits);

    /* 3. Initialize accumulator */
    memset(out, 0, hidden_dim * sizeof(float));

    /* Scratch buffers for expert SwiGLU */
    float *gate_buf   = (float *)malloc(expert_intermediate_dim * sizeof(float));
    float *up_buf     = (float *)malloc(expert_intermediate_dim * sizeof(float));
    float *act_buf    = (float *)malloc(expert_intermediate_dim * sizeof(float));
    float *expert_out = (float *)malloc(hidden_dim * sizeof(float));

    if (!gate_buf || !up_buf || !act_buf || !expert_out) {
        if (gate_buf) free(gate_buf);
        if (up_buf) free(up_buf);
        if (act_buf) free(act_buf);
        if (expert_out) free(expert_out);
        return 0;
    }

    size_t gate_slice_bytes = (size_t)expert_intermediate_dim * (hidden_dim / QK_K) * sizeof(block_q6_K);
    size_t up_slice_bytes   = (size_t)expert_intermediate_dim * (hidden_dim / QK_K) * sizeof(block_q6_K);
    size_t down_slice_bytes = (size_t)hidden_dim * (expert_intermediate_dim / QK_K) * sizeof(block_q6_K);

    /* 4. Execute Selected Active Experts */
    for (int i = 0; i < num_active; i++) {
        int e = selected[i].expert_id;
        float weight = selected[i].weight;

        uint64_t gate_off = (uint64_t)e * gate_slice_bytes;
        uint64_t up_off   = (uint64_t)e * up_slice_bytes;
        uint64_t down_off = (uint64_t)e * down_slice_bytes;

        /* gate = W_gate[e] * x */
        gemv_q6k_offset(model, w->gate_exps, gate_off, x, gate_buf, expert_intermediate_dim, hidden_dim);
        /* up = W_up[e] * x */
        gemv_q6k_offset(model, w->up_exps, up_off, x, up_buf, expert_intermediate_dim, hidden_dim);

        /* act = silu(gate) * up */
        for (int k = 0; k < expert_intermediate_dim; k++) {
            act_buf[k] = silu(gate_buf[k]) * up_buf[k];
        }

        /* expert_out = W_down[e] * act */
        gemv_q6k_offset(model, w->down_exps, down_off, act_buf, expert_out, hidden_dim, expert_intermediate_dim);

        /* Accumulate weighted expert output */
        for (int d = 0; d < hidden_dim; d++) {
            out[d] += weight * expert_out[d];
        }
    }

    free(gate_buf);
    free(up_buf);
    free(act_buf);
    free(expert_out);

    /* 5. Shared Expert (if present) */
    if (w->gate_shexp && w->up_shexp && w->down_shexp && shared_intermediate_dim > 0) {
        float *shexp_out = (float *)malloc(hidden_dim * sizeof(float));
        if (shexp_out) {
            ffn_swiglu(model, w->gate_shexp, w->up_shexp, w->down_shexp,
                       x, hidden_dim, shared_intermediate_dim, shexp_out);
            for (int d = 0; d < hidden_dim; d++) {
                out[d] += shexp_out[d];
            }
            free(shexp_out);
        }
    }

    return 1;
}
