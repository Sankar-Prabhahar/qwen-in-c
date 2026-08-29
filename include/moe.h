#ifndef MOE_H
#define MOE_H

#include <stdint.h>
#include "model.h"
#include "tensor.h"

typedef struct {
    int expert_id;
    float weight;
} ActiveExpert;

typedef struct {
    Tensor *router_weight;     /* Router gating weights [hidden_dim, num_experts] */
    Tensor *gate_exps;         /* All experts gate [hidden_dim, expert_intermediate, num_experts] */
    Tensor *up_exps;           /* All experts up [hidden_dim, expert_intermediate, num_experts] */
    Tensor *down_exps;         /* All experts down [expert_intermediate, hidden_dim, num_experts] */

    /* Shared experts (optional) */
    Tensor *gate_shexp;
    Tensor *up_shexp;
    Tensor *down_shexp;
} MoEWeights;

/*
 * moe_route:
 * Computes router gating scores, finds top-k active experts,
 * and normalizes their weights via softmax.
 */
void moe_route(const float *router_logits,
               int num_experts,
               int num_active,
               ActiveExpert *selected_experts);

/*
 * moe_forward:
 * Executes one complete Mixture-of-Experts FFN layer.
 */
int moe_forward(const Model *model,
                const MoEWeights *w,
                const float *x,
                int hidden_dim,
                int expert_intermediate_dim,
                int num_experts,
                int num_active,
                int shared_intermediate_dim,
                float *out);

#endif
