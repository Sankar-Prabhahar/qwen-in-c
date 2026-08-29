#ifndef MOE_H
#define MOE_H

#include "model.h"
#include "tensor.h"

typedef struct {
    int expert_id;
    float weight;
} ActiveExpert;

/*
 * moe_route:
 * Computes router gating scores, finds top-k active experts,
 * and normalizes their weights via softmax.
 */
void moe_route(const float *router_logits,
               int num_experts,
               int num_active,
               ActiveExpert *selected_experts);

#endif
