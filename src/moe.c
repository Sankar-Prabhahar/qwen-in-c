#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "moe.h"
#include "softmax.h"

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
