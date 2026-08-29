#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "ffn.h"
#include "gemv_q6k.h"

static inline float silu(float x)
{
    return x / (1.0f + expf(-x));
}

int ffn_swiglu(const Model *model,
               const Tensor *gate_tensor,
               const Tensor *up_tensor,
               const Tensor *down_tensor,
               const float *x,
               int hidden_dim,
               int intermediate_dim,
               float *out)
{
    if (!model || !gate_tensor || !up_tensor || !down_tensor || !x || !out) return 0;

    float *gate = (float *)malloc(intermediate_dim * sizeof(float));
    float *up   = (float *)malloc(intermediate_dim * sizeof(float));
    float *act  = (float *)malloc(intermediate_dim * sizeof(float));

    if (!gate || !up || !act) {
        if (gate) free(gate);
        if (up) free(up);
        if (act) free(act);
        return 0;
    }

    /* 1. gate = W_gate * x (in: hidden_dim, out: intermediate_dim) */
    if (!gemv_q6k(model, gate_tensor, x, gate, intermediate_dim, hidden_dim)) {
        free(gate); free(up); free(act);
        return 0;
    }

    /* 2. up = W_up * x (in: hidden_dim, out: intermediate_dim) */
    if (!gemv_q6k(model, up_tensor, x, up, intermediate_dim, hidden_dim)) {
        free(gate); free(up); free(act);
        return 0;
    }

    /* 3. act = silu(gate) * up */
    for (int i = 0; i < intermediate_dim; i++) {
        act[i] = silu(gate[i]) * up[i];
    }

    /* 4. out = W_down * act (in: intermediate_dim, out: hidden_dim) */
    if (!gemv_q6k(model, down_tensor, act, out, hidden_dim, intermediate_dim)) {
        free(gate); free(up); free(act);
        return 0;
    }

    free(gate);
    free(up);
    free(act);
    return 1;
}
