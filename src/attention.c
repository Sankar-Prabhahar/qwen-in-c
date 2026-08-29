#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "attention.h"
#include "simd.h"
#include "softmax.h"

void attention_gqa(const float *Q,
                   const ModelKVCache *cache,
                   int layer,
                   int pos,
                   int n_heads,
                   int n_kv_heads,
                   int head_dim,
                   float *out)
{
    if (!Q || !cache || !out || pos < 0) return;

    int seq_len = pos + 1;
    int n_rep = n_heads / n_kv_heads;
    float inv_sqrt = 1.0f / sqrtf((float)head_dim);

    /* Allocate score buffer on stack or heap */
    float score_stack[256];
    float *scores = (seq_len <= 256) ? score_stack : (float *)malloc(seq_len * sizeof(float));

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / n_rep;
        const float *q = Q + h * head_dim;
        float *head_out = out + h * head_dim;

        /* 1. Compute Q * K^T scores for tokens 0..pos */
        for (int t = 0; t < seq_len; t++) {
            const float *k_step = kv_cache_get_k(cache, layer, t);
            const float *k = k_step + kv_h * head_dim;
            scores[t] = dot_product_avx2(q, k, head_dim) * inv_sqrt;
        }

        /* 2. Softmax normalization */
        softmax(scores, seq_len);

        /* 3. Weighted sum over V */
        memset(head_out, 0, head_dim * sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            const float *v_step = kv_cache_get_v(cache, layer, t);
            const float *v = v_step + kv_h * head_dim;
            float weight = scores[t];

            for (int d = 0; d < head_dim; d++) {
                head_out[d] += weight * v[d];
            }
        }
    }

    if (scores != score_stack) {
        free(scores);
    }
}