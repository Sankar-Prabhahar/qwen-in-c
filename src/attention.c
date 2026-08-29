#include <math.h>
#include <stdlib.h>
#include "kv_cache.h"
#include "attention.h"
#include "simd.h"
#include "softmax.h"

void attention_head(const float *Q,
                    const float *K,
                    const float *V,
                    float *out,
                    int seq_len,
                    int head_dim)
{
    float *scores = (float*)malloc(seq_len * sizeof(float));

    float inv = 1.0f / sqrtf((float)head_dim);

    for(int i=0;i<seq_len;i++)
        scores[i] = dot_product_avx2(Q, K + i*head_dim, head_dim) * inv;

    softmax(scores, seq_len);

    for(int d=0; d<head_dim; d++)
        out[d] = 0.0f;

    for(int i=0;i<seq_len;i++)
        for(int d=0; d<head_dim; d++)
            out[d] += scores[i] * V[i*head_dim + d];

    free(scores);
}
void attention_cached(const float *Q,
                      KVCache *cache,
                      float *out)
{
    float *scores = (float*)malloc(cache->len * sizeof(float));

    float inv = 1.0f / sqrtf((float)cache->head_dim);

    for(int i=0;i<cache->len;i++)
        scores[i] = dot_product_avx2(
            Q,
            cache->K + i * cache->head_dim,
            cache->head_dim
        ) * inv;

    softmax(scores, cache->len);

    for(int d=0; d<cache->head_dim; d++)
        out[d] = 0.0f;

    for(int i=0;i<cache->len;i++)
        for(int d=0; d<cache->head_dim; d++)
            out[d] += scores[i] *
                cache->V[i * cache->head_dim + d];

    free(scores);
}