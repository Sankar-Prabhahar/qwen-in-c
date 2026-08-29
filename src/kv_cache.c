#include <stdlib.h>
#include <string.h>
#include "kv_cache.h"

int kv_init(KVCache *cache, int max_seq, int head_dim)
{
    cache->max_seq = max_seq;
    cache->head_dim = head_dim;
    cache->len = 0;

    cache->K = malloc(sizeof(float) * max_seq * head_dim);
    cache->V = malloc(sizeof(float) * max_seq * head_dim);

    return cache->K && cache->V;
}

void kv_push(KVCache *cache, const float *K, const float *V)
{
    memcpy(cache->K + cache->len * cache->head_dim,
           K,
           sizeof(float) * cache->head_dim);

    memcpy(cache->V + cache->len * cache->head_dim,
           V,
           sizeof(float) * cache->head_dim);

    cache->len++;
}

void kv_free(KVCache *cache)
{
    free(cache->K);
    free(cache->V);
}