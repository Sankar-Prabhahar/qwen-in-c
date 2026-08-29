#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kv_cache.h"

int kv_cache_init(ModelKVCache *cache,
                  int n_layers,
                  int max_seq,
                  int n_kv_heads,
                  int head_dim)
{
    if (!cache) return 0;

    cache->n_layers = n_layers;
    cache->max_seq = max_seq;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;

    cache->layers = (LayerKVCache *)calloc(n_layers, sizeof(LayerKVCache));
    if (!cache->layers) return 0;

    int kv_dim = n_kv_heads * head_dim;
    size_t layer_floats = (size_t)max_seq * kv_dim;

    for (int l = 0; l < n_layers; l++) {
        cache->layers[l].max_seq = max_seq;
        cache->layers[l].n_kv_heads = n_kv_heads;
        cache->layers[l].head_dim = head_dim;
        cache->layers[l].kv_dim = kv_dim;
        cache->layers[l].current_len = 0;

        cache->layers[l].k = (float *)calloc(layer_floats, sizeof(float));
        cache->layers[l].v = (float *)calloc(layer_floats, sizeof(float));

        if (!cache->layers[l].k || !cache->layers[l].v) {
            kv_cache_free(cache);
            return 0;
        }
    }

    return 1;
}

void kv_cache_put(ModelKVCache *cache,
                  int layer,
                  int pos,
                  const float *k_vec,
                  const float *v_vec)
{
    if (!cache || layer < 0 || layer >= cache->n_layers) return;
    LayerKVCache *l = &cache->layers[layer];
    if (pos < 0 || pos >= l->max_seq) return;

    size_t offset = (size_t)pos * l->kv_dim;
    memcpy(l->k + offset, k_vec, l->kv_dim * sizeof(float));
    memcpy(l->v + offset, v_vec, l->kv_dim * sizeof(float));

    if (pos + 1 > l->current_len) {
        l->current_len = pos + 1;
    }
}

const float* kv_cache_get_k(const ModelKVCache *cache, int layer, int pos)
{
    if (!cache || layer < 0 || layer >= cache->n_layers) return NULL;
    const LayerKVCache *l = &cache->layers[layer];
    if (pos < 0 || pos >= l->max_seq) return NULL;
    return l->k + (size_t)pos * l->kv_dim;
}

const float* kv_cache_get_v(const ModelKVCache *cache, int layer, int pos)
{
    if (!cache || layer < 0 || layer >= cache->n_layers) return NULL;
    const LayerKVCache *l = &cache->layers[layer];
    if (pos < 0 || pos >= l->max_seq) return NULL;
    return l->v + (size_t)pos * l->kv_dim;
}

void kv_cache_free(ModelKVCache *cache)
{
    if (!cache || !cache->layers) return;

    for (int l = 0; l < cache->n_layers; l++) {
        if (cache->layers[l].k) free(cache->layers[l].k);
        if (cache->layers[l].v) free(cache->layers[l].v);
    }

    free(cache->layers);
    cache->layers = NULL;
}