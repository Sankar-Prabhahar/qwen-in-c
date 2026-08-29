#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stddef.h>

/*
 * LayerKVCache stores the key and value states for a single transformer layer
 * across all KV heads up to max_seq_len.
 */
typedef struct {
    float *k;          /* Shape: [max_seq, n_kv_heads * head_dim] */
    float *v;          /* Shape: [max_seq, n_kv_heads * head_dim] */
    int max_seq;
    int n_kv_heads;
    int head_dim;
    int kv_dim;        /* n_kv_heads * head_dim */
    int current_len;
} LayerKVCache;

/*
 * ModelKVCache stores KV caches for all layers in the transformer.
 */
typedef struct {
    LayerKVCache *layers;
    int n_layers;
    int max_seq;
    int n_kv_heads;
    int head_dim;
} ModelKVCache;

int kv_cache_init(ModelKVCache *cache,
                  int n_layers,
                  int max_seq,
                  int n_kv_heads,
                  int head_dim);

void kv_cache_put(ModelKVCache *cache,
                  int layer,
                  int pos,
                  const float *k_vec,
                  const float *v_vec);

const float* kv_cache_get_k(const ModelKVCache *cache, int layer, int pos);
const float* kv_cache_get_v(const ModelKVCache *cache, int layer, int pos);

void kv_cache_free(ModelKVCache *cache);

#endif