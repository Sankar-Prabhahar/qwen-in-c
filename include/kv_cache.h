#ifndef KV_CACHE_H
#define KV_CACHE_H

typedef struct{
    float *K;
    float *V;
    int max_seq;
    int head_dim;
    int len;
} KVCache;

void attention_head(const float *Q,
                    const float *K,
                    const float *V,
                    float *out,
                    int seq_len,
                    int head_dim);

void attention_cached(const float *Q,
                      KVCache *cache,
                      float *out);
int kv_init(KVCache *cache, int max_seq, int head_dim);
void kv_push(KVCache *cache, const float *K, const float *V);
void kv_free(KVCache *cache);
#endif