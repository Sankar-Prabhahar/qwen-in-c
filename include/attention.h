#ifndef ATTENTION_H
#define ATTENTION_H

#include "kv_cache.h"

/*
 * attention_gqa:
 * Multi-head / Grouped-Query Attention for the token at current position `pos`.
 *
 * Parameters:
 *   Q:          Query vectors for all query heads [n_heads * head_dim]
 *   cache:      Model KV cache storing past K and V projections
 *   layer:      Layer index
 *   pos:        Current token position (0-indexed, sequence length = pos + 1)
 *   n_heads:    Number of query heads (e.g. 32)
 *   n_kv_heads: Number of key/value heads (e.g. 4)
 *   head_dim:   Dimension of each head (e.g. 64)
 *   out:        Output buffer [n_heads * head_dim]
 */
void attention_gqa(const float *Q,
                   const ModelKVCache *cache,
                   int layer,
                   int pos,
                   int n_heads,
                   int n_kv_heads,
                   int head_dim,
                   float *out);

#endif