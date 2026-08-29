#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "model.h"
#include "tensor.h"

/*
 * embedding_lookup:
 * Given a model and the token_embd tensor, extracts and dequantizes
 * the hidden_size embedding vector for token_id into out_vec.
 *
 * Returns 1 on success, 0 on error.
 */
int embedding_lookup(const Model *model,
                     const Tensor *embd_tensor,
                     int token_id,
                     float *out_vec);

#endif
