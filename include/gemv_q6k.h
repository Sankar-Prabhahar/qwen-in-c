#ifndef GEMV_Q6K_H
#define GEMV_Q6K_H

#include "model.h"
#include "tensor.h"
#include "q6k.h"

/*
 * gemv_q6k:
 * Compute output[rows] = W * x[cols] where W is stored as Q6_K blocks.
 *
 * The weight tensor has shape [cols, rows] (GGUF stores col-major for weights:
 * dims[0]=in_features, dims[1]=out_features).
 * Each row of W is (cols / QK_K) consecutive block_q6_K blocks.
 *
 * output[r] = dot(decode(W_row_r), x)
 *
 * Returns 1 on success.
 */
int gemv_q6k(const Model *model,
             const Tensor *weight,
             const float *x,
             float *output,
             int rows,
             int cols);

#endif
