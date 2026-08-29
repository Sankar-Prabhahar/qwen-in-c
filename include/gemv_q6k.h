#ifndef GEMV_Q6K_H
#define GEMV_Q6K_H

#include <stdint.h>
#include "model.h"
#include "tensor.h"
#include "q6k.h"

/*
 * gemv_q6k:
 * Compute output[rows] = W * x[cols] where W is stored as Q6_K blocks.
 */
int gemv_q6k(const Model *model,
             const Tensor *weight,
             const float *x,
             float *output,
             int rows,
             int cols);

/*
 * gemv_q6k_offset:
 * Compute GEMV on a slice of a multi-dimensional tensor (e.g. expert slice in MoE).
 */
int gemv_q6k_offset(const Model *model,
                    const Tensor *weight,
                    uint64_t extra_byte_offset,
                    const float *x,
                    float *output,
                    int rows,
                    int cols);

#endif
