#ifndef GEMV_H
#define GEMV_H

void gemv_naive(const float *matrix,
                const float *vector,
                float *output,
                int rows,
                int cols);
void gemv_avx2(const float *matrix,
               const float *vector,
               float *output,
               int rows,
               int cols);
#endif