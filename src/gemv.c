#include "gemv.h"
#include <immintrin.h>
void gemv_naive(const float *matrix,
                const float *vector,
                float *output,
                int rows,
                int cols)
{
    for(int r = 0; r < rows; r++){

        float sum = 0.0f;

        for(int c = 0; c < cols; c++)
            sum += matrix[r * cols + c] * vector[c];

        output[r] = sum;
    }
}
void gemv_avx2(const float *matrix,
               const float *vector,
               float *output,
               int rows,
               int cols)
{
    for(int r = 0; r < rows; r++){

        __m256 sum = _mm256_setzero_ps();

        int c;

        for(c = 0; c <= cols - 8; c += 8){

            __m256 m = _mm256_loadu_ps(&matrix[r * cols + c]);
            __m256 v = _mm256_loadu_ps(&vector[c]);

            sum = _mm256_fmadd_ps(m, v, sum);
        }

        float temp[8];
        _mm256_storeu_ps(temp, sum);

        float result = 0.0f;

        for(int i = 0; i < 8; i++)
            result += temp[i];

        for(; c < cols; c++)
            result += matrix[r * cols + c] * vector[c];

        output[r] = result;
    }
}