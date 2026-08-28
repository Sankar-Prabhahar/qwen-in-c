#include <immintrin.h>
#include "simd.h"

float dot_product_avx2(const float *a,
                       const float *b,
                       int n)
{
    __m256 sum = _mm256_setzero_ps();

    int i;

    for(i = 0; i <= n - 8; i += 8){

        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);

        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum);

    float result = 0.0f;

    for(int j = 0; j < 8; j++)
        result += temp[j];

    for(; i < n; i++)
        result += a[i] * b[i];

    return result;
}