#include <immintrin.h>
#include <stdio.h>

void test_avx2(){

    float a[8]={1,2,3,4,5,6,7,8};
    float b[8]={1,1,1,1,1,1,1,1};
    float c[8];

    __m256 va=_mm256_loadu_ps(a);
    __m256 vb=_mm256_loadu_ps(b);
    __m256 vc=_mm256_add_ps(va,vb);

    _mm256_storeu_ps(c,vc);

    printf("AVX2:");

    for(int i=0;i<8;i++)
        printf(" %.0f",c[i]);

    printf("\n");
}