#include <math.h>
#include "rmsnorm.h"

void rmsnorm(float *out,
             const float *x,
             const float *weight,
             int n,
             float eps)
{
    float ss = 0.0f;

    for(int i=0;i<n;i++)
        ss += x[i] * x[i];

    float scale = 1.0f / sqrtf(ss / n + eps);

    for(int i=0;i<n;i++)
        out[i] = x[i] * scale * weight[i];
}