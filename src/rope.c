#include <math.h>
#include "rope.h"

void rope_apply(float *x,
                int head_dim,
                int position,
                float theta)
{
    for(int i = 0; i < head_dim; i += 2){

        float freq = powf(theta, -(float)i / head_dim);
        float angle = position * freq;

        float c = cosf(angle);
        float s = sinf(angle);

        float a = x[i];
        float b = x[i + 1];

        x[i]     = a * c - b * s;
        x[i + 1] = a * s + b * c;
    }
}