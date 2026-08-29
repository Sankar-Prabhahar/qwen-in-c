#include <math.h>
#include "softmax.h"

void softmax(float *x, int n)
{
    float max = x[0];

    for(int i=1;i<n;i++)
        if(x[i] > max) max = x[i];

    float sum = 0.0f;

    for(int i=0;i<n;i++){
        x[i] = expf(x[i] - max);
        sum += x[i];
    }

    for(int i=0;i<n;i++)
        x[i] /= sum;
}