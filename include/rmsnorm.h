#ifndef RMSNORM_H
#define RMSNORM_H

void rmsnorm(float *out,
             const float *x,
             const float *weight,
             int n,
             float eps);

#endif