#ifndef SAMPLER_H
#define SAMPLER_H

#include <stdint.h>

typedef struct {
    float temperature;  /* 0.0 for greedy argmax, >0 for stochastic */
    int top_k;          /* e.g. 40, 0 to disable */
    float top_p;        /* e.g. 0.9, 1.0 to disable */
    uint64_t rng_seed;
} SamplerConfig;

int sample_token(float *logits, int vocab_size, SamplerConfig *config);

#endif
