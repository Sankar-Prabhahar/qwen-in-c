#ifndef ATTENTION_H
#define ATTENTION_H

void attention_head(const float *Q,
                    const float *K,
                    const float *V,
                    float *out,
                    int seq_len,
                    int head_dim);

#endif