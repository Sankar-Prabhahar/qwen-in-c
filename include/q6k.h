#ifndef Q6K_H
#define Q6K_H
#include <stddef.h>
#include <stdint.h>

#define QK_K 256

typedef struct {

    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;

} block_q6_K;

/* Decode one Q6_K block into 256 float values */
void q6k_decode_block(const block_q6_K *block,
                      float *output);
float fp16_to_float(uint16_t h);
/* FP16 helper */
float q6k_get_scale(const block_q6_K *block);
void q6k_dump_block(const block_q6_K *block);
void q6k_extract_low4(const block_q6_K *block, uint8_t out[256]);
void q6k_extract_high2(const block_q6_K *block, uint8_t out[256]);
void q6k_decode_block_exact(const block_q6_K *block, float out[QK_K]);

#endif