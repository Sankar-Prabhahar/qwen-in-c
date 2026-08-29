#ifndef Q6K_H
#define Q6K_H

#include <stddef.h>
#include <stdint.h>

#define QK_K 256

typedef struct {
    uint8_t ql[128];     /* quants, lower 4 bits (128 bytes) */
    uint8_t qh[64];      /* quants, upper 2 bits (64 bytes)  */
    int8_t  scales[16];  /* scales, 8-bit signed (16 bytes)  */
    uint16_t d;          /* fp16 super-block scale (2 bytes) */
} block_q6_K;

/* FP16 helper */
float fp16_to_float(uint16_t h);
float q6k_get_scale(const block_q6_K *block);

/* Decode a single Q6_K block (256 float values) */
void q6k_decode_block(const block_q6_K *block, float *output);

/* Dequantize row of k elements (k must be a multiple of QK_K = 256) */
void dequantize_row_q6_k(const block_q6_K *x, float *y, int k);

void q6k_dump_block(const block_q6_K *block);

#endif