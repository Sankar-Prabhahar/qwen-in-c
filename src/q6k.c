#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "q6k.h"

/* Convert IEEE-754 half precision (FP16) to float (FP32) */
float fp16_to_float(uint16_t h)
{
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x03FF;

    uint32_t bits;

    if (exp == 0) {
        if (frac == 0) {
            bits = sign;
        } else {
            /* Normalize subnormal numbers */
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03FF;
            bits = sign | ((exp + 113) << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000 | (frac << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (frac << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

float q6k_get_scale(const block_q6_K *block)
{
    return fp16_to_float(block->d);
}

/*
 * Exact GGML dequantize_row_q6_k implementation.
 * k is the total number of weights (must be a multiple of QK_K = 256).
 */
void dequantize_row_q6_k(const block_q6_K *x, float *y, int k)
{
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = fp16_to_float(x[i].d);

        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t  *sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;

                y[l +  0] = d * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

void q6k_decode_block(const block_q6_K *block, float *output)
{
    dequantize_row_q6_k(block, output, QK_K);
}

void q6k_dump_block(const block_q6_K *block)
{
    printf("d = %.6f\n", q6k_get_scale(block));

    printf("First 8 ql bytes: ");
    for (int i = 0; i < 8; i++) printf("%02X ", block->ql[i]);
    printf("\n");

    printf("First 8 qh bytes: ");
    for (int i = 0; i < 8; i++) printf("%02X ", block->qh[i]);
    printf("\n");

    printf("First 8 scales: ");
    for (int i = 0; i < 8; i++) printf("%d ", block->scales[i]);
    printf("\n");
}