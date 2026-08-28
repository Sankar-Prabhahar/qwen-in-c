#include <stdint.h>
#include <string.h>

#include "q6k.h"

/* Convert IEEE-754 half precision (FP16) to float */
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
void q6k_decode_block(const block_q6_K *block, float *output)
{
    float d = q6k_get_scale(block);

    uint8_t low[256], high[256];

    q6k_extract_low4(block, low);
    q6k_extract_high2(block, high);

    for(int i=0;i<256;i++){
        int q = ((int)high[i] << 4) | low[i];
        q -= 32;
        output[i] = q * d;
    }
}

#include <stdio.h>

void q6k_dump_block(const block_q6_K *block)
{
    printf("d = %.6f\n", q6k_get_scale(block));

    printf("First 8 ql bytes: ");
    for(int i=0;i<8;i++) printf("%02X ", block->ql[i]);
    printf("\n");

    printf("First 8 qh bytes: ");
    for(int i=0;i<8;i++) printf("%02X ", block->qh[i]);
    printf("\n");

    printf("First 8 scales: ");
    for(int i=0;i<8;i++) printf("%d ", block->scales[i]);
    printf("\n");
}
void q6k_extract_low4(const block_q6_K *block, uint8_t out[256])
{
    for(int i=0;i<64;i++){
        out[2*i]     =  block->ql[i]       & 0x0F;
        out[2*i + 1] = (block->ql[i] >> 4) & 0x0F;
    }

    for(int i=128;i<256;i++)
        out[i]=0;
}

void q6k_extract_high2(const block_q6_K *block, uint8_t out[256])
{
    memset(out,0,256);

    for(int i=0;i<64;i++){

        uint8_t b = block->qh[i];

        out[4*i]     =  b       & 0x03;
        out[4*i + 1] = (b >> 2) & 0x03;
        out[4*i + 2] = (b >> 4) & 0x03;
        out[4*i + 3] = (b >> 6) & 0x03;
    }
}
static inline int q6k_unpack_weight(const block_q6_K *block, int i)
{
    uint8_t lo = (block->ql[i >> 1] >> ((i & 1) * 4)) & 0x0F;
    uint8_t hi = (block->qh[i >> 2] >> ((i & 3) * 2)) & 0x03;

    return ((hi << 4) | lo) - 32;
}
void q6k_decode_block_exact(const block_q6_K *block, float out[QK_K])
{
    for(int g = 0; g < 16; g++) {

        float scale = q6k_get_scale(block) * (float)block->scales[g];

        for(int j = 0; j < 16; j++) {
            int idx = g * 16 + j;
            out[idx] = q6k_unpack_weight(block, idx) * scale;
        }
    }

    for(int i = 256; i < QK_K; i++)
        out[i] = 0.0f;
}