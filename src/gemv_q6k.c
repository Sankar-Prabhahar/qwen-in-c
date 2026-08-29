#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "gemv_q6k.h"

/*
 * Decode one Q6_K block (256 floats) directly into a temporary buffer,
 * then compute the dot product with x[256] using AVX2 FMA.
 *
 * The inner decode+dot is fused to avoid allocating a full 256-float buffer
 * for every block — we accumulate 8 floats at a time via AVX2.
 */
static float dot_q6k_block_avx2(const block_q6_K *blk, const float *x)
{
    const float d       = fp16_to_float(blk->d);
    const uint8_t *ql   = blk->ql;
    const uint8_t *qh   = blk->qh;
    const int8_t  *sc   = blk->scales;

    __m256 acc = _mm256_setzero_ps();

    /*
     * Two halves of 128 weights each.
     * Inner loop processes 8 weights at a time using scalar decode + AVX2 dot.
     * (Full vectorised decode of 6-bit integers is complex; scalar decode +
     *  AVX2 accumulation is clean and still much faster than naive float GEMV.)
     */
    float tmp[8];

    for (int half = 0; half < 2; half++) {
        const uint8_t *ql_h = ql + half * 64;
        const uint8_t *qh_h = qh + half * 32;
        const int8_t  *sc_h = sc + half * 8;
        const float   *x_h  = x  + half * 128;

        for (int l = 0; l < 32; l += 8) {
            /* Decode 8 groups of 4 values each (q1, q2, q3, q4 per l) */
            /* q1 = ql[l+ 0] lo4 | qh[l] bits[1:0] << 4   → output[l+ 0] */
            /* q2 = ql[l+32] lo4 | qh[l] bits[3:2] << 4   → output[l+32] */
            /* q3 = ql[l+ 0] hi4 | qh[l] bits[5:4] << 4   → output[l+64] */
            /* q4 = ql[l+32] hi4 | qh[l] bits[7:6] << 4   → output[l+96] */

            /* Unroll 8 l values */
            for (int i = 0; i < 8; i++) {
                int li = l + i;
                int cur_is = li / 16;
                float s1 = d * (float)sc_h[cur_is + 0];
                float s2 = d * (float)sc_h[cur_is + 2];
                float s3 = d * (float)sc_h[cur_is + 4];
                float s4 = d * (float)sc_h[cur_is + 6];

                float w1 = s1 * (float)(int)(((ql_h[li]      & 0x0F) | (((qh_h[li] >> 0) & 3) << 4)) - 32);
                float w2 = s2 * (float)(int)(((ql_h[li + 32]  & 0x0F) | (((qh_h[li] >> 2) & 3) << 4)) - 32);
                float w3 = s3 * (float)(int)(((ql_h[li]       >>  4)  | (((qh_h[li] >> 4) & 3) << 4)) - 32);
                float w4 = s4 * (float)(int)(((ql_h[li + 32]  >>  4)  | (((qh_h[li] >> 6) & 3) << 4)) - 32);

                /* Multiply each decoded weight by its corresponding x element */
                tmp[i] = w1 * x_h[li] + w2 * x_h[li + 32] + w3 * x_h[li + 64] + w4 * x_h[li + 96];
            }

            __m256 v = _mm256_loadu_ps(tmp);
            acc = _mm256_add_ps(acc, v);
        }
    }

    /* Horizontal sum */
    float buf[8];
    _mm256_storeu_ps(buf, acc);
    float sum = 0.0f;
    for (int i = 0; i < 8; i++) sum += buf[i];
    return sum;
}

int gemv_q6k(const Model *model,
             const Tensor *weight,
             const float *x,
             float *output,
             int rows,
             int cols)
{
    if (!model || !weight || !x || !output) return 0;
    if (weight->type != 14) {
        printf("gemv_q6k: expected Q6_K (type 14), got type %u\n", weight->type);
        return 0;
    }
    if (cols % QK_K != 0) {
        printf("gemv_q6k: cols %d not multiple of %d\n", cols, QK_K);
        return 0;
    }

    int blocks_per_row = cols / QK_K;
    size_t row_bytes   = (size_t)blocks_per_row * sizeof(block_q6_K);

    const uint8_t *base = model->data + model->data_start + weight->offset;

    for (int r = 0; r < rows; r++) {
        const block_q6_K *row = (const block_q6_K *)(base + (size_t)r * row_bytes);
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            sum += dot_q6k_block_avx2(&row[b], x + b * QK_K);
        }
        output[r] = sum;
    }

    return 1;
}
