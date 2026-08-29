#include <stdio.h>
#include <string.h>
#include "embedding.h"
#include "q6k.h"

int embedding_lookup(const Model *model,
                     const Tensor *embd_tensor,
                     int token_id,
                     float *out_vec)
{
    if (!model || !embd_tensor || !out_vec) {
        return 0;
    }

    if (token_id < 0 || (uint64_t)token_id >= embd_tensor->dims[1]) {
        printf("embedding_lookup: token_id %d out of bounds (vocab_size = %llu)\n",
               token_id, (unsigned long long)embd_tensor->dims[1]);
        return 0;
    }

    int hidden_dim = (int)embd_tensor->dims[0];

    if (embd_tensor->type == 14) { /* GGML_TYPE_Q6_K */
        if (hidden_dim % QK_K != 0) {
            printf("embedding_lookup: hidden_dim %d not multiple of %d\n", hidden_dim, QK_K);
            return 0;
        }

        int blocks_per_row = hidden_dim / QK_K;
        size_t row_bytes = (size_t)blocks_per_row * sizeof(block_q6_K);
        size_t offset = (size_t)token_id * row_bytes;

        const uint8_t *src = model->data + model->data_start + embd_tensor->offset + offset;
        dequantize_row_q6_k((const block_q6_K *)src, out_vec, hidden_dim);
        return 1;
    } else if (embd_tensor->type == 0) { /* GGML_TYPE_F32 */
        size_t row_bytes = (size_t)hidden_dim * sizeof(float);
        size_t offset = (size_t)token_id * row_bytes;
        const uint8_t *src = model->data + model->data_start + embd_tensor->offset + offset;
        memcpy(out_vec, src, row_bytes);
        return 1;
    } else if (embd_tensor->type == 1) { /* GGML_TYPE_F16 */
        size_t row_bytes = (size_t)hidden_dim * sizeof(uint16_t);
        size_t offset = (size_t)token_id * row_bytes;
        const uint16_t *src = (const uint16_t *)(model->data + model->data_start + embd_tensor->offset + offset);
        for (int i = 0; i < hidden_dim; i++) {
            out_vec[i] = fp16_to_float(src[i]);
        }
        return 1;
    }

    printf("embedding_lookup: unsupported tensor type %u\n", embd_tensor->type);
    return 0;
}
