#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#include "model.h"
#include "mmap.h"
#include "gguf.h"
#include "tensor.h"
#include "tensor_loader.h"
#include "tensor_index.h"
#include "tensor_data.h"
#include "simd.h"
#include "gemv.h"
#include "rmsnorm.h"
#include "rope.h"
#include "softmax.h"
#include "q6k.h"
#include "quant.h"
#include "embedding.h"
#include "attention.h"
#include "kv_cache.h"

void test_avx2();

static float dot_product_naive(const float *a, const float *b, int n)
{
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

int main(int argc, char **argv)
{
    printf("=== Qwen30B-in-C ===\n");

    /* 1. SIMD Sanity */
    test_avx2();

    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    float dot_ans = dot_product_avx2(a, b, 8);
    printf("AVX2 Dot Product: %.0f\n", dot_ans);

    /* 2. M5: RoPE Verification */
    float rope_vec[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    rope_apply(rope_vec, 8, 1, 10000.0f);
    printf("RoPE: ");
    for (int i = 0; i < 8; i++) {
        printf("%.4f ", rope_vec[i]);
    }
    printf("\n");

    /* 3. M6: Softmax Verification */
    float sm[3] = {1.0f, 2.0f, 3.0f};
    softmax(sm, 3);
    printf("Softmax: ");
    for (int i = 0; i < 3; i++) {
        printf("%.4f ", sm[i]);
    }
    printf("\n");

    /* 4. RMSNorm Verification */
    float x[4] = {1, 2, 3, 4};
    float w[4] = {1, 1, 1, 1};
    float y[4];
    rmsnorm(y, x, w, 4, 1e-5f);
    printf("RMSNorm: ");
    for (int i = 0; i < 4; i++) {
        printf("%.4f ", y[i]);
    }
    printf("\n");

    /* 5. GEMV Sanity & Benchmark */
    float matrix[4] = {1, 2, 3, 4};
    float vector[2] = {5, 6};
    float output[2];
    gemv_naive(matrix, vector, output, 2, 2);
    printf("GEMV Test: %.0f, %.0f\n", output[0], output[1]);

    #define ROWS 256
    #define COLS 256
    static float big_matrix[ROWS * COLS];
    static float big_vector[COLS];
    static float out_naive[ROWS];
    static float out_avx[ROWS];

    for (int i = 0; i < ROWS * COLS; i++) {
        big_matrix[i] = (float)(i % 17) * 0.1f;
    }
    for (int i = 0; i < COLS; i++) {
        big_vector[i] = (float)(i % 13) * 0.2f;
    }

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    QueryPerformanceCounter(&start);
    for (int i = 0; i < 100; i++) {
        gemv_naive(big_matrix, big_vector, out_naive, ROWS, COLS);
    }
    QueryPerformanceCounter(&end);
    double gemv_naive_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    QueryPerformanceCounter(&start);
    for (int i = 0; i < 100; i++) {
        gemv_avx2(big_matrix, big_vector, out_avx, ROWS, COLS);
    }
    QueryPerformanceCounter(&end);
    double gemv_avx_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    printf("\nGEMV Benchmark (256x256)\n");
    printf("Naive : %.2f ms\n", gemv_naive_ms);
    printf("AVX2  : %.2f ms\n", gemv_avx_ms);
    printf("Check: %.3f vs %.3f\n", out_naive[0], out_avx[0]);

    /* Dot product benchmark */
    volatile float sink = 0.0f;
    QueryPerformanceCounter(&start);
    for (int i = 0; i < 1000000; i++) {
        sink += dot_product_naive(a, b, 8);
    }
    QueryPerformanceCounter(&end);
    double naive_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    QueryPerformanceCounter(&start);
    for (int i = 0; i < 1000000; i++) {
        sink += dot_product_avx2(a, b, 8);
    }
    QueryPerformanceCounter(&end);
    double avx_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    printf("Dot Product Bench (1M iters) - Naive: %.2f ms, AVX2: %.2f ms (sink: %.1f)\n",
           naive_ms, avx_ms, sink);

    /* 6. KV Cache & Attention Sanity */
    KVCache cache;
    kv_init(&cache, 8, 4);
    float q_vec[4] = {1, 0, 0, 0};
    float k1[4] = {1, 0, 0, 0};
    float v1[4] = {10, 20, 30, 40};
    float k2[4] = {0, 1, 0, 0};
    float v2[4] = {50, 60, 70, 80};
    kv_push(&cache, k1, v1);
    kv_push(&cache, k2, v2);

    float attn_out[4];
    attention_cached(q_vec, &cache, attn_out);
    printf("Cached Attention: ");
    for (int i = 0; i < 4; i++) {
        printf("%.3f ", attn_out[i]);
    }
    printf("\n");
    kv_free(&cache);

    /* 7. Model Inspection & Inference Pipeline Validation */
    if (argc < 2) {
        printf("\nUsage: qwen30b <model.gguf>\n");
        return 0;
    }

    printf("\nModel: %s\n", argv[1]);
    Model model;
    if (!model_map(argv[1], &model)) {
        printf("Failed to map model.\n");
        return 1;
    }

    GGUFHeader header;
    gguf_read_header(&model, &header);

    printf("GGUF Header -> Magic: %s, Version: %u, Tensors: %llu, Metadata: %llu\n",
           header.magic, header.version,
           (unsigned long long)header.tensor_count,
           (unsigned long long)header.metadata_count);

    gguf_print_architecture(&model, &header);

    TensorIndex index;
    build_tensor_index(&model, &header, &index);
    printf("Indexed tensors: %llu\n", (unsigned long long)index.count);
    printf("data_start = %llu\n", (unsigned long long)model.data_start);

    /* 8. M8: Real GGML Q6_K Decoder Test on Real Tensor Weight */
    Tensor *q_tensor = find_tensor(&index, "blk.0.attn_q.weight");
    if (q_tensor) {
        printf("\nFound blk.0.attn_q.weight (Type: %u, Offset: %llu, Shape: %llu x %llu)\n",
               q_tensor->type, (unsigned long long)q_tensor->offset,
               (unsigned long long)q_tensor->dims[0],
               (unsigned long long)q_tensor->dims[1]);

        block_q6_K real_block;
        load_q6k_block(&model, q_tensor, &real_block);
        q6k_dump_block(&real_block);

        float decoded[QK_K];
        q6k_decode_block(&real_block, decoded);

        printf("GGML Q6_K Decoded (first 8 values):\n");
        for (int i = 0; i < 8; i++) {
            printf("%.6f ", decoded[i]);
        }
        printf("\n");
    }

    /* 9. M7: Token Embedding Lookup Test */
    Tensor *embd_tensor = find_tensor(&index, "token_embd.weight");
    if (embd_tensor) {
        printf("\nFound token_embd.weight (Type: %u, Shape: %llu x %llu)\n",
               embd_tensor->type,
               (unsigned long long)embd_tensor->dims[0],
               (unsigned long long)embd_tensor->dims[1]);

        int hidden_dim = (int)embd_tensor->dims[0];
        float *token_vec = (float *)malloc(hidden_dim * sizeof(float));

        if (token_vec) {
            /* Test token ID 1 (e.g. <s>) */
            if (embedding_lookup(&model, embd_tensor, 1, token_vec)) {
                float norm_sq = 0.0f;
                for (int i = 0; i < hidden_dim; i++) norm_sq += token_vec[i] * token_vec[i];
                printf("Token ID 1 Embedding (Norm: %.4f, First 6 weights): ", sqrtf(norm_sq));
                for (int i = 0; i < 6; i++) printf("%.6f ", token_vec[i]);
                printf("\n");
            }

            /* Test token ID 450 (e.g. "Hello") */
            if (embedding_lookup(&model, embd_tensor, 450, token_vec)) {
                float norm_sq = 0.0f;
                for (int i = 0; i < hidden_dim; i++) norm_sq += token_vec[i] * token_vec[i];
                printf("Token ID 450 Embedding (Norm: %.4f, First 6 weights): ", sqrtf(norm_sq));
                for (int i = 0; i < 6; i++) printf("%.6f ", token_vec[i]);
                printf("\n");
            }

            free(token_vec);
        }
    }

    model_unmap(&model);
    return 0;
}