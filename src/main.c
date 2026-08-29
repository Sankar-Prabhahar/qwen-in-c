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
#include "gemv_q6k.h"
#include "rmsnorm.h"
#include "rope.h"
#include "softmax.h"
#include "q6k.h"
#include "quant.h"
#include "embedding.h"
#include "attention.h"
#include "kv_cache.h"
#include "ffn.h"
#include "block.h"

void test_avx2();

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

    /* 5. GEMV Benchmark */
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

    TensorIndex index;
    build_tensor_index(&model, &header, &index);
    printf("Indexed tensors: %llu\n", (unsigned long long)index.count);
    printf("data_start = %llu\n", (unsigned long long)model.data_start);

    /* 6. M7: Token Embedding Lookup Test */
    Tensor *embd_tensor = find_tensor(&index, "token_embd.weight");
    int hidden_dim = (int)embd_tensor->dims[0];
    float *x_token = (float *)malloc(hidden_dim * sizeof(float));

    if (embedding_lookup(&model, embd_tensor, 450, x_token)) {
        float norm_sq = 0.0f;
        for (int i = 0; i < hidden_dim; i++) norm_sq += x_token[i] * x_token[i];
        printf("\n[M7] Token ID 450 Embedding (Norm: %.4f, First 4 weights): ", sqrtf(norm_sq));
        for (int i = 0; i < 4; i++) printf("%.6f ", x_token[i]);
        printf("\n");
    }

    /* 7. M9: Real Weight GEMV (Q Projection) */
    Tensor *q_tensor = find_tensor(&index, "blk.0.attn_q.weight");
    int q_in_dim = (int)q_tensor->dims[0];
    int q_out_dim = (int)q_tensor->dims[1];
    float *q_proj_out = (float *)malloc(q_out_dim * sizeof(float));

    QueryPerformanceCounter(&start);
    gemv_q6k(&model, q_tensor, x_token, q_proj_out, q_out_dim, q_in_dim);
    QueryPerformanceCounter(&end);
    double gemv_q6k_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

    float q_norm_sq = 0.0f;
    for (int i = 0; i < q_out_dim; i++) q_norm_sq += q_proj_out[i] * q_proj_out[i];
    printf("\n[M9] Real Weight GEMV (blk.0.attn_q.weight x token_450):\n");
    printf("     Time: %.2f ms | Norm: %.4f | First 8 values: ", gemv_q6k_ms, sqrtf(q_norm_sq));
    for (int i = 0; i < 8; i++) printf("%.6f ", q_proj_out[i]);
    printf("\n");

    /* 8. M13: Full Transformer Block 0 Forward Pass */
    TransformerBlockWeights block0;
    if (load_block_weights(&index, 0, &block0)) {
        printf("\n[M13] Loaded all 9 tensor weights for Transformer Block 0 successfully.\n");

        ModelKVCache kv_cache;
        int max_seq = 256;
        int n_heads = 32;
        int n_kv_heads = 4;
        int head_dim = 64;
        int intermediate_dim = 5632;
        float eps = 1e-5f;
        float theta = 10000.0f;

        kv_cache_init(&kv_cache, 1, max_seq, n_kv_heads, head_dim);

        /* Clone token embedding to pass into block 0 */
        float *hidden_state = (float *)malloc(hidden_dim * sizeof(float));
        memcpy(hidden_state, x_token, hidden_dim * sizeof(float));

        QueryPerformanceCounter(&start);
        int ok = transformer_block_forward(&model, &block0, &kv_cache, 0, 0,
                                           hidden_state, hidden_dim, n_heads,
                                           n_kv_heads, head_dim, intermediate_dim,
                                           eps, theta);
        QueryPerformanceCounter(&end);
        double block_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

        if (ok) {
            float block_norm_sq = 0.0f;
            for (int i = 0; i < hidden_dim; i++) block_norm_sq += hidden_state[i] * hidden_state[i];
            printf("[M13] Block 0 Forward Pass Completed (Time: %.2f ms, Output Norm: %.4f)\n",
                   block_ms, sqrtf(block_norm_sq));
            printf("     First 6 output activations: ");
            for (int i = 0; i < 6; i++) printf("%.6f ", hidden_state[i]);
            printf("\n");
        }

        free(hidden_state);
        kv_cache_free(&kv_cache);
    } else {
        printf("Failed to locate all weights for Transformer Block 0.\n");
    }

    free(x_token);
    free(q_proj_out);
    model_unmap(&model);
    return 0;
}