#include <stdio.h>
#include "tensor_loader.h"
#include "gguf.h"
#include "model.h"
#include "mmap.h"
#include "rmsnorm.h"
#include "tensor_data.h"
#include "quant.h"
#include "gemv.h"
#include "tensor_index.h"
#include <windows.h>
#include "simd.h"
#include "q6k.h"
void test_avx2();
static float dot_product_naive(const float *a,
                               const float *b,
                               int n)
{
    float sum = 0.0f;

    for(int i = 0; i < n; i++)
        sum += a[i] * b[i];

    return sum;
}

int main(int argc, char **argv)

{
    printf("=== Qwen30B-in-C ===\n");

    test_avx2();

block_q6_K test_block = {0};
test_block.d = 0x3C00;

printf("Q6K Scale: %.1f\n", q6k_get_scale(&test_block));

float a[8] = {1,2,3,4,5,6,7,8};
float b[8] = {8,7,6,5,4,3,2,1};

float answer = dot_product_avx2(a, b, 8);
float decoded[QK_K];

q6k_decode_block(&test_block, decoded);

printf("Q6K Decode Test: %.1f %.1f %.1f %.1f\n",
       decoded[0], decoded[1], decoded[2], decoded[3]);

printf("AVX2 Dot Product: %.0f\n", answer);
/* GEMV sanity test */
printf("Q6K Block Size: %zu bytes\n", sizeof(block_q6_K));
q6k_dump_block(&test_block);
float matrix[4] = {
    1, 2,
    3, 4
};

float vector[2] = {5, 6};
float output[2];

gemv_naive(matrix, vector, output, 2, 2);

printf("GEMV Test\n");
printf("%.0f\n", output[0]);
printf("%.0f\n", output[1]);
/* GEMV benchmark (256 x 256) */

#define ROWS 256
#define COLS 256

static float big_matrix[ROWS * COLS];
static float big_vector[COLS];
static float out_naive[ROWS];
static float out_avx[ROWS];

for(int i = 0; i < ROWS * COLS; i++)
    big_matrix[i] = (float)(i % 17) * 0.1f;

for(int i = 0; i < COLS; i++)
    big_vector[i] = (float)(i % 13) * 0.2f;

LARGE_INTEGER freq2, start2, end2;
QueryPerformanceFrequency(&freq2);

/* Naive GEMV */
QueryPerformanceCounter(&start2);

for(int i = 0; i < 100; i++)
    gemv_naive(big_matrix, big_vector, out_naive, ROWS, COLS);

QueryPerformanceCounter(&end2);

double gemv_naive_ms =
    (double)(end2.QuadPart - start2.QuadPart) * 1000.0 / freq2.QuadPart;

/* AVX2 GEMV */
QueryPerformanceCounter(&start2);

for(int i = 0; i < 100; i++)
    gemv_avx2(big_matrix, big_vector, out_avx, ROWS, COLS);

QueryPerformanceCounter(&end2);

double gemv_avx_ms =
    (double)(end2.QuadPart - start2.QuadPart) * 1000.0 / freq2.QuadPart;

printf("\nGEMV Benchmark (256x256)\n");
printf("Naive : %.2f ms\n", gemv_naive_ms);
printf("AVX2  : %.2f ms\n", gemv_avx_ms);

/* Verify both give the same answer */
printf("Check: %.3f vs %.3f\n", out_naive[0], out_avx[0]);
LARGE_INTEGER freq, start, end;
QueryPerformanceFrequency(&freq);

volatile float sink = 0.0f;

/* Normal version */
QueryPerformanceCounter(&start);

for(int i = 0; i < 1000000; i++)
    sink += dot_product_naive(a, b, 8);

QueryPerformanceCounter(&end);

double naive_ms =
    (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

/* AVX2 version */
QueryPerformanceCounter(&start);

for(int i = 0; i < 1000000; i++)
    sink += dot_product_avx2(a, b, 8);

QueryPerformanceCounter(&end);

double avx_ms =
    (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart;

printf("Naive: %.2f ms\n", naive_ms);
printf("AVX2 : %.2f ms\n", avx_ms);
    if (argc < 2) {
        printf("Usage:\n");
        printf("qwen30b model.gguf\n");
        return 1;
    }

    printf("Model: %s\n\n", argv[1]);

    Model model;

    if (!model_map(argv[1], &model)) {
        printf("Failed to map model.\n");
        return 1;
    }

    GGUFHeader header;
    gguf_read_header(&model, &header);

    printf("GGUF Header\n");
    printf("Magic: %s\n", header.magic);
    printf("Version: %u\n", header.version);
    printf("Tensors: %llu\n", (unsigned long long)header.tensor_count);
    printf("Metadata: %llu\n", (unsigned long long)header.metadata_count);
    gguf_print_architecture(&model, &header);
    gguf_list_tensors(&model, &header);
    Tensor tensor;

if(load_first_tensor(&model, &header, &tensor)){

    printf("\nFirst Tensor Details\n");
    printf("Name: %s\n", tensor.name);
    printf("Dimensions: %u\n", tensor.n_dims);

    printf("Shape: ");
    for(uint32_t i = 0; i < tensor.n_dims; i++){
        printf("%llu",
            (unsigned long long)tensor.dims[i]);

        if(i + 1 < tensor.n_dims)
            printf(" x ");
    }

    printf("\nType: %u\n", tensor.type);
    printf("Offset: %llu\n",
        (unsigned long long)tensor.offset);
    describe_quant_type(tensor.type);
}
print_tensor_bytes(&model, &header, &tensor);
TensorIndex index;

build_tensor_index(&model, &header, &index);

printf("\nTensor Index\n");
printf("Indexed tensors: %llu\n",
    (unsigned long long)index.count);
Tensor *found = find_tensor(&index, "blk.0.attn_q.weight");
block_q6_K real_block;

if(load_q6k_block(&model, found, &real_block)){

    printf("\nReal Q6_K Block\n");
    printf("Size: %zu\n", sizeof(real_block));

    q6k_dump_block(&real_block);
}
float real_decoded[QK_K];

q6k_decode_block(&real_block, real_decoded);

printf("First 8 decoded values:\n");
for(int i=0;i<8;i++)
    printf("%.6f ", real_decoded[i]);
printf("\n");
uint8_t low4[256], high2[256];

q6k_extract_low4(&real_block, low4);
q6k_extract_high2(&real_block, high2);

printf("Raw 6-bit components (first 8):\n");
printf("Signed q values (first 8):\n");
for(int i=0;i<8;i++){

    int q = ((int)high2[i] << 4) | low4[i];
    q -= 32;

    printf("%d ", q);
    printf("First 8 decoded values:\n");
for(int i=0;i<8;i++)
    printf("%.6f ", real_decoded[i]);
printf("\n");

printf("Recovered q (first 8):\n");
for(int i=0;i<8;i++)
    printf("%d ", (int)(real_decoded[i] / q6k_get_scale(&real_block)));
printf("\n");
}
printf("\n");
printf("Recovered q (first 8): ");
for(int i=0;i<8;i++)
    printf("%d ", (int)(real_decoded[i] / q6k_get_scale(&real_block)));
printf("\n");
float exact[QK_K];

q6k_decode_block_exact(&real_block, exact);

printf("Exact decoder (first 8): ");
for(int i = 0; i < 8; i++)
    printf("%.6f ", exact[i]);
printf("\n");
float x[4] = {1,2,3,4};
float w[4] = {1,1,1,1};
float y[4];

rmsnorm(y, x, w, 4, 1e-5f);

printf("RMSNorm: ");
for(int i=0;i<4;i++)
    printf("%.4f ", y[i]);
printf("\n");
if(found){

    printf("\nTensor Lookup Success\n");
    printf("Name: %s\n", found->name);
    printf("Dimensions: %u\n", found->n_dims);

    printf("Shape: ");
    for(uint32_t i = 0; i < found->n_dims; i++){

        printf("%llu",
            (unsigned long long)found->dims[i]);

        if(i + 1 < found->n_dims)
            printf(" x ");
    }

    printf("\nType: %u\n", found->type);
    printf("Offset: %llu\n",
        (unsigned long long)found->offset);

}else{

    printf("\nTensor not found.\n");
}
    model_unmap(&model);

    return 0;
}