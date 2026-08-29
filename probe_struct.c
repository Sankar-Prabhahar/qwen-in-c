#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    uint16_t d;
} block_q6_K;
int main(void) {
    printf("sizeof(block_q6_K) = %zu\n", sizeof(block_q6_K));
    printf("offsetof ql     = %zu\n", offsetof(block_q6_K, ql));
    printf("offsetof qh     = %zu\n", offsetof(block_q6_K, qh));
    printf("offsetof scales = %zu\n", offsetof(block_q6_K, scales));
    printf("offsetof d      = %zu\n", offsetof(block_q6_K, d));
    return 0;
}
