#include <stdio.h>

#include "gguf.h"

void test_avx2(void);

int main(int argc, char **argv)
{
    GGUFHeader header;

    printf("=== Qwen30B-in-C ===\n");
    test_avx2();

    if (argc < 2) {
        printf("Usage: %s model.gguf\n", argv[0]);
        return 1;
    }

    printf("Model: %s\n", argv[1]);

    if (!gguf_read_header(argv[1], &header)) {
        printf("Failed to open model.\n");
        return 1;
    }

    printf("\nGGUF Header\n");
    printf("Magic: %s\n", header.magic);
    printf("Version: %u\n", header.version);
    printf("Tensors: %llu\n", (unsigned long long)header.tensor_count);
    printf("Metadata: %llu\n", (unsigned long long)header.metadata_count);

    return 0;
}