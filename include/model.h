#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>

typedef struct{
    char magic[5];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
} GGUFHeader;

int gguf_read_header(const char *path, GGUFHeader *header);

#endif