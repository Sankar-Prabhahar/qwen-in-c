#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include "model.h"

typedef struct{
    char magic[5];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
} GGUFHeader;

int gguf_read_header(Model *model, GGUFHeader *header);
void gguf_print_architecture(Model *model, GGUFHeader *header);
void gguf_list_tensors(Model *model, GGUFHeader *header);
#endif