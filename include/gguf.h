#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include "model.h"

typedef struct {
    char magic[5];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
} GGUFHeader;

int gguf_read_header(Model *model, GGUFHeader *header);
void gguf_print_architecture(Model *model, GGUFHeader *header);
void gguf_list_tensors(Model *model, GGUFHeader *header);

int gguf_get_meta_u32(const Model *model, const GGUFHeader *header, const char *key, uint32_t *val);
int gguf_get_meta_u64(const Model *model, const GGUFHeader *header, const char *key, uint64_t *val);
int gguf_get_meta_f32(const Model *model, const GGUFHeader *header, const char *key, float *val);
int gguf_get_meta_str(const Model *model, const GGUFHeader *header, const char *key, char *out, size_t max_len);

#endif