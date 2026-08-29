#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "gguf.h"

static uint32_t read_u32(const uint8_t *p) { return *(const uint32_t *)p; }
static uint64_t read_u64(const uint8_t *p) { return *(const uint64_t *)p; }

static const uint8_t* skip_value(const uint8_t *p, uint32_t type)
{
    switch (type) {
        case 0: case 1: case 7: return p + 1;
        case 2: case 3: return p + 2;
        case 4: case 5: case 6: return p + 4;
        case 10: case 11: case 12: return p + 8;
        case 8: {
            uint64_t len = read_u64(p);
            return p + 8 + len;
        }
        case 9: {
            uint32_t elem_type = read_u32(p); p += 4;
            uint64_t count = read_u64(p); p += 8;
            for (uint64_t i = 0; i < count; i++) {
                p = skip_value(p, elem_type);
            }
            return p;
        }
        default: return p;
    }
}

static const uint8_t* read_string(const uint8_t *p, char *out, uint64_t max_len)
{
    uint64_t len = read_u64(p);
    p += 8;

    uint64_t copy = (len < max_len - 1) ? len : max_len - 1;
    memcpy(out, p, copy);
    out[copy] = '\0';

    return p + len;
}

int gguf_read_header(Model *model, GGUFHeader *header)
{
    if (!model || !header || !model->data) return 0;

    const uint8_t *p = model->data;
    memcpy(header->magic, p, 4);
    header->magic[4] = '\0';
    p += 4;

    header->version = read_u32(p);
    p += 4;

    header->tensor_count = read_u64(p);
    p += 8;

    header->metadata_count = read_u64(p);
    return 1;
}

static const uint8_t* gguf_find_meta_value(const Model *model, const GGUFHeader *header, const char *target_key, uint32_t *out_type)
{
    if (!model || !header || !target_key) return NULL;
    const uint8_t *p = model->data + 24;

    for (uint64_t i = 0; i < header->metadata_count; i++) {
        char key[256];
        p = read_string(p, key, sizeof(key));

        uint32_t val_type = read_u32(p);
        p += 4;

        if (strcmp(key, target_key) == 0) {
            if (out_type) *out_type = val_type;
            return p;
        }

        p = skip_value(p, val_type);
    }
    return NULL;
}

int gguf_get_meta_u32(const Model *model, const GGUFHeader *header, const char *key, uint32_t *val)
{
    uint32_t type = 0;
    const uint8_t *p = gguf_find_meta_value(model, header, key, &type);
    if (!p) return 0;

    if (type == 4 || type == 5) { /* UINT32 or INT32 */
        *val = read_u32(p);
        return 1;
    } else if (type == 10 || type == 11) { /* UINT64 or INT64 */
        *val = (uint32_t)read_u64(p);
        return 1;
    }
    return 0;
}

int gguf_get_meta_u64(const Model *model, const GGUFHeader *header, const char *key, uint64_t *val)
{
    uint32_t type = 0;
    const uint8_t *p = gguf_find_meta_value(model, header, key, &type);
    if (!p) return 0;

    if (type == 10 || type == 11) {
        *val = read_u64(p);
        return 1;
    } else if (type == 4 || type == 5) {
        *val = (uint64_t)read_u32(p);
        return 1;
    }
    return 0;
}

int gguf_get_meta_f32(const Model *model, const GGUFHeader *header, const char *key, float *val)
{
    uint32_t type = 0;
    const uint8_t *p = gguf_find_meta_value(model, header, key, &type);
    if (!p) return 0;

    if (type == 6) { /* FLOAT32 */
        memcpy(val, p, sizeof(float));
        return 1;
    } else if (type == 12) { /* FLOAT64 */
        double d = *(const double *)p;
        *val = (float)d;
        return 1;
    }
    return 0;
}

int gguf_get_meta_str(const Model *model, const GGUFHeader *header, const char *key, char *out, size_t max_len)
{
    uint32_t type = 0;
    const uint8_t *p = gguf_find_meta_value(model, header, key, &type);
    if (!p || type != 8) return 0;

    read_string(p, out, max_len);
    return 1;
}

void gguf_print_architecture(Model *model, GGUFHeader *header)
{
    const uint8_t *p = model->data + 24;
    printf("\nMetadata keys:\n");

    for (uint64_t i = 0; i < header->metadata_count; i++) {
        char key[256];
        p = read_string(p, key, sizeof(key));

        uint32_t value_type = read_u32(p);
        p += 4;

        printf("%2llu. %s\n", (unsigned long long)(i + 1), key);
        p = skip_value(p, value_type);
    }
}

void gguf_list_tensors(Model *model, GGUFHeader *header)
{
    const uint8_t *p = model->data + 24;
    for (uint64_t i = 0; i < header->metadata_count; i++) {
        char key[256];
        p = read_string(p, key, sizeof(key));
        uint32_t value_type = read_u32(p);
        p += 4;
        p = skip_value(p, value_type);
    }

    printf("\nFirst 10 tensors:\n");
    uint64_t limit = (header->tensor_count < 10) ? header->tensor_count : 10;

    for (uint64_t t = 0; t < limit; t++) {
        char name[256];
        p = read_string(p, name, sizeof(name));
        uint32_t n_dims = read_u32(p);
        p += 4;
        p += (uint64_t)n_dims * 8;
        p += 4; /* type */
        p += 8; /* offset */
        printf("%2llu. %s\n", (unsigned long long)(t + 1), name);
    }
}