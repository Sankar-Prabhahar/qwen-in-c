#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "gguf.h"
static const uint8_t* skip_value(const uint8_t *p, uint32_t type);
static uint32_t read_u32(const uint8_t *p){
    return *(const uint32_t*)p;
}

static uint64_t read_u64(const uint8_t *p){
    return *(const uint64_t*)p;
}

int gguf_read_header(Model *model, GGUFHeader *header){

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
static const uint8_t* skip_u32(const uint8_t *p){
    return p + 4;
}

static const uint8_t* skip_u64(const uint8_t *p){
    return p + 8;
}

static const uint8_t* read_string(const uint8_t *p, char *out, uint64_t max_len){

    uint64_t len = read_u64(p);
    p += 8;

    uint64_t copy = len;

    if(copy >= max_len)
        copy = max_len - 1;

    memcpy(out, p, copy);
    out[copy] = '\0';

    return p + len;
}

void gguf_print_architecture(Model *model, GGUFHeader *header){

    const uint8_t *p = model->data + 24;

    printf("\nMetadata keys:\n");

    for(uint64_t i = 0; i < header->metadata_count; i++){

        char key[256];

        p = read_string(p, key, sizeof(key));

        uint32_t value_type = read_u32(p);
        p += 4;

        printf("%2llu. %s\n", (unsigned long long)(i + 1), key);

        p = skip_value(p, value_type);
    }
}
static const uint8_t* skip_value(const uint8_t *p, uint32_t type){

    switch(type){

        case 0: return p + 1;              // UINT8
        case 1: return p + 1;              // INT8
        case 2: return p + 2;              // UINT16
        case 3: return p + 2;              // INT16
        case 4: return p + 4;              // UINT32
        case 5: return p + 4;              // INT32
        case 6: return p + 4;              // FLOAT32
        case 7: return p + 1;              // BOOL
        case 8:{                           // STRING
            uint64_t len = read_u64(p);
            return p + 8 + len;
        }
        case 9:{                           // ARRAY
            uint32_t elem_type = read_u32(p);
            p += 4;

            uint64_t count = read_u64(p);
            p += 8;

            for(uint64_t i=0;i<count;i++)
                p = skip_value(p, elem_type);

            return p;
        }
        case 10: return p + 8;             // UINT64
        case 11: return p + 8;             // INT64
        case 12: return p + 8;             // FLOAT64

        default:
            return p;
    }
}
void gguf_list_tensors(Model *model, GGUFHeader *header){

    const uint8_t *p = model->data + 24;

    /* Skip all metadata entries */
    for(uint64_t i = 0; i < header->metadata_count; i++){

        char key[256];

        p = read_string(p, key, sizeof(key));

        uint32_t value_type = read_u32(p);
        p += 4;

        p = skip_value(p, value_type);
    }

    printf("\nFirst 10 tensors:\n");

    uint64_t limit = header->tensor_count;
    if(limit > 10) limit = 10;

    for(uint64_t t = 0; t < limit; t++){

        char name[256];

        p = read_string(p, name, sizeof(name));

        uint32_t n_dims = read_u32(p);
        p += 4;

        for(uint32_t d = 0; d < n_dims; d++)
            p += 8;   /* skip each dimension */

        p += 4;       /* tensor type */
        p += 8;       /* tensor offset */

        printf("%2llu. %s\n",
            (unsigned long long)(t + 1),
            name);
    }
}