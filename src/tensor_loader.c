#include <stdint.h>
#include <string.h>

#include "tensor_loader.h"

static uint32_t read_u32(const uint8_t *p){
    return *(const uint32_t*)p;
}

static uint64_t read_u64(const uint8_t *p){
    return *(const uint64_t*)p;
}

static const uint8_t* skip_string(const uint8_t *p){
    uint64_t len = read_u64(p);
    return p + 8 + len;
}

static const uint8_t* skip_value(const uint8_t *p, uint32_t type){

    switch(type){
        case 0:
        case 1:
        case 7: return p + 1;

        case 2:
        case 3: return p + 2;

        case 4:
        case 5:
        case 6: return p + 4;

        case 10:
        case 11:
        case 12: return p + 8;

        case 8:{
            uint64_t len = read_u64(p);
            return p + 8 + len;
        }

        case 9:{
            uint32_t elem = read_u32(p);
            p += 4;

            uint64_t count = read_u64(p);
            p += 8;

            for(uint64_t i=0;i<count;i++)
                p = skip_value(p, elem);

            return p;
        }

        default:
            return p;
    }
}

int load_first_tensor(Model *model, GGUFHeader *header, Tensor *tensor){

    const uint8_t *p = model->data + 24;

    /* Skip metadata */
    for(uint64_t i=0;i<header->metadata_count;i++){

        uint64_t len = read_u64(p);
        p += 8 + len;

        uint32_t type = read_u32(p);
        p += 4;

        p = skip_value(p, type);
    }

    /* Read first tensor */

    uint64_t len = read_u64(p);
    p += 8;

    memcpy(tensor->name, p, len);
    tensor->name[len] = '\0';
    p += len;

    tensor->n_dims = read_u32(p);
    p += 4;

    for(uint32_t i=0;i<tensor->n_dims && i<MAX_DIMS;i++){

        tensor->dims[i] = read_u64(p);
        p += 8;
    }

tensor->type = read_u32(p);
p += 4;

tensor->offset = read_u64(p);
p += 8;

/* Skip the remaining tensor directory entries */
for(uint64_t t = 1; t < header->tensor_count; t++){

    uint64_t name_len = read_u64(p);
    p += 8 + name_len;

    uint32_t dims = read_u32(p);
    p += 4;

    p += (uint64_t)dims * 8;   /* skip dimensions */

    p += 4;                    /* skip tensor type */
    p += 8;                    /* skip tensor offset */
}

/* GGUF aligns tensor data to 32 bytes */
uintptr_t addr = (uintptr_t)p;
addr = (addr + 31) & ~(uintptr_t)31;

model->data_start = (uint64_t)(addr - (uintptr_t)model->data);

return 1;
}