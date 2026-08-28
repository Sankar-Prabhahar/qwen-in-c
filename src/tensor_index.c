#include <stdint.h>
#include <string.h>

#include "tensor_index.h"

static uint32_t read_u32(const uint8_t *p){
    return *(const uint32_t*)p;
}

static uint64_t read_u64(const uint8_t *p){
    return *(const uint64_t*)p;
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

int build_tensor_index(Model *model, GGUFHeader *header, TensorIndex *index){

    const uint8_t *p = model->data + 24;

    index->count = 0;

    /* Skip metadata */
    for(uint64_t i=0;i<header->metadata_count;i++){

        uint64_t len = read_u64(p);
        p += 8 + len;

        uint32_t type = read_u32(p);
        p += 4;

        p = skip_value(p, type);
    }

    /* Read every tensor entry */
    for(uint64_t t=0;
        t<header->tensor_count && t<MAX_TENSORS;
        t++){

        Tensor *tensor = &index->tensors[t];

        uint64_t name_len = read_u64(p);
        p += 8;

        memcpy(tensor->name, p, name_len);
        tensor->name[name_len] = '\0';
        p += name_len;

        tensor->n_dims = read_u32(p);
        p += 4;

        for(uint32_t d=0;
            d<tensor->n_dims && d<MAX_DIMS;
            d++){

            tensor->dims[d] = read_u64(p);
            p += 8;
        }

        p += (uint64_t)(tensor->n_dims > MAX_DIMS ?
            tensor->n_dims - MAX_DIMS : 0) * 8;

        tensor->type = read_u32(p);
        p += 4;

        tensor->offset = read_u64(p);
        p += 8;

        index->count++;
    }

    return 1;
}

Tensor* find_tensor(TensorIndex *index, const char *name){

    for(uint64_t i=0;i<index->count;i++){

        if(strcmp(index->tensors[i].name, name)==0)
            return &index->tensors[i];
    }

    return NULL;
}