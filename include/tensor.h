#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>

#define MAX_DIMS 4

typedef struct{

    char name[256];

    uint32_t n_dims;

    uint64_t dims[MAX_DIMS];

    uint32_t type;

    uint64_t offset;

} Tensor;

#endif