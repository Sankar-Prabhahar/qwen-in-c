#ifndef TENSOR_INDEX_H
#define TENSOR_INDEX_H

#include "model.h"
#include "gguf.h"
#include "tensor.h"

#define MAX_TENSORS 1000

typedef struct{
    Tensor tensors[MAX_TENSORS];
    uint64_t count;
} TensorIndex;

int build_tensor_index(Model *model, GGUFHeader *header, TensorIndex *index);

Tensor* find_tensor(TensorIndex *index, const char *name);

#endif