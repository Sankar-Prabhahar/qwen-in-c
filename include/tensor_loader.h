#ifndef TENSOR_LOADER_H
#define TENSOR_LOADER_H

#include "model.h"
#include "gguf.h"
#include "tensor.h"

int load_first_tensor(Model *model, GGUFHeader *header, Tensor *tensor);

#endif