#ifndef TENSOR_DATA_H
#define TENSOR_DATA_H

#include "model.h"
#include "gguf.h"
#include "tensor.h"
#include "q6k.h"


int load_q6k_block(Model *model,
                    Tensor *tensor,
                    block_q6_K *block);
void print_tensor_bytes(Model *model, GGUFHeader *header, Tensor *tensor);

#endif