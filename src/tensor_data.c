#include <stdio.h>
#include <stdint.h>

#include "tensor_data.h"

void print_tensor_bytes(Model *model, GGUFHeader *header, Tensor *tensor){

    (void)header;   /* silence unused warning */

    const uint8_t *data = model->data + model->data_start + tensor->offset;
    printf("\nFirst 16 tensor bytes:\n");

    for(int i = 0; i < 16; i++){
        printf("%02X ", data[i]);
    }

    printf("\n");
}
int load_q6k_block(Model *model,
                   Tensor *tensor,
                   block_q6_K *block)
{
    const uint8_t *src =
        model->data +
        model->data_start +
        tensor->offset;

    memcpy(block, src, sizeof(block_q6_K));

    return 1;
}