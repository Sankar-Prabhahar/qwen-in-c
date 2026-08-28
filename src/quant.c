#include <stdio.h>
#include "quant.h"

void describe_quant_type(uint32_t type){

    printf("\nQuantization Info\n");

    switch(type){

        case 0:
            printf("Format: F32\n");
            printf("Bits per weight: 32\n");
            break;

        case 1:
            printf("Format: F16\n");
            printf("Bits per weight: 16\n");
            break;

        case 2:
            printf("Format: Q4_0\n");
            printf("Bits per weight: 4\n");
            break;

        case 3:
            printf("Format: Q4_1\n");
            printf("Bits per weight: 4\n");
            break;

        case 14:
             printf("Format: Q6_K\n");
            printf("Bits per weight: 6\n");
            printf("Block quantization: K-Quant\n");
            break;

        default:
            printf("Format: Unknown (%u)\n", type);
            break;
    }
}