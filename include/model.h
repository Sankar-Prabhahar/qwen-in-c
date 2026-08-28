#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <windows.h>

typedef struct {
    uint8_t *data;
    uint64_t file_size;
    uint64_t data_start;
    HANDLE file;
    HANDLE mapping;
} Model;

#endif