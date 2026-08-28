#ifndef MMAP_H
#define MMAP_H

#include "model.h"

int model_map(const char *path, Model *model);
void model_unmap(Model *model);

#endif