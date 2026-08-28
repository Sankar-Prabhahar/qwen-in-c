#include <windows.h>
#include "mmap.h"

int model_map(const char *path, Model *model) {

    model->file = CreateFileA(
        path,
        GENERIC_READ,
        FILE_SHARE_READ,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (model->file == INVALID_HANDLE_VALUE)
        return 0;

    LARGE_INTEGER size;
    GetFileSizeEx(model->file, &size);

    model->mapping = CreateFileMappingA(
        model->file,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );

    if (!model->mapping) {
        CloseHandle(model->file);
        return 0;
    }

    model->data = (uint8_t *)MapViewOfFile(
        model->mapping,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!model->data) {
        CloseHandle(model->mapping);
        CloseHandle(model->file);
        return 0;
    }

    model->file_size = size.QuadPart;

    return 1;
}

void model_unmap(Model *model) {
    UnmapViewOfFile(model->data);
    CloseHandle(model->mapping);
    CloseHandle(model->file);
}