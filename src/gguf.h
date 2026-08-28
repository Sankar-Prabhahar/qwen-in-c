#include <stdio.h>
#include <stdint.h>
#include <string.h>
#if defined(__has_include)
#  if __has_include("../include/gguf.c")
#    include "../include/gguf.c"
#  else
typedef struct {
    char magic[5];
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
} GGUFHeader;
#  endif
#else
#  include "../include/gguf.c"
#endif

int gguf_read_header(const char *path, GGUFHeader *header){

    FILE *fp=fopen(path,"rb");

    if(!fp){
        return 0;
    }

    fread(header->magic,1,4,fp);
    header->magic[4]='\0';

    fread(&header->version,sizeof(uint32_t),1,fp);
    fread(&header->tensor_count,sizeof(uint64_t),1,fp);
    fread(&header->metadata_count,sizeof(uint64_t),1,fp);

    fclose(fp);

    return 1;
}