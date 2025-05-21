#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

float* load_data(const char* filename, size_t* size);
void free_data(float* data);

#endif
