#include "data_handler.h"
#include <stdio.h>
#include <stdlib.h>

float* load_data(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return NULL;
    }

    // Placeholder: Read data into array
    *size = 10; // Dummy size
    float* data = (float*)malloc(*size * sizeof(float));
    for (size_t i = 0; i < *size; i++) {
        data[i] = (float)i; // Dummy data
    }

    fclose(file);
    return data;
}

void free_data(float* data) {
    if (data) free(data);
}
