#include "data_handler.h"
#include <stdio.h>
#include <stdbool.h>

bool validate_data(const char* filename) {
    size_t size;
    float* data = load_data(filename, &size);
    if (!data) {
        fprintf(stderr, "Data validation failed: Could not load file %s\n", filename);
        return false;
    }
    
    // Placeholder: Check for NaN or invalid values
    for (size_t i = 0; i < size; i++) {
        if (data[i] < 0) { // Example validation rule
            fprintf(stderr, "Invalid data detected at index %zu: %f\n", i, data[i]);
            free_data(data);
            return false;
        }
    }
    
    free_data(data);
    printf("Data validation passed for %s\n", filename);
    return true;
}
