#include "data_validator.h"
#include <assert.h>
#include <stdio.h>

void test_data_validator() {
    // Create dummy file with valid data
    FILE* file = fopen("dummy.txt", "w");
    fprintf(file, "1.0 2.0 3.0");
    fclose(file);
    
    assert(validate_data("dummy.txt") == true);
    
    // Create dummy file with invalid data
    file = fopen("invalid.txt", "w");
    fprintf(file, "-1.0 2.0 3.0");
    fclose(file);
    
    assert(validate_data("invalid.txt") == false);
    
    remove("dummy.txt");
    remove("invalid.txt");
}

int main() {
    test_data_validator();
    printf("Data validator test passed\n");
    return 0;
}
