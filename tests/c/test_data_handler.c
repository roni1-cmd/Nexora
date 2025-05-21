#include "data_handler.h"
#include <assert.h>
#include <stdio.h>

void test_load_data() {
    size_t size;
    float* data = load_data("dummy.txt", &size);
    assert(data != NULL);
    assert(size == 10);
    for (size_t i = 0; i < size; i++) {
        assert(data[i] == (float)i);
    }
    free_data(data);
}

int main() {
    test_load_data();
    printf("Data handler test passed\n");
    return 0;
}
