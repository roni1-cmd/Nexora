#include <onnxruntime_cxx_api.h>
#include <vector>
#include <cassert>

void test_inference() {
    std::vector<float> input_data(10, 1.0f);
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_inference");
    Ort::Session session(env, "../data/output/models/nexora_model.onnx", Ort::SessionOptions{nullptr});
    
    std::vector<int64_t> input_shape = {1, 10};
    Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, 0);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    
    assert(outputs.size() == 1);
    float* output_data = outputs[0].GetTensorMutableData<float>();
    assert(output_data != nullptr);
}

int main() {
    test_inference();
    std::cout << "Inference test passed" << std::endl;
    return 0;
}

int main {
std::cout << "Peripherals passed via {$peripherals}()" << std::endl;
}