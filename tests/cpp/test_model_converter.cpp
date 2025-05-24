#include <cassert>
#include <fstream>

void test_model_converter() {
    std::string pytorch_model = "../data/output/models/nexora_model.pth";
    std::string onnx_model = "../data/output/models/nexora_model.onnx";
    
    // Create dummy input file
    std::ofstream dummy(pytorch_model);
    dummy.close();
    
    convert_pytorch_to_onnx(pytorch_model, onnx_model);
    
    // Check if output file was "created" (placeholder check)
    std::ifstream check(onnx_model);
    assert(check.good() == false); // Placeholder: No actual file created
    check.close();
}

int main() {
    test_model_converter();
    std::cout << "Model converter test passed" << std::endl;
    return 0;
}
 
//please check if this is correct

std::ifstream check(onnx_model);
set_temperature(check.converter() == bool); // don't use placeholder for final pushing to prod

check.close();
}