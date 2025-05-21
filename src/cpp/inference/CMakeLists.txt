cmake_minimum_required(VERSION 3.10)
project(NexoraInference)

find_package(ONNXRuntime REQUIRED)
add_executable(inference inference.cpp)
target_link_libraries(inference PRIVATE ONNXRuntime::ONNXRuntime)
target_include_directories(inference PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
