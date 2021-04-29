cmake_minimum_required (VERSION 3.9)

project(deterministic LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

add_library(deterministic SHARED
  bob.cu
)

target_compile_features(deterministic PRIVATE cxx_std_14)
target_link_libraries(deterministic PRIVATE ${TORCH_LIBRARIES} Python::Python)