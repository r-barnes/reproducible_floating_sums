cmake_minimum_required (VERSION 3.9)

project(deterministic LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

# add_library(deterministic SHARED
#   kahan.cu
# )
# target_compile_features(deterministic PRIVATE cxx_std_14)
# target_link_libraries(deterministic PRIVATE ${TORCH_LIBRARIES} Python::Python)
# target_compile_options(deterministic PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)

# add_library(demmel SHARED
#   demmel.cu
# )
# target_compile_features(demmel PRIVATE cxx_std_14)
# target_link_libraries(demmel PRIVATE ${TORCH_LIBRARIES} Python::Python)
# target_compile_options(demmel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)

add_library(repro_blas_cuda SHARED
  repro_blas_cuda.cu
)
target_compile_features(repro_blas_cuda PRIVATE cxx_std_14)
target_link_libraries(repro_blas_cuda PRIVATE ${TORCH_LIBRARIES} Python::Python)
target_compile_options(repro_blas_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)