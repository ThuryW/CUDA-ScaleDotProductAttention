cmake_minimum_required(VERSION 3.18)
project(ScaledDotProductAttention LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Find CUDA package
find_package(CUDA REQUIRED)

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})

include_directories( ./src/include )

# Add executable for the matrix multiplication test
add_executable(test_matmul tests/test_matmul.cu)

set(SOURCE
    src/main.cu
    src/matmul.cu
    src/inout.cu
    src/utils.cu
    src/softmax.cu
)

add_executable(main ${SOURCE})

# Link against CUDA and cuBLAS libraries
target_link_libraries(test_matmul ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})
target_link_libraries(main ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY})

# Specify CUDA architecture (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Enable separable compilation for CUDA
set_target_properties(test_matmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)