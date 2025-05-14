cmake_minimum_required(VERSION 3.18)
project(CudaSquare LANGUAGES CXX CUDA)

message(STATUS "CUDA Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")

include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        /usr/local/cuda/include  # Adjust path if necessary
)
set(CMAKE_CXX_STANDARD 17)
add_executable(cuda_square square.cu)
set_target_properties(cuda_square PROPERTIES CUDA_ARCHITECTURES "89")