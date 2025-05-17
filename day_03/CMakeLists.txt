cmake_minimum_required(VERSION 3.18)
project(matrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
add_executable(matrix matrix.cu)
