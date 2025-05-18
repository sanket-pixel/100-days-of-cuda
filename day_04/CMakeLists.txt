cmake_minimum_required(VERSION 3.18)
project(euclidian LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
add_executable(euclidian euclidian.cu)
