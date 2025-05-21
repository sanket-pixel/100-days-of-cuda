cmake_minimum_required(VERSION 3.18)
project(grayscale LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

add_executable(grayscale grayscale.cu)


include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(grayscale ${OpenCV_LIBS})