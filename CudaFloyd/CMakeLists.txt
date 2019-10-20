cmake_minimum_required(VERSION 3.12)
set(CMAKE_C_COMPILER "gcc-7")
set(CMAKE_CXX_COMPILER "g++-7")

project(CudaFloyd)
set(PROJECT_NAME CudaFloyd)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 17)

file(GLOB SOURCE_FILES src/* src/*/*)

cuda_add_executable(CudaFloyd ${SOURCE_FILES})

