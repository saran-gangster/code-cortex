cmake_minimum_required(VERSION 3.24)
project(aeroguard_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers or libnvinfer were not found. Use the TensorRT shipped with JetPack.")
endif()

add_executable(aeroguard_native
    src/main.cpp
    src/tensorrt_runner.cpp
    src/cuda_preprocess.cu
    src/cuda_postprocess.cu)

target_include_directories(aeroguard_native PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})

target_link_libraries(aeroguard_native PRIVATE
    ${TENSORRT_LIBRARY}
    CUDA::cudart
    ${OpenCV_LIBS})

target_compile_options(aeroguard_native PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(aeroguard_native PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)

