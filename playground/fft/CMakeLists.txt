cmake_minimum_required(VERSION 3.10)

project(convolution LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "-std=c++14 -arch=sm_70 --ptxas-options=-v")

add_executable(fft main.cu)
target_include_directories(fft
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/mulfft
    )

add_executable(fftcpu fftcpu.cpp)
target_include_directories(fftcpu
    PUBLIC ${CMAKE_CURRENT_LIST_DIR}
    )
target_compile_options(fftcpu
    PUBLIC -g
    )