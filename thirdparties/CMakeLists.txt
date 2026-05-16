SET(ENABLE_TEST OFF)
if(NOT DEFINED USE_SPQLIOS_INTL)
  set(USE_SPQLIOS_INTL OFF CACHE BOOL "Use interleaved-format SPQLIOS")
endif()
if(NOT DEFINED USE_AVX512)
  set(USE_AVX512 OFF CACHE BOOL "Use AVX512 ver. SPQLIOS")
endif()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/TFHEpp/include)
add_subdirectory(TFHEpp)
target_compile_definitions(tfhe++ INTERFACE ${TFHEpp_DEFINITIONS})

# tfhe-rs negacyclic FFT lives in thirdparties/tfhe-rs-fft/ (BSD 3-Clause Clear License, ZAMA)
# Its .cu source is compiled as part of cufhe_gpu (see src/CMakeLists.txt) for
# CUDA __device__ variable linkage; header included via target_include_directories.
