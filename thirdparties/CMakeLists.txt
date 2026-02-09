SET(ENABLE_TEST OFF)
add_subdirectory(TFHEpp)

# Add GPU-NTT library
set(GPUNTT_BUILD_EXAMPLES OFF CACHE BOOL "Build GPU-NTT Examples" FORCE)
set(GPUNTT_BUILD_BENCHMARKS OFF CACHE BOOL "Build GPU-NTT Benchmarks" FORCE)
add_subdirectory(GPU-NTT)

if(USE_GPU_FFT)
  set(GPUFFT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GPUFFT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  add_subdirectory(GPU-FFT)
endif()

# tfhe-rs negacyclic FFT lives in thirdparties/tfhe-rs-fft/ (BSD 3-Clause Clear License, ZAMA)
# Its .cu source is compiled as part of cufhe_gpu (see src/CMakeLists.txt) for
# CUDA __device__ variable linkage; header included via target_include_directories.