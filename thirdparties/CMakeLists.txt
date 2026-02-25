SET(ENABLE_TEST OFF)
add_subdirectory(TFHEpp)

# tfhe-rs negacyclic FFT lives in thirdparties/tfhe-rs-fft/ (BSD 3-Clause Clear License, ZAMA)
# Its .cu source is compiled as part of cufhe_gpu (see src/CMakeLists.txt) for
# CUDA __device__ variable linkage; header included via target_include_directories.