# Uses small modulus NTT with GPU-NTT library for NTT primitives
set(CUFHE_GPU_SOURCES
    bootstrap_gpu.cu
    cufhe_gates_gpu.cu
    cufhe_gpu.cu
    keyswitch_gpu.cu
    ntt_small_modulus.cu  # Small modulus NTT tables (~2^31)
)
if(USE_FFT)
    list(APPEND CUFHE_GPU_SOURCES fft_negacyclic.cu)  # Negacyclic FFT twiddle factors
endif()
add_library(cufhe_gpu ${CUFHE_GPU_SOURCES})

target_link_libraries(cufhe_gpu PUBLIC ntt)  # GPU-NTT library
target_link_libraries(cufhe_gpu INTERFACE tfhe++)
target_include_directories(cufhe_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/cereal/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/spqlios
    ${PROJECT_SOURCE_DIR}/thirdparties/GPU-NTT/src/include
    )
set_target_properties(cufhe_gpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
set_target_properties(cufhe_gpu PROPERTIES CUDA_ARCHITECTURES 80) #A100

# Enable CUDA separable compilation for __device__ variable linking across TUs
set_target_properties(cufhe_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cufhe_gpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
