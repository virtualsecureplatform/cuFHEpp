set(CUFHE_GPU_SOURCES
    bootstrap_gpu.cu
    annihilate_gpu.cu
    circuitbootstrapping_gpu.cu
    cufhe_gates_gpu.cu
    cufhe_gpu.cu
    keyswitch_gpu.cu
    aes_gpu.cu
    ascon_gpu.cu
    ntt_small_modulus.cu  # Small modulus NTT tables (~2^31)
)
if(USE_FFT AND NOT USE_GPU_FFT)
    # Twiddle data from tfhe-rs (BSD 3-Clause Clear, ZAMA)
    # Must be in same link unit for CUDA __device__ variable resolution
    list(APPEND CUFHE_GPU_SOURCES
        ${PROJECT_SOURCE_DIR}/thirdparties/tfhe-rs-fft/fft_negacyclic.cu)
endif()
add_library(cufhe_gpu ${CUFHE_GPU_SOURCES})
if(USE_AVX512)
    target_compile_options(cufhe_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f>)
endif()

target_link_libraries(cufhe_gpu PUBLIC tfhe++)
target_include_directories(cufhe_gpu
    PUBLIC ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/cereal/include
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/randen
    ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/spqlios
    )
if(USE_FFT AND NOT USE_GPU_FFT)
    target_include_directories(cufhe_gpu
        PUBLIC ${PROJECT_SOURCE_DIR}/thirdparties/tfhe-rs-fft)
endif()
set_target_properties(cufhe_gpu PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)

# Enable CUDA separable compilation for __device__ variable linking across TUs
set_target_properties(cufhe_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cufhe_gpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
