set(CUFHE_GPU_SOURCES
    bootstrap_gpu.cu
    cufhe_gates_gpu.cu
    cufhe_gpu.cu
    keyswitch_gpu.cu
    ntt_small_modulus.cu  # Small modulus NTT tables (~2^31)
)
if(NOT USE_CONCRETE)
    list(APPEND CUFHE_GPU_SOURCES
        annihilate_gpu.cu
        circuitbootstrapping_gpu.cu)
endif()
if(USE_FFT AND NOT USE_CONCRETE)
    list(APPEND CUFHE_GPU_SOURCES
        aes_gpu.cu
        ascon_gpu.cu)
endif()
if(USE_FFT AND NOT USE_GPU_FFT)
    # Twiddle data from tfhe-rs (BSD 3-Clause Clear, ZAMA)
    # Must be in same link unit for CUDA __device__ variable resolution
    list(APPEND CUFHE_GPU_SOURCES
        ${PROJECT_SOURCE_DIR}/thirdparties/tfhe-rs-fft/fft_negacyclic.cu)
endif()
cufhe_set_gpu_sources(${CUFHE_GPU_SOURCES})
add_library(cufhe_gpu ${CUFHE_GPU_SOURCES})
if(USE_AVX512)
    target_compile_options(cufhe_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f>
        $<$<COMPILE_LANGUAGE:HIP>:-mavx512f>)
endif()

target_link_libraries(cufhe_gpu PUBLIC tfhe++ ${CUFHE_GPU_RUNTIME_TARGET})
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
if(CUFHE_GPU_BACKEND STREQUAL "CUDA")
    target_compile_options(cufhe_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>)
    set_target_properties(cufhe_gpu PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
cufhe_configure_gpu_target(cufhe_gpu)
if(CUFHE_GPU_BACKEND STREQUAL "HIP")
    target_link_options(cufhe_gpu INTERFACE
        $<$<LINK_LANGUAGE:HIP>:--hip-link;-fgpu-rdc>)
endif()
