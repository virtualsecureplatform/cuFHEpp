set(CUFHE_TEST_GPU_SOURCES
    test_backend_gate_bench.cu
    test_small_modulus.cu
    test_fft_roundtrip.cu
    test_accumulate_diag.cu
    test_annihilate_gpu.cu
    test_circuitbootstrapping_gpu.cu
    test_keyswitch_diag.cu
    test_compare_cpu_gpu.cu
    test_single_cmux.cu
    test_blind_rotate_compare.cu
    test_fft_perf.cu
    test_aes_decrypt_gpu.cu
    test_ascon_xof_gpu.cu)
cufhe_set_gpu_sources(${CUFHE_TEST_GPU_SOURCES})

function(cufhe_enable_cuda_separable_compilation target)
    if(CUFHE_GPU_BACKEND STREQUAL "CUDA")
        set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    endif()
endfunction()

add_executable(test_gate_gpu test_gate_gpu.cc)
target_include_directories(test_gate_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

add_executable(test_mux_batch_gpu test_mux_batch_gpu.cc)
target_include_directories(test_mux_batch_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_mux_batch_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

if(USE_BLOCK_BINARY)
    add_executable(test_block_binary_gpu test_block_binary_gpu.cc)
    target_include_directories(test_block_binary_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    target_link_libraries(test_block_binary_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

add_executable(test_gate_gpu_multi test_gate_gpu_multi.cc)
target_include_directories(test_gate_gpu_multi PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_multi cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

add_executable(test_intensive test_intensive.cc)
target_include_directories(test_intensive PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_intensive cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

if(NOT USE_BLOCK_BINARY)
    add_executable(test_perf test_perf.cc)
    target_include_directories(test_perf PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    target_link_libraries(test_perf cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

    add_executable(test_backend_gate_bench test_backend_gate_bench.cu)
    target_include_directories(test_backend_gate_bench PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_backend_gate_bench)
    target_link_libraries(test_backend_gate_bench cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(NOT USE_FFT)
    add_executable(test_small_modulus test_small_modulus.cu)
    target_include_directories(test_small_modulus PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_small_modulus)
    target_link_libraries(test_small_modulus cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

add_executable(test_gate_gpu_lvl02 test_gate_gpu_lvl02.cc)
target_include_directories(test_gate_gpu_lvl02 PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_lvl02 cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

if(USE_FFT AND USE_GPU_FFT)
    add_executable(test_fft_roundtrip test_fft_roundtrip.cu)
    target_include_directories(test_fft_roundtrip PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_fft_roundtrip)
    target_link_libraries(test_fft_roundtrip cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(USE_FFT AND USE_GPU_FFT AND NOT USE_BLOCK_BINARY)
    add_executable(test_accumulate_diag test_accumulate_diag.cu)
    target_include_directories(test_accumulate_diag PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_accumulate_diag)
    target_link_libraries(test_accumulate_diag cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

# The Tangor/Iyokan integration passes TFHEpp's split-Fourier TRGSWFFT
# representation directly to cuFHEpp. Keep a focused regression test for that
# adapter separate from the native cuFHEpp CMUX tests.
if(USE_FFT AND USE_GPU_FFT)
    add_executable(test_tfhepp_cmux_cuda test_tfhepp_cmux_cuda.cu)
    target_include_directories(test_tfhepp_cmux_cuda PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_tfhepp_cmux_cuda)
    target_link_libraries(test_tfhepp_cmux_cuda cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

    # This is the per-cell bootstrap used after a RAM/ROM CMUX update.  Keep
    # it available in the block-binary configuration used by Iyokan/Tangor.
    add_executable(test_tfhepp_ram_bootstrap_cuda test_tfhepp_ram_bootstrap_cuda.cu)
    target_include_directories(test_tfhepp_ram_bootstrap_cuda PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_tfhepp_ram_bootstrap_cuda)
    target_link_libraries(test_tfhepp_ram_bootstrap_cuda cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(NOT USE_CONCRETE)
    add_executable(test_annihilate_gpu test_annihilate_gpu.cu)
    target_include_directories(test_annihilate_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_annihilate_gpu)
    target_link_libraries(test_annihilate_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

    add_executable(test_circuitbootstrapping_gpu test_circuitbootstrapping_gpu.cu)
    target_include_directories(test_circuitbootstrapping_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_circuitbootstrapping_gpu)
    target_link_libraries(test_circuitbootstrapping_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

add_executable(test_keyswitch_diag test_keyswitch_diag.cu)
target_include_directories(test_keyswitch_diag PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
cufhe_enable_cuda_separable_compilation(test_keyswitch_diag)
target_link_libraries(test_keyswitch_diag cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

add_executable(test_cpu_lvl02 test_cpu_lvl02.cc)
target_include_directories(test_cpu_lvl02 PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
target_link_libraries(test_cpu_lvl02 cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

if(NOT USE_BLOCK_BINARY)
    add_executable(test_compare_cpu_gpu test_compare_cpu_gpu.cu)
    target_include_directories(test_compare_cpu_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_compare_cpu_gpu)
    target_link_libraries(test_compare_cpu_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(USE_FFT AND USE_GPU_FFT AND NOT USE_BLOCK_BINARY)
    add_executable(test_single_cmux test_single_cmux.cu)
    target_include_directories(test_single_cmux PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_single_cmux)
    target_link_libraries(test_single_cmux cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

    add_executable(test_blind_rotate_compare test_blind_rotate_compare.cu)
    target_include_directories(test_blind_rotate_compare PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_blind_rotate_compare)
    target_link_libraries(test_blind_rotate_compare cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(USE_FFT AND USE_GPU_FFT)
    add_executable(test_fft_perf test_fft_perf.cu)
    target_include_directories(test_fft_perf PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_fft_perf)
    target_link_libraries(test_fft_perf cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(USE_FFT AND NOT USE_CONCRETE)
    add_executable(test_aes_decrypt_gpu
        test_aes_decrypt_gpu.cu
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/AES/src/AES.cpp)
    target_include_directories(test_aes_decrypt_gpu PRIVATE ${CUFHE_GPU_INCLUDE_DIRECTORIES})
    cufhe_enable_cuda_separable_compilation(test_aes_decrypt_gpu)
    target_link_libraries(test_aes_decrypt_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})

    add_executable(test_ascon_xof_gpu
        test_ascon_xof_gpu.cu
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/ascon-c/crypto_hash/asconxof128/ref/hash.c)
    target_include_directories(test_ascon_xof_gpu
        PRIVATE
        ${CUFHE_GPU_INCLUDE_DIRECTORIES}
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/ascon-c/tests)
    cufhe_enable_cuda_separable_compilation(test_ascon_xof_gpu)
    target_link_libraries(test_ascon_xof_gpu cufhe_gpu ${CUFHE_GPU_RUNTIME_TARGET})
endif()

if(CUFHE_GPU_BACKEND STREQUAL "HIP")
    get_property(CUFHE_TEST_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS CUFHE_TEST_TARGETS)
        cufhe_configure_gpu_target(${target})
    endforeach()
endif()
