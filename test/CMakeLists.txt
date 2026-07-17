add_executable(test_gate_gpu test_gate_gpu.cc)
target_include_directories(test_gate_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu cufhe_gpu ${CUDA_LIBRARIES})

if(USE_BLOCK_BINARY)
    add_executable(test_block_binary_gpu test_block_binary_gpu.cc)
    target_include_directories(test_block_binary_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(test_block_binary_gpu cufhe_gpu ${CUDA_LIBRARIES})
endif()

add_executable(test_gate_gpu_multi test_gate_gpu_multi.cc)
target_include_directories(test_gate_gpu_multi PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_multi cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_intensive test_intensive.cc)
target_include_directories(test_intensive PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_intensive cufhe_gpu ${CUDA_LIBRARIES})

if(NOT USE_BLOCK_BINARY)
    add_executable(test_perf test_perf.cc)
    target_include_directories(test_perf PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_libraries(test_perf cufhe_gpu ${CUDA_LIBRARIES})

    add_executable(test_backend_gate_bench test_backend_gate_bench.cu)
    target_include_directories(test_backend_gate_bench PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_backend_gate_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_backend_gate_bench cufhe_gpu ${CUDA_LIBRARIES})
endif()

add_executable(test_gate_gpu_lvl02 test_gate_gpu_lvl02.cc)
target_include_directories(test_gate_gpu_lvl02 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_gate_gpu_lvl02 cufhe_gpu ${CUDA_LIBRARIES})

if(USE_FFT AND USE_GPU_FFT)
    add_executable(test_fft_roundtrip test_fft_roundtrip.cu)
    target_include_directories(test_fft_roundtrip PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_fft_roundtrip PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_fft_roundtrip cufhe_gpu ${CUDA_LIBRARIES})
endif()

if(USE_FFT AND USE_GPU_FFT AND NOT USE_BLOCK_BINARY)
    add_executable(test_accumulate_diag test_accumulate_diag.cu)
    target_include_directories(test_accumulate_diag PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_accumulate_diag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_accumulate_diag cufhe_gpu ${CUDA_LIBRARIES})
endif()

if(NOT USE_CONCRETE)
    add_executable(test_annihilate_gpu test_annihilate_gpu.cu)
    target_include_directories(test_annihilate_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_annihilate_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_annihilate_gpu cufhe_gpu ${CUDA_LIBRARIES})

    add_executable(test_circuitbootstrapping_gpu test_circuitbootstrapping_gpu.cu)
    target_include_directories(test_circuitbootstrapping_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_circuitbootstrapping_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_circuitbootstrapping_gpu cufhe_gpu ${CUDA_LIBRARIES})
endif()

add_executable(test_keyswitch_diag test_keyswitch_diag.cu)
target_include_directories(test_keyswitch_diag PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(test_keyswitch_diag PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_keyswitch_diag cufhe_gpu ${CUDA_LIBRARIES})

add_executable(test_cpu_lvl02 test_cpu_lvl02.cc)
target_include_directories(test_cpu_lvl02 PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(test_cpu_lvl02 cufhe_gpu ${CUDA_LIBRARIES})

if(NOT USE_BLOCK_BINARY)
    add_executable(test_compare_cpu_gpu test_compare_cpu_gpu.cu)
    target_include_directories(test_compare_cpu_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_compare_cpu_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_compare_cpu_gpu cufhe_gpu ${CUDA_LIBRARIES})
endif()

if(USE_FFT AND USE_GPU_FFT AND NOT USE_BLOCK_BINARY)
    add_executable(test_single_cmux test_single_cmux.cu)
    target_include_directories(test_single_cmux PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_single_cmux PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_single_cmux cufhe_gpu ${CUDA_LIBRARIES})

    add_executable(test_blind_rotate_compare test_blind_rotate_compare.cu)
    target_include_directories(test_blind_rotate_compare PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_blind_rotate_compare PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_blind_rotate_compare cufhe_gpu ${CUDA_LIBRARIES})
endif()

if(USE_FFT AND USE_GPU_FFT)
    add_executable(test_fft_perf test_fft_perf.cu)
    target_include_directories(test_fft_perf PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_fft_perf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_fft_perf cufhe_gpu ${CUDA_LIBRARIES})
endif()

if(USE_FFT AND NOT USE_CONCRETE)
    add_executable(test_aes_decrypt_gpu
        test_aes_decrypt_gpu.cu
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/AES/src/AES.cpp)
    target_include_directories(test_aes_decrypt_gpu PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    set_target_properties(test_aes_decrypt_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_aes_decrypt_gpu cufhe_gpu ${CUDA_LIBRARIES})

    add_executable(test_ascon_xof_gpu
        test_ascon_xof_gpu.cu
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/ascon-c/crypto_hash/asconxof128/ref/hash.c)
    target_include_directories(test_ascon_xof_gpu
        PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${PROJECT_SOURCE_DIR}/thirdparties/TFHEpp/thirdparties/ascon-c/tests)
    set_target_properties(test_ascon_xof_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(test_ascon_xof_gpu cufhe_gpu ${CUDA_LIBRARIES})
endif()
