cmake_minimum_required(VERSION 3.21)

set(CUFHE_GPU_BACKEND "CUDA" CACHE STRING
    "GPU backend to use (CUDA or HIP)")
set_property(CACHE CUFHE_GPU_BACKEND PROPERTY STRINGS CUDA HIP ROCM)
string(TOUPPER "${CUFHE_GPU_BACKEND}" CUFHE_GPU_BACKEND)
if(CUFHE_GPU_BACKEND STREQUAL "ROCM")
  set(CUFHE_GPU_BACKEND "HIP")
endif()

if(CUFHE_GPU_BACKEND STREQUAL "HIP")
  if(NOT DEFINED CMAKE_HIP_ARCHITECTURES AND NOT DEFINED ENV{HIPARCHS})
    set(CMAKE_HIP_ARCHITECTURES "gfx1201" CACHE STRING
        "AMD GPU architectures to build for. gfx1201 targets Radeon AI PRO R9700.")
  endif()
  project(cuFHE LANGUAGES HIP CXX C)
  add_compile_definitions(CUFHE_USE_HIP=1)
elseif(CUFHE_GPU_BACKEND STREQUAL "CUDA")
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "80;89" CACHE STRING
        "CUDA architectures to build for. 80 supports A100; 89 supports RTX 4070.")
  endif()
  project(cuFHE LANGUAGES CUDA CXX C)
  find_package(CUDAToolkit REQUIRED)
  set(CUFHE_GPU_RUNTIME_TARGET CUDA::cudart)
  set(CUFHE_GPU_INCLUDE_DIRECTORIES
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  message(FATAL_ERROR
      "Unsupported CUFHE_GPU_BACKEND='${CUFHE_GPU_BACKEND}'. Use CUDA or HIP.")
endif()
message(STATUS "cuFHEpp GPU backend: ${CUFHE_GPU_BACKEND}")

if(CUFHE_GPU_BACKEND STREQUAL "HIP")
  find_path(CUFHE_HIP_INCLUDE_DIR hip/hip_runtime.h
      HINTS ${CMAKE_HIP_COMPILER_ROCM_ROOT} $ENV{ROCM_PATH} /opt/rocm
      PATH_SUFFIXES include
      REQUIRED)
  find_library(CUFHE_HIP_RUNTIME_LIBRARY amdhip64
      HINTS ${CMAKE_HIP_COMPILER_ROCM_ROOT} $ENV{ROCM_PATH} /opt/rocm
      PATH_SUFFIXES lib lib64
      REQUIRED)
  add_library(cufhe_hip_runtime INTERFACE)
  target_compile_definitions(cufhe_hip_runtime INTERFACE
      CUFHE_USE_HIP=1 __HIP_PLATFORM_AMD__=1)
  target_include_directories(cufhe_hip_runtime SYSTEM INTERFACE
      ${CUFHE_HIP_INCLUDE_DIR})
  target_link_libraries(cufhe_hip_runtime INTERFACE
      ${CUFHE_HIP_RUNTIME_LIBRARY})
  set(CUFHE_GPU_RUNTIME_TARGET cufhe_hip_runtime)
  set(CUFHE_GPU_INCLUDE_DIRECTORIES ${CUFHE_HIP_INCLUDE_DIR})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

function(cufhe_set_gpu_sources)
  if(CUFHE_GPU_BACKEND STREQUAL "HIP")
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
  endif()
endfunction()

function(cufhe_configure_gpu_target target)
  if(CUFHE_GPU_BACKEND STREQUAL "HIP")
    # Device globals are referenced across translation units, so HIP needs
    # relocatable device code during both compilation and the final link.
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>
        $<$<COMPILE_LANGUAGE:HIP>:-include>
        $<$<COMPILE_LANGUAGE:HIP>:${PROJECT_SOURCE_DIR}/include/hip_compat_math.cuh>
        $<$<COMPILE_LANGUAGE:HIP>:-Wno-unused-value>)
    target_link_options(${target} PRIVATE
        $<$<LINK_LANGUAGE:HIP>:--hip-link;-fgpu-rdc>)
  else()
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  endif()
endfunction()

option(USE_BLAKE3 "Use blake3 as CSPRNG" ON)
option(USE_RANDEN "Use randen as CSPRNG" OFF)
option(USE_80BIT_SECURITY "Use 80bit security parameter(faster)" OFF)
option(USE_CGGI19 "Use the parameter set proposed in CGGI19" OFF)
option(USE_BLOCK_BINARY "Use TFHEpp parameters with block-binary lvl0 keys" OFF)
option(USE_CONCRETE "Use the parameter set proposed in CONCRETE" OFF)
option(USE_KEY_BUNDLE "Use key bundle algorithm (2-bit multi-bit bootstrapping)" ON)
option(USE_SUBSET_KEY "Use subset key switching" OFF)
option(USE_FFT "Use negacyclic FFT instead of NTT (tfhe-rs style)" ON)

if(USE_BLOCK_BINARY)
  if(USE_KEY_BUNDLE)
    message(STATUS "Block-binary keys are incompatible with key bundling; disabling USE_KEY_BUNDLE")
    set(USE_KEY_BUNDLE OFF CACHE BOOL
        "Use key bundle algorithm (2-bit multi-bit bootstrapping)" FORCE)
  endif()
  set(USE_SUBSET_KEY ON CACHE BOOL "Use subset key switching" FORCE)
endif()

if(USE_BLAKE3 AND USE_RANDEN)
  message(FATAL_ERROR "USE_BLAKE3 and USE_RANDEN are mutually exclusive")
endif()

if(USE_BLAKE3)
  add_compile_definitions(USE_BLAKE3)
elseif(USE_RANDEN)
  add_compile_definitions(USE_RANDEN)
endif()

if(USE_KEY_BUNDLE)
  add_compile_definitions(USE_KEY_BUNDLE)
endif()

if(USE_SUBSET_KEY)
  add_compile_definitions(USE_SUBSET_KEY)
endif()

if(USE_FFT)
  add_compile_definitions(USE_FFT)
endif()

option(USE_GPU_FFT "Use GPU-FFT library for negacyclic FFT" ON)
if(NOT USE_FFT AND USE_GPU_FFT)
  message(STATUS "USE_FFT=OFF selects the NTT backend; disabling USE_GPU_FFT")
  set(USE_GPU_FFT OFF CACHE BOOL
      "Use GPU-FFT library for negacyclic FFT" FORCE)
endif()
if(USE_GPU_FFT)
  add_compile_definitions(USE_GPU_FFT)
endif()

if(USE_BLOCK_BINARY AND USE_FFT AND NOT USE_GPU_FFT)
  message(FATAL_ERROR
      "Block-binary FFT builds require USE_GPU_FFT=ON; use USE_FFT=OFF for NTT")
endif()

# Ports of the ROCm-tuned transform paths to CUDA builds.  Always active on
# HIP.  Defaults follow RTX 4070 (sm_89) benchmarks: the paired FFT is ~20%
# faster, register accumulators are a modest NTT win, and the low-LDS layout is
# required for lvl02 MUX on GPUs with a ~99 KiB shared-memory cap (Ada).  The
# warp-shuffle tail stays off: NVIDIA's 64-register ceiling for 1024-thread
# blocks makes it spill and lose occupancy.
option(USE_CUDA_WARP_TRANSFORM
       "Run the final NTT butterfly stages in registers via warp shuffles" OFF)
option(USE_CUDA_REGISTER_NTT_ACCUM
       "Keep pointwise multiply-accumulate results in registers" ON)
option(USE_CUDA_LOW_LDS_BOOTSTRAP
       "Use the low-shared-memory bootstrap scratch layout" ON)
option(USE_CUDA_PAIRED_GPU_FFT
       "Run two GPU-FFT transforms per block in the KeyBundle path" ON)
# Second-generation lvl1 NTT optimizations (RTX 4070 tuning): together they
# cut lvl1 gate time by ~29%.  MIN_BLOCKS applies its three-block launch bound
# only to sm_89+; Ampere keeps unconstrained register allocation.
option(USE_CUDA_PAIRED_NTT
       "Transform two decomposition digits per lvl1 NTT pass" ON)
option(USE_CUDA_SHOUP_NTT
       "Use Shoup precomputed-quotient twiddle multiplication in the lvl1 NTT"
       ON)
option(USE_CUDA_MIN_BLOCKS
       "Force three resident lvl1 NTT gate blocks on sm_89+ via launch bounds"
       ON)
foreach(cuda_transform_opt
        USE_CUDA_WARP_TRANSFORM USE_CUDA_REGISTER_NTT_ACCUM
        USE_CUDA_LOW_LDS_BOOTSTRAP USE_CUDA_PAIRED_GPU_FFT
        USE_CUDA_PAIRED_NTT USE_CUDA_SHOUP_NTT USE_CUDA_MIN_BLOCKS)
  if(${cuda_transform_opt})
    add_compile_definitions(${cuda_transform_opt})
  endif()
endforeach()

if(USE_80BIT_SECURITY)
  add_compile_definitions(USE_80BIT_SECURITY)
elseif(USE_CGGI19)
  add_compile_definitions(USE_CGGI19)
elseif(USE_BLOCK_BINARY)
  add_compile_definitions(USE_BLOCK_BINARY)
elseif(USE_CONCRETE)
  add_compile_definitions(USE_CONCRETE)
endif()

# For OpenMP
find_package(OpenMP)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

OPTION(ENABLE_TEST "Build tests" OFF)
OPTION(ENABLE_TUTORIAL "Build tutorial" OFF)

add_subdirectory(src)
add_subdirectory(thirdparties)

if(ENABLE_TEST)
    add_subdirectory(test)
endif()
