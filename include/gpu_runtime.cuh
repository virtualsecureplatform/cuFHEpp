#pragma once

// Keep the existing CUDA-facing API source-compatible while allowing the
// same headers and kernels to be compiled by HIP for AMD GPUs.
#if defined(CUFHE_USE_HIP)
#include <hip/hip_runtime.h>

using cudaDeviceProp = hipDeviceProp_t;
using cudaError = hipError_t;
using cudaError_t = hipError_t;
using cudaEvent_t = hipEvent_t;
using cudaStream_t = hipStream_t;

#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorNotReady hipErrorNotReady
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaFree hipFree
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
    hipFuncAttributeMaxDynamicSharedMemorySize
#define CUFHE_GPU_FUNCTION(...) \
    reinterpret_cast<const void*>(HIP_KERNEL_NAME(__VA_ARGS__))

// Template argument commas are visible to the preprocessor. Dispatch by the
// resulting argument count so CUDA-style kernel names can still be passed
// directly to cudaFuncSetAttribute.
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_3(function, attribute, value) \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(function), attribute, value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_4(f1, f2, attribute, value) \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2), attribute, value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_5(f1, f2, f3, attribute, value) \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3), attribute, value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_6(f1, f2, f3, f4, attribute, value) \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3, f4), attribute, value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_7(f1, f2, f3, f4, f5, attribute,   \
                                          value)                           \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3, f4, f5), attribute, \
                        value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_8(f1, f2, f3, f4, f5, f6, attribute,   \
                                          value)                               \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3, f4, f5, f6), attribute, \
                        value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_9(f1, f2, f3, f4, f5, f6, f7,   \
                                          attribute, value)             \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3, f4, f5, f6, f7), \
                        attribute, value)
#define CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_10(f1, f2, f3, f4, f5, f6, f7, f8,  \
                                           attribute, value)                \
    hipFuncSetAttribute(CUFHE_GPU_FUNCTION(f1, f2, f3, f4, f5, f6, f7, f8), \
                        attribute, value)
#define CUFHE_DETAIL_SELECT_FUNC_ATTRIBUTE(_1, _2, _3, _4, _5, _6, _7, _8, _9, \
                                           _10, name, ...)                     \
    name
#define cudaFuncSetAttribute(...)                                             \
    CUFHE_DETAIL_SELECT_FUNC_ATTRIBUTE(                                       \
        __VA_ARGS__, CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_10,                      \
        CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_9, CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_8, \
        CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_7, CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_6, \
        CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_5, CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_4, \
        CUFHE_DETAIL_HIP_FUNC_ATTRIBUTE_3)                                    \
    (__VA_ARGS__)
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterDefault hipHostRegisterDefault
#define cudaHostUnregister hipHostUnregister
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyToSymbol(symbol, ...) \
    hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemset hipMemset
#define cudaSetDevice hipSetDevice
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaSuccess hipSuccess

#else
#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#define CUFHE_GPU_FUNCTION(...) __VA_ARGS__
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CUFHE_GPU_DEVICE_COMPILER 1
#endif
