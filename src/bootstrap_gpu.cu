/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <include/utils_gpu.cuh>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
#include <cooperative_groups.h>
#endif

namespace cufhe {

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
namespace {
std::vector<cudaMemPool_t> cooperative_scratch_pools;
void InitializeCooperativeScratchPools(int gpu_count);
void DeleteCooperativeScratchPools();
}
#endif

using namespace std;
using namespace TFHEpp;

vector<NTTValue*> bk_ntts;
vector<CuNTTHandler<>*> ntt_handlers;

// lvl02 storage (lvl0 -> lvl2 bootstrapping)
vector<NTTValueFor<TFHEpp::lvl2param::n, 64>*> bk_ntts_lvl02;
vector<CuNTTHandler<TFHEpp::lvl2param::n, 64>*> ntt_handlers_lvl02;

template <class P>
auto& BootstrappingKeyStorage()
{
    if constexpr (sizeof(typename P::T) == 8) {
        return bk_ntts_lvl02;
    }
    else {
        static_assert(sizeof(typename P::T) == 4,
                      "Unsupported bootstrapping ring size");
        return bk_ntts;
    }
}

#ifdef USE_FFT

// ============================================================================
// FFT mode: BSK conversion to Fourier domain
// ============================================================================

#ifdef USE_GPU_FFT

/**
 * __TRGSW2FFT__ (GPU-FFT variant): Convert BSK polynomial to Fourier domain
 * Uses fold + twist + GPUFFTForward
 * Generic for both uint32_t (lvl1) and uint64_t (lvl2) Torus types
 */
template <class P = TFHEpp::lvl1param>
__global__ void __TRGSW2FFT__(NTTValue* const bk_fft,
                              const typename P::T* const bk,
                              CuNTTHandler<P::n, sizeof(typename P::T) * 8> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh_fft[HALF_N];

    const int in_index =
        blockIdx.z * (BootstrappingTRGSWRows<P> * (P::k + 1) * N) +
        blockIdx.y * N;
    const int out_index =
        blockIdx.z * (BootstrappingTRGSWRows<P> * (P::k + 1) * HALF_N) +
        blockIdx.y * HALF_N;

    const uint32_t tid = threadIdx.x;

    // Load + normalize + fold + twist
    constexpr double norm =
        1.0 /
        static_cast<double>(
            1ULL << (std::numeric_limits<typename P::T>::digits / 2)) /
        static_cast<double>(
            1ULL << (std::numeric_limits<typename P::T>::digits -
                     std::numeric_limits<typename P::T>::digits / 2));
    if (tid < HALF_N) {
        double re =
            static_cast<double>(static_cast<std::make_signed_t<typename P::T>>(
                bk[in_index + tid])) *
            norm;
        double im =
            static_cast<double>(static_cast<std::make_signed_t<typename P::T>>(
                bk[in_index + tid + HALF_N])) *
            norm;
        double2 folded = {re, im};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    if (tid < HALF_N) {
        bk_fft[out_index + tid] = sh_fft[tid];
    }
}

#else  // !USE_GPU_FFT (tfhe-rs FFT)

/**
 * __TRGSW2FFT__ (tfhe-rs variant): Convert BSK polynomial to Fourier domain
 * Uses simple pack + NSMFFT_direct
 */
template <class P = TFHEpp::lvl1param>
__global__ void __TRGSW2FFT__(NTTValue* const bk_fft,
                              const typename P::T* const bk,
                              CuNTTHandler<P::n, sizeof(typename P::T) * 8> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const int in_index =
        blockIdx.z * (BootstrappingTRGSWRows<P> * (P::k + 1) * N) +
        blockIdx.y * N;
    const int out_index =
        blockIdx.z * (BootstrappingTRGSWRows<P> * (P::k + 1) * HALF_N) +
        blockIdx.y * HALF_N;

    const uint32_t tid = threadIdx.x;

    constexpr double norm = 1.0 / 4294967296.0;  // 1 / 2^32
    if (tid < HALF_N) {
        sh_fft[tid] = {
            static_cast<double>(static_cast<int32_t>(bk[in_index + tid])) *
                norm,
            static_cast<double>(
                static_cast<int32_t>(bk[in_index + tid + HALF_N])) *
                norm};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    }
    else {
        for (int s = 0; s < 11; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        bk_fft[out_index + tid] = sh_fft[tid];
    }
}

#endif  // USE_GPU_FFT

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt,
               const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream& st)
{
    cudaSetDevice(st.device_id());
    TFHEpp::lvl1param::T* d_trgsw;
    cudaMalloc((void**)&d_trgsw, sizeof(trgsw));
    cudaMemcpyAsync(d_trgsw, trgsw.data(), sizeof(trgsw),
                    cudaMemcpyHostToDevice, st.st());

    constexpr uint32_t num_threads = lvl1param::n >> 1;  // N/2
    dim3 grid(1, BootstrappingTRGSWRows<lvl1param> * (lvl1param::k + 1), 1);
    dim3 block(num_threads);
    __TRGSW2FFT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    cudaMemcpyAsync(trgswntt.trgswhost.data(),
                    trgswntt.trgswdevices[st.device_id()],
                    cuFHETRGSWNTTlvl1::kNumElements * sizeof(NTTValue),
                    cudaMemcpyDeviceToHost, st.st());
    cudaFree(d_trgsw);
}

void InitializeNTThandlers(const int gpuNum)
{
#ifdef USE_GPU_FFT
    // GPU-FFT mode: generate tables and allocate per-GPU device memory
    CuNTTHandler<>::Create();
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        ntt_handlers.push_back(new CuNTTHandler<>());
        ntt_handlers[i]->SetDevicePointers(i);
        cudaDeviceSynchronize();
        CuCheckError();
    }
#else
    // tfhe-rs FFT mode: twiddles are in __device__ memory, no per-GPU
    // initialization needed
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        ntt_handlers.push_back(new CuNTTHandler<>());
        cudaDeviceSynchronize();
        CuCheckError();
    }
#endif
}

template <class P>
void BootstrappingKeyToNTT(const BootstrappingKey<P>& bk, const int gpuNum)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;

    auto& bk_storage = BootstrappingKeyStorage<typename P::targetP>();

    bk_storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        // FFT BSK: n_lwe * (k+1)*l * (k+1) * (N/2) double2 elements
        size_t fft_elems = static_cast<size_t>(P::domainP::n) *
                           BootstrappingTRGSWRows<typename P::targetP> *
                           (P::targetP::k + 1) * HALF_N;
        cudaMalloc((void**)&bk_storage[i], sizeof(NTTValue) * fft_elems);

        typename P::targetP::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        // Grid: 1 x ((k+1)*l*(k+1)) x n_lwe
        dim3 grid(
            1,
            BootstrappingTRGSWRows<typename P::targetP> * (P::targetP::k + 1),
            P::domainP::n);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        if constexpr (sizeof(typename P::targetP::T) == 8) {
            __TRGSW2FFT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers_lvl02[i]);
        }
        else {
            __TRGSW2FFT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers[i]);
        }
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                           \
    template void BootstrappingKeyToNTT<P>(const BootstrappingKey<P>& bk, \
                                           const int gpuNum)
INST(TFHEpp::lvl01param);
INST(TFHEpp::lvl02param);
INST(TFHEpp::lvlh2param);
#ifdef USE_DIFFERENT_BR_PARAM
INST(TFHEpp::cblvl02param);
INST(TFHEpp::cblvlh2param);
#endif
#undef INST

#ifdef USE_KEY_BUNDLE
template <class P>
void BootstrappingKeyBundleToNTT(const BootstrappingKey<P>& bk,
                                 const int gpuNum)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t num_pairs = P::domainP::k * P::domainP::n / P::Addends;
    constexpr uint32_t bk_elements_per_pair = (1 << P::Addends) - 1;  // 3
    constexpr uint32_t trgsw_polys =
        BootstrappingTRGSWRows<typename P::targetP> * (P::targetP::k + 1);
    constexpr size_t total_fft_elems = static_cast<size_t>(num_pairs) *
                                       bk_elements_per_pair * trgsw_polys *
                                       HALF_N;

    auto& bk_storage = BootstrappingKeyStorage<typename P::targetP>();

    bk_storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_storage[i], sizeof(NTTValue) * total_fft_elems);

        typename P::targetP::T* d_bk;
        size_t bk_byte_size = sizeof(bk);
        cudaMalloc((void**)&d_bk, bk_byte_size);
        cudaMemcpy(d_bk, bk.data(), bk_byte_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        // Grid: 1 x trgsw_polys x (num_pairs * bk_elements_per_pair)
        dim3 grid(1, trgsw_polys, num_pairs * bk_elements_per_pair);
        dim3 block(HALF_N);
        if constexpr (sizeof(typename P::targetP::T) == 8) {
            __TRGSW2FFT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers_lvl02[i]);
        }
        else {
            __TRGSW2FFT__<<<grid, block>>>(bk_storage[i], d_bk,
                                           *ntt_handlers[i]);
        }
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                   \
    template void BootstrappingKeyBundleToNTT<P>( \
        const BootstrappingKey<P>& bk, const int gpuNum)
INST(TFHEpp::lvl01param);
INST(TFHEpp::lvl02param);
INST(TFHEpp::lvlh2param);
#ifdef USE_DIFFERENT_BR_PARAM
INST(TFHEpp::cblvl02param);
INST(TFHEpp::cblvlh2param);
#endif
#undef INST
#endif  // USE_KEY_BUNDLE

void DeleteBootstrappingKeyNTT(const int gpuNum)
{
    for (size_t i = 0; i < bk_ntts.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts[i]);
        delete ntt_handlers[i];
    }
    ntt_handlers.clear();
}

void InitializeNTThandlers_lvl02(const int gpuNum)
{
    CuNTTHandler<TFHEpp::lvl2param::n, 64>::Create();
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        ntt_handlers_lvl02.push_back(new CuNTTHandler<TFHEpp::lvl2param::n, 64>());
        ntt_handlers_lvl02[i]->SetDevicePointers(i);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

void DeleteBootstrappingKeyNTT_lvl02(const int gpuNum)
{
    for (size_t i = 0; i < bk_ntts_lvl02.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts_lvl02[i]);
    }
    bk_ntts_lvl02.clear();
    for (size_t i = 0; i < ntt_handlers_lvl02.size(); i++) {
        delete ntt_handlers_lvl02[i];
    }
    ntt_handlers_lvl02.clear();
    CuNTTHandler<TFHEpp::lvl2param::n, 64>::Destroy();
}

void BootstrappingKeyFlatToNTT_lvl02(
    const TFHEpp::TRGSW<TFHEpp::lvl2param>* flat_bk,
    const uint32_t num_elements, const int gpuNum)
{
    using tgtP = TFHEpp::lvl2param;
    constexpr uint32_t N = tgtP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t trgsw_polys =
        BootstrappingTRGSWRows<tgtP> * (tgtP::k + 1);

    size_t fft_elems = static_cast<size_t>(num_elements) * trgsw_polys * HALF_N;

    bk_ntts_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts_lvl02[i], sizeof(NTTValue) * fft_elems);

        // Copy flat TRGSW array to device
        size_t bk_bytes =
            static_cast<size_t>(num_elements) * sizeof(TFHEpp::TRGSW<tgtP>);
        typename tgtP::T* d_bk;
        cudaMalloc((void**)&d_bk, bk_bytes);
        cudaMemcpy(d_bk, flat_bk, bk_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(1, trgsw_polys, num_elements);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        __TRGSW2FFT__<tgtP>
            <<<grid, block>>>(bk_ntts_lvl02[i], d_bk, *ntt_handlers_lvl02[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}

#else  // !USE_FFT

// ============================================================================
// NTT mode: BSK conversion to NTT domain (existing implementation)
// ============================================================================

template <class P = TFHEpp::lvl1param>
__global__ void __TRGSW2NTT__(NTTValueFor<P::n, sizeof(typename P::T) * 8>* const bk_ntt,
                              const typename P::T* const bk,
                              CuNTTHandler<P::n, sizeof(typename P::T) * 8> ntt)
{
    __shared__ NTTValueFor<P::n, sizeof(typename P::T) * 8> sh_temp[P::n];
    const int index =
        blockIdx.z * (BootstrappingTRGSWRows<P> * (P::k + 1) * P::n) +
        blockIdx.y * (P::k + 1) * P::n + blockIdx.x * P::n;
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    // Load and convert each element: Torus -> NTT mod
    for (int i = tid; i < P::n; i += bdim) {
        sh_temp[i] = torus_to_ntt_mod<P::n, sizeof(typename P::T) * 8>(bk[index + i]);
    }
    __syncthreads();

    // Forward NTT
    SmallForwardNTT<P::nbit, sizeof(typename P::T) * 8>(sh_temp, ntt.forward_root_, tid);

    // Store result
    for (int i = tid; i < P::n; i += bdim) {
        bk_ntt[index + i] = sh_temp[i];
    }
}

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt,
               const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream& st)
{
    cudaSetDevice(st.device_id());
    TFHEpp::lvl1param::T* d_trgsw;
    cudaMalloc((void**)&d_trgsw, sizeof(trgsw));
    cudaMemcpyAsync(d_trgsw, trgsw.data(), sizeof(trgsw),
                    cudaMemcpyHostToDevice, st.st());

    constexpr uint32_t num_threads = lvl1param::n >> 1;  // N/2
    dim3 grid(lvl1param::k + 1, BootstrappingTRGSWRows<lvl1param>, 1);
    dim3 block(num_threads);
    __TRGSW2NTT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    // Copy NTT results back to host (NTTValue sized)
    cudaMemcpyAsync(trgswntt.trgswhost.data(),
                    trgswntt.trgswdevices[st.device_id()],
                    cuFHETRGSWNTTlvl1::kNumElements * sizeof(NTTValue),
                    cudaMemcpyDeviceToHost, st.st());
    cudaFree(d_trgsw);
}

void InitializeNTThandlers(const int gpuNum)
{
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        ntt_handlers.push_back(new CuNTTHandler<>());
        ntt_handlers[i]->Create();
        ntt_handlers[i]->CreateConstant();
        ntt_handlers[i]->SetDevicePointers(
            i);  // Set device pointers for this GPU
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

template <class P>
void BootstrappingKeyToNTT(const BootstrappingKey<P>& bk, const int gpuNum)
{
    constexpr uint32_t N = P::targetP::n;

    auto& bk_storage = BootstrappingKeyStorage<typename P::targetP>();

    bk_storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_storage[i],
                   sizeof(NTTValueFor<N, sizeof(typename P::targetP::T) * 8>) * P::domainP::n *
                       BootstrappingTRGSWRows<typename P::targetP> *
                       (P::targetP::k + 1) * N);

        typename P::targetP::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(P::targetP::k + 1,
                  BootstrappingTRGSWRows<typename P::targetP>, P::domainP::n);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        if constexpr (sizeof(typename P::targetP::T) == 8) {
            __TRGSW2NTT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers_lvl02[i]);
        }
        else {
            __TRGSW2NTT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers[i]);
        }
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                           \
    template void BootstrappingKeyToNTT<P>(const BootstrappingKey<P>& bk, \
                                           const int gpuNum)
INST(TFHEpp::lvl01param);
INST(TFHEpp::lvl02param);
INST(TFHEpp::lvlh2param);
#ifdef USE_DIFFERENT_BR_PARAM
INST(TFHEpp::cblvl02param);
INST(TFHEpp::cblvlh2param);
#endif
#undef INST

#ifdef USE_KEY_BUNDLE
template <class P>
void BootstrappingKeyBundleToNTT(const BootstrappingKey<P>& bk,
                                 const int gpuNum)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t num_pairs = P::domainP::k * P::domainP::n / P::Addends;
    constexpr uint32_t bk_elements_per_pair =
        (1 << P::Addends) - 1;  // 3 for Addends=2
    constexpr uint32_t trgsw_polys =
        BootstrappingTRGSWRows<typename P::targetP> * (P::targetP::k + 1);
    constexpr size_t total_ntt_elems =
        static_cast<size_t>(num_pairs) * bk_elements_per_pair * trgsw_polys * N;

    auto& bk_storage = BootstrappingKeyStorage<typename P::targetP>();

    bk_storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_storage[i],
                   sizeof(NTTValueFor<N, sizeof(typename P::targetP::T) * 8>) * total_ntt_elems);

        typename P::targetP::T* d_bk;
        size_t bk_byte_size = sizeof(bk);
        cudaMalloc((void**)&d_bk, bk_byte_size);
        cudaMemcpy(d_bk, bk.data(), bk_byte_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(P::targetP::k + 1,
                  BootstrappingTRGSWRows<typename P::targetP>,
                  num_pairs * bk_elements_per_pair);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        if constexpr (sizeof(typename P::targetP::T) == 8) {
            __TRGSW2NTT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers_lvl02[i]);
        }
        else {
            __TRGSW2NTT__<typename P::targetP>
                <<<grid, block>>>(bk_storage[i], d_bk, *ntt_handlers[i]);
        }
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                   \
    template void BootstrappingKeyBundleToNTT<P>( \
        const BootstrappingKey<P>& bk, const int gpuNum)
INST(TFHEpp::lvl01param);
INST(TFHEpp::lvl02param);
INST(TFHEpp::lvlh2param);
#ifdef USE_DIFFERENT_BR_PARAM
INST(TFHEpp::cblvl02param);
INST(TFHEpp::cblvlh2param);
#endif
#undef INST
#endif  // USE_KEY_BUNDLE

void DeleteBootstrappingKeyNTT(const int gpuNum)
{
    for (size_t i = 0; i < bk_ntts.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts[i]);

        ntt_handlers[i]->Destroy();
        delete ntt_handlers[i];
    }
    ntt_handlers.clear();
}

void InitializeNTThandlers_lvl02(const int gpuNum)
{
    CuNTTHandler<TFHEpp::lvl2param::n, 64>::Create();
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
    InitializeCooperativeScratchPools(gpuNum);
#endif
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        ntt_handlers_lvl02.push_back(new CuNTTHandler<TFHEpp::lvl2param::n, 64>());
        ntt_handlers_lvl02[i]->SetDevicePointers(i);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

void DeleteBootstrappingKeyNTT_lvl02(const int gpuNum)
{
    (void)gpuNum;
    for (size_t i = 0; i < bk_ntts_lvl02.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts_lvl02[i]);
    }
    bk_ntts_lvl02.clear();

    for (size_t i = 0; i < ntt_handlers_lvl02.size(); i++) {
        delete ntt_handlers_lvl02[i];
    }
    ntt_handlers_lvl02.clear();
    CuNTTHandler<TFHEpp::lvl2param::n, 64>::Destroy();
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
    DeleteCooperativeScratchPools();
#endif
}

void BootstrappingKeyFlatToNTT_lvl02(
    const TFHEpp::TRGSW<TFHEpp::lvl2param>* flat_bk,
    const uint32_t num_elements, const int gpuNum)
{
    using tgtP = TFHEpp::lvl2param;
    constexpr uint32_t N = tgtP::n;
    constexpr uint32_t trgsw_polys =
        BootstrappingTRGSWRows<tgtP> * (tgtP::k + 1);

    size_t ntt_elems = static_cast<size_t>(num_elements) * trgsw_polys * N;

    bk_ntts_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts_lvl02[i],
                   sizeof(NTTValueFor<N, 64>) * ntt_elems);

        size_t bk_bytes =
            static_cast<size_t>(num_elements) * sizeof(TFHEpp::TRGSW<tgtP>);
        typename tgtP::T* d_bk;
        cudaMalloc((void**)&d_bk, bk_bytes);
        cudaMemcpy(d_bk, flat_bk, bk_bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(tgtP::k + 1, BootstrappingTRGSWRows<tgtP>, num_elements);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        __TRGSW2NTT__<tgtP>
            <<<grid, block>>>(bk_ntts_lvl02[i], d_bk, *ntt_handlers_lvl02[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}

#endif  // USE_FFT

// ============================================================================
// CMUX operation
// ============================================================================

#ifdef USE_FFT

#ifdef USE_GPU_FFT

__global__
__launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const NTTValue* const tgsw_fft,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256

    extern __shared__ NTTValue sh_cmux[];
    double2* const sh_fft = &sh_cmux[0];
    double2* const sh_accum = &sh_cmux[HALF_N];

    for (int i = tid; i < (lvl1param::k + 1) * HALF_N; i += NUM_THREADS) {
        sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    constexpr typename lvl1param::T roundoffset =
        1ULL << (std::numeric_limits<typename lvl1param::T>::digits -
                 lvl1param::l * lvl1param::Bgbit - 1);

    for (int j = 0; j <= lvl1param::k; j++) {
        for (int digit = 0; digit < lvl1param::l; digit++) {
            if (tid < HALF_N) {
                typename lvl1param::T temp_re = trlwe1[j * N + tid] -
                                                trlwe0[j * N + tid] +
                                                decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename lvl1param::T>::digits -
                       (digit + 1) * lvl1param::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                typename lvl1param::T temp_im = trlwe1[j * N + tid + HALF_N] -
                                                trlwe0[j * N + tid + HALF_N] +
                                                decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename lvl1param::T>::digits -
                       (digit + 1) * lvl1param::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                // Fold + twist
                double2 folded = {static_cast<double>(digit_re),
                                  static_cast<double>(digit_im)};
                double2 tw = __ldg(&ntt.twist_[tid]);
                sh_fft[tid] = folded * tw;
            }
            __syncthreads();

            if (tid < FFT_THREADS) {
                GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                    __syncthreads();
            }

            int digit_linear = j * lvl1param::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
#pragma unroll
                for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                    double2 bk_val = __ldg(
                        &tgsw_fft[((lvl1param::k + 1) * digit_linear + out_k) *
                                      HALF_N +
                                  tid]);
                    sh_accum[out_k * HALF_N + tid] += fft_val * bk_val;
                }
            }
            __syncthreads();
        }
    }

    constexpr double denorm = 4294967296.0;  // 2^32
    for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];
        if (tid < FFT_THREADS) {
            GPUFFTInverse<N>(sh_inv, ntt.inverse_root_, tid);
        }
        else {
            for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            out[k_idx * N + tid] = trlwe0[k_idx * N + tid] +
                                   static_cast<uint32_t>(static_cast<int64_t>(
                                       llrint(val.x * denorm)));
            out[k_idx * N + tid + HALF_N] =
                trlwe0[k_idx * N + tid + HALF_N] +
                static_cast<uint32_t>(
                    static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}

#else  // !USE_GPU_FFT (tfhe-rs FFT CMUX)

__global__
__launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const NTTValue* const tgsw_fft,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    extern __shared__ NTTValue sh_cmux[];
    double2* const sh_fft = &sh_cmux[0];
    double2* const sh_accum = &sh_cmux[HALF_N];

    for (int i = tid; i < (lvl1param::k + 1) * HALF_N; i += NUM_THREADS) {
        sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    constexpr typename lvl1param::T roundoffset =
        1ULL << (std::numeric_limits<typename lvl1param::T>::digits -
                 lvl1param::l * lvl1param::Bgbit - 1);

    for (int j = 0; j <= lvl1param::k; j++) {
        for (int digit = 0; digit < lvl1param::l; digit++) {
            if (tid < HALF_N) {
                typename lvl1param::T temp_re = trlwe1[j * N + tid] -
                                                trlwe0[j * N + tid] +
                                                decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename lvl1param::T>::digits -
                       (digit + 1) * lvl1param::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                typename lvl1param::T temp_im = trlwe1[j * N + tid + HALF_N] -
                                                trlwe0[j * N + tid + HALF_N] +
                                                decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename lvl1param::T>::digits -
                       (digit + 1) * lvl1param::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                sh_fft[tid] = {static_cast<double>(digit_re),
                               static_cast<double>(digit_im)};
            }
            __syncthreads();

            if (tid < FFT_THREADS) {
                NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
            }
            else {
                for (int s = 0; s < 11; s++) __syncthreads();
            }

            int digit_linear = j * lvl1param::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
#pragma unroll
                for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                    double2 bk_val = __ldg(
                        &tgsw_fft[((lvl1param::k + 1) * digit_linear + out_k) *
                                      HALF_N +
                                  tid]);
                    sh_accum[out_k * HALF_N + tid] += fft_val * bk_val;
                }
            }
            __syncthreads();
        }
    }

    constexpr double denorm = 4294967296.0;  // 2^32
    for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];
        if (tid < FFT_THREADS) {
            NSMFFT_inverse<HalfDegree<Degree<N>>>(sh_inv);
        }
        else {
            for (int s = 0; s < 11; s++) __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            out[k_idx * N + tid] = trlwe0[k_idx * N + tid] +
                                   static_cast<uint32_t>(static_cast<int64_t>(
                                       llrint(val.x * denorm)));
            out[k_idx * N + tid + HALF_N] =
                trlwe0[k_idx * N + tid + HALF_N] +
                static_cast<uint32_t>(
                    static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT

__global__
__launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const NTTValue* const tgsw_ntt,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = lvl1param::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512

    extern __shared__ NTTValue sh_cmux[];
    NTTValue* const sh_work =
        &sh_cmux[0];  // Working buffer for NTT (N elements)
    NTTValue* const sh_accum =
        &sh_cmux[N];  // Accumulated results ((k+1)*N elements)

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
        for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
            sh_accum[k_idx * N + tid] = 0;
            sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
        }
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    constexpr typename lvl1param::T roundoffset =
        1ULL << (std::numeric_limits<typename lvl1param::T>::digits -
                 lvl1param::l * lvl1param::Bgbit - 1);

    // Process each TRLWE component and decomposition level sequentially
    for (int j = 0; j <= lvl1param::k; j++) {
        for (int digit = 0; digit < lvl1param::l; digit++) {
            // Step 1: Decompose (trlwe1 - trlwe0) for component j, digit
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    typename lvl1param::T temp = trlwe1[j * N + i] -
                                                 trlwe0[j * N + i] +
                                                 decomp_offset + roundoffset;
                    int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename lvl1param::T>::digits -
                           (digit + 1) * lvl1param::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = signed_int_to_ntt_mod<N>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                SmallForwardNTT<lvl1param::nbit>(sh_work, ntt.forward_root_,
                                                 tid);
            }
            else {
                for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                    __syncthreads();
            }

            // Step 3: Multiply with TRGSW_NTT and accumulate
            int digit_linear = j * lvl1param::l + digit;
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    NTTValue ntt_val = sh_work[i];
#pragma unroll
                    for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                        NTTValue bk_val = __ldg(
                            &tgsw_ntt[(((lvl1param::k + 1) * digit_linear +
                                        out_k)
                                       << lvl1param::nbit) +
                                      i]);
                        sh_accum[out_k * N + i] = small_mod_madd<N>(
                            ntt_val, bk_val, sh_accum[out_k * N + i]);
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT on accumulated results, modswitch, add to trlwe0,
    // write out Operate directly on sh_accum to avoid copying to sh_work
    for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
        NTTValue* const sh_ntt_buf = &sh_accum[k_idx * N];

        // Inverse NTT directly on accumulator buffer
        if (tid < NUM_THREADS) {
            SmallInverseNTT<lvl1param::nbit>(sh_ntt_buf, ntt.inverse_root_,
                                             ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        // Convert with modulus switching and add to trlwe0
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                out[k_idx * N + i] = trlwe0[k_idx * N + i] +
                                     ntt_mod_to_torus32<N>(sh_ntt_buf[i]);
            }
        }
        __syncthreads();
    }
}

#endif  // USE_FFT

template <class brP, class iksP>
__global__
__launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>, MIN_BLOCKS4HOMGATE<typename brP::targetP>) void __Bootstrap__(
    typename iksP::targetP::T* const out,
    const typename brP::domainP::T* const in, const typename brP::targetP::T mu,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
    const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    __shared__
        typename brP::targetP::T tlwe[(brP::targetP::k + 1) * brP::targetP::n];

#if defined(USE_CUDA_PAIRED_NTT_WARP) && !defined(USE_FFT) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr size_t prepared_bytes =
        ((iksP::domainP::k * iksP::domainP::n -
          (use_subset_key_switch<iksP> ? iksP::targetP::k * iksP::targetP::n : 0)) *
             iksP::t + 1) * sizeof(uint32_t);
    if constexpr (std::is_same_v<iksP, TFHEpp::lvl10param> &&
                  prepared_bytes <= MEM4HOMGATE<typename brP::targetP>) {
        extern __shared__ char generic_scratch[];
        __BlindRotate__<brP, false, false, 0, true, /*KeySwitchAfter31=*/true>(
            tlwe, in, mu, bk, ntt);
        KeySwitchPrepared32<iksP, NUM_THREAD4HOMGATE<typename brP::targetP>, true>(
            out, tlwe, ksk, reinterpret_cast<uint32_t*>(generic_scratch));
    } else
#endif
    {
        __BlindRotate__<brP>(tlwe, in, mu, bk, ntt);
        KeySwitch<iksP>(out, tlwe, ksk);
    }
}

// template <class iksP, class bkP>
// __global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __Bootstrap__(
//     typename iksP::domainP::T* const out, const typename iksP::domainP::T*
//     const in, const typename bkP::targetP::T mu, const NTTValue* const bk,
//     const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
// {
//     __shared__ typename bkP::targetP::T
//     tlwe[iksP::targetP::k*iksP::targetP::n+1];

//     KeySwitch<iksP>(tlwe, in, ksk);
//     __threadfence();
//     __BlindRotate__<bkP>(out,tlwe,mu,bk,ntt);
//     __threadfence();
// }

template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename P:: targetP>, MIN_BLOCKS4HOMGATE<typename P:: targetP>) void __BlindRotateGlobal__(TFHEpp::lvl1param::T* const
                                                     out,
                                                 const TFHEpp::lvl0param::
                                                     T* const in,
                                                 const TFHEpp::lvl1param::T mu,
                                                 const NTTValue* const bk,
#ifdef USE_KEY_BUNDLE
                                                 const NTTValue* const
                                                     one_trgsw_ntt,
                                                 const NTTValue* const xai_ntt,
#endif
                                                 const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8>
                                                     ntt)
{
#ifdef USE_KEY_BUNDLE
    __BlindRotateKeyBundle__<P>(out, in, mu, bk, one_trgsw_ntt, xai_ntt, ntt);
#else
    __BlindRotate__<P>(out, in, mu, bk, ntt);
#endif
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp:: lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp:: lvl1param>) void __SEIandBootstrap2TRLWE__(TFHEpp::lvl1param::
                                                           T* const out,
                                                       const TFHEpp::lvl1param::
                                                           T* const in,
                                                       const TFHEpp::lvl1param::
                                                           T mu,
                                                       const NTTValue* const bk,
                                                       const TFHEpp::lvl0param::
                                                           T* const ksk,
#ifdef USE_KEY_BUNDLE
                                                       const NTTValue* const
                                                           one_trgsw_ntt,
                                                       const NTTValue* const
                                                           xai_ntt,
#endif
                                                       const CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
#ifdef USE_FFT
    // Shared memory layout for FFT mode:
    // Accumulate uses (N/2 + (k+1)*N/2) double2 = (k+2)*N/2 double2
    // After that: tlwe needs (k+1)*N uint32_t, tlwelvl0 needs lvl0 TLWE
    // uint32_t Convert byte offset: (k+2)*N/2 double2 = (k+2)*N/2 * 16 bytes In
    // NTTValue (double2) units: (k+2)*N/2
    constexpr size_t acc_ntt_elems = (lvl1param::k + 2) * (lvl1param::n / 2);
#ifndef USE_KEY_BUNDLE
    NTTValue* sh_acc_ntt = &sh[0];
#endif
    TFHEpp::lvl1param::T* tlwe = (TFHEpp::lvl1param::T*)&sh[acc_ntt_elems];
    // tlwe occupies (k+1)*N uint32_t = (k+1)*N*4 bytes = (k+1)*N/4 double2
    // elements
    constexpr size_t tlwe_in_nttval =
        ((lvl1param::k + 1) * lvl1param::n * sizeof(uint32_t) +
         sizeof(NTTValue) - 1) /
        sizeof(NTTValue);
    lvl0param::T* tlwelvl0 = (lvl0param::T*)&sh[acc_ntt_elems + tlwe_in_nttval];
#else
    // Shared memory layout for NTT mode (all uint32_t / NTTValue):
    // [0, (k+2)*N): sh_acc_ntt for Accumulate (working buf + accum)
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE with (k+1)*N elements)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE)
#ifndef USE_KEY_BUNDLE
    NTTValue* sh_acc_ntt = &sh[0];
#endif
    TFHEpp::lvl1param::T* tlwe =
        (TFHEpp::lvl1param::T*)&sh[(lvl1param::k + 2) * lvl1param::n];
    lvl0param::T* tlwelvl0 =
        (lvl0param::T*)&sh[(lvl1param::k + 2 + lvl1param::k + 1) *
                           lvl1param::n];
#endif

#if defined(USE_CUDA_PAIRED_NTT_WARP) && !defined(USE_FFT) && !defined(USE_KEY_BUNDLE) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr size_t prepared_bytes =
        ((lvl10param::domainP::k * lvl10param::domainP::n -
          (use_subset_key_switch<lvl10param> ? lvl10param::targetP::k * lvl10param::targetP::n : 0)) *
             lvl10param::t + 1) * sizeof(uint32_t);
    if constexpr (prepared_bytes <= (lvl1param::k + 2) * lvl1param::n * sizeof(NTTValue)) {
        KeySwitchPrepared32<lvl10param, NUM_THREAD4HOMGATE<lvl1param>, true>(
            tlwelvl0, in, ksk, reinterpret_cast<uint32_t*>(sh));
    } else
#endif
    {
        KeySwitch<lvl10param>(tlwelvl0, in, ksk);
    }
    __syncthreads();

#ifdef USE_KEY_BUNDLE
    __BlindRotateKeyBundle__<lvl01param>(tlwe, tlwelvl0, mu, bk, one_trgsw_ntt,
                                         xai_ntt, ntt);
#else
    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    uint32_t bar =
        2 * TFHEpp::lvl1param::n -
        (tlwelvl0[TFHEpp::lvl0param::k * TFHEpp::lvl0param::n] >>
         (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 1 -
          TFHEpp::lvl1param::nbit));
    RotatedTestVector<lvl1param>(tlwe, bar, mu);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {
        constexpr typename TFHEpp::lvl0param::T roundoffset =
            1ULL
            << (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 2 -
                TFHEpp::lvl1param::nbit);
        bar = modSwitchFromTorus<lvl01param>(tlwelvl0[i] + roundoffset);
#ifdef USE_FFT
        constexpr size_t trgsw_fft_size = (lvl1param::k + 1) * lvl1param::l *
                                          (lvl1param::k + 1) *
                                          (lvl1param::n / 2);
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar, bk + i * trgsw_fft_size,
                               ntt);
#else
#if defined(USE_CUDA_PAIRED_NTT_WARP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        AccumulateLegacyPairedNTT<lvl01param>(tlwe, sh_acc_ntt, bar,
#else
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar,
#endif
                               bk + (i << lvl1param::nbit) *
                                        (lvl1param::k + 1) *
                                        (lvl1param::k + 1) * lvl1param::l,
                               ntt);
#endif
    }
#endif
    __syncthreads();
    for (int i = 0; i < (lvl1param::k + 1) * lvl1param::n; i++) {
        out[i] = tlwe[i];
    }
}

template <class P, uint index>
__device__ inline typename P::T __SampleExtractValue__(
    const typename P::T* const in, const uint i)
{
    constexpr uint nmask = P::n - 1;
    if (i == P::k * P::n) return in[P::k * P::n + index];

    const uint k = i >> P::nbit;
    const uint n = i & nmask;
    return n <= index ? in[k * P::n + index - n]
                      : -in[k * P::n + P::n + index - n];
}

template <class P, uint index>
__device__ inline void __SampleExtractIndex__(typename P::T* const res,
                                              const typename P::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (uint i = tid; i <= P::k * P::n; i += bdim)
        res[i] = __SampleExtractValue__<P, index>(in, i);
}

// Combine a sample extraction directly with a previously extracted TLWE.
// This lets the low-LDS Mux path reuse one TRLWE buffer for both blind
// rotations instead of keeping two 32 KiB lvl02 ciphertexts in shared memory.
template <class P, uint index, bool negate>
__device__ inline void __SampleExtractCombine__(typename P::T* const res,
                                                const typename P::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (uint i = tid; i <= P::k * P::n; i += bdim) {
        const typename P::T value = __SampleExtractValue__<P, index>(in, i);

        if constexpr (negate)
            res[i] = -res[i] - value;
        else
            res[i] += value;
    }
}

template <class P, uint index>
__device__ inline void __SampleExtractToRegisters__(typename P::T* const res,
                                                    const typename P::T* in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    uint slot = 0;
    for (uint i = tid; i <= P::k * P::n; i += bdim)
        res[slot++] = __SampleExtractValue__<P, index>(in, i);
}

template <class P, uint index, bool negate>
__device__ inline void __SampleExtractRegistersCombine__(
    typename P::T* const res, const typename P::T* const saved,
    const typename P::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    uint slot = 0;
    for (uint i = tid; i <= P::k * P::n; i += bdim) {
        const typename P::T value = __SampleExtractValue__<P, index>(in, i);
        if constexpr (negate)
            res[i] = -saved[slot++] - value;
        else
            res[i] = saved[slot++] + value;
    }
}

// Reuse the blind-rotation workspace for prepared signed row addresses after
// capturing its input, including the already-combined TLWE in MUX/NMux.
template <class iksP, class brP, bool ExtractFromTRLWE>
__device__ inline void __KeySwitchAfterBlindRotate__(
    typename iksP::targetP::T* out, const typename iksP::domainP::T* input,
    const typename iksP::targetP::T* ksk, char* dyn_sh)
{
#if defined(USE_CUDA_PAIRED_NTT_WARP) && !defined(USE_FFT) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr size_t prepared_bytes =
        (iksP::domainP::k * iksP::domainP::n * iksP::t + 1) * sizeof(uint16_t);
    if constexpr (std::is_same_v<iksP, TFHEpp::lvl20param> &&
                  prepared_bytes <= MEM4HOMGATE_DYN<typename brP::targetP>) {
        KeySwitchPrepared16<iksP, NUM_THREAD4HOMGATE<typename brP::targetP>,
                            ExtractFromTRLWE>(
            out, input, ksk, reinterpret_cast<uint16_t*>(dyn_sh));
    } else if constexpr (std::is_same_v<iksP, TFHEpp::lvl10param> &&
        ((iksP::domainP::k * iksP::domainP::n -
          (use_subset_key_switch<iksP> ? iksP::targetP::k * iksP::targetP::n : 0)) *
             iksP::t + 1) * sizeof(uint32_t) <= MEM4HOMGATE_DYN<typename brP::targetP>) {
        KeySwitchPrepared32<iksP, NUM_THREAD4HOMGATE<typename brP::targetP>, ExtractFromTRLWE>(
            out, input, ksk, reinterpret_cast<uint32_t*>(dyn_sh));
    } else
#endif
    {
        const typename iksP::domainP::T* tlwe = input;
        if constexpr (ExtractFromTRLWE) {
            auto* extracted = reinterpret_cast<typename iksP::domainP::T*>(dyn_sh);
            __SampleExtractIndex__<typename iksP::domainP, 0>(extracted, input);
            __syncthreads();
            tlwe = extracted;
        }
        KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    }
}

// The MUX workspace prefix is dead before each blind rotation. Keep the
// prepared key rows inside that prefix so neither TRLWE buffer is overwritten.
template <class iksP, class brP, int Sign0, int Sign1, auto Offset>
__device__ __forceinline__ void __KeySwitchMuxPreAdd__(
    typename iksP::targetP::T* out, const typename iksP::domainP::T* in0,
    const typename iksP::domainP::T* in1, const typename iksP::targetP::T* ksk,
    char* scratch)
{
#if defined(USE_CUDA_PAIRED_NTT_WARP) && !defined(USE_FFT) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(CUFHE_USE_HIP)
    constexpr size_t count = (iksP::domainP::k * iksP::domainP::n -
        (use_subset_key_switch<iksP> ? iksP::targetP::k * iksP::targetP::n : 0)) * iksP::t;
    if constexpr (std::is_same_v<iksP, TFHEpp::lvl10param> &&
                  count * sizeof(uint32_t) <= MEM4HOMGATE<typename brP::targetP>) {
        IdentityKeySwitchPreAddPrepared<iksP, Sign0, Sign1, Offset>(
            out, in0, in1, ksk, reinterpret_cast<uint32_t*>(scratch));
    } else if constexpr (std::is_same_v<iksP, TFHEpp::lvl20param> &&
                         count * sizeof(uint16_t) <= MEM4HOMGATE<typename brP::targetP>) {
        IdentityKeySwitchPreAddPrepared16<iksP, Sign0, Sign1, Offset>(
            out, in0, in1, ksk, reinterpret_cast<uint16_t*>(scratch));
    } else
#endif
    {
        IdentityKeySwitchPreAdd<iksP, Sign0, Sign1, Offset>(out, in0, in1, ksk);
    }
}

// Exact paired transforms for eligible raw binary gates; this is a hardware
// schedule choice, with the cryptographic parameter types unchanged.
template <class brP>
constexpr bool USE_RAW_PAIRED_NTT64 =
#ifndef USE_FFT
    USE_PAIRED_NTT64<typename brP::targetP> && brP::domainP::ell == 2;
#else
    false;
#endif

template <class brP>
constexpr size_t MEM4RAWBINARYGATE_DYN =
#ifdef USE_FFT
    MEM4HOMGATE_DYN<typename brP::targetP>;
#else
    MEM4HOMGATE_DYN<typename brP::targetP, false, USE_RAW_PAIRED_NTT64<brP>>;
#endif

template <class iksP, class brP, std::make_signed_t<typename brP::targetP::T> μ,
          int casign, int cbsign, auto offset, bool StageMonomials64 = false, bool PairUnstaged64 = false>
__device__ inline void __HomGate__(typename brP::targetP::T* const out,
                                   const typename iksP::domainP::T* const in0,
                                   const typename iksP::domainP::T* const in1,
                                   const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    __shared__
        typename iksP::targetP::T tlwe[iksP::targetP::k * iksP::targetP::n + 1];

    extern __shared__ char dyn_sh[];
    constexpr int prepared_count = (iksP::domainP::k * iksP::domainP::n -
        (use_subset_key_switch<iksP> ? iksP::targetP::k * iksP::targetP::n : 0)) * iksP::t;
#if defined(USE_CUDA_PAIRED_NTT_WARP) && !defined(USE_FFT) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (std::is_same_v<iksP, TFHEpp::lvl10param> &&
                  prepared_count * sizeof(uint32_t) <= MEM4HOMGATE<typename brP::targetP>) {
        IdentityKeySwitchPreAddPrepared<iksP, casign, cbsign, offset>(
            tlwe, in0, in1, ksk, reinterpret_cast<uint32_t*>(dyn_sh));
    } else if constexpr (std::is_same_v<iksP, TFHEpp::lvl20param> &&
                         prepared_count * sizeof(uint16_t) <=
                             MEM4HOMGATE_DYN<typename brP::targetP>) {
        IdentityKeySwitchPreAddPrepared16<iksP, casign, cbsign, offset>(
            tlwe, in0, in1, ksk, reinterpret_cast<uint16_t*>(dyn_sh));
    } else
#endif
    {
        IdentityKeySwitchPreAdd<iksP, casign, cbsign, offset>(tlwe, in0, in1, ksk);
    }
    __syncthreads();

#ifdef USE_FFT
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
#else
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP, StageMonomials64, PairUnstaged64>;
#endif
    typename brP::targetP::T* trlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);

#ifdef USE_FFT
    __BlindRotate__<brP, StageMonomials64>(trlwe, tlwe, μ, bk, ntt);
#else
    __BlindRotate__<brP, StageMonomials64, PairUnstaged64, 0, true>(trlwe, tlwe, μ, bk, ntt);
#endif
    __SampleExtractIndex__<typename brP::targetP, 0>(out, trlwe);
}

template <class brP, std::make_signed_t<typename brP::targetP::T> μ, class iksP,
          int casign, int cbsign, auto offset, bool CacheOutputExponents = true>
__device__ inline void __HomGate__(typename iksP::targetP::T* const out,
                                   const typename brP::domainP::T* const in0,
                                   const typename brP::domainP::T* const in1,
                                   const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    typename brP::targetP::T* trlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);

    // Select the measured cache only for this single-blind-rotation binary schedule.
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
    if constexpr (CacheOutputExponents && std::is_same_v<brP, TFHEpp::lvl02param> &&
                  std::is_same_v<typename iksP::targetP, typename brP::domainP>) {
        __BlindRotatePreAdd__<brP, casign, cbsign, offset, true, true, true>(
            trlwe, in0, in1, bk, ntt, out);
    } else
#endif
    {
        __BlindRotatePreAdd__<brP, casign, cbsign, offset, true, true>(
            trlwe, in0, in1, bk, ntt);
    }

    __KeySwitchAfterBlindRotate__<iksP, brP, true>(out, trlwe, ksk, dyn_sh);
}

#ifdef USE_KEY_BUNDLE
// Key-bundle HomGate variants (IKS-BR order)
template <class iksP, class brP, std::make_signed_t<typename brP::targetP::T> μ,
          int casign, int cbsign, auto offset>
__device__ inline void __HomGateKeyBundle__(
    typename brP::targetP::T* const out,
    const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
    const typename iksP::targetP::T* const ksk,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const one_trgsw_ntt,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const xai_ntt,
    const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    __shared__
        typename iksP::targetP::T tlwe[iksP::targetP::k * iksP::targetP::n + 1];

    IdentityKeySwitchPreAdd<iksP, casign, cbsign, offset>(tlwe, in0, in1, ksk);
    __syncthreads();

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    typename brP::targetP::T* trlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);

    __BlindRotateKeyBundle__<brP>(trlwe, tlwe, μ, bk, one_trgsw_ntt, xai_ntt,
                                  ntt);
    __SampleExtractIndex__<typename brP::targetP, 0>(out, trlwe);
}

// Key-bundle HomGate variants (BR-IKS order)
template <class brP, std::make_signed_t<typename brP::targetP::T> μ, class iksP,
          int casign, int cbsign, auto offset>
__device__ inline void __HomGateKeyBundle__(
    typename iksP::targetP::T* const out,
    const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
    const typename iksP::targetP::T* const ksk,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const one_trgsw_ntt,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const xai_ntt,
    const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    typename brP::targetP::T* trlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);

    __BlindRotatePreAddKeyBundle__<brP, casign, cbsign, offset>(
        trlwe, in0, in1, bk, one_trgsw_ntt, xai_ntt, ntt);

    // After BlindRotate, FFT workspace is free -- reuse for tlwe
    typename brP::targetP::T* tlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh);
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
}
#endif  // USE_KEY_BUNDLE

// br iks ver.
template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NandBootstrap__(typename iksP::targetP::T* const
                                                 out,
                                             const typename brP::domainP::
                                                 T* const in0,
                                             const typename brP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                       ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NorBootstrap__(typename iksP::targetP::T* const
                                                out,
                                            const typename brP::domainP::
                                                T* const in0,
                                            const typename brP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                        ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __XnorBootstrap__(typename iksP::targetP::T* const
                                                 out,
                                             const typename brP::domainP::
                                                 T* const in0,
                                             const typename brP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<brP, μ, iksP, -2, -2, -2 * brP::domainP::μ>(out, in0, in1, bk,
                                                            ksk, ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndBootstrap__(typename iksP::targetP::T* const
                                                out,
                                            const typename brP::domainP::
                                                T* const in0,
                                            const typename brP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                      ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrBootstrap__(typename iksP::targetP::T* const out,
                                           const typename brP::domainP::T* const
                                               in0,
                                           const typename brP::domainP::T* const
                                               in1,
                                           NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                           const typename iksP::targetP::
                                               T* const ksk,
                                           const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                               ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                     ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __XorBootstrap__(typename iksP::targetP::T* const
                                                out,
                                            const typename brP::domainP::
                                                T* const in0,
                                            const typename brP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<brP, μ, iksP, 2, 2, 2 * brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                         ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndNYBootstrap__(typename iksP::targetP::T* const
                                                  out,
                                              const typename brP::domainP::
                                                  T* const in0,
                                              const typename brP::domainP::
                                                  T* const in1,
                                              NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                       ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndYNBootstrap__(typename iksP::targetP::T* const
                                                  out,
                                              const typename brP::domainP::
                                                  T* const in0,
                                              const typename brP::domainP::
                                                  T* const in1,
                                              NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                       ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrNYBootstrap__(typename iksP::targetP::T* const
                                                 out,
                                             const typename brP::domainP::
                                                 T* const in0,
                                             const typename brP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                      ntt);
}

template <class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ,
          class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrYNBootstrap__(typename iksP::targetP::T* const
                                                 out,
                                             const typename brP::domainP::
                                                 T* const in0,
                                             const typename brP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk,
                                                      ntt);
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template <class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __MuxBootstrap__(typename iksP::targetP::T* const
                                                out,
                                            const typename brP::domainP::
                                                T* const inc,
                                            const typename brP::domainP::
                                                T* const in1,
                                            const typename brP::domainP::
                                                T* const in0,
                                            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    extern __shared__ char dyn_sh[];
#if !defined(USE_FFT) && !defined(CUFHE_USE_HIP) &&     defined(CUFHE_WARP_TRANSFORM_TAIL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // Preserve all original inputs through both rotations; no output workspace.
    constexpr bool adjacent64 =
        std::is_same_v<brP, TFHEpp::lvl02param> && UseWarpNTTTail<64>;
#else
    constexpr bool adjacent64 = false;
#endif
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* trlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* trlwe0 =
        low_lds ? trlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ,
                          adjacent64, false, false, adjacent64>(
        trlwe1, inc, in1, bk, ntt);
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    typename brP::targetP::T* tlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh);

    if constexpr (low_lds) {
        typename brP::targetP::T saved[2 * brP::targetP::k + 1];
        __SampleExtractToRegisters__<typename brP::targetP, 0>(saved, trlwe1);
        __syncthreads();

        __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ,
                              adjacent64, false, false, adjacent64>(
            trlwe1, inc, in0, bk, ntt);
        __SampleExtractRegistersCombine__<typename brP::targetP, 0, false>(
            tlwe, saved, trlwe1);
        if (tid == 0) tlwe[brP::targetP::k * brP::targetP::n] += μ;
        __syncthreads();
    }
    else {
        __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ,
                              adjacent64, false, false, adjacent64>(
            trlwe0, inc, in0, bk, ntt);

        constexpr uint32_t trlwe_size = (brP::targetP::k + 1) * brP::targetP::n;
#pragma unroll
        for (int i = tid; i < trlwe_size; i += bdim) trlwe1[i] += trlwe0[i];
        if (tid == 0) trlwe1[brP::targetP::k * brP::targetP::n] += μ;
        __syncthreads();

        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
        __syncthreads();
    }

    __KeySwitchAfterBlindRotate__<iksP, brP, false>(out, tlwe, ksk, dyn_sh);
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template <class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NMuxBootstrap__(typename iksP::targetP::T* const
                                                 out,
                                             const typename brP::domainP::
                                                 T* const inc,
                                             const typename brP::domainP::
                                                 T* const in1,
                                             const typename brP::domainP::
                                                 T* const in0,
                                             const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    extern __shared__ char dyn_sh[];
#if !defined(USE_FFT) && !defined(CUFHE_USE_HIP) &&     defined(CUFHE_WARP_TRANSFORM_TAIL) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // Preserve all original inputs through both rotations; no output workspace.
    constexpr bool adjacent64 =
        std::is_same_v<brP, TFHEpp::lvl02param> && UseWarpNTTTail<64>;
#else
    constexpr bool adjacent64 = false;
#endif
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* trlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* trlwe0 =
        low_lds ? trlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ,
                          adjacent64, false, false, adjacent64>(
        trlwe1, inc, in1, bk, ntt);
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    typename brP::targetP::T* tlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh);

    if constexpr (low_lds) {
        typename brP::targetP::T saved[2 * brP::targetP::k + 1];
        __SampleExtractToRegisters__<typename brP::targetP, 0>(saved, trlwe1);
        __syncthreads();

        __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ,
                              adjacent64, false, false, adjacent64>(
            trlwe1, inc, in0, bk, ntt);
        __SampleExtractRegistersCombine__<typename brP::targetP, 0, true>(
            tlwe, saved, trlwe1);
        if (tid == 0) tlwe[brP::targetP::k * brP::targetP::n] -= μ;
        __syncthreads();
    }
    else {
        __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ,
                              adjacent64, false, false, adjacent64>(
            trlwe0, inc, in0, bk, ntt);

        constexpr uint32_t trlwe_size = (brP::targetP::k + 1) * brP::targetP::n;
#pragma unroll
        for (int i = tid; i < trlwe_size; i += bdim)
            trlwe1[i] = -trlwe1[i] - trlwe0[i];
        if (tid == 0) trlwe1[brP::targetP::k * brP::targetP::n] -= μ;
        __syncthreads();

        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
        __syncthreads();
    }

    __KeySwitchAfterBlindRotate__<iksP, brP, false>(out, tlwe, ksk, dyn_sh);
}

// Public NAND batch: each block runs the unchanged scalar fused operation.
template <class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>,
                            MIN_BLOCKS4HOMGATE<typename brP::targetP>)
void __NandBootstrapBatch__(
    typename iksP::targetP::T* out, size_t out_stride,
    const typename brP::domainP::T* in0, size_t in0_stride,
    const typename brP::domainP::T* in1, size_t in1_stride,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
    const typename iksP::targetP::T* ksk,
#ifdef USE_KEY_BUNDLE
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* one,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* xai,
#endif
    CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    const size_t i = blockIdx.x;
#ifdef USE_KEY_BUNDLE
    __HomGateKeyBundle__<brP, μ, iksP, -1, -1, brP::domainP::μ>(
        out + i * out_stride, in0 + i * in0_stride, in1 + i * in1_stride,
        bk, ksk, one, xai, ntt);
#else
    __HomGate__<brP, μ, iksP, -1, -1, brP::domainP::μ, false>(
        out + i * out_stride, in0 + i * in0_stride, in1 + i * in1_stride,
        bk, ksk, ntt);
#endif
}

// All blocks share the evaluation key, while each block owns its TLWE/TRLWE
// workspace and evaluates an independent ciphertext. No common-mask assumption.
template <class iksP, class brP, typename brP::targetP::T μ, bool StageMonomials64 = false>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>,
                            MIN_BLOCKS4HOMGATE<typename brP::targetP>)
void __NandBootstrapBatch__(
    typename brP::targetP::T* out, size_t out_stride,
    const typename iksP::domainP::T* in0, size_t in0_stride,
    const typename iksP::domainP::T* in1, size_t in1_stride,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
    const typename iksP::targetP::T* ksk,
#ifdef USE_KEY_BUNDLE
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* one,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* xai,
#endif
    CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    const size_t i = blockIdx.x;
#ifdef USE_KEY_BUNDLE
    __HomGateKeyBundle__<iksP, brP, μ, -1, -1, iksP::domainP::μ>(
        out + i * out_stride, in0 + i * in0_stride, in1 + i * in1_stride,
        bk, ksk, one, xai, ntt);
#else
    __HomGate__<iksP, brP, μ, -1, -1, iksP::domainP::μ, StageMonomials64,
                !StageMonomials64 && USE_RAW_PAIRED_NTT64<brP>>(
        out + i * out_stride, in0 + i * in0_stride, in1 + i * in1_stride,
        bk, ksk, ntt);
#endif
}

// iks br ver.
template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NandBootstrap__(typename brP::targetP::T* const
                                                 out,
                                             const typename iksP::domainP::
                                                 T* const in0,
                                             const typename iksP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NorBootstrap__(typename brP::targetP::T* const out,
                                            const typename iksP::domainP::
                                                T* const in0,
                                            const typename iksP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, -iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __XnorBootstrap__(typename brP::targetP::T* const
                                                 out,
                                             const typename iksP::domainP::
                                                 T* const in0,
                                             const typename iksP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<iksP, brP, μ, -2, -2, -2 * iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndBootstrap__(typename brP::targetP::T* const out,
                                            const typename iksP::domainP::
                                                T* const in0,
                                            const typename iksP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, -iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrBootstrap__(typename brP::targetP::T* const out,
                                           const typename iksP::domainP::
                                               T* const in0,
                                           const typename iksP::domainP::
                                               T* const in1,
                                           NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                           const typename iksP::targetP::
                                               T* const ksk,
                                           const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                               ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __XorBootstrap__(typename brP::targetP::T* const out,
                                            const typename iksP::domainP::
                                                T* const in0,
                                            const typename iksP::domainP::
                                                T* const in1,
                                            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __HomGate__<iksP, brP, μ, 2, 2, 2 * iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndNYBootstrap__(typename brP::targetP::T* const
                                                  out,
                                              const typename iksP::domainP::
                                                  T* const in0,
                                              const typename iksP::domainP::
                                                  T* const in1,
                                              NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, -iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __AndYNBootstrap__(typename brP::targetP::T* const
                                                  out,
                                              const typename iksP::domainP::
                                                  T* const in0,
                                              const typename iksP::domainP::
                                                  T* const in1,
                                              NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, -iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrNYBootstrap__(typename brP::targetP::T* const
                                                 out,
                                             const typename iksP::domainP::
                                                 T* const in0,
                                             const typename iksP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param,
          typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __OrYNBootstrap__(typename brP::targetP::T* const
                                                 out,
                                             const typename iksP::domainP::
                                                 T* const in0,
                                             const typename iksP::domainP::
                                                 T* const in1,
                                             NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, iksP::domainP::μ, false,
                USE_RAW_PAIRED_NTT64<brP>>(
        out, in0, in1, bk, ksk, ntt);
}

#ifdef USE_KEY_BUNDLE
// Key-bundle gate kernels (BR-IKS order)
#define DEFINE_KB_GATE_BRIKS(Name, casign_val, cbsign_val, offset_expr) \
    template <class brP = TFHEpp::lvl01param,                           \
              typename brP::targetP::T μ = TFHEpp::lvl1param::μ,        \
              class iksP = TFHEpp::lvl10param>                          \
    __global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>, MIN_BLOCKS4HOMGATE<typename brP::targetP>) void                 \
        __##Name##BootstrapKB__(                                        \
            typename iksP::targetP::T* const out,                       \
            const typename brP::domainP::T* const in0,                  \
            const typename brP::domainP::T* const in1,                  \
            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,                           \
            const typename iksP::targetP::T* const ksk,                 \
            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const one_trgsw_ntt,    \
            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const xai_ntt,          \
            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)                    \
    {                                                                   \
        __HomGateKeyBundle__<brP, μ, iksP, casign_val, cbsign_val,      \
                             offset_expr>(out, in0, in1, bk, ksk,       \
                                          one_trgsw_ntt, xai_ntt, ntt); \
    }

DEFINE_KB_GATE_BRIKS(Nand, -1, -1, brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(Nor, -1, -1, -brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(Xnor, -2, -2, -2 * brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(And, 1, 1, -brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(Or, 1, 1, brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(Xor, 2, 2, 2 * brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(AndNY, -1, 1, -brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(AndYN, 1, -1, -brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(OrNY, -1, 1, brP::domainP::μ)
DEFINE_KB_GATE_BRIKS(OrYN, 1, -1, brP::domainP::μ)
#undef DEFINE_KB_GATE_BRIKS

// Key-bundle gate kernels (IKS-BR order)
#define DEFINE_KB_GATE_IKSBR(Name, casign_val, cbsign_val, offset_expr)        \
    template <class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, \
              typename brP::targetP::T μ = TFHEpp::lvl1param::μ>               \
    __global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>, MIN_BLOCKS4HOMGATE<typename brP::targetP>) void                        \
        __##Name##BootstrapKB__(                                               \
            typename brP::targetP::T* const out,                               \
            const typename iksP::domainP::T* const in0,                        \
            const typename iksP::domainP::T* const in1,                        \
            NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,                                  \
            const typename iksP::targetP::T* const ksk,                        \
            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const one_trgsw_ntt,           \
            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const xai_ntt,                 \
            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)                           \
    {                                                                          \
        __HomGateKeyBundle__<iksP, brP, μ, casign_val, cbsign_val,             \
                             offset_expr>(out, in0, in1, bk, ksk,              \
                                          one_trgsw_ntt, xai_ntt, ntt);        \
    }

DEFINE_KB_GATE_IKSBR(Nand, -1, -1, iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(Nor, -1, -1, -iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(Xnor, -2, -2, -2 * iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(And, 1, 1, -iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(Or, 1, 1, iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(Xor, 2, 2, 2 * iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(AndNY, -1, 1, -iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(AndYN, 1, -1, -iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(OrNY, -1, 1, iksP::domainP::μ)
DEFINE_KB_GATE_IKSBR(OrYN, 1, -1, iksP::domainP::μ)
#undef DEFINE_KB_GATE_IKSBR

// Key-bundle Mux (BR-IKS order)
template <class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __MuxBootstrapKB__(typename iksP::targetP::T* const
                                                  out,
                                              const typename brP::domainP::
                                                  T* const inc,
                                              const typename brP::domainP::
                                                  T* const in1,
                                              const typename brP::domainP::
                                                  T* const in0,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                  one_trgsw_ntt,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                  xai_ntt,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* trlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* trlwe0 =
        low_lds ? trlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __BlindRotatePreAddKeyBundle__<brP, 1, 1, -brP::domainP::μ>(
        trlwe1, inc, in1, bk, one_trgsw_ntt, xai_ntt, ntt);
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    typename brP::targetP::T* tlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh);

    if constexpr (low_lds) {
        typename brP::targetP::T saved[2 * brP::targetP::k + 1];
        __SampleExtractToRegisters__<typename brP::targetP, 0>(saved, trlwe1);
        __syncthreads();

        __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(
            trlwe1, inc, in0, bk, one_trgsw_ntt, xai_ntt, ntt);
        __SampleExtractRegistersCombine__<typename brP::targetP, 0, false>(
            tlwe, saved, trlwe1);
        if (tid == 0) tlwe[brP::targetP::k * brP::targetP::n] += μ;
        __syncthreads();
    }
    else {
        __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(
            trlwe0, inc, in0, bk, one_trgsw_ntt, xai_ntt, ntt);

        constexpr uint32_t trlwe_size = (brP::targetP::k + 1) * brP::targetP::n;
#pragma unroll
        for (int i = tid; i < trlwe_size; i += bdim) trlwe1[i] += trlwe0[i];
        if (tid == 0) trlwe1[brP::targetP::k * brP::targetP::n] += μ;
        __syncthreads();

        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
        __syncthreads();
    }

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
}

// Key-bundle NMux (BR-IKS order)
template <class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NMuxBootstrapKB__(typename iksP::targetP::T* const
                                                   out,
                                               const typename brP::domainP::
                                                   T* const inc,
                                               const typename brP::domainP::
                                                   T* const in1,
                                               const typename brP::domainP::
                                                   T* const in0,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                               const typename iksP::targetP::
                                                   T* const ksk,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                   one_trgsw_ntt,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                   xai_ntt,
                                               const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                   ntt)
{
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* trlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* trlwe0 =
        low_lds ? trlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __BlindRotatePreAddKeyBundle__<brP, 1, 1, -brP::domainP::μ>(
        trlwe1, inc, in1, bk, one_trgsw_ntt, xai_ntt, ntt);
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    typename brP::targetP::T* tlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh);

    if constexpr (low_lds) {
        typename brP::targetP::T saved[2 * brP::targetP::k + 1];
        __SampleExtractToRegisters__<typename brP::targetP, 0>(saved, trlwe1);
        __syncthreads();

        __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(
            trlwe1, inc, in0, bk, one_trgsw_ntt, xai_ntt, ntt);
        __SampleExtractRegistersCombine__<typename brP::targetP, 0, true>(
            tlwe, saved, trlwe1);
        if (tid == 0) tlwe[brP::targetP::k * brP::targetP::n] -= μ;
        __syncthreads();
    }
    else {
        __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(
            trlwe0, inc, in0, bk, one_trgsw_ntt, xai_ntt, ntt);

        constexpr uint32_t trlwe_size = (brP::targetP::k + 1) * brP::targetP::n;
#pragma unroll
        for (int i = tid; i < trlwe_size; i += bdim)
            trlwe1[i] = -trlwe1[i] - trlwe0[i];
        if (tid == 0) trlwe1[brP::targetP::k * brP::targetP::n] -= μ;
        __syncthreads();

        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
        __syncthreads();
    }

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
}

// Key-bundle Mux (IKS-BR order)
template <class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __MuxBootstrapKB__(typename brP::targetP::T* const
                                                  out,
                                              const typename iksP::domainP::
                                                  T* const inc,
                                              const typename iksP::domainP::
                                                  T* const in1,
                                              const typename iksP::domainP::
                                                  T* const in0,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                              const typename iksP::targetP::
                                                  T* const ksk,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                  one_trgsw_ntt,
                                              const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                  xai_ntt,
                                              const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                  ntt)
{
    __shared__ typename iksP::targetP::T
        tlwelvl0[iksP::targetP::k * iksP::targetP::n + 1];

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* tlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* tlwe0 =
        low_lds ? tlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1,
                                                           ksk);
    __syncthreads();
    __BlindRotateKeyBundle__<brP>(tlwe1, tlwelvl0, μ, bk, one_trgsw_ntt,
                                  xai_ntt, ntt);
    __SampleExtractIndex__<typename brP::targetP, 0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0,
                                                            ksk);
    __syncthreads();
    __BlindRotateKeyBundle__<brP>(tlwe0, tlwelvl0, μ, bk, one_trgsw_ntt,
                                  xai_ntt, ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    if constexpr (low_lds) {
        __SampleExtractCombine__<typename brP::targetP, 0, false>(out, tlwe0);
    }
    else {
        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe1, tlwe0);
        __syncthreads();

        constexpr uint32_t tlwe_size = brP::targetP::k * brP::targetP::n + 1;
#pragma unroll
        for (int i = tid; i < tlwe_size; i += bdim) out[i] += tlwe1[i];
    }
    if (tid == 0) out[brP::targetP::k * brP::targetP::n] += μ;
}

// Key-bundle NMux (IKS-BR order)
template <class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NMuxBootstrapKB__(typename brP::targetP::T* const
                                                   out,
                                               const typename iksP::domainP::
                                                   T* const inc,
                                               const typename iksP::domainP::
                                                   T* const in1,
                                               const typename iksP::domainP::
                                                   T* const in0,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                               const typename iksP::targetP::
                                                   T* const ksk,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                   one_trgsw_ntt,
                                               const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const
                                                   xai_ntt,
                                               const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                   ntt)
{
    __shared__ typename iksP::targetP::T
        tlwelvl0[iksP::targetP::k * iksP::targetP::n + 1];

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* tlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* tlwe0 =
        low_lds ? tlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1,
                                                           ksk);
    __syncthreads();
    __BlindRotateKeyBundle__<brP>(tlwe1, tlwelvl0, μ, bk, one_trgsw_ntt,
                                  xai_ntt, ntt);
    __SampleExtractIndex__<typename brP::targetP, 0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0,
                                                            ksk);
    __syncthreads();
    __BlindRotateKeyBundle__<brP>(tlwe0, tlwelvl0, μ, bk, one_trgsw_ntt,
                                  xai_ntt, ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    if constexpr (low_lds) {
        __SampleExtractCombine__<typename brP::targetP, 0, true>(out, tlwe0);
    }
    else {
        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe1, tlwe0);
        __syncthreads();

        constexpr uint32_t tlwe_size = brP::targetP::k * brP::targetP::n + 1;
#pragma unroll
        for (int i = tid; i < tlwe_size; i += bdim) out[i] = -out[i] - tlwe1[i];
    }
    if (tid == 0) out[brP::targetP::k * brP::targetP::n] -= μ;
}
#endif  // USE_KEY_BUNDLE

template <class P>
__global__
__launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp::lvl1param>) void __CopyBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k * P::n; i += bdim) out[i] = in[i];
    __syncthreads();
}

template <class P>
__global__
__launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MIN_BLOCKS4HOMGATE<TFHEpp::lvl1param>) void __NotBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k * P::n; i += bdim) out[i] = -in[i];
    __syncthreads();
}

// Finish raw MUX/NMux after an AND has written the first partial output.
// A separate CTA launch keeps each blind rotation's register lifetime local.
template <class iksP, class brP, typename brP::targetP::T mu, bool Negate>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>,
                            MIN_BLOCKS4HOMGATE<typename brP::targetP>)
void __RawMuxCombine__(typename brP::targetP::T* out,
    const typename iksP::domainP::T* selector,
    const typename iksP::domainP::T* data_zero,
    const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* bk,
    const typename iksP::targetP::T* ksk,
    CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8> ntt)
{
    using P = typename brP::targetP;
    __shared__ typename iksP::targetP::T tlwe[iksP::targetP::k * iksP::targetP::n + 1];
    extern __shared__ char dyn_sh[];
    constexpr size_t prefix = MEM4RAWBINARYGATE_DYN<brP> -
        (P::k + 1) * P::n * sizeof(typename P::T);
    auto* trlwe = reinterpret_cast<typename P::T*>(dyn_sh + prefix);
    __KeySwitchMuxPreAdd__<iksP, brP, -1, 1, -iksP::domainP::μ>(
        tlwe, selector, data_zero, ksk, dyn_sh);
    __syncthreads();
    __BlindRotate__<brP, false, USE_RAW_PAIRED_NTT64<brP>>(
        trlwe, tlwe, mu, bk, ntt);
    __SampleExtractCombine__<P, 0, Negate>(out, trlwe);
    if (threadIdx.x == 0) {
        if constexpr (Negate) out[P::k * P::n] -= mu;
        else out[P::k * P::n] += mu;
    }
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template <class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __MuxBootstrap__(typename brP::targetP::T* const out,
                                            const typename iksP::domainP::
                                                T* const inc,
                                            const typename iksP::domainP::
                                                T* const in1,
                                            const typename iksP::domainP::
                                                T* const in0,
                                            const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                            const typename iksP::targetP::
                                                T* const ksk,
                                            const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                ntt)
{
    __shared__ typename iksP::targetP::T
        tlwelvl0[iksP::targetP::k * iksP::targetP::n + 1];

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* tlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* tlwe0 =
        low_lds ? tlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __KeySwitchMuxPreAdd__<iksP, brP, 1, 1, -iksP::domainP::μ>(
        tlwelvl0, inc, in1, ksk, dyn_sh);
    __syncthreads();
    // Exact carry lowering benefits this scalar 64-bit MUX schedule.
    __BlindRotate__<brP, false, false,
                    sizeof(typename brP::targetP::T) == 8>(tlwe1, tlwelvl0, μ, bk, ntt);

    __KeySwitchMuxPreAdd__<iksP, brP, -1, 1, -iksP::domainP::μ>(
        tlwelvl0, inc, in0, ksk, dyn_sh);
    __syncthreads();
    // Both key switches have consumed their inputs before any aliased write.
    __SampleExtractIndex__<typename brP::targetP, 0>(out, tlwe1);
    // The low-LDS layout reuses tlwe1 for the second blind rotation.
    __syncthreads();
    __BlindRotate__<brP, false, false,
                    sizeof(typename brP::targetP::T) == 8>(tlwe0, tlwelvl0, μ, bk, ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    if constexpr (low_lds) {
        __SampleExtractCombine__<typename brP::targetP, 0, false>(out, tlwe0);
    }
    else {
        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe1, tlwe0);
        __syncthreads();

        constexpr uint32_t tlwe_size = brP::targetP::k * brP::targetP::n + 1;
#pragma unroll
        for (int i = tid; i < tlwe_size; i += bdim) out[i] += tlwe1[i];
    }
    if (tid == 0) out[brP::targetP::k * brP::targetP::n] += μ;
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template <class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP:: targetP>, MIN_BLOCKS4HOMGATE<typename brP:: targetP>) void __NMuxBootstrap__(typename brP::targetP::T* const
                                                 out,
                                             const typename iksP::domainP::
                                                 T* const inc,
                                             const typename iksP::domainP::
                                                 T* const in1,
                                             const typename iksP::domainP::
                                                 T* const in0,
                                             const NTTValueFor<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>* const bk,
                                             const typename iksP::targetP::
                                                 T* const ksk,
                                             const CuNTTHandler<brP::targetP::n, sizeof(typename brP::targetP::T) * 8>
                                                 ntt)
{
    __shared__ typename iksP::targetP::T
        tlwelvl0[iksP::targetP::k * iksP::targetP::n + 1];

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    constexpr bool low_lds = USE_LOW_LDS_BOOTSTRAP<typename brP::targetP>;
    constexpr size_t trlwe_bytes = (brP::targetP::k + 1) * brP::targetP::n *
                                   sizeof(typename brP::targetP::T);
    typename brP::targetP::T* tlwe1 =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);
    typename brP::targetP::T* tlwe0 =
        low_lds ? tlwe1
                : reinterpret_cast<typename brP::targetP::T*>(
                      dyn_sh + fft_bytes + trlwe_bytes);

    __KeySwitchMuxPreAdd__<iksP, brP, 1, 1, -iksP::domainP::μ>(
        tlwelvl0, inc, in1, ksk, dyn_sh);
    __syncthreads();
    // Exact carry lowering benefits this scalar 64-bit MUX schedule.
    __BlindRotate__<brP, false, false,
                    sizeof(typename brP::targetP::T) == 8>(tlwe1, tlwelvl0, μ, bk, ntt);

    __KeySwitchMuxPreAdd__<iksP, brP, -1, 1, -iksP::domainP::μ>(
        tlwelvl0, inc, in0, ksk, dyn_sh);
    __syncthreads();
    // Both key switches have consumed their inputs before any aliased write.
    __SampleExtractIndex__<typename brP::targetP, 0>(out, tlwe1);
    // The low-LDS layout reuses tlwe1 for the second blind rotation.
    __syncthreads();
    __BlindRotate__<brP, false, false,
                    sizeof(typename brP::targetP::T) == 8>(tlwe0, tlwelvl0, μ, bk, ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    if constexpr (low_lds) {
        __SampleExtractCombine__<typename brP::targetP, 0, true>(out, tlwe0);
    }
    else {
        __SampleExtractIndex__<typename brP::targetP, 0>(tlwe1, tlwe0);
        __syncthreads();

        constexpr uint32_t tlwe_size = brP::targetP::k * brP::targetP::n + 1;
#pragma unroll
        for (int i = tid; i < tlwe_size; i += bdim) out[i] = -out[i] - tlwe1[i];
    }
    if (tid == 0) out[brP::targetP::k * brP::targetP::n] -= μ;
}

void Bootstrap(TFHEpp::lvl0param::T* const out,
               const TFHEpp::lvl0param::T* const in, const lvl1param::T mu,
               const cudaStream_t st, const int gpuNum)
{
    constexpr size_t shared_bytes = MEM4HOMGATE<TFHEpp::lvl1param>;
    CuSafeCall(cudaFuncSetAttribute(__Bootstrap__<lvl01param, lvl10param>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    __Bootstrap__<lvl01param, lvl10param>
        <<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, shared_bytes, st>>>(
            out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum],
            *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CMUXNTTkernel(TFHEpp::lvl1param::T* const res, const NTTValue* const cs,
                   TFHEpp::lvl1param::T* const c1,
                   TFHEpp::lvl1param::T* const c0, cudaStream_t st,
                   const int gpuNum)
{
    constexpr size_t shmem_size =
        (TFHEpp::lvl1param::k + 2) * TFHEpp::lvl1param::n * sizeof(NTTValue);
    cudaFuncSetAttribute(
        __CMUXNTT__, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    __CMUXNTT__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, shmem_size, st>>>(
        res, cs, c1, c0, *ntt_handlers[gpuNum]);
    CuCheckError();
}

void BootstrapTLWE2TRLWE(TFHEpp::lvl1param::T* const out,
                         const TFHEpp::lvl0param::T* const in,
                         const lvl1param::T mu, const cudaStream_t st,
                         const int gpuNum)
{
    cudaFuncSetAttribute(__BlindRotateGlobal__<TFHEpp::lvl01param>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __BlindRotateGlobal__<TFHEpp::lvl01param>
        <<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>,
           MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(out, in, mu, bk_ntts[gpuNum],
#ifdef USE_KEY_BUNDLE
                                                 one_trgsw_ntt_devs[gpuNum],
                                                 xai_ntt_devs[gpuNum],
#endif
                                                 *ntt_handlers[gpuNum]);
    CuCheckError();
}

void SEIandBootstrap2TRLWE(TFHEpp::lvl1param::T* const out,
                           const TFHEpp::lvl1param::T* const in,
                           const lvl1param::T mu, const cudaStream_t st,
                           const int gpuNum)
{
    // Shared memory layout (all uint32_t / NTTValue):
    // [0, (k+2)*N): Accumulate working space
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE, ceil((n0+1)/2+1) uint32_ts)
    constexpr size_t shmem_elems =
        (lvl1param::k + 2 + lvl1param::k + 1) * lvl1param::n +
        (lvl0param::n + 1) / 2 + 1;
    constexpr size_t shmem_size = shmem_elems * sizeof(NTTValue);
    cudaFuncSetAttribute(__SEIandBootstrap2TRLWE__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_size);
    __SEIandBootstrap2TRLWE__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>,
                                shmem_size, st>>>(
        out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum],
#ifdef USE_KEY_BUNDLE
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
#endif
        *ntt_handlers[gpuNum]);
    CuCheckError();
}

template <class brP, typename brP::targetP::T μ, class iksP>
void NandBootstrap(typename iksP::targetP::T* const out,
                   const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NandBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NandBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NandBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NandBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NandBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NandBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NandBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void NandBootstrap<brP, μ, iksP>(                            \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void NandBootstrap(typename brP::targetP::T* const out,
                   const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NandBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NandBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NandBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NandBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NandBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NandBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NandBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void NandBootstrap<iksP, brP, μ>(                             \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void NandBootstrapBatch(
    typename iksP::targetP::T* out, size_t out_stride,
    const typename brP::domainP::T* in0, size_t in0_stride,
    const typename brP::domainP::T* in1, size_t in1_stride,
    uint32_t count, cudaStream_t st, int gpuNum)
{
    if (count == 0) return;
    constexpr size_t out_elements = iksP::targetP::k * iksP::targetP::n + 1;
    constexpr size_t in_elements = brP::domainP::k * brP::domainP::n + 1;
    if (out_stride < out_elements || in0_stride < in_elements ||
        in1_stride < in_elements)
        throw std::invalid_argument("NandBootstrapBatch: TLWE stride is too small");
    CuSafeCall(cudaSetDevice(gpuNum));
    if (count == 1) {
        NandBootstrap<brP, μ, iksP>(out, in0, in1, st, gpuNum);
        return;
    }
    auto& handlers = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return ntt_handlers_lvl02;
        else
            return ntt_handlers;
    }();
    auto& keys = BootstrappingKeyStorage<typename brP::targetP>();
    auto& switching_keys = sizeof(typename brP::targetP::T) == 8
                               ? ksk_devs_lvl20 : ksk_devs;
#ifdef USE_KEY_BUNDLE
    auto& ones = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return one_trgsw_ntt_devs_lvl02;
        else
            return one_trgsw_ntt_devs;
    }();
    auto& monomials = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return xai_ntt_devs_lvl02;
        else
            return xai_ntt_devs;
    }();
#endif
    constexpr size_t shared_bytes = MEM4HOMGATE_DYN<typename brP::targetP>;
    CuSafeCall(cudaFuncSetAttribute(__NandBootstrapBatch__<brP, μ, iksP>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    __NandBootstrapBatch__<brP, μ, iksP>
        <<<count, NUM_THREAD4HOMGATE<typename brP::targetP>, shared_bytes, st>>>(
            out, out_stride, in0, in0_stride, in1, in1_stride,
            keys[gpuNum], switching_keys[gpuNum],
#ifdef USE_KEY_BUNDLE
            ones[gpuNum], monomials[gpuNum],
#endif
            *handlers[gpuNum]);
    CuCheckError();
}

template void NandBootstrapBatch<TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param>(
    TFHEpp::lvl0param::T*, size_t, const TFHEpp::lvl0param::T*, size_t,
    const TFHEpp::lvl0param::T*, size_t, uint32_t, cudaStream_t, int);
template void NandBootstrapBatch<TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param>(
    TFHEpp::lvl0param::T*, size_t, const TFHEpp::lvl0param::T*, size_t,
    const TFHEpp::lvl0param::T*, size_t, uint32_t, cudaStream_t, int);

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
namespace {
namespace cg = cooperative_groups;
using CoopBR = TFHEpp::lvl02param;
using CoopKS = TFHEpp::lvl20param;
using CoopTarget = CoopBR::targetP;
using CoopWord = CoopTarget::T;
constexpr int CoopN = CoopTarget::n;
constexpr int CoopRows = BootstrappingTRGSWRows<CoopTarget>;
constexpr int CoopThreads = CoopN / 2;
constexpr size_t CoopKeyWords = size_t(CoopRows) * (CoopTarget::k + 1) * CoopN;
constexpr size_t CoopScratchBytes =
    (CoopRows + 1) * (CoopTarget::k + 1) * CoopN * sizeof(CoopWord) +
    CoopBR::domainP::n * sizeof(CoopBR::domainP::T);
// Exponents are uint16_t; round the next sample's state up to uint64_t alignment.
constexpr size_t CoopScratchWords =
    (CoopScratchBytes + sizeof(CoopWord) - 1) / sizeof(CoopWord);
constexpr size_t CoopSharedBytes = 65536;
static_assert(CoopN == 1024 && CoopTarget::k == 2 && CoopRows == 9 &&
              CoopBR::domainP::ell == 2);
static_assert(CoopTarget::l == CoopTarget::lₐ &&
              CoopTarget::Bgbit == CoopTarget::Bgₐbit);

// Small independent batches trade extra SMs and global partial sums for latency.
// Every sample has nine row CTAs. All CTAs participate in every grid barrier;
// the host only launches a grid that can be resident in full.
__global__ __launch_bounds__(CoopThreads, 1)
void CooperativeNandBatchKernel(
    CoopWord* out, size_t out_stride,
    const CoopWord* in0, size_t in0_stride,
    const CoopWord* in1, size_t in1_stride,
    const uint64_t* bk, const CoopKS::targetP::T* ksk,
    CuNTTHandler<CoopN, 64> ntt, CoopWord* scratch)
{
    auto grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int row = blockIdx.x % CoopRows, sample = blockIdx.x / CoopRows;
    out += sample * out_stride;
    in0 += sample * in0_stride;
    in1 += sample * in1_stride;
    scratch += sample * CoopScratchWords;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* work = reinterpret_cast<uint64_t*>(dynamic_shared);
    __shared__ CoopBR::domainP::T lwe[CoopBR::domainP::n + 1];
    auto* state = scratch;
    auto* partial = state + (CoopTarget::k + 1) * CoopN;
    auto* powers = reinterpret_cast<CoopBR::domainP::T*>(
        partial + CoopRows * (CoopTarget::k + 1) * CoopN);
    // Mutable cross-CTA state must not use the read-only cache (__ldg).
    volatile CoopWord* state_read = state;
    volatile CoopWord* partial_rw = partial;
    volatile CoopBR::domainP::T* power_read = powers;
    if (row == 0) {
        IdentityKeySwitchPreAddPrepared16<CoopKS, -1, -1, CoopKS::domainP::μ>(
            lwe, in0, in1, ksk, reinterpret_cast<uint16_t*>(dynamic_shared));
        __syncthreads();
        const auto correction = BlockBinaryModSwitchCorrection<CoopBR>(lwe);
        constexpr CoopBR::domainP::T roundoffset = CoopBR::domainP::T{1} <<
            (std::numeric_limits<CoopBR::domainP::T>::digits - 2 - CoopTarget::nbit);
        const CoopBR::domainP::T corrected =
            lwe[CoopBR::domainP::n] - correction / 2 + roundoffset;
        const uint32_t bar = 2 * CoopN - modSwitchFromTorus<CoopBR>(corrected);
        RotatedTestVector<CoopTarget>(state, bar, CoopTarget::μ);
        for (int i = tid; i < CoopBR::domainP::n; i += CoopThreads)
            powers[i] = modSwitchFromTorus<CoopBR>(lwe[i] + roundoffset);
    }
    grid.sync();
    constexpr CoopWord offset = offsetgen<CoopTarget>() +
        (CoopWord{1} << (64 - CoopTarget::l * CoopTarget::Bgbit - 1));
    for (uint32_t block = 0;
         block < CoopBR::domainP::n / CoopBR::domainP::ell; ++block) {
        const uint32_t a0 = power_read[2 * block], a1 = power_read[2 * block + 1];
        const int component = row / CoopTarget::l, digit = row % CoopTarget::l;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
            const CoopWord value = state_read[component * CoopN + i] + offset;
            const int32_t decomp = int32_t((value >>
                (64 - (digit + 1) * CoopTarget::Bgbit)) & (CoopTarget::Bg - 1)) -
                CoopTarget::Bg / 2;
            work[i] = signed_int_to_ntt_mod<CoopN, 64>(decomp);
        }
        __syncthreads();
        SmallForwardNTT<CoopTarget::nbit, 64, false, 2>(work, ntt.forward_root_, tid);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
            const auto x0 = __ldg(block_xai_fft_lvl02 + a0 * CoopN + i);
            const auto x1 = __ldg(block_xai_fft_lvl02 + a1 * CoopN + i);
            const auto d0 = small_mod64_mult<2>(work[i], x0);
            const auto d1 = small_mod64_mult<2>(work[i], x1);
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                const size_t local = (row * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                const size_t index = block * 2 * CoopKeyWords + local;
                partial_rw[local] = small_mod64_madd2<2>(
                    d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords), 0);
            }
        }
        grid.sync();
        if (row <= CoopTarget::k) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                uint64_t sum = 0;
#pragma unroll
                for (int source = 0; source < CoopRows; ++source)
                    sum = small_mod64_add<2>(sum,
                        partial_rw[(source * (CoopTarget::k + 1) + row) * CoopN + i]);
                work[i] = sum;
            }
            __syncthreads();
            SmallInverseNTT<CoopTarget::nbit, 64, 2>(
                work, ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                state[row * CoopN + i] = state_read[row * CoopN + i] +
                    ntt_mod_to_torus64<CoopN, 64, true>(work[i]);
            }
        }
        grid.sync();
    }
    if (row == 0) __SampleExtractIndex__<CoopTarget, 0>(out, state);
}

// Three row groups reduce cross-block partial storage for medium batches.
constexpr int CoopThreeGroups = 3;
constexpr size_t CoopThreeScratchBytes =
    (CoopThreeGroups + 1) * (CoopTarget::k + 1) * CoopN * sizeof(CoopWord) +
    CoopBR::domainP::n * sizeof(CoopBR::domainP::T);
constexpr size_t CoopThreeScratchWords =
    (CoopThreeScratchBytes + sizeof(CoopWord) - 1) / sizeof(CoopWord);

__global__ __launch_bounds__(CoopThreads, 1)
void CooperativeNandBatchThreeKernel(
    CoopWord* out, size_t out_stride,
    const CoopWord* in0, size_t in0_stride,
    const CoopWord* in1, size_t in1_stride,
    const uint64_t* bk, const CoopKS::targetP::T* ksk,
    CuNTTHandler<CoopN, 64> ntt, CoopWord* scratch)
{
    auto grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int row = blockIdx.x % CoopThreeGroups, sample = blockIdx.x / CoopThreeGroups;
    out += sample * out_stride;
    in0 += sample * in0_stride;
    in1 += sample * in1_stride;
    scratch += sample * CoopThreeScratchWords;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* work = reinterpret_cast<uint64_t*>(dynamic_shared);
    __shared__ CoopBR::domainP::T lwe[CoopBR::domainP::n + 1];
    auto* state = scratch;
    auto* partial = state + (CoopTarget::k + 1) * CoopN;
    auto* powers = reinterpret_cast<CoopBR::domainP::T*>(
        partial + CoopThreeGroups * (CoopTarget::k + 1) * CoopN);
    // Mutable cross-CTA state must not use the read-only cache (__ldg).
    volatile CoopWord* state_read = state;
    volatile CoopWord* partial_rw = partial;
    volatile CoopBR::domainP::T* power_read = powers;
    if (row == 0) {
        IdentityKeySwitchPreAddPrepared16<CoopKS, -1, -1, CoopKS::domainP::μ>(
            lwe, in0, in1, ksk, reinterpret_cast<uint16_t*>(dynamic_shared));
        __syncthreads();
        const auto correction = BlockBinaryModSwitchCorrection<CoopBR>(lwe);
        constexpr CoopBR::domainP::T roundoffset = CoopBR::domainP::T{1} <<
            (std::numeric_limits<CoopBR::domainP::T>::digits - 2 - CoopTarget::nbit);
        const CoopBR::domainP::T corrected =
            lwe[CoopBR::domainP::n] - correction / 2 + roundoffset;
        const uint32_t bar = 2 * CoopN - modSwitchFromTorus<CoopBR>(corrected);
        RotatedTestVector<CoopTarget>(state, bar, CoopTarget::μ);
        for (int i = tid; i < CoopBR::domainP::n; i += CoopThreads)
            powers[i] = modSwitchFromTorus<CoopBR>(lwe[i] + roundoffset);
    }
    grid.sync();
    constexpr CoopWord offset = offsetgen<CoopTarget>() +
        (CoopWord{1} << (64 - CoopTarget::l * CoopTarget::Bgbit - 1));
    for (uint32_t block = 0;
         block < CoopBR::domainP::n / CoopBR::domainP::ell; ++block) {
        const uint32_t a0 = power_read[2 * block], a1 = power_read[2 * block + 1];
        auto* sums = work + 2 * CoopN;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                sums[outpoly * CoopN + i] = 0;
        }
#pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int first = row * CoopRows / CoopThreeGroups + 2 * step;
            const int limit = (row + 1) * CoopRows / CoopThreeGroups;
            if (first < limit) {
                const bool paired = first + 1 < limit;
#pragma unroll
                for (int which = 0; which < 2; ++which) {
                    if (which == 0 || paired) {
                        const int source_row = first + which;
                        const int component = source_row / CoopTarget::l;
                        const int digit = source_row % CoopTarget::l;
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int i = tid + e * CoopThreads;
                            const CoopWord value = state_read[component * CoopN + i] + offset;
                            const int32_t decomp = int32_t((value >>
                                (64 - (digit + 1) * CoopTarget::Bgbit)) & (CoopTarget::Bg - 1)) -
                                CoopTarget::Bg / 2;
                            work[which * CoopN + i] = signed_int_to_ntt_mod<CoopN, 64>(decomp);
                        }
                    }
                }
                __syncthreads();
                if (paired)
                    SmallForwardNTTPair64<CoopTarget::nbit, false, 2, false>(
                        work, work + CoopN, ntt.forward_root_, tid);
                else
                    SmallForwardNTT<CoopTarget::nbit, 64, false, 2>(work, ntt.forward_root_, tid);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = tid + e * CoopThreads;
                    const auto x0 = __ldg(block_xai_fft_lvl02 + a0 * CoopN + i);
                    const auto x1 = __ldg(block_xai_fft_lvl02 + a1 * CoopN + i);
                    const auto d0 = small_mod64_mult<2>(work[i], x0);
                    const auto d1 = small_mod64_mult<2>(work[i], x1);
                    if (paired) {
                        const auto d2 = small_mod64_mult<2>(work[CoopN + i], x0);
                        const auto d3 = small_mod64_mult<2>(work[CoopN + i], x1);
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd4<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                d2, __ldg(bk + index + (CoopTarget::k + 1) * CoopN),
                                d3, __ldg(bk + index + CoopKeyWords + (CoopTarget::k + 1) * CoopN),
                                sums[outpoly * CoopN + i]);
                        }
                    } else {
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd2<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                sums[outpoly * CoopN + i]);
                        }
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                partial_rw[(row * (CoopTarget::k + 1) + outpoly) * CoopN + i] =
                    sums[outpoly * CoopN + i];
        }
        grid.sync();
        if (row <= CoopTarget::k) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                uint64_t sum = 0;
#pragma unroll
                for (int source = 0; source < CoopThreeGroups; ++source)
                    sum = small_mod64_add<2>(sum,
                        partial_rw[(source * (CoopTarget::k + 1) + row) * CoopN + i]);
                work[i] = sum;
            }
            __syncthreads();
            SmallInverseNTT<CoopTarget::nbit, 64, 2>(
                work, ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                state[row * CoopN + i] = state_read[row * CoopN + i] +
                    ntt_mod_to_torus64<CoopN, 64, true>(work[i]);
            }
        }
        grid.sync();
    }
    if (row == 0) __SampleExtractIndex__<CoopTarget, 0>(out, state);
}

// Two blocks cover batches beyond the three-block resident limit. Rows 0..3
// use two forward pairs and own inverse polynomials 0/1; rows 4..8 use two
// forward pairs plus one scalar transform and own inverse polynomial 2.
constexpr int CoopTwoGroups = 2;
constexpr size_t CoopTwoScratchBytes =
    (CoopTwoGroups + 1) * (CoopTarget::k + 1) * CoopN * sizeof(CoopWord) +
    CoopBR::domainP::n * sizeof(CoopBR::domainP::T);
constexpr size_t CoopTwoScratchWords =
    (CoopTwoScratchBytes + sizeof(CoopWord) - 1) / sizeof(CoopWord);

__global__ __launch_bounds__(CoopThreads, 1)
void CooperativeNandBatchTwoKernel(
    CoopWord* out, size_t out_stride,
    const CoopWord* in0, size_t in0_stride,
    const CoopWord* in1, size_t in1_stride,
    const uint64_t* bk, const CoopKS::targetP::T* ksk,
    CuNTTHandler<CoopN, 64> ntt, CoopWord* scratch)
{
    auto grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int row = blockIdx.x % CoopTwoGroups, sample = blockIdx.x / CoopTwoGroups;
    out += sample * out_stride;
    in0 += sample * in0_stride;
    in1 += sample * in1_stride;
    scratch += sample * CoopTwoScratchWords;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* work = reinterpret_cast<uint64_t*>(dynamic_shared);
    __shared__ CoopBR::domainP::T lwe[CoopBR::domainP::n + 1];
    auto* state = scratch;
    auto* partial = state + (CoopTarget::k + 1) * CoopN;
    auto* powers = reinterpret_cast<CoopBR::domainP::T*>(
        partial + CoopTwoGroups * (CoopTarget::k + 1) * CoopN);
    // Mutable cross-CTA state must not use the read-only cache (__ldg).
    volatile CoopWord* state_read = state;
    volatile CoopWord* partial_rw = partial;
    volatile CoopBR::domainP::T* power_read = powers;
    if (row == 0) {
        IdentityKeySwitchPreAddPrepared16<CoopKS, -1, -1, CoopKS::domainP::μ>(
            lwe, in0, in1, ksk, reinterpret_cast<uint16_t*>(dynamic_shared));
        __syncthreads();
        const auto correction = BlockBinaryModSwitchCorrection<CoopBR>(lwe);
        constexpr CoopBR::domainP::T roundoffset = CoopBR::domainP::T{1} <<
            (std::numeric_limits<CoopBR::domainP::T>::digits - 2 - CoopTarget::nbit);
        const CoopBR::domainP::T corrected =
            lwe[CoopBR::domainP::n] - correction / 2 + roundoffset;
        const uint32_t bar = 2 * CoopN - modSwitchFromTorus<CoopBR>(corrected);
        RotatedTestVector<CoopTarget>(state, bar, CoopTarget::μ);
        for (int i = tid; i < CoopBR::domainP::n; i += CoopThreads)
            powers[i] = modSwitchFromTorus<CoopBR>(lwe[i] + roundoffset);
    }
    grid.sync();
    constexpr CoopWord offset = offsetgen<CoopTarget>() +
        (CoopWord{1} << (64 - CoopTarget::l * CoopTarget::Bgbit - 1));
    for (uint32_t block = 0;
         block < CoopBR::domainP::n / CoopBR::domainP::ell; ++block) {
        const uint32_t a0 = power_read[2 * block], a1 = power_read[2 * block + 1];
        auto* sums = work + 2 * CoopN;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                sums[outpoly * CoopN + i] = 0;
        }
#pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int first = row * CoopRows / CoopTwoGroups + 2 * step;
            const int limit = (row + 1) * CoopRows / CoopTwoGroups;
            if (first < limit) {
                const bool paired = first + 1 < limit;
#pragma unroll
                for (int which = 0; which < 2; ++which) {
                    if (which == 0 || paired) {
                        const int source_row = first + which;
                        const int component = source_row / CoopTarget::l;
                        const int digit = source_row % CoopTarget::l;
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int i = tid + e * CoopThreads;
                            const CoopWord value = state_read[component * CoopN + i] + offset;
                            const int32_t decomp = int32_t((value >>
                                (64 - (digit + 1) * CoopTarget::Bgbit)) & (CoopTarget::Bg - 1)) -
                                CoopTarget::Bg / 2;
                            work[which * CoopN + i] = signed_int_to_ntt_mod<CoopN, 64>(decomp);
                        }
                    }
                }
                __syncthreads();
                if (paired)
                    SmallForwardNTTPair64<CoopTarget::nbit, false, 2, false>(
                        work, work + CoopN, ntt.forward_root_, tid);
                else
                    SmallForwardNTT<CoopTarget::nbit, 64, false, 2>(work, ntt.forward_root_, tid);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = tid + e * CoopThreads;
                    const auto x0 = __ldg(block_xai_fft_lvl02 + a0 * CoopN + i);
                    const auto x1 = __ldg(block_xai_fft_lvl02 + a1 * CoopN + i);
                    const auto d0 = small_mod64_mult<2>(work[i], x0);
                    const auto d1 = small_mod64_mult<2>(work[i], x1);
                    if (paired) {
                        const auto d2 = small_mod64_mult<2>(work[CoopN + i], x0);
                        const auto d3 = small_mod64_mult<2>(work[CoopN + i], x1);
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd4<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                d2, __ldg(bk + index + (CoopTarget::k + 1) * CoopN),
                                d3, __ldg(bk + index + CoopKeyWords + (CoopTarget::k + 1) * CoopN),
                                sums[outpoly * CoopN + i]);
                        }
                    } else {
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd2<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                sums[outpoly * CoopN + i]);
                        }
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                partial_rw[(row * (CoopTarget::k + 1) + outpoly) * CoopN + i] =
                    sums[outpoly * CoopN + i];
        }
        grid.sync();
        if (row == 0) {
#pragma unroll
            for (int poly = 0; poly < 2; ++poly) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = tid + e * CoopThreads;
                    uint64_t sum = 0;
#pragma unroll
                    for (int source = 0; source < CoopTwoGroups; ++source)
                        sum = small_mod64_add<2>(sum,
                            partial_rw[(source * (CoopTarget::k + 1) + poly) * CoopN + i]);
                    work[poly * CoopN + i] = sum;
                }
            }
            __syncthreads();
            SmallInverseNTTPair64<CoopTarget::nbit, 2, false>(
                work, work + CoopN, ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
            for (int poly = 0; poly < 2; ++poly) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = tid + e * CoopThreads;
                    state[poly * CoopN + i] = state_read[poly * CoopN + i] +
                        ntt_mod_to_torus64<CoopN, 64, true>(work[poly * CoopN + i]);
                }
            }
        } else {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                uint64_t sum = 0;
#pragma unroll
                for (int source = 0; source < CoopTwoGroups; ++source)
                    sum = small_mod64_add<2>(sum,
                        partial_rw[(source * (CoopTarget::k + 1) + 2) * CoopN + i]);
                work[i] = sum;
            }
            __syncthreads();
            SmallInverseNTT<CoopTarget::nbit, 64, 2>(
                work, ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                state[2 * CoopN + i] = state_read[2 * CoopN + i] +
                    ntt_mod_to_torus64<CoopN, 64, true>(work[i]);
            }
        }
        grid.sync();
    }
    if (row == 0) __SampleExtractIndex__<CoopTarget, 0>(out, state);
}

// Five groups partition the nine existing rows into groups of one or two.
constexpr int CoopFiveGroups = 5;
constexpr size_t CoopFiveScratchBytes =
    (CoopFiveGroups + 1) * (CoopTarget::k + 1) * CoopN * sizeof(CoopWord) +
    CoopBR::domainP::n * sizeof(CoopBR::domainP::T);
constexpr size_t CoopFiveScratchWords =
    (CoopFiveScratchBytes + sizeof(CoopWord) - 1) / sizeof(CoopWord);

__global__ __launch_bounds__(CoopThreads, 1)
void CooperativeNandBatchFiveKernel(
    CoopWord* out, size_t out_stride,
    const CoopWord* in0, size_t in0_stride,
    const CoopWord* in1, size_t in1_stride,
    const uint64_t* bk, const CoopKS::targetP::T* ksk,
    CuNTTHandler<CoopN, 64> ntt, CoopWord* scratch)
{
    auto grid = cg::this_grid();
    const int tid = threadIdx.x;
    const int row = blockIdx.x % CoopFiveGroups, sample = blockIdx.x / CoopFiveGroups;
    out += sample * out_stride;
    in0 += sample * in0_stride;
    in1 += sample * in1_stride;
    scratch += sample * CoopFiveScratchWords;
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* work = reinterpret_cast<uint64_t*>(dynamic_shared);
    __shared__ CoopBR::domainP::T lwe[CoopBR::domainP::n + 1];
    auto* state = scratch;
    auto* partial = state + (CoopTarget::k + 1) * CoopN;
    auto* powers = reinterpret_cast<CoopBR::domainP::T*>(
        partial + CoopFiveGroups * (CoopTarget::k + 1) * CoopN);
    // Mutable cross-CTA state must not use the read-only cache (__ldg).
    volatile CoopWord* state_read = state;
    volatile CoopWord* partial_rw = partial;
    volatile CoopBR::domainP::T* power_read = powers;
    if (row == 0) {
        IdentityKeySwitchPreAddPrepared16<CoopKS, -1, -1, CoopKS::domainP::μ>(
            lwe, in0, in1, ksk, reinterpret_cast<uint16_t*>(dynamic_shared));
        __syncthreads();
        const auto correction = BlockBinaryModSwitchCorrection<CoopBR>(lwe);
        constexpr CoopBR::domainP::T roundoffset = CoopBR::domainP::T{1} <<
            (std::numeric_limits<CoopBR::domainP::T>::digits - 2 - CoopTarget::nbit);
        const CoopBR::domainP::T corrected =
            lwe[CoopBR::domainP::n] - correction / 2 + roundoffset;
        const uint32_t bar = 2 * CoopN - modSwitchFromTorus<CoopBR>(corrected);
        RotatedTestVector<CoopTarget>(state, bar, CoopTarget::μ);
        for (int i = tid; i < CoopBR::domainP::n; i += CoopThreads)
            powers[i] = modSwitchFromTorus<CoopBR>(lwe[i] + roundoffset);
    }
    grid.sync();
    constexpr CoopWord offset = offsetgen<CoopTarget>() +
        (CoopWord{1} << (64 - CoopTarget::l * CoopTarget::Bgbit - 1));
    for (uint32_t block = 0;
         block < CoopBR::domainP::n / CoopBR::domainP::ell; ++block) {
        const uint32_t a0 = power_read[2 * block], a1 = power_read[2 * block + 1];
        auto* sums = work + 2 * CoopN;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                sums[outpoly * CoopN + i] = 0;
        }
#pragma unroll
        for (int step = 0; step < 1; ++step) {
            const int first = row * CoopRows / CoopFiveGroups + 2 * step;
            const int limit = (row + 1) * CoopRows / CoopFiveGroups;
            if (first < limit) {
                const bool paired = first + 1 < limit;
#pragma unroll
                for (int which = 0; which < 2; ++which) {
                    if (which == 0 || paired) {
                        const int source_row = first + which;
                        const int component = source_row / CoopTarget::l;
                        const int digit = source_row % CoopTarget::l;
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int i = tid + e * CoopThreads;
                            const CoopWord value = state_read[component * CoopN + i] + offset;
                            const int32_t decomp = int32_t((value >>
                                (64 - (digit + 1) * CoopTarget::Bgbit)) & (CoopTarget::Bg - 1)) -
                                CoopTarget::Bg / 2;
                            work[which * CoopN + i] = signed_int_to_ntt_mod<CoopN, 64>(decomp);
                        }
                    }
                }
                __syncthreads();
                if (paired)
                    SmallForwardNTTPair64<CoopTarget::nbit, false, 2, false>(
                        work, work + CoopN, ntt.forward_root_, tid);
                else
                    SmallForwardNTT<CoopTarget::nbit, 64, false, 2>(work, ntt.forward_root_, tid);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int i = tid + e * CoopThreads;
                    const auto x0 = __ldg(block_xai_fft_lvl02 + a0 * CoopN + i);
                    const auto x1 = __ldg(block_xai_fft_lvl02 + a1 * CoopN + i);
                    const auto d0 = small_mod64_mult<2>(work[i], x0);
                    const auto d1 = small_mod64_mult<2>(work[i], x1);
                    if (paired) {
                        const auto d2 = small_mod64_mult<2>(work[CoopN + i], x0);
                        const auto d3 = small_mod64_mult<2>(work[CoopN + i], x1);
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd4<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                d2, __ldg(bk + index + (CoopTarget::k + 1) * CoopN),
                                d3, __ldg(bk + index + CoopKeyWords + (CoopTarget::k + 1) * CoopN),
                                sums[outpoly * CoopN + i]);
                        }
                    } else {
#pragma unroll
                        for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly) {
                            const size_t index = block * 2 * CoopKeyWords +
                                (first * (CoopTarget::k + 1) + outpoly) * CoopN + i;
                            sums[outpoly * CoopN + i] = small_mod64_madd2<2>(
                                d0, __ldg(bk + index), d1, __ldg(bk + index + CoopKeyWords),
                                sums[outpoly * CoopN + i]);
                        }
                    }
                }
                __syncthreads();
            }
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = tid + e * CoopThreads;
#pragma unroll
            for (int outpoly = 0; outpoly <= CoopTarget::k; ++outpoly)
                partial_rw[(row * (CoopTarget::k + 1) + outpoly) * CoopN + i] =
                    sums[outpoly * CoopN + i];
        }
        grid.sync();
        if (row <= CoopTarget::k) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                uint64_t sum = 0;
#pragma unroll
                for (int source = 0; source < CoopFiveGroups; ++source)
                    sum = small_mod64_add<2>(sum,
                        partial_rw[(source * (CoopTarget::k + 1) + row) * CoopN + i]);
                work[i] = sum;
            }
            __syncthreads();
            SmallInverseNTT<CoopTarget::nbit, 64, 2>(
                work, ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int i = tid + e * CoopThreads;
                state[row * CoopN + i] = state_read[row * CoopN + i] +
                    ntt_mod_to_torus64<CoopN, 64, true>(work[i]);
            }
        }
        grid.sync();
    }
    if (row == 0) __SampleExtractIndex__<CoopTarget, 0>(out, state);
}

// Keep a bounded cache for this implementation without changing the caller's
// current/default CUDA memory pool. Scratch allocations remain stream ordered.
void InitializeCooperativeScratchPools(int gpu_count)
{
    cooperative_scratch_pools.resize(gpu_count, nullptr);
    for (int gpu = 0; gpu < gpu_count; ++gpu) {
        CuSafeCall(cudaSetDevice(gpu));
        cudaDeviceProp prop{};
        CuSafeCall(cudaGetDeviceProperties(&prop, gpu));
        if (prop.major != 8 || prop.minor != 9 || !prop.cooperativeLaunch) continue;
        int supported = 0;
        CuSafeCall(cudaDeviceGetAttribute(
            &supported, cudaDevAttrMemoryPoolsSupported, gpu));
        if (!supported || cooperative_scratch_pools[gpu]) continue;
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.handleTypes = cudaMemHandleTypeNone;
        properties.location.type = cudaMemLocationTypeDevice;
        properties.location.id = gpu;
        cudaMemPool_t pool = nullptr;
        CuSafeCall(cudaMemPoolCreate(&pool, &properties));
        const uint64_t max_scratch = std::max(
            uint64_t(prop.multiProcessorCount / CoopRows) * CoopScratchWords,
            std::max(uint64_t(prop.multiProcessorCount / CoopThreeGroups) * CoopThreeScratchWords,
                     uint64_t(prop.multiProcessorCount / CoopTwoGroups) * CoopTwoScratchWords)) *
            sizeof(CoopWord);
        // CUDA may reserve much more backing memory than the requested scratch.
        // Learn that reservation while priming one maximum-size call, rather
        // than assuming an allocator granularity or modifying the caller pool.
        void* priming = nullptr;
        CuSafeCall(cudaMallocFromPoolAsync(&priming, max_scratch, pool, nullptr));
        CuSafeCall(cudaStreamSynchronize(nullptr));
        uint64_t retained_bytes = 0;
        CuSafeCall(cudaMemPoolGetAttribute(
            pool, cudaMemPoolAttrReservedMemCurrent, &retained_bytes));
        CuSafeCall(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &retained_bytes));
        CuSafeCall(cudaFreeAsync(priming, nullptr));
        CuSafeCall(cudaStreamSynchronize(nullptr));
        cooperative_scratch_pools[gpu] = pool;
    }
}

void DeleteCooperativeScratchPools()
{
    for (size_t gpu = 0; gpu < cooperative_scratch_pools.size(); ++gpu) {
        if (!cooperative_scratch_pools[gpu]) continue;
        CuSafeCall(cudaSetDevice(gpu));
        CuSafeCall(cudaMemPoolDestroy(cooperative_scratch_pools[gpu]));
    }
    cooperative_scratch_pools.clear();
}

bool TryCooperativeNandBatch(
    CoopWord* out, size_t out_stride,
    const CoopWord* in0, size_t in0_stride,
    const CoopWord* in1, size_t in1_stride,
    uint32_t count, cudaStream_t stream, int gpu)
{
    cudaDeviceProp prop{};
    CuSafeCall(cudaGetDeviceProperties(&prop, gpu));
    // Only the measured SM89 configuration selects this latency schedule.
    if (prop.major != 8 || prop.minor != 9 || !prop.cooperativeLaunch ||
        count > uint32_t(prop.multiProcessorCount / CoopTwoGroups))
        return false;
    int pools = 0;
    CuSafeCall(cudaDeviceGetAttribute(&pools, cudaDevAttrMemoryPoolsSupported, gpu));
    if (!pools) return false;
    const bool five_groups = count > uint32_t(prop.multiProcessorCount / CoopRows) &&
                            count <= uint32_t(prop.multiProcessorCount / CoopFiveGroups);
    const bool three_groups = count > uint32_t(prop.multiProcessorCount / CoopFiveGroups) &&
                             count <= uint32_t(prop.multiProcessorCount / CoopThreeGroups);
    const bool two_groups = count > uint32_t(prop.multiProcessorCount / CoopThreeGroups);
    const int groups = two_groups ? CoopTwoGroups : three_groups ? CoopThreeGroups :
                       five_groups ? CoopFiveGroups : CoopRows;
    const size_t scratch_words = two_groups ? CoopTwoScratchWords :
                                 three_groups ? CoopThreeScratchWords :
                                 five_groups ? CoopFiveScratchWords : CoopScratchWords;
    const void* kernel = two_groups ? (const void*)CooperativeNandBatchTwoKernel :
                         three_groups ? (const void*)CooperativeNandBatchThreeKernel :
                         five_groups ? (const void*)CooperativeNandBatchFiveKernel :
                                      (const void*)CooperativeNandBatchKernel;
    // Reserve enough shared memory to place each row CTA on a separate SM.
    CuSafeCall(cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, CoopSharedBytes));
    int active = 0;
    CuSafeCall(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, kernel, CoopThreads, CoopSharedBytes));
    if (count > uint32_t(active * prop.multiProcessorCount / groups)) return false;

    // Stream-ordered allocation/free gives concurrent calls independent state
    // and becomes allocation/free graph nodes during stream capture.
    CoopWord* workspace = nullptr;
    CuSafeCall(cudaMallocFromPoolAsync(&workspace,
        count * scratch_words * sizeof(CoopWord),
        cooperative_scratch_pools.at(gpu), stream));
    auto* bk = bk_ntts_lvl02[gpu];
    auto* ksk = ksk_devs_lvl20[gpu];
    auto ntt = *ntt_handlers_lvl02[gpu];
    void* args[] = {&out, &out_stride, &in0, &in0_stride, &in1, &in1_stride,
                    &bk, &ksk, &ntt, &workspace};
    const auto launch = cudaLaunchCooperativeKernel(
        kernel, dim3(groups * count),
        dim3(CoopThreads), args, CoopSharedBytes, stream);
    const auto release = cudaFreeAsync(workspace, stream);
    CuSafeCall(launch);
    CuSafeCall(release);
    return true;
}
}  // namespace
#endif

template <class iksP, class brP, typename brP::targetP::T μ>
void NandBootstrapBatch(
    typename brP::targetP::T* out, size_t out_stride,
    const typename iksP::domainP::T* in0, size_t in0_stride,
    const typename iksP::domainP::T* in1, size_t in1_stride,
    uint32_t count, cudaStream_t st, int gpuNum)
{
    if (count == 0) return;
    constexpr size_t out_elements = brP::targetP::k * brP::targetP::n + 1;
    constexpr size_t in_elements = iksP::domainP::k * iksP::domainP::n + 1;
    if (out_stride < out_elements || in0_stride < in_elements ||
        in1_stride < in_elements)
        throw std::invalid_argument("NandBootstrapBatch: TLWE stride is too small");
    CuSafeCall(cudaSetDevice(gpuNum));
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE) && \
    defined(USE_CUDA_PAIRED_NTT) && defined(USE_CUDA_WARP_TRANSFORM64) && \
    CUDART_VERSION >= 11040
    if constexpr (std::is_same_v<iksP, TFHEpp::lvl20param> &&
                  std::is_same_v<brP, TFHEpp::lvl02param> &&
                  USE_RAW_PAIRED_NTT64<brP> && μ == brP::targetP::μ) {
        if (TryCooperativeNandBatch(out, out_stride, in0, in0_stride,
                                   in1, in1_stride, count, st, gpuNum))
            return;
    }
#endif

    auto& handlers = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return ntt_handlers_lvl02;
        else
            return ntt_handlers;
    }();
    auto& keys = BootstrappingKeyStorage<typename brP::targetP>();
    auto& switching_keys = sizeof(typename brP::targetP::T) == 8
                               ? ksk_devs_lvl20 : ksk_devs;
#ifdef USE_KEY_BUNDLE
    auto& ones = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return one_trgsw_ntt_devs_lvl02;
        else
            return one_trgsw_ntt_devs;
    }();
    auto& monomials = []() -> auto& {
        if constexpr (sizeof(typename brP::targetP::T) == 8)
            return xai_ntt_devs_lvl02;
        else
            return xai_ntt_devs;
    }();
#endif
    auto launch = [&](auto stage_tag) {
        constexpr bool staged = decltype(stage_tag)::value;
#ifdef USE_FFT
        constexpr size_t shared_bytes = MEM4HOMGATE_DYN<typename brP::targetP>;
#else
        constexpr size_t shared_bytes = MEM4HOMGATE_DYN<typename brP::targetP, staged,
            !staged && USE_RAW_PAIRED_NTT64<brP>>;
#endif
        CuSafeCall(cudaFuncSetAttribute(
            __NandBootstrapBatch__<iksP, brP, μ, staged>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes));
        __NandBootstrapBatch__<iksP, brP, μ, staged>
            <<<count, NUM_THREAD4HOMGATE<typename brP::targetP>,
               shared_bytes, st>>>(
                out, out_stride, in0, in0_stride, in1, in1_stride,
                keys[gpuNum], switching_keys[gpuNum],
#ifdef USE_KEY_BUNDLE
                ones[gpuNum], monomials[gpuNum],
#endif
                *handlers[gpuNum]);
    };
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && \
    !defined(CUFHE_USE_HIP) && !defined(USE_KEY_BUNDLE)
    if constexpr (USE_ASYNC_MONOMIALS64<typename brP::targetP> &&
                  brP::domainP::ell == 2) {
        int multiprocessors = 0;
        CuSafeCall(cudaDeviceGetAttribute(&multiprocessors,
            cudaDevAttrMultiProcessorCount, gpuNum));
        // Retain staging outside the measured SM89 raw-pair configuration.
        if (count >= static_cast<uint32_t>(multiprocessors)) {
            bool prefer_raw_pair = false;
            if constexpr (USE_RAW_PAIRED_NTT64<brP>) {
                int major = 0, minor = 0;
                CuSafeCall(cudaDeviceGetAttribute(&major,
                    cudaDevAttrComputeCapabilityMajor, gpuNum));
                CuSafeCall(cudaDeviceGetAttribute(&minor,
                    cudaDevAttrComputeCapabilityMinor, gpuNum));
                // On SM89 the cached raw pair wins even at full occupancy.
                prefer_raw_pair = major == 8 && minor == 9;
            }
            if (!prefer_raw_pair) {
                launch(std::true_type{});
                CuCheckError();
                return;
            }
        }
    }
#endif
    launch(std::false_type{});
    CuCheckError();
}

#define INST(iksP, brP, mu)                                                \
    template void NandBootstrapBatch<iksP, brP, mu>(                         \
        typename brP::targetP::T*, size_t,                                  \
        const typename iksP::domainP::T*, size_t,                            \
        const typename iksP::domainP::T*, size_t,                            \
        uint32_t, cudaStream_t, int)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST



template <class brP, typename brP::targetP::T μ, class iksP>
void OrBootstrap(typename iksP::targetP::T* const out,
                 const typename brP::domainP::T* const in0,
                 const typename brP::domainP::T* const in1,
                 const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void OrBootstrap<brP, μ, iksP>(                              \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void OrBootstrap(typename brP::targetP::T* const out,
                 const typename iksP::domainP::T* const in0,
                 const typename iksP::domainP::T* const in1,
                 const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void OrBootstrap<iksP, brP, μ>(                               \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void OrYNBootstrap(typename iksP::targetP::T* const out,
                   const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrYNBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrYNBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrYNBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrYNBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void OrYNBootstrap<brP, μ, iksP>(                            \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void OrYNBootstrap(typename brP::targetP::T* const out,
                   const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrYNBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrYNBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrYNBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrYNBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void OrYNBootstrap<iksP, brP, μ>(                             \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void OrNYBootstrap(typename iksP::targetP::T* const out,
                   const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrNYBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrNYBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrNYBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrNYBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void OrNYBootstrap<brP, μ, iksP>(                            \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void OrNYBootstrap(typename brP::targetP::T* const out,
                   const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__OrNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrNYBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__OrNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __OrNYBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__OrNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __OrNYBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __OrNYBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void OrNYBootstrap<iksP, brP, μ>(                             \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void AndBootstrap(typename iksP::targetP::T* const out,
                  const typename brP::domainP::T* const in0,
                  const typename brP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void AndBootstrap<brP, μ, iksP>(                             \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void AndBootstrap(typename brP::targetP::T* const out,
                  const typename iksP::domainP::T* const in0,
                  const typename iksP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void AndBootstrap<iksP, brP, μ>(                              \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void AndYNBootstrap(typename iksP::targetP::T* const out,
                    const typename brP::domainP::T* const in0,
                    const typename brP::domainP::T* const in1,
                    const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndYNBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndYNBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndYNBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndYNBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void AndYNBootstrap<brP, μ, iksP>(                           \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void AndYNBootstrap(typename brP::targetP::T* const out,
                    const typename iksP::domainP::T* const in0,
                    const typename iksP::domainP::T* const in1,
                    const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndYNBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndYNBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndYNBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndYNBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void AndYNBootstrap<iksP, brP, μ>(                            \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void AndNYBootstrap(typename iksP::targetP::T* const out,
                    const typename brP::domainP::T* const in0,
                    const typename brP::domainP::T* const in1,
                    const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndNYBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndNYBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndNYBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndNYBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void AndNYBootstrap<brP, μ, iksP>(                           \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void AndNYBootstrap(typename brP::targetP::T* const out,
                    const typename iksP::domainP::T* const in0,
                    const typename iksP::domainP::T* const in1,
                    const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__AndNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndNYBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__AndNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __AndNYBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__AndNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __AndNYBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __AndNYBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void AndNYBootstrap<iksP, brP, μ>(                            \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void NorBootstrap(typename iksP::targetP::T* const out,
                  const typename brP::domainP::T* const in0,
                  const typename brP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void NorBootstrap<brP, μ, iksP>(                             \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void NorBootstrap(typename brP::targetP::T* const out,
                  const typename iksP::domainP::T* const in0,
                  const typename iksP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __NorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void NorBootstrap<iksP, brP, μ>(                              \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void XorBootstrap(typename iksP::targetP::T* const out,
                  const typename brP::domainP::T* const in0,
                  const typename brP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__XorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__XorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__XorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __XorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __XorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void XorBootstrap<brP, μ, iksP>(                             \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void XorBootstrap(typename brP::targetP::T* const out,
                  const typename iksP::domainP::T* const in0,
                  const typename iksP::domainP::T* const in1,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__XorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__XorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__XorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __XorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __XorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void XorBootstrap<iksP, brP, μ>(                              \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void XnorBootstrap(typename iksP::targetP::T* const out,
                   const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__XnorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XnorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__XnorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XnorBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__XnorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __XnorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __XnorBootstrap__<brP, brP::targetP::μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void XnorBootstrap<brP, μ, iksP>(                            \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const in0,                        \
        const typename brP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void XnorBootstrap(typename brP::targetP::T* const out,
                   const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__XnorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XnorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                one_trgsw_ntt_devs_lvl02[gpuNum], xai_ntt_devs_lvl02[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__XnorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4HOMGATE_DYN<typename brP::targetP>);
        __XnorBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4HOMGATE_DYN<typename brP::targetP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__XnorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4RAWBINARYGATE_DYN<brP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __XnorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __XnorBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4RAWBINARYGATE_DYN<brP>, st>>>(
                out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void XnorBootstrap<iksP, brP, μ>(                             \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const in0,                        \
        const typename iksP::domainP::T* const in1, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class P>
void CopyBootstrap(typename P::T* const out, const typename P::T* const in,
                   const cudaStream_t st, const int gpuNum)
{
    __CopyBootstrap__<P>
        <<<1, std::min(P::n + 1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0,
           st>>>(out, in);
    CuCheckError();
}
#define INST(P)                                                   \
    template void CopyBootstrap<P>(typename P::T* const out,      \
                                   const typename P::T* const in, \
                                   const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template <class P>
void NotBootstrap(typename P::T* const out, const typename P::T* const in,
                  const cudaStream_t st, const int gpuNum)
{
    __NotBootstrap__<P>
        <<<1, std::min(P::n + 1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0,
           st>>>(out, in);
    CuCheckError();
}
#define INST(P)                                                  \
    template void NotBootstrap<P>(typename P::T* const out,      \
                                  const typename P::T* const in, \
                                  const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template <class brP, typename brP::targetP::T μ, class iksP>
void MuxBootstrap(typename iksP::targetP::T* const out,
                  const typename brP::domainP::T* const inc,
                  const typename brP::domainP::T* const in1,
                  const typename brP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__MuxBootstrapKB__<brP, μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __MuxBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], one_trgsw_ntt_devs_lvl02[gpuNum],
                xai_ntt_devs_lvl02[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__MuxBootstrapKB__<brP, μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __MuxBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__MuxBootstrap__<brP, μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4MUXGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __MuxBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __MuxBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void MuxBootstrap<brP, μ, iksP>(                             \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const inc,                        \
        const typename brP::domainP::T* const in1,                        \
        const typename brP::domainP::T* const in0, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST
template <class iksP, class brP, typename brP::targetP::T μ>
void MuxBootstrap(typename brP::targetP::T* const out,
                  const typename iksP::domainP::T* const inc,
                  const typename iksP::domainP::T* const in1,
                  const typename iksP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#if !defined(USE_KEY_BUNDLE) && !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 10000
    if constexpr (USE_RAW_PAIRED_NTT64<brP>) {
        cudaStreamCaptureStatus capture_status;
        CuSafeCall(cudaStreamIsCapturing(st, &capture_status));
        // The first graph node writes out before the second reads these inputs.
        if (capture_status == cudaStreamCaptureStatusActive &&
            out != inc && out != in0) {
            AndBootstrap<iksP, brP, μ>(out, inc, in1, st, gpuNum);
            constexpr size_t shared_bytes = MEM4RAWBINARYGATE_DYN<brP>;
            CuSafeCall(cudaFuncSetAttribute(__RawMuxCombine__<iksP, brP, μ, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            __RawMuxCombine__<iksP, brP, μ, false>
                <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, shared_bytes, st>>>(
                    out, inc, in0, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                    *ntt_handlers_lvl02[gpuNum]);
            CuCheckError();
            return;
        }
    }
#endif

#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__MuxBootstrapKB__<iksP, brP, μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __MuxBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], one_trgsw_ntt_devs_lvl02[gpuNum],
                xai_ntt_devs_lvl02[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__MuxBootstrapKB__<iksP, brP, μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __MuxBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__MuxBootstrap__<iksP, brP, μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4MUXGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __MuxBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __MuxBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void MuxBootstrap<iksP, brP, μ>(                              \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const inc,                        \
        const typename iksP::domainP::T* const in1,                        \
        const typename iksP::domainP::T* const in0, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST

template <class iksP, class brP, typename brP::targetP::T μ>
void NMuxBootstrap(typename brP::targetP::T* const out,
                   const typename iksP::domainP::T* const inc,
                   const typename iksP::domainP::T* const in1,
                   const typename iksP::domainP::T* const in0,
                   const cudaStream_t st, const int gpuNum)
{
#if !defined(USE_KEY_BUNDLE) && !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 10000
    if constexpr (USE_RAW_PAIRED_NTT64<brP>) {
        cudaStreamCaptureStatus capture_status;
        CuSafeCall(cudaStreamIsCapturing(st, &capture_status));
        // The first graph node writes out before the second reads these inputs.
        if (capture_status == cudaStreamCaptureStatusActive &&
            out != inc && out != in0) {
            AndBootstrap<iksP, brP, μ>(out, inc, in1, st, gpuNum);
            constexpr size_t shared_bytes = MEM4RAWBINARYGATE_DYN<brP>;
            CuSafeCall(cudaFuncSetAttribute(__RawMuxCombine__<iksP, brP, μ, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            __RawMuxCombine__<iksP, brP, μ, true>
                <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, shared_bytes, st>>>(
                    out, inc, in0, bk_ntts_lvl02[gpuNum], ksk_devs_lvl20[gpuNum],
                    *ntt_handlers_lvl02[gpuNum]);
            CuCheckError();
            return;
        }
    }
#endif

#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NMuxBootstrapKB__<iksP, brP, μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __NMuxBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], one_trgsw_ntt_devs_lvl02[gpuNum],
                xai_ntt_devs_lvl02[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NMuxBootstrapKB__<iksP, brP, μ>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __NMuxBootstrapKB__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NMuxBootstrap__<iksP, brP, μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4MUXGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NMuxBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NMuxBootstrap__<iksP, brP, μ>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                 \
    template void NMuxBootstrap<iksP, brP, μ>(                             \
        typename brP::targetP::T* const out,                               \
        const typename iksP::domainP::T* const inc,                        \
        const typename iksP::domainP::T* const in1,                        \
        const typename iksP::domainP::T* const in0, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
INST(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::lvl2param::μ);
#undef INST
template <class brP, typename brP::targetP::T μ, class iksP>
void NMuxBootstrap(typename iksP::targetP::T* const out,
                   const typename brP::domainP::T* const inc,
                   const typename brP::domainP::T* const in1,
                   const typename brP::domainP::T* const in0,
                   const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        cudaFuncSetAttribute(__NMuxBootstrapKB__<brP, μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __NMuxBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], one_trgsw_ntt_devs_lvl02[gpuNum],
                xai_ntt_devs_lvl02[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        cudaFuncSetAttribute(__NMuxBootstrapKB__<brP, μ, iksP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             MEM4MUXGATE_DYN<typename brP::targetP>);
        __NMuxBootstrapKB__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#else
    cudaFuncSetAttribute(__NMuxBootstrap__<brP, μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4MUXGATE_DYN<typename brP::targetP>);
    if constexpr (sizeof(typename brP::targetP::T) == 8) {
        __NMuxBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts_lvl02[gpuNum],
                ksk_devs_lvl20[gpuNum], *ntt_handlers_lvl02[gpuNum]);
    }
    else {
        __NMuxBootstrap__<brP, μ, iksP>
            <<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
               MEM4MUXGATE_DYN<typename brP::targetP>, st>>>(
                out, inc, in1, in0, bk_ntts[gpuNum], ksk_devs[gpuNum],
                *ntt_handlers[gpuNum]);
    }
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
    template void NMuxBootstrap<brP, μ, iksP>(                            \
        typename iksP::targetP::T* const out,                             \
        const typename brP::domainP::T* const inc,                        \
        const typename brP::domainP::T* const in1,                        \
        const typename brP::domainP::T* const in0, const cudaStream_t st, \
        const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
INST(TFHEpp::lvl02param, TFHEpp::lvl2param::μ, TFHEpp::lvl20param);
#undef INST

}  // namespace cufhe
