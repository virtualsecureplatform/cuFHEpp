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


#include <include/bootstrap_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/utils_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <limits>
#include <vector>
#include <algorithm>

namespace cufhe {

using namespace std;
using namespace TFHEpp;

vector<NTTValue*> bk_ntts;
vector<CuNTTHandler<>*> ntt_handlers;

#ifdef USE_FFT

// ============================================================================
// FFT mode: BSK conversion to Fourier domain
// ============================================================================

#ifdef USE_GPU_FFT

/**
 * __TRGSW2FFT__ (GPU-FFT variant): Convert BSK polynomial to Fourier domain
 * Uses fold + twist + GPUFFTForward512
 */
template<class P = TFHEpp::lvl1param>
__global__ void __TRGSW2FFT__(NTTValue* const bk_fft, const typename P::T* const bk,
                              CuNTTHandler<> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256

    __shared__ double2 sh_fft[HALF_N];

    const int in_index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * N) +
                         blockIdx.y * N;
    const int out_index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * HALF_N) +
                          blockIdx.y * HALF_N;

    const uint32_t tid = threadIdx.x;

    // Load + normalize + fold + twist
    constexpr double norm = 1.0 / 4294967296.0;  // 1/2^32
    if (tid < HALF_N) {
        double re = static_cast<double>(static_cast<int32_t>(bk[in_index + tid])) * norm;
        double im = static_cast<double>(static_cast<int32_t>(bk[in_index + tid + HALF_N])) * norm;
        double2 folded = {re, im};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
    } else {
        for (int s = 0; s < 5; s++) __syncthreads();
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
template<class P = TFHEpp::lvl1param>
__global__ void __TRGSW2FFT__(NTTValue* const bk_fft, const typename P::T* const bk,
                              CuNTTHandler<> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const int in_index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * N) +
                         blockIdx.y * N;
    const int out_index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * HALF_N) +
                          blockIdx.y * HALF_N;

    const uint32_t tid = threadIdx.x;

    constexpr double norm = 1.0 / 4294967296.0;  // 1 / 2^32
    if (tid < HALF_N) {
        sh_fft[tid] = {static_cast<double>(static_cast<int32_t>(bk[in_index + tid])) * norm,
                       static_cast<double>(static_cast<int32_t>(bk[in_index + tid + HALF_N])) * norm};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    } else {
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
    dim3 grid(1, (lvl1param::k+1) * lvl1param::l * (lvl1param::k+1), 1);
    dim3 block(num_threads);
    __TRGSW2FFT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    cudaMemcpyAsync(
        trgswntt.trgswhost.data(), trgswntt.trgswdevices[st.device_id()],
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
    // tfhe-rs FFT mode: twiddles are in __device__ memory, no per-GPU initialization needed
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        ntt_handlers.push_back(new CuNTTHandler<>());
        cudaDeviceSynchronize();
        CuCheckError();
    }
#endif
}

template<class P>
void BootstrappingKeyToNTT(const BootstrappingKey<P>& bk,
                           const int gpuNum)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;

    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        // FFT BSK: n_lwe * (k+1)*l * (k+1) * (N/2) double2 elements
        size_t fft_elems = static_cast<size_t>(P::domainP::n) *
                           (P::targetP::k+1) * P::targetP::l *
                           (P::targetP::k+1) * HALF_N;
        cudaMalloc((void**)&bk_ntts[i], sizeof(NTTValue) * fft_elems);

        typename P::targetP::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        // Grid: 1 x ((k+1)*l*(k+1)) x n_lwe
        dim3 grid(1, (P::targetP::k+1) * P::targetP::l * (P::targetP::k+1), P::domainP::n);
        dim3 block(N >> NTT_THREAD_UNITBIT);
        __TRGSW2FFT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                \
template void BootstrappingKeyToNTT<P>(const BootstrappingKey<P>& bk, \
                           const int gpuNum)
INST(TFHEpp::lvl01param);
#undef INST

#ifdef USE_KEY_BUNDLE
template<class P>
void BootstrappingKeyBundleToNTT(const BootstrappingKey<P>& bk,
                                  const int gpuNum)
{
    constexpr uint32_t HALF_N = P::targetP::n >> 1;
    constexpr uint32_t num_pairs = P::domainP::k * P::domainP::n / P::Addends;
    constexpr uint32_t bk_elements_per_pair = (1 << P::Addends) - 1;  // 3
    constexpr uint32_t trgsw_polys = (P::targetP::k + 1) * P::targetP::l * (P::targetP::k + 1);
    constexpr size_t total_fft_elems = static_cast<size_t>(num_pairs) * bk_elements_per_pair *
                                       trgsw_polys * HALF_N;

    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts[i], sizeof(NTTValue) * total_fft_elems);

        typename P::targetP::T* d_bk;
        size_t bk_byte_size = sizeof(bk);
        cudaMalloc((void**)&d_bk, bk_byte_size);
        cudaMemcpy(d_bk, bk.data(), bk_byte_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        // Grid: 1 x trgsw_polys x (num_pairs * bk_elements_per_pair)
        dim3 grid(1, trgsw_polys, num_pairs * bk_elements_per_pair);
        dim3 block(HALF_N);
        __TRGSW2FFT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                \
template void BootstrappingKeyBundleToNTT<P>(const BootstrappingKey<P>& bk, \
                           const int gpuNum)
INST(TFHEpp::lvl01param);
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

#else  // !USE_FFT

// ============================================================================
// NTT mode: BSK conversion to NTT domain (existing implementation)
// ============================================================================

template<class P = TFHEpp::lvl1param>
__global__ void __TRGSW2NTT__(NTTValue* const bk_ntt, const typename P::T* const bk,
                              CuNTTHandler<> ntt)
{
    __shared__ NTTValue sh_temp[P::n];
    const int index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * P::n) +
                      blockIdx.y * (P::k+1) * P::n + blockIdx.x * P::n;
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    // Load and convert each element: Torus -> NTT mod
    for (int i = tid; i < P::n; i += bdim) {
        sh_temp[i] = torus32_to_ntt_mod(bk[index + i]);
    }
    __syncthreads();
    // Forward NTT
    if constexpr (P::n == 1024) {
        SmallForwardNTT32_1024(sh_temp, ntt.forward_root_, tid);
    }
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
    dim3 grid(lvl1param::k+1, (lvl1param::k+1) * lvl1param::l, 1);
    dim3 block(num_threads);
    __TRGSW2NTT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    // Copy NTT results back to host (NTTValue sized)
    cudaMemcpyAsync(
        trgswntt.trgswhost.data(), trgswntt.trgswdevices[st.device_id()],
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
        ntt_handlers[i]->SetDevicePointers(i);  // Set device pointers for this GPU
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

template<class P>
void BootstrappingKeyToNTT(const BootstrappingKey<P>& bk,
                           const int gpuNum)
{
    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts[i], sizeof(NTTValue) * P::domainP::n *
                                            (P::targetP::k+1) * P::targetP::l *
                                            (P::targetP::k+1) * P::targetP::n);

        typename P::targetP::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(P::targetP::k+1, (P::targetP::k+1) * P::targetP::l, P::domainP::n);
        dim3 block(P::targetP::n >> NTT_THREAD_UNITBIT);
        __TRGSW2NTT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                \
template void BootstrappingKeyToNTT<P>(const BootstrappingKey<P>& bk, \
                           const int gpuNum)
INST(TFHEpp::lvl01param);
#undef INST

#ifdef USE_KEY_BUNDLE
template<class P>
void BootstrappingKeyBundleToNTT(const BootstrappingKey<P>& bk,
                                  const int gpuNum)
{
    constexpr uint32_t num_pairs = P::domainP::k * P::domainP::n / P::Addends;
    constexpr uint32_t bk_elements_per_pair = (1 << P::Addends) - 1;  // 3 for Addends=2
    constexpr uint32_t trgsw_polys = (P::targetP::k + 1) * P::targetP::l * (P::targetP::k + 1);
    constexpr size_t total_ntt_elems = num_pairs * bk_elements_per_pair *
                                       trgsw_polys * P::targetP::n;

    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts[i], sizeof(NTTValue) * total_ntt_elems);

        // Upload BK data and NTT-transform it
        // BK layout: bk[pair_idx][bk_elem] is a TRGSW
        // Each TRGSW has (k+1)*l * (k+1) polynomials of N elements
        typename P::targetP::T* d_bk;
        size_t bk_byte_size = sizeof(bk);
        cudaMalloc((void**)&d_bk, bk_byte_size);
        cudaMemcpy(d_bk, bk.data(), bk_byte_size, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        // Grid: (k+1) x ((k+1)*l) x (num_pairs * bk_elements_per_pair)
        dim3 grid(P::targetP::k + 1, (P::targetP::k + 1) * P::targetP::l,
                  num_pairs * bk_elements_per_pair);
        dim3 block(P::targetP::n >> NTT_THREAD_UNITBIT);
        __TRGSW2NTT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                \
template void BootstrappingKeyBundleToNTT<P>(const BootstrappingKey<P>& bk, \
                           const int gpuNum)
INST(TFHEpp::lvl01param);
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

#endif  // USE_FFT

// ============================================================================
// CMUX operation
// ============================================================================

#ifdef USE_FFT

#ifdef USE_GPU_FFT

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
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
                GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
            } else {
                for (int s = 0; s < 5; s++) __syncthreads();
            }

            int digit_linear = j * lvl1param::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
                #pragma unroll
                for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                    double2 bk_val = __ldg(&tgsw_fft[
                        ((lvl1param::k + 1) * digit_linear + out_k) * HALF_N + tid]);
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
            GPUFFTInverse512(sh_inv, ntt.inverse_root_, ntt.n_inverse_, tid);
        } else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            out[k_idx * N + tid]          = trlwe0[k_idx * N + tid] +
                static_cast<uint32_t>(static_cast<int64_t>(llrint(val.x * denorm)));
            out[k_idx * N + tid + HALF_N] = trlwe0[k_idx * N + tid + HALF_N] +
                static_cast<uint32_t>(static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}

#else  // !USE_GPU_FFT (tfhe-rs FFT CMUX)

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
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
            } else {
                for (int s = 0; s < 11; s++) __syncthreads();
            }

            int digit_linear = j * lvl1param::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
                #pragma unroll
                for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                    double2 bk_val = __ldg(&tgsw_fft[
                        ((lvl1param::k + 1) * digit_linear + out_k) * HALF_N + tid]);
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
        } else {
            for (int s = 0; s < 11; s++) __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            out[k_idx * N + tid]          = trlwe0[k_idx * N + tid] +
                static_cast<uint32_t>(static_cast<int64_t>(llrint(val.x * denorm)));
            out[k_idx * N + tid + HALF_N] = trlwe0[k_idx * N + tid + HALF_N] +
                static_cast<uint32_t>(static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const NTTValue* const tgsw_ntt,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = lvl1param::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512

    extern __shared__ NTTValue sh_cmux[];
    NTTValue* const sh_work = &sh_cmux[0];         // Working buffer for NTT (N elements)
    NTTValue* const sh_accum = &sh_cmux[N];        // Accumulated results ((k+1)*N elements)

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
                    sh_work[i] = (digit_val < 0) ? (small_ntt::P + digit_val) : static_cast<uint32_t>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                SmallForwardNTT32_1024(sh_work, ntt.forward_root_, tid);
            } else {
                for (int s = 0; s < 5; s++) __syncthreads();
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
                        NTTValue bk_val = __ldg(&tgsw_ntt[(((lvl1param::k + 1) * digit_linear + out_k) << lvl1param::nbit) + i]);
                        sh_accum[out_k * N + i] = small_mod_add(sh_accum[out_k * N + i], small_mod_mult(ntt_val, bk_val));
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT on accumulated results, modswitch, add to trlwe0, write out
    // Operate directly on sh_accum to avoid copying to sh_work
    for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
        NTTValue* const sh_ntt_buf = &sh_accum[k_idx * N];

        // Inverse NTT directly on accumulator buffer
        if (tid < NUM_THREADS) {
            SmallInverseNTT32_1024(sh_ntt_buf, ntt.inverse_root_, ntt.n_inverse_, tid);
        } else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Convert with modulus switching and add to trlwe0
        constexpr uint32_t half_mod = small_ntt::P / 2;
        if (tid < NUM_THREADS) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                uint32_t val = sh_ntt_buf[i];
                int32_t signed_val = (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P) : static_cast<int32_t>(val);
                out[k_idx * N + i] = trlwe0[k_idx * N + i] + ntt_mod_to_torus32(signed_val);
            }
        }
        __syncthreads();
    }
}

#endif  // USE_FFT

template <class brP, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __Bootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in,
    const typename brP::targetP::T mu, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T tlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    __BlindRotate__<brP>(tlwe,in,mu,bk,ntt);
    KeySwitch<iksP>(out, tlwe, ksk);
    __threadfence();
}

// template <class iksP, class bkP>
// __global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __Bootstrap__(
//     typename iksP::domainP::T* const out, const typename iksP::domainP::T* const in,
//     const typename bkP::targetP::T mu, const NTTValue* const bk,
//     const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
// {
//     __shared__ typename bkP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 

//     KeySwitch<iksP>(tlwe, in, ksk);
//     __threadfence();
//     __BlindRotate__<bkP>(out,tlwe,mu,bk,ntt);
//     __threadfence();
// }

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename P::targetP>) void __BlindRotateGlobal__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
    const TFHEpp::lvl1param::T mu, const NTTValue* const bk, const CuNTTHandler<> ntt)
{
    __BlindRotate__<P>(out, in, mu, bk, ntt);
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __SEIandBootstrap2TRLWE__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
    const TFHEpp::lvl1param::T mu, const NTTValue* const bk, const TFHEpp::lvl0param::T* const ksk,
    const CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
#ifdef USE_FFT
    // Shared memory layout for FFT mode:
    // Accumulate uses (N/2 + (k+1)*N/2) double2 = (k+2)*N/2 double2
    // After that: tlwe needs (k+1)*N uint32_t, tlwelvl0 needs lvl0 TLWE uint32_t
    // Convert byte offset: (k+2)*N/2 double2 = (k+2)*N/2 * 16 bytes
    // In NTTValue (double2) units: (k+2)*N/2
    constexpr size_t acc_ntt_elems = (lvl1param::k + 2) * (lvl1param::n / 2);
    NTTValue* sh_acc_ntt = &sh[0];
    TFHEpp::lvl1param::T* tlwe =
        (TFHEpp::lvl1param::T*)&sh[acc_ntt_elems];
    // tlwe occupies (k+1)*N uint32_t = (k+1)*N*4 bytes = (k+1)*N/4 double2 elements
    constexpr size_t tlwe_in_nttval = ((lvl1param::k + 1) * lvl1param::n * sizeof(uint32_t) + sizeof(NTTValue) - 1) / sizeof(NTTValue);
    lvl0param::T* tlwelvl0 =
        (lvl0param::T*)&sh[acc_ntt_elems + tlwe_in_nttval];
#else
    // Shared memory layout for NTT mode (all uint32_t / NTTValue):
    // [0, (k+2)*N): sh_acc_ntt for Accumulate (working buf + accum)
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE with (k+1)*N elements)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE)
    NTTValue* sh_acc_ntt = &sh[0];
    TFHEpp::lvl1param::T* tlwe =
        (TFHEpp::lvl1param::T*)&sh[(lvl1param::k + 2) * lvl1param::n];
    lvl0param::T* tlwelvl0 =
        (lvl0param::T*)&sh[(lvl1param::k + 2 + lvl1param::k + 1) * lvl1param::n];
#endif

    KeySwitch<lvl10param>(tlwelvl0, in, ksk);
    __syncthreads();

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register uint32_t bar = 2 * TFHEpp::lvl1param::n - (tlwelvl0[TFHEpp::lvl0param::k * TFHEpp::lvl0param::n] >> (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 1 - TFHEpp::lvl1param::nbit));
    RotatedTestVector<lvl1param>(tlwe, bar, mu);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {
        constexpr typename TFHEpp::lvl0param::T roundoffset =
                1ULL << (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 2 -
                        TFHEpp::lvl1param::nbit);
        bar = modSwitchFromTorus<lvl01param>(tlwelvl0[i]+roundoffset);
#ifdef USE_FFT
        constexpr size_t trgsw_fft_size = (lvl1param::k+1) * lvl1param::l *
                                           (lvl1param::k+1) * (lvl1param::n / 2);
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar,
                   bk + i * trgsw_fft_size, ntt);
#else
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
#endif
    }
    __syncthreads();
    for (int i = 0; i < (lvl1param::k+1) * lvl1param::n; i++) {
        out[i] = tlwe[i];
    }
    __threadfence();
}

template<class P, uint index>
__device__ inline void __SampleExtractIndex__(typename P::T* const res, const typename P::T* const in){
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint nmask = P::n-1; 
    for (uint i = tid; i <= P::k*P::n; i += bdim) {
        if (i == P::k*P::n){
            res[P::k*P::n] = in[P::k*P::n+index];
        }else {
            const uint k = i >> P::nbit; 
            const uint n = i & nmask;
            if (n  <= index) res[i] = in[k*P::n + index - n];
            else res[i] = -in[k*P::n + P::n + index-n];
        }
    }
}

template <class iksP, class brP, std::make_signed_t<typename brP::targetP::T> μ, int casign, int cbsign, std::make_signed_t<typename iksP::domainP::T> offset>
__device__ inline void __HomGate__(typename brP::targetP::T* const out,
                                   const typename iksP::domainP::T* const in0,
                                   const typename iksP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 

    IdentityKeySwitchPreAdd<iksP, casign, cbsign, offset>(tlwe, in0, in1, ksk);
    __syncthreads();

    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    __BlindRotate__<brP>(trlwe, tlwe, μ, bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out,trlwe);
    __threadfence();
}

template <class brP, std::make_signed_t<typename brP::targetP::T> μ, class iksP, int casign, int cbsign, std::make_signed_t<typename brP::domainP::T> offset>
__device__ inline void __HomGate__(typename iksP::targetP::T* const out,
                                   const typename brP::domainP::T* const in0,
                                   const typename brP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, casign,cbsign,offset>(trlwe,in0,in1,bk,ntt);

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

#ifdef USE_KEY_BUNDLE
// Key-bundle HomGate variants (IKS-BR order)
template <class iksP, class brP, std::make_signed_t<typename brP::targetP::T> μ, int casign, int cbsign, std::make_signed_t<typename iksP::domainP::T> offset>
__device__ inline void __HomGateKeyBundle__(typename brP::targetP::T* const out,
                                   const typename iksP::domainP::T* const in0,
                                   const typename iksP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const NTTValue* const one_trgsw_ntt,
                                   const NTTValue* const xai_ntt,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1];

    IdentityKeySwitchPreAdd<iksP, casign, cbsign, offset>(tlwe, in0, in1, ksk);
    __syncthreads();

    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotateKeyBundle__<brP>(trlwe, tlwe, μ, bk, one_trgsw_ntt, xai_ntt, ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out,trlwe);
    __threadfence();
}

// Key-bundle HomGate variants (BR-IKS order)
template <class brP, std::make_signed_t<typename brP::targetP::T> μ, class iksP, int casign, int cbsign, std::make_signed_t<typename brP::domainP::T> offset>
__device__ inline void __HomGateKeyBundle__(typename iksP::targetP::T* const out,
                                   const typename brP::domainP::T* const in0,
                                   const typename brP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const NTTValue* const one_trgsw_ntt,
                                   const NTTValue* const xai_ntt,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAddKeyBundle__<brP, casign,cbsign,offset>(trlwe,in0,in1,bk,one_trgsw_ntt,xai_ntt,ntt);

    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}
#endif  // USE_KEY_BUNDLE

// br iks ver.
template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NandBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XnorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -2, -2, -2 * brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 2, 2, 2 * brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,ntt);
    __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Add all (k+1)*n elements of the TRLWE ciphertexts
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] += trlwe0[i];
    }
    // Add μ to b[0] (the constant term of polynomial b)
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] += μ;
    }

    __syncthreads();

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,ntt);
    __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Negate and add all (k+1)*n elements of the TRLWE ciphertexts
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] = -trlwe1[i] - trlwe0[i];
    }
    // Subtract μ from b[0] (the constant term of polynomial b)
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] -= μ;
    }

    __syncthreads();

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// iks br ver.
template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NandBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XnorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -2, -2, -2 * iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 2, 2, 2 * iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndNYBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndYNBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrNYBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrYNBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

#ifdef USE_KEY_BUNDLE
// Key-bundle gate kernels (BR-IKS order)
#define DEFINE_KB_GATE_BRIKS(Name, casign_val, cbsign_val, offset_expr) \
template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param> \
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __##Name##BootstrapKB__( \
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk, \
    const NTTValue* const one_trgsw_ntt, const NTTValue* const xai_ntt, \
    const CuNTTHandler<> ntt) \
{ \
    __HomGateKeyBundle__<brP, μ, iksP, casign_val, cbsign_val, offset_expr>(out, in0, in1, bk, ksk, one_trgsw_ntt, xai_ntt, ntt); \
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
#define DEFINE_KB_GATE_IKSBR(Name, casign_val, cbsign_val, offset_expr) \
template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ> \
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __##Name##BootstrapKB__( \
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk, \
    const NTTValue* const one_trgsw_ntt, const NTTValue* const xai_ntt, \
    const CuNTTHandler<> ntt) \
{ \
    __HomGateKeyBundle__<iksP, brP, μ, casign_val, cbsign_val, offset_expr>(out, in0, in1, bk, ksk, one_trgsw_ntt, xai_ntt, ntt); \
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
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrapKB__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const NTTValue* const one_trgsw_ntt,
    const NTTValue* const xai_ntt, const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAddKeyBundle__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,one_trgsw_ntt,xai_ntt,ntt);
    __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,one_trgsw_ntt,xai_ntt,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] += trlwe0[i];
    }
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] += μ;
    }
    __syncthreads();

    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// Key-bundle NMux (BR-IKS order)
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrapKB__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const NTTValue* const one_trgsw_ntt,
    const NTTValue* const xai_ntt, const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAddKeyBundle__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,one_trgsw_ntt,xai_ntt,ntt);
    __BlindRotatePreAddKeyBundle__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,one_trgsw_ntt,xai_ntt,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] = -trlwe1[i] - trlwe0[i];
    }
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] -= μ;
    }
    __syncthreads();

    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// Key-bundle Mux (IKS-BR order)
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrapKB__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const NTTValue* const one_trgsw_ntt,
    const NTTValue* const xai_ntt, const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1];

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotateKeyBundle__<brP>(tlwe1,tlwelvl0,μ,bk,one_trgsw_ntt,xai_ntt,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotateKeyBundle__<brP>(tlwe0,tlwelvl0,μ,bk,one_trgsw_ntt,xai_ntt,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);

    __syncthreads();

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] += tlwe1[i];
    }
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] += μ;
    }
    __threadfence();
}

// Key-bundle NMux (IKS-BR order)
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrapKB__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const NTTValue* const one_trgsw_ntt,
    const NTTValue* const xai_ntt, const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1];

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotateKeyBundle__<brP>(tlwe1,tlwelvl0,μ,bk,one_trgsw_ntt,xai_ntt,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotateKeyBundle__<brP>(tlwe0,tlwelvl0,μ,bk,one_trgsw_ntt,xai_ntt,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);

    __syncthreads();

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] = -out[i] - tlwe1[i];
    }
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] -= μ;
    }
    __threadfence();
}
#endif  // USE_KEY_BUNDLE

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CopyBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k*P::n; i += bdim) 
        out[i] = in[i];
    __syncthreads();
    __threadfence();
}

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __NotBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k*P::n; i += bdim) 
        out[i] = -in[i];
    __syncthreads();
    __threadfence();
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1]; 

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n]; 
    __BlindRotate__<brP>(tlwe1,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n]; 
    __BlindRotate__<brP>(tlwe0,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);
    
    __syncthreads();

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Add TLWE ciphertexts: k*n elements for 'a' parts plus 1 for 'b'
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] += tlwe1[i];
    }
    // Add μ to b (the last element)
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] += μ;
    }
    __threadfence();
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1];

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotate__<brP>(tlwe1,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotate__<brP>(tlwe0,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);

    __syncthreads();


    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Negate and add TLWE ciphertexts: k*n elements for 'a' parts plus 1 for 'b'
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] = -out[i] - tlwe1[i];
    }
    // Subtract μ from b (the last element)
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] -= μ;
    }
    __threadfence();
}

void Bootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in,
               const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    __Bootstrap__<lvl01param,lvl10param><<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, 0, st>>>(
        out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CMUXNTTkernel(TFHEpp::lvl1param::T* const res, const NTTValue* const cs,
                   TFHEpp::lvl1param::T* const c1,
                   TFHEpp::lvl1param::T* const c0, cudaStream_t st,
                   const int gpuNum)
{
    constexpr size_t shmem_size = (TFHEpp::lvl1param::k + 2) * TFHEpp::lvl1param::n * sizeof(NTTValue);
    cudaFuncSetAttribute(__CMUXNTT__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_size);
    __CMUXNTT__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, shmem_size, st>>>(
        res, cs, c1, c0, *ntt_handlers[gpuNum]);
    CuCheckError();
}

void BootstrapTLWE2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
                         const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__BlindRotateGlobal__<TFHEpp::lvl01param>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __BlindRotateGlobal__<TFHEpp::lvl01param><<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in, mu, bk_ntts[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void SEIandBootstrap2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
                           const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    // Shared memory layout (all uint32_t / NTTValue):
    // [0, (k+2)*N): Accumulate working space
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE, ceil((n0+1)/2+1) uint32_ts)
    constexpr size_t shmem_elems = (lvl1param::k + 2 + lvl1param::k + 1) * lvl1param::n +
                                   (lvl0param::n + 1) / 2 + 1;
    constexpr size_t shmem_size = shmem_elems * sizeof(NTTValue);
    cudaFuncSetAttribute(
        __SEIandBootstrap2TRLWE__, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size);
    __SEIandBootstrap2TRLWE__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>,
                              shmem_size, st>>>
        (out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

template<class brP, typename brP::targetP::T μ, class iksP>
void NandBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NandBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NandBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrap__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NandBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void NandBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NandBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NandBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NandBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrYNBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrYNBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrYNBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrYNBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrNYBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrNYBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrNYBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__OrNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__OrNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrNYBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndYNBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndYNBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndYNBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndYNBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndYNBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndYNBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndNYBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndNYBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndNYBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndNYBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__AndNYBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__AndNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndNYBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void NorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void NorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void XorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__XorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__XorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void XorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void XorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__XorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__XorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void XorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void XnorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__XnorBootstrapKB__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrapKB__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__XnorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void XnorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void XnorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__XnorBootstrapKB__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        one_trgsw_ntt_devs[gpuNum], xai_ntt_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__XnorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void XnorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class P>
void CopyBootstrap(typename P::T* const out, const typename P::T* const in,
                   const cudaStream_t st, const int gpuNum)
{
    __CopyBootstrap__<P><<<1, std::min(P::n + 1,NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0, st>>>(out, in);
    CuCheckError();
}
#define INST(P) \
template void CopyBootstrap<P>(typename P::T* const out, const typename P::T* const in, \
                   const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template<class P>
void NotBootstrap(typename P::T* const out, const typename P::T* const in,
                  const cudaStream_t st, const int gpuNum)
{
    __NotBootstrap__<P><<<1, std::min(P::n + 1,NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0, st>>>(out, in);
    CuCheckError();
}
#define INST(P) \
template void NotBootstrap<P>(typename P::T* const out, const typename P::T* const in, \
                   const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void MuxBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
                  const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__MuxBootstrapKB__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrapKB__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], one_trgsw_ntt_devs[gpuNum],
                             xai_ntt_devs[gpuNum], *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__MuxBootstrap__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrap__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void MuxBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc, \
                                       const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, \
                                       const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void MuxBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__MuxBootstrapKB__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrapKB__<iksP,brP,μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], one_trgsw_ntt_devs[gpuNum],
                             xai_ntt_devs[gpuNum], *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__MuxBootstrap__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrap__<iksP,brP,μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void MuxBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc, \
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, \
                  const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class iksP, class brP, typename brP::targetP::T μ>
void NMuxBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NMuxBootstrapKB__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrapKB__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], one_trgsw_ntt_devs[gpuNum],
                             xai_ntt_devs[gpuNum], *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NMuxBootstrap__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NMuxBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc, \
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0,  \
                  const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST
template<class brP, typename brP::targetP::T μ, class iksP>
void NMuxBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
                  const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
#ifdef USE_KEY_BUNDLE
    cudaFuncSetAttribute(__NMuxBootstrapKB__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrapKB__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], one_trgsw_ntt_devs[gpuNum],
                             xai_ntt_devs[gpuNum], *ntt_handlers[gpuNum]);
#else
    cudaFuncSetAttribute(__NMuxBootstrap__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrap__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
#endif
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NMuxBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc, \
                                       const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, \
                                       const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST

}  // namespace cufhe
