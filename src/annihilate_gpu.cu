#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <include/annihilate_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <key.hpp>
#include <trgsw.hpp>
#include <utils.hpp>

namespace cufhe {

extern std::vector<CuNTTHandler<>*> ntt_handlers;
extern std::vector<CuNTTHandler<TFHEpp::lvl2param::n>*> ntt_handlers_lvl02;

std::vector<NTTValue*> ahk_ntts_lvl1;
std::vector<NTTValue*> ahk_ntts_lvl2;

namespace {

template <class P>
constexpr bool is_lvl1_ring_v = P::n == TFHEpp::lvl1param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl1param::T);

template <class P>
constexpr bool is_lvl2_ring_v = P::n == TFHEpp::lvl2param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl2param::T);

template <class P>
std::vector<NTTValue*>& AnnihilateKeyStorage()
{
    if constexpr (is_lvl1_ring_v<P>)
        return ahk_ntts_lvl1;
    else if constexpr (is_lvl2_ring_v<P>)
        return ahk_ntts_lvl2;
    else
        static_assert(is_lvl1_ring_v<P> || is_lvl2_ring_v<P>,
                      "Unsupported annihilate parameter ring");
}

template <class P>
CuNTTHandler<P::n>* AnnihilateHandler(const int gpuNum)
{
    if constexpr (is_lvl1_ring_v<P>) {
        return reinterpret_cast<CuNTTHandler<P::n>*>(ntt_handlers[gpuNum]);
    }
    else if constexpr (is_lvl2_ring_v<P>) {
        return reinterpret_cast<CuNTTHandler<P::n>*>(
            ntt_handlers_lvl02[gpuNum]);
    }
    else {
        static_assert(is_lvl1_ring_v<P> || is_lvl2_ring_v<P>,
                      "Unsupported annihilate parameter ring");
    }
}

template <class P>
constexpr size_t EvalAutoKeyElements()
{
    return static_cast<size_t>(P::k) * P::l * (P::k + 1) * (P::n / 2);
}

template <class P>
constexpr size_t AnnihilateKeyElements()
{
    return static_cast<size_t>(P::nbit) * EvalAutoKeyElements<P>();
}

}  // namespace

#if defined(USE_FFT)
template <class P>
__global__ void __HalfTRGSWPolynomialToFFT__(NTTValue* const out,
                                            const typename P::T* const in,
                                            CuNTTHandler<P::n> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t half_n = N / 2;
#ifdef USE_GPU_FFT
    constexpr uint32_t fft_threads = half_n >> 1;
#else
    constexpr uint32_t fft_threads = half_n / (Degree<N>::opt / 2);
#endif

    __shared__ double2 sh_fft[half_n];

    const uint32_t tid = ThisThreadRankInBlock();
    const size_t row = blockIdx.x;
    const size_t in_index = row * N;
    const size_t out_index = row * half_n;
    constexpr double norm =
        1.0 /
        static_cast<double>(
            1ULL << (std::numeric_limits<typename P::T>::digits / 2)) /
        static_cast<double>(
            1ULL << (std::numeric_limits<typename P::T>::digits -
                     std::numeric_limits<typename P::T>::digits / 2));

    if (tid < half_n) {
        const double re = static_cast<double>(
                              static_cast<std::make_signed_t<typename P::T>>(
                                  in[in_index + tid])) *
                          norm;
        const double im = static_cast<double>(
                              static_cast<std::make_signed_t<typename P::T>>(
                                  in[in_index + tid + half_n])) *
                          norm;
        NTTValue folded = {re, im};
#ifdef USE_GPU_FFT
        folded *= __ldg(&ntt.twist_[tid]);
#endif
        sh_fft[tid] = folded;
    }
    __syncthreads();

    if (tid < fft_threads) {
#ifdef USE_GPU_FFT
        if constexpr (N == 1024) {
            GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
        }
        else if constexpr (N == 2048) {
            GPUFFTForward1024(sh_fft, ntt.forward_root_, tid);
        }
#else
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
    }
    else {
#ifdef USE_GPU_FFT
        for (int s = 0; s < 3; s++) __syncthreads();
#else
        for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
            __syncthreads();
#endif
    }

    if (tid < half_n) out[out_index + tid] = sh_fft[tid];
}
#endif  // USE_FFT

template <class P>
__global__ void __DivideTRLWEBy2__(typename P::T* const trlwe)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t total = (P::k + 1) * N;
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < total; i += bdim) trlwe[i] >>= 1;
}

template <class P>
__global__ void __TRLWEAddInPlace__(typename P::T* const out,
                                    const typename P::T* const addend)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t total = (P::k + 1) * N;
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < total; i += bdim) out[i] += addend[i];
}

template <class P>
__device__ inline void __AutomorphismPolynomial__(
    typename P::T* const out, const typename P::T* const in, const uint32_t d)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t Nmask = N - 1;
    constexpr uint32_t signmask = N;
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < N; i += bdim) {
        const uint32_t index = i * d;
        const typename P::T value = in[i];
        out[index & Nmask] = (index & signmask) ? -value : value;
    }
}

#if defined(USE_FFT)
template <class P>
__device__ inline typename P::T __TorusFromDouble__(const double value)
{
    if constexpr (sizeof(typename P::T) == 4) {
        return static_cast<typename P::T>(
            static_cast<int32_t>(llrint(value * 4294967296.0)));
    }
    else {
        return static_cast<typename P::T>(
            double_to_torus64(value * 18446744073709551616.0));
    }
}

template <class P>
__device__ inline void __ExternalProductPolyHalfTRGSWFFT__(
    typename P::T* const out, const typename P::T* const poly,
    const typename P::T* const auto_b, const NTTValue* const halftrgswfft,
    NTTValue* const sh_acc_ntt, const CuNTTHandler<P::n> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t half_n = N / 2;
    constexpr uint32_t num_threads = N / 2;

#ifdef USE_GPU_FFT
    constexpr uint32_t fft_threads = half_n >> 1;
#else
    constexpr uint32_t fft_threads =
        half_n / (Degree<N>::opt / 2);
#endif

    NTTValue* const sh_fft = &sh_acc_ntt[0];
    NTTValue* const sh_accum = &sh_acc_ntt[half_n];

    for (uint32_t i = tid; i < (P::k + 1) * half_n; i += num_threads)
        sh_accum[i] = {0.0, 0.0};
    __syncthreads();

    constexpr typename P::T decomp_offset = offsetgen<P>();
    constexpr int remaining_bits =
        std::numeric_limits<typename P::T>::digits - P::l * P::Bgbit;
    constexpr typename P::T roundoffset =
        remaining_bits > 0
            ? (static_cast<typename P::T>(1) << (remaining_bits - 1))
            : static_cast<typename P::T>(0);
    constexpr typename P::T decomp_mask =
        (static_cast<typename P::T>(1) << P::Bgbit) - 1;
    constexpr typename P::T decomp_half =
        static_cast<typename P::T>(1) << (P::Bgbit - 1);

    for (uint32_t digit = 0; digit < P::l; digit++) {
        if (tid < half_n) {
            typename P::T temp_re = poly[tid] + decomp_offset + roundoffset;
            typename P::T temp_im =
                poly[tid + half_n] + decomp_offset + roundoffset;
            const int32_t digit_re = static_cast<int32_t>(
                ((temp_re >>
                  (std::numeric_limits<typename P::T>::digits -
                   (digit + 1) * P::Bgbit)) &
                 decomp_mask) -
                decomp_half);
            const int32_t digit_im = static_cast<int32_t>(
                ((temp_im >>
                  (std::numeric_limits<typename P::T>::digits -
                   (digit + 1) * P::Bgbit)) &
                 decomp_mask) -
                decomp_half);
            NTTValue folded = {static_cast<double>(digit_re),
                               static_cast<double>(digit_im)};
#ifdef USE_GPU_FFT
            folded *= __ldg(&ntt.twist_[tid]);
#endif
            sh_fft[tid] = folded;
        }
        __syncthreads();

        if (tid < fft_threads) {
#ifdef USE_GPU_FFT
            if constexpr (N == 1024) {
                GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
            }
            else if constexpr (N == 2048) {
                GPUFFTForward1024(sh_fft, ntt.forward_root_, tid);
            }
#else
            NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
        }
        else {
#ifdef USE_GPU_FFT
            for (int s = 0; s < 3; s++) __syncthreads();
#else
            for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
                __syncthreads();
#endif
        }

        if (tid < half_n) {
            const NTTValue fft_val = sh_fft[tid];
            for (uint32_t out_k = 0; out_k <= P::k; out_k++) {
                const size_t key_offset =
                    (static_cast<size_t>(digit) * (P::k + 1) + out_k) *
                        half_n +
                    tid;
                const NTTValue key_val = __ldg(&halftrgswfft[key_offset]);
                sh_accum[out_k * half_n + tid] += fft_val * key_val;
            }
        }
        __syncthreads();
    }

    for (uint32_t k_idx = 0; k_idx <= P::k; k_idx++) {
        NTTValue* const sh_inv = &sh_accum[k_idx * half_n];
        if (tid < fft_threads) {
#ifdef USE_GPU_FFT
            if constexpr (N == 1024) {
                GPUFFTInverse512(sh_inv, ntt.inverse_root_, tid);
            }
            else if constexpr (N == 2048) {
                GPUFFTInverse1024(sh_inv, ntt.inverse_root_, tid);
            }
#else
            NSMFFT_inverse<HalfDegree<Degree<N>>>(sh_inv);
#endif
        }
        else {
#ifdef USE_GPU_FFT
            for (int s = 0; s < 3; s++) __syncthreads();
#else
            for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
                __syncthreads();
#endif
        }

        if (tid < half_n) {
            NTTValue val = sh_inv[tid];
#ifdef USE_GPU_FFT
            val *= __ldg(&ntt.untwist_[tid]);
#endif
            const typename P::T lo = __TorusFromDouble__<P>(val.x);
            const typename P::T hi = __TorusFromDouble__<P>(val.y);
            if (k_idx == P::k) {
                out[k_idx * N + tid] = auto_b[tid] - lo;
                out[k_idx * N + tid + half_n] =
                    auto_b[tid + half_n] - hi;
            }
            else {
                out[k_idx * N + tid] = -lo;
                out[k_idx * N + tid + half_n] = -hi;
            }
        }
        __syncthreads();
    }
}
#endif  // USE_FFT

template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<P>)
void __EvalAutoKernel__(typename P::T* const out,
                        const typename P::T* const in, const uint32_t d,
                        const NTTValue* const evalautokey,
                        const CuNTTHandler<P::n> ntt)
{
    static_assert(P::k == 1,
                  "CUDA EvalAuto currently supports GLWE dimension 1");
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<P>;
    auto* sh_acc_ntt = reinterpret_cast<NTTValue*>(dyn_sh);
    auto* auto_a = reinterpret_cast<typename P::T*>(dyn_sh + fft_bytes);
    auto* auto_b = auto_a + P::n;

    __AutomorphismPolynomial__<P>(auto_a, in, d);
    __AutomorphismPolynomial__<P>(auto_b, in + P::n, d);
    __syncthreads();

#if defined(USE_FFT)
    __ExternalProductPolyHalfTRGSWFFT__<P>(out, auto_a, auto_b, evalautokey,
                                           sh_acc_ntt, ntt);
#endif
}

template <class P>
__global__ void __CopyTRLWEBatch__(typename P::T* const out,
                                   const size_t out_stride,
                                   const typename P::T* const in,
                                   const size_t in_stride,
                                   const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;

    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t total = (P::k + 1) * P::n;
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_in = in + batch * in_stride;
    for (uint32_t i = tid; i < total; i += bdim) batch_out[i] = batch_in[i];
}

template <class P>
__global__ void __DivideTRLWEBy2Batch__(typename P::T* const trlwe,
                                        const size_t stride,
                                        const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;

    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t total = (P::k + 1) * P::n;
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_trlwe = trlwe + batch * stride;
    for (uint32_t i = tid; i < total; i += bdim) batch_trlwe[i] >>= 1;
}

template <class P>
__global__ void __TRLWEAddInPlaceBatch__(typename P::T* const out,
                                         const size_t out_stride,
                                         const typename P::T* const addend,
                                         const size_t addend_stride,
                                         const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;

    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t total = (P::k + 1) * P::n;
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_addend = addend + batch * addend_stride;
    for (uint32_t i = tid; i < total; i += bdim)
        batch_out[i] += batch_addend[i];
}

template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<P>)
void __EvalAutoBatchKernel__(typename P::T* const out,
                             const size_t out_stride,
                             const typename P::T* const in,
                             const size_t in_stride, const uint32_t d,
                             const NTTValue* const evalautokey,
                             const CuNTTHandler<P::n> ntt,
                             const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;

    static_assert(P::k == 1,
                  "CUDA EvalAuto currently supports GLWE dimension 1");
    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<P>;
    auto* sh_acc_ntt = reinterpret_cast<NTTValue*>(dyn_sh);
    auto* auto_a = reinterpret_cast<typename P::T*>(dyn_sh + fft_bytes);
    auto* auto_b = auto_a + P::n;

    const typename P::T* const batch_in = in + batch * in_stride;
    typename P::T* const batch_out = out + batch * out_stride;
    __AutomorphismPolynomial__<P>(auto_a, batch_in, d);
    __AutomorphismPolynomial__<P>(auto_b, batch_in + P::n, d);
    __syncthreads();

#if defined(USE_FFT)
    __ExternalProductPolyHalfTRGSWFFT__<P>(
        batch_out, auto_a, auto_b, evalautokey, sh_acc_ntt, ntt);
#endif
}

template <class P>
void AnnihilateKeyPolynomialGen(AnnihilateKeyPolynomial<P>& ahk,
                                const TFHEpp::Key<P>& key)
{
    for (uint32_t bit = 0; bit < P::nbit; bit++) {
        const uint32_t d = (1U << (bit + 1)) + 1U;
        for (uint32_t key_idx = 0; key_idx < P::k; key_idx++) {
            TFHEpp::Polynomial<P> partkey{};
            for (uint32_t i = 0; i < P::n; i++)
                partkey[i] = key[key_idx * P::n + i];

            TFHEpp::Polynomial<P> autokey{};
            TFHEpp::Automorphism<P>(autokey, partkey, d);
            TFHEpp::halftrgswSymEncrypt<P>(ahk[bit][key_idx], autokey, key);
        }
    }
}

template <class P>
void AnnihilateKeyPolynomialGen(AnnihilateKeyPolynomial<P>& ahk,
                                const TFHEpp::SecretKey& sk)
{
    AnnihilateKeyPolynomialGen<P>(ahk, sk.key.get<P>());
}

template <class P>
void AnnihilateKeyPolynomialToDevice(const AnnihilateKeyPolynomial<P>& ahk,
                                     const int gpuNum)
{
    static_assert(P::k == 1,
                  "CUDA annihilate currently supports GLWE dimension 1");
    static_assert(P::l̅ == 1 && P::l̅ₐ == 1,
                  "CUDA annihilate currently supports standard decomposition");
    auto& storage = AnnihilateKeyStorage<P>();
    constexpr uint32_t rows = P::nbit * P::k * P::l * (P::k + 1);
    constexpr size_t poly_elems = static_cast<size_t>(rows) * P::n;
    const size_t poly_bytes = poly_elems * sizeof(typename P::T);
    const size_t fft_bytes = AnnihilateKeyElements<P>() * sizeof(NTTValue);

    std::vector<typename P::T> packed(poly_elems);
    size_t row = 0;
    for (uint32_t bit = 0; bit < P::nbit; bit++) {
        for (uint32_t key_idx = 0; key_idx < P::k; key_idx++) {
            for (uint32_t digit = 0; digit < P::l; digit++) {
                for (uint32_t out_k = 0; out_k <= P::k; out_k++) {
                    const auto& poly = ahk[bit][key_idx][digit][out_k];
                    for (uint32_t i = 0; i < P::n; i++)
                        packed[row * P::n + i] = poly[i];
                    row++;
                }
            }
        }
    }

    storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        if (storage[i] != nullptr) CuSafeCall(cudaFree(storage[i]));
        CuSafeCall(cudaMalloc(&storage[i], fft_bytes));
        typename P::T* d_poly = nullptr;
        CuSafeCall(cudaMalloc(&d_poly, poly_bytes));
        CuSafeCall(cudaMemcpy(d_poly, packed.data(), poly_bytes,
                              cudaMemcpyHostToDevice));
#if defined(USE_FFT)
        __HalfTRGSWPolynomialToFFT__<P><<<rows, P::n / 2>>>(
            storage[i], d_poly, *AnnihilateHandler<P>(i));
#endif
        CuCheckError();
        CuSafeCall(cudaFree(d_poly));
    }
}

template <class P>
void DeleteAnnihilateKey(const int gpuNum)
{
    auto& storage = AnnihilateKeyStorage<P>();
    for (size_t i = 0; i < storage.size(); i++) {
        cudaSetDevice(i);
        cudaFree(storage[i]);
    }
    storage.clear();
}

template <class P>
void AnnihilateKeySwitchingWithWorkspace(
    typename P::T* const out, const typename P::T* const in,
    typename P::T* const evaledauto, const cudaStream_t st, const int gpuNum)
{
    static_assert(P::k == 1,
                  "CUDA annihilate currently supports GLWE dimension 1");
    cudaSetDevice(gpuNum);
    auto& storage = AnnihilateKeyStorage<P>();
    const NTTValue* const ahk = storage[gpuNum];
    auto* const handler = AnnihilateHandler<P>(gpuNum);

    constexpr size_t trlwe_bytes = (P::k + 1) * P::n * sizeof(typename P::T);
    CuSafeCall(cudaMemcpyAsync(out, in, trlwe_bytes, cudaMemcpyDeviceToDevice,
                               st));

    constexpr size_t evalauto_key_elems = EvalAutoKeyElements<P>();
    constexpr size_t shmem = MEM4HOMGATE<P> + 2 * P::n * sizeof(typename P::T);
    static bool evalauto_attribute_set = false;
    if (!evalauto_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __EvalAutoKernel__<P>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        evalauto_attribute_set = true;
    }

    for (uint32_t bit = 0; bit < P::nbit; bit++) {
        __DivideTRLWEBy2__<P><<<1, NUM_THREAD4HOMGATE<P>, 0, st>>>(out);
        const uint32_t d = (1U << (bit + 1)) + 1U;
        __EvalAutoKernel__<P><<<1, NUM_THREAD4HOMGATE<P>, shmem, st>>>(
            evaledauto, out, d, ahk + bit * evalauto_key_elems, *handler);
        __TRLWEAddInPlace__<P><<<1, NUM_THREAD4HOMGATE<P>, 0, st>>>(
            out, evaledauto);
    }

    CuCheckError();
}

template <class P>
void AnnihilateKeySwitchingBatchWithWorkspace(
    typename P::T* const out, const size_t out_stride,
    const typename P::T* const in, const size_t in_stride,
    typename P::T* const evaledauto, const size_t evaledauto_stride,
    const size_t batch_count, const cudaStream_t st, const int gpuNum)
{
    static_assert(P::k == 1,
                  "CUDA annihilate currently supports GLWE dimension 1");
    if (batch_count == 0) return;

    cudaSetDevice(gpuNum);
    auto& storage = AnnihilateKeyStorage<P>();
    const NTTValue* const ahk = storage[gpuNum];
    auto* const handler = AnnihilateHandler<P>(gpuNum);

    constexpr size_t evalauto_key_elems = EvalAutoKeyElements<P>();
    constexpr size_t shmem = MEM4HOMGATE<P> + 2 * P::n * sizeof(typename P::T);
    static bool evalauto_batch_attribute_set = false;
    if (!evalauto_batch_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __EvalAutoBatchKernel__<P>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        evalauto_batch_attribute_set = true;
    }

    __CopyTRLWEBatch__<P>
        <<<batch_count, NUM_THREAD4HOMGATE<P>, 0, st>>>(
            out, out_stride, in, in_stride, batch_count);
    for (uint32_t bit = 0; bit < P::nbit; bit++) {
        __DivideTRLWEBy2Batch__<P>
            <<<batch_count, NUM_THREAD4HOMGATE<P>, 0, st>>>(
                out, out_stride, batch_count);
        const uint32_t d = (1U << (bit + 1)) + 1U;
        __EvalAutoBatchKernel__<P>
            <<<batch_count, NUM_THREAD4HOMGATE<P>, shmem, st>>>(
                evaledauto, evaledauto_stride, out, out_stride, d,
                ahk + bit * evalauto_key_elems, *handler, batch_count);
        __TRLWEAddInPlaceBatch__<P>
            <<<batch_count, NUM_THREAD4HOMGATE<P>, 0, st>>>(
                out, out_stride, evaledauto, evaledauto_stride, batch_count);
    }

    CuCheckError();
}

template <class P>
void AnnihilateKeySwitching(typename P::T* const out,
                            const typename P::T* const in,
                            const cudaStream_t st, const int gpuNum)
{
    constexpr size_t trlwe_bytes = (P::k + 1) * P::n * sizeof(typename P::T);
    typename P::T* evaledauto = nullptr;
    CuSafeCall(cudaMalloc(&evaledauto, trlwe_bytes));
    AnnihilateKeySwitchingWithWorkspace<P>(out, in, evaledauto, st, gpuNum);
    CuSafeCall(cudaFree(evaledauto));
}

#define INST(P)                                                              \
    template void AnnihilateKeyPolynomialGen<P>(                             \
        AnnihilateKeyPolynomial<P>&, const TFHEpp::Key<P>&);                 \
    template void AnnihilateKeyPolynomialGen<P>(                             \
        AnnihilateKeyPolynomial<P>&, const TFHEpp::SecretKey&);              \
    template void AnnihilateKeyPolynomialToDevice<P>(                        \
        const AnnihilateKeyPolynomial<P>&, const int);                       \
    template void DeleteAnnihilateKey<P>(const int);                         \
    template void AnnihilateKeySwitchingWithWorkspace<P>(                    \
        typename P::T* const, const typename P::T* const, typename P::T* const, \
        const cudaStream_t, const int);                                       \
    template void AnnihilateKeySwitchingBatchWithWorkspace<P>(               \
        typename P::T* const, const size_t, const typename P::T* const,       \
        const size_t, typename P::T* const, const size_t, const size_t,       \
        const cudaStream_t, const int);                                      \
    template void AnnihilateKeySwitching<P>(                                 \
        typename P::T* const, const typename P::T* const, const cudaStream_t, \
        const int)

INST(TFHEpp::lvl1param);
INST(TFHEpp::AHlvl1param);
INST(TFHEpp::lvl2param);
INST(TFHEpp::AHlvl2param);

#undef INST

}  // namespace cufhe
