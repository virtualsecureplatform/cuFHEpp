#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <tfhe/key.hpp>
#include <tfhe/trgsw.hpp>

namespace cufhe {

extern std::vector<NTTValue*> bk_ntts;
extern std::vector<NTTValue*> bk_ntts_lvl02;
extern std::vector<CuNTTHandler<>*> ntt_handlers;
extern std::vector<CuNTTHandler<TFHEpp::lvl2param::n>*> ntt_handlers_lvl02;

std::vector<NTTValue*> cbsk_ntts_lvl1;
std::vector<NTTValue*> cbsk_ntts_lvl2;

namespace {

__host__ __device__ constexpr uint32_t BitsNeeded(const uint32_t data)
{
    uint32_t value = data;
    uint32_t bits = 0;
    while (value != 0) {
        bits++;
        value >>= 1;
    }
    return bits;
}

template <class P>
constexpr bool is_lvl1_ring_v = P::n == TFHEpp::lvl1param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl1param::T);

template <class P>
constexpr bool is_lvl2_ring_v = P::n == TFHEpp::lvl2param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl2param::T);

template <class P>
std::vector<NTTValue*>& CBswitchingKeyStorage()
{
    if constexpr (is_lvl1_ring_v<P>)
        return cbsk_ntts_lvl1;
    else if constexpr (is_lvl2_ring_v<P>)
        return cbsk_ntts_lvl2;
    else
        static_assert(is_lvl1_ring_v<P> || is_lvl2_ring_v<P>,
                      "Unsupported CB switching key ring");
}

template <class P>
CuNTTHandler<P::n>* RingHandler(const int gpuNum)
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
                      "Unsupported ring handler");
    }
}

template <class brP>
NTTValue* BootstrappingKeyStorage(const int gpuNum)
{
    if constexpr (brP::targetP::n == TFHEpp::lvl2param::n)
        return bk_ntts_lvl02[gpuNum];
    else
        return bk_ntts[gpuNum];
}

#ifdef USE_KEY_BUNDLE
template <class P>
NTTValue* OneTRGSWStorage(const int gpuNum)
{
    if constexpr (P::n == TFHEpp::lvl2param::n)
        return one_trgsw_ntt_devs_lvl02[gpuNum];
    else
        return one_trgsw_ntt_devs[gpuNum];
}

template <class P>
NTTValue* XaiStorage(const int gpuNum)
{
    if constexpr (P::n == TFHEpp::lvl2param::n)
        return xai_ntt_devs_lvl02[gpuNum];
    else
        return xai_ntt_devs[gpuNum];
}
#endif

template <class iksP>
typename iksP::targetP::T* KeySwitchingKeyStorage(const int gpuNum)
{
    if constexpr (iksP::domainP::n == TFHEpp::lvl2param::n)
        return reinterpret_cast<typename iksP::targetP::T*>(
            ksk_devs_lvl20[gpuNum]);
    else
        return reinterpret_cast<typename iksP::targetP::T*>(
            ksk_devs[gpuNum]);
}

template <class P>
__host__ __device__
constexpr uint32_t TRGSWRows()
{
    return P::k * P::lₐ * P::l̅ₐ + P::l * P::l̅;
}

template <class P>
__host__ __device__
constexpr size_t TRGSWFFTElements()
{
#if defined(USE_FFT)
    constexpr uint32_t transform_size = P::n / 2;
#else
    constexpr uint32_t transform_size = P::n;
#endif
    return static_cast<size_t>(TRGSWRows<P>()) * (P::k + 1) *
           transform_size;
}

template <class P>
constexpr size_t CBswitchingKeyElements()
{
    return static_cast<size_t>(P::k) * TRGSWFFTElements<P>();
}

template <class P>
__host__ __device__ constexpr size_t TRLWEElements()
{
    return static_cast<size_t>(P::k + 1) * P::n;
}

template <class P>
constexpr size_t TRGSWElements()
{
    return static_cast<size_t>(TRGSWRows<P>()) * TRLWEElements<P>();
}

template <class P, bool IsNonce>
__device__ constexpr typename P::T ExternalProductOffset()
{
    constexpr uint32_t levels = IsNonce ? P::lₐ : P::l;
    constexpr uint32_t bgbit = IsNonce ? P::Bgₐbit : P::Bgbit;
    constexpr typename P::T bg = IsNonce ? P::Bgₐ : P::Bg;
    typename P::T offset = 0;
    for (uint32_t i = 1; i <= levels; i++)
        offset += (bg / 2) *
                  (static_cast<typename P::T>(1)
                   << (std::numeric_limits<typename P::T>::digits -
                       i * bgbit));
    return offset;
}

}  // namespace

#if defined(USE_FFT)
template <class P>
__global__ void __TRGSWPolynomialToFFT__(NTTValue* const out,
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
        GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
#else
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
    }
    else {
#ifdef USE_GPU_FFT
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
#else
        for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
            __syncthreads();
#endif
    }

    if (tid < half_n) out[out_index + tid] = sh_fft[tid];
}
#else
template <class P>
__global__ void __TRGSWPolynomialToNTT__(NTTValue* const out,
                                        const typename P::T* const in,
                                        CuNTTHandler<P::n> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t num_threads = N / 2;
    __shared__ NTTValue sh_ntt[N];

    const uint32_t tid = ThisThreadRankInBlock();
    const size_t row = blockIdx.x;
    const size_t row_offset = row * N;

    if (tid < num_threads) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const uint32_t i = tid + e * num_threads;
            sh_ntt[i] = torus_to_ntt_mod<N>(in[row_offset + i]);
        }
    }
    __syncthreads();

    if (tid < num_threads) {
        SmallForwardNTT<P::nbit>(sh_ntt, ntt.forward_root_, tid);
    }
    else {
        for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
            __syncthreads();
    }

    if (tid < num_threads) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const uint32_t i = tid + e * num_threads;
            out[row_offset + i] = sh_ntt[i];
        }
    }
}
#endif  // USE_FFT

template <class P, uint32_t num_out>
__device__ inline typename P::T CBTestVectorValue(const uint32_t index)
{
    constexpr uint32_t bitwidth = BitsNeeded(num_out - 1);
    constexpr uint32_t mask = (1U << bitwidth) - 1;
    const uint32_t digit = index & mask;
    return static_cast<typename P::T>(1)
           << (std::numeric_limits<typename P::T>::digits -
               (digit + 1) * P::Bgbit - 1);
}

template <class P, uint32_t num_out>
__device__ inline typename P::T RotatedCBTestVectorValue(
    const uint32_t index, const uint32_t exponent)
{
    constexpr uint32_t N = P::n;
    if (exponent == 0 || exponent == 2 * N)
        return CBTestVectorValue<P, num_out>(index);
    if (exponent < N) {
        const uint32_t src = (index - exponent) & (N - 1);
        const typename P::T value = CBTestVectorValue<P, num_out>(src);
        return index < exponent ? -value : value;
    }

    const uint32_t shifted = exponent - N;
    const uint32_t src = (index - shifted) & (N - 1);
    const typename P::T value = CBTestVectorValue<P, num_out>(src);
    return index < shifted ? value : -value;
}

template <class brP, uint32_t num_out>
__device__ inline uint32_t CBModSwitch(
    const typename brP::domainP::T value)
{
    constexpr uint32_t bitwidth = BitsNeeded(num_out - 1);
    return (value >>
            (std::numeric_limits<typename brP::domainP::T>::digits - 1 -
             brP::targetP::nbit + bitwidth))
           << bitwidth;
}

template <class brP, uint32_t num_out>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>)
void __BlindRotateCBKernel__(typename brP::targetP::T* const out,
                             const typename brP::domainP::T* const in,
                             const NTTValue* const bk,
#ifdef USE_KEY_BUNDLE
                             const NTTValue* const one_trgsw_ntt,
                             const NTTValue* const xai_ntt,
#endif
                             const CuNTTHandler<brP::targetP::n> ntt)
{
    using targetP = typename brP::targetP;
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t total = (targetP::k + 1) * N;
    const uint32_t bdim = ThisBlockSize();
    constexpr typename brP::domainP::T roundoffset =
        static_cast<typename brP::domainP::T>(1)
        << (std::numeric_limits<typename brP::domainP::T>::digits - 2 -
            targetP::nbit + BitsNeeded(num_out - 1));
    constexpr uint32_t mod_shift =
        std::numeric_limits<typename brP::domainP::T>::digits - 1 -
        targetP::nbit;
    constexpr uint32_t input_count = brP::domainP::k * brP::domainP::n;
    using SignedDomainT = std::make_signed_t<typename brP::domainP::T>;
    __shared__ long long residual[NUM_THREAD4HOMGATE<targetP>];

    long long local_residual = 0;
    for (uint32_t i = tid; i < input_count; i += bdim) {
        const uint32_t moded = CBModSwitch<brP, num_out>(in[i] + roundoffset);
        const typename brP::domainP::T rounded =
            static_cast<typename brP::domainP::T>(moded << mod_shift);
        local_residual += static_cast<SignedDomainT>(in[i] - rounded);
    }
    residual[tid] = local_residual;
    __syncthreads();
    for (uint32_t stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) residual[tid] += residual[tid + stride];
        __syncthreads();
    }

    const typename brP::domainP::T corrected_b =
        static_cast<typename brP::domainP::T>(
            in[brP::domainP::k * brP::domainP::n] - residual[0] / 2 +
            roundoffset);
    const uint32_t bbar = 2 * N - CBModSwitch<brP, num_out>(corrected_b);
    for (uint32_t i = tid; i < total; i += bdim) {
        const uint32_t k_idx = i >> targetP::nbit;
        const uint32_t n = i & (N - 1);
        out[i] = k_idx == targetP::k
                     ? RotatedCBTestVectorValue<targetP, num_out>(n, bbar)
                     : 0;
    }
    __syncthreads();

    extern __shared__ NTTValue sh[];
    NTTValue* const sh_acc_ntt = &sh[0];

#ifdef USE_KEY_BUNDLE
    constexpr uint32_t num_pairs =
        brP::domainP::k * brP::domainP::n / brP::Addends;
#ifdef USE_FFT
    constexpr size_t trgsw_size =
        (targetP::k + 1) * (targetP::k + 1) * targetP::l * (targetP::n / 2);
#else
    constexpr size_t trgsw_size =
        (targetP::k + 1) * (targetP::k + 1) * targetP::l * targetP::n;
#endif
    for (uint32_t i = 0; i < num_pairs; i++) {
        const uint32_t bara0 =
            CBModSwitch<brP, num_out>(in[2 * i] + roundoffset);
        const uint32_t bara1 =
            CBModSwitch<brP, num_out>(in[2 * i + 1] + roundoffset);
        const NTTValue* const bk0 = bk + i * 3 * trgsw_size;
        const NTTValue* const bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValue* const bk2 = bk + (i * 3 + 2) * trgsw_size;
        AccumulateKeyBundle<brP>(out, sh_acc_ntt, bara0, bara1, bk0, bk1, bk2,
                                 one_trgsw_ntt, xai_ntt, ntt);
    }
#else
#ifdef USE_FFT
    constexpr size_t trgsw_size =
        (targetP::k + 1) * targetP::l * (targetP::k + 1) * (targetP::n / 2);
#else
    constexpr size_t trgsw_size =
        (targetP::k + 1) * targetP::l * (targetP::k + 1) * targetP::n;
#endif
    for (uint32_t i = 0; i < brP::domainP::k * brP::domainP::n; i++) {
        const uint32_t abar =
            CBModSwitch<brP, num_out>(in[i] + roundoffset);
        if (abar == 0) continue;
        Accumulate<brP>(out, sh_acc_ntt, abar, bk + i * trgsw_size, ntt);
    }
#endif
}

template <class brP, uint32_t num_out>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>)
void __BlindRotateCBBatchKernel__(typename brP::targetP::T* const out,
                                  const size_t out_stride,
                                  const typename brP::domainP::T* const in,
                                  const size_t in_stride,
                                  const NTTValue* const bk,
#ifdef USE_KEY_BUNDLE
                                  const NTTValue* const one_trgsw_ntt,
                                  const NTTValue* const xai_ntt,
#endif
                                  const CuNTTHandler<brP::targetP::n> ntt,
                                  const size_t batch_count)
{
    using targetP = typename brP::targetP;
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;

    typename targetP::T* const batch_out = out + batch * out_stride;
    const typename brP::domainP::T* const batch_in = in + batch * in_stride;
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t total = (targetP::k + 1) * N;
    const uint32_t bdim = ThisBlockSize();
    constexpr typename brP::domainP::T roundoffset =
        static_cast<typename brP::domainP::T>(1)
        << (std::numeric_limits<typename brP::domainP::T>::digits - 2 -
            targetP::nbit + BitsNeeded(num_out - 1));
    constexpr uint32_t mod_shift =
        std::numeric_limits<typename brP::domainP::T>::digits - 1 -
        targetP::nbit;
    constexpr uint32_t input_count = brP::domainP::k * brP::domainP::n;
    using SignedDomainT = std::make_signed_t<typename brP::domainP::T>;
    __shared__ long long residual[NUM_THREAD4HOMGATE<targetP>];

    long long local_residual = 0;
    for (uint32_t i = tid; i < input_count; i += bdim) {
        const uint32_t moded =
            CBModSwitch<brP, num_out>(batch_in[i] + roundoffset);
        const typename brP::domainP::T rounded =
            static_cast<typename brP::domainP::T>(moded << mod_shift);
        local_residual += static_cast<SignedDomainT>(batch_in[i] - rounded);
    }
    residual[tid] = local_residual;
    __syncthreads();
    for (uint32_t stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) residual[tid] += residual[tid + stride];
        __syncthreads();
    }

    const typename brP::domainP::T corrected_b =
        static_cast<typename brP::domainP::T>(
            batch_in[brP::domainP::k * brP::domainP::n] -
            residual[0] / 2 + roundoffset);
    const uint32_t bbar = 2 * N - CBModSwitch<brP, num_out>(corrected_b);
    for (uint32_t i = tid; i < total; i += bdim) {
        const uint32_t k_idx = i >> targetP::nbit;
        const uint32_t n = i & (N - 1);
        batch_out[i] = k_idx == targetP::k
                           ? RotatedCBTestVectorValue<targetP, num_out>(n,
                                                                         bbar)
                           : 0;
    }
    __syncthreads();

    extern __shared__ NTTValue sh[];
    NTTValue* const sh_acc_ntt = &sh[0];

#ifdef USE_KEY_BUNDLE
    constexpr uint32_t num_pairs =
        brP::domainP::k * brP::domainP::n / brP::Addends;
#ifdef USE_FFT
    constexpr size_t trgsw_size =
        (targetP::k + 1) * (targetP::k + 1) * targetP::l * (targetP::n / 2);
#else
    constexpr size_t trgsw_size =
        (targetP::k + 1) * (targetP::k + 1) * targetP::l * targetP::n;
#endif
    for (uint32_t i = 0; i < num_pairs; i++) {
        const uint32_t bara0 =
            CBModSwitch<brP, num_out>(batch_in[2 * i] + roundoffset);
        const uint32_t bara1 =
            CBModSwitch<brP, num_out>(batch_in[2 * i + 1] + roundoffset);
        const NTTValue* const bk0 = bk + i * 3 * trgsw_size;
        const NTTValue* const bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValue* const bk2 = bk + (i * 3 + 2) * trgsw_size;
        AccumulateKeyBundle<brP>(batch_out, sh_acc_ntt, bara0, bara1, bk0,
                                 bk1, bk2, one_trgsw_ntt, xai_ntt, ntt);
    }
#else
#ifdef USE_FFT
    constexpr size_t trgsw_size =
        (targetP::k + 1) * targetP::l * (targetP::k + 1) * (targetP::n / 2);
#else
    constexpr size_t trgsw_size =
        (targetP::k + 1) * targetP::l * (targetP::k + 1) * targetP::n;
#endif
    for (uint32_t i = 0; i < brP::domainP::k * brP::domainP::n; i++) {
        const uint32_t abar =
            CBModSwitch<brP, num_out>(batch_in[i] + roundoffset);
        if (abar == 0) continue;
        Accumulate<brP>(batch_out, sh_acc_ntt, abar, bk + i * trgsw_size, ntt);
    }
#endif
}

template <class P, uint32_t num_out>
__device__ inline typename P::T SampleExtractValue(
    const typename P::T* const trlwe, const uint32_t index,
    const uint32_t tlwe_index)
{
    constexpr uint32_t N = P::n;
    if (tlwe_index == P::k * N) return trlwe[P::k * N + index];
    const uint32_t k_idx = tlwe_index >> P::nbit;
    const uint32_t n = tlwe_index & (N - 1);
    return n <= index ? trlwe[k_idx * N + index - n]
                      : -trlwe[k_idx * N + N + index - n];
}

template <class P, uint32_t num_out>
__global__ void __InvSampleExtractCBKernel__(typename P::T* const out,
                                             const typename P::T* const acc)
{
    const uint32_t digit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t N = P::n;
    constexpr size_t trlwe_elems = TRLWEElements<P>();
    typename P::T* const trlwe = out + static_cast<size_t>(digit) * trlwe_elems;

    for (uint32_t i = tid; i < trlwe_elems; i += bdim) {
        const uint32_t k_idx = i >> P::nbit;
        const uint32_t n = i & (N - 1);
        if (k_idx < P::k) {
            if (n == 0)
                trlwe[i] = SampleExtractValue<P, num_out>(
                    acc, digit, k_idx * N);
            else
                trlwe[i] = -SampleExtractValue<P, num_out>(
                    acc, digit, k_idx * N + (N - n));
        }
        else {
            if (n == 0) {
                const typename P::T offset =
                    static_cast<typename P::T>(1)
                    << (std::numeric_limits<typename P::T>::digits -
                        (digit + 1) * P::Bgbit - 1);
                trlwe[i] = acc[P::k * N + digit] + offset;
            }
            else {
                trlwe[i] = 0;
            }
        }
    }
}

template <class P, uint32_t num_out>
__global__ void __InvSampleExtractCBBatchKernel__(
    typename P::T* const out, const size_t out_stride,
    const typename P::T* const acc, const size_t acc_stride,
    const size_t batch_count)
{
    const uint32_t digit = blockIdx.x;
    const size_t batch = blockIdx.y;
    if (batch >= batch_count) return;

    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t N = P::n;
    constexpr size_t trlwe_elems = TRLWEElements<P>();
    typename P::T* const trlwe =
        out + batch * out_stride + static_cast<size_t>(digit) * trlwe_elems;
    const typename P::T* const batch_acc = acc + batch * acc_stride;

    for (uint32_t i = tid; i < trlwe_elems; i += bdim) {
        const uint32_t k_idx = i >> P::nbit;
        const uint32_t n = i & (N - 1);
        if (k_idx < P::k) {
            if (n == 0)
                trlwe[i] = SampleExtractValue<P, num_out>(
                    batch_acc, digit, k_idx * N);
            else
                trlwe[i] = -SampleExtractValue<P, num_out>(
                    batch_acc, digit, k_idx * N + (N - n));
        }
        else {
            if (n == 0) {
                const typename P::T offset =
                    static_cast<typename P::T>(1)
                    << (std::numeric_limits<typename P::T>::digits -
                        (digit + 1) * P::Bgbit - 1);
                trlwe[i] = batch_acc[P::k * N + digit] + offset;
            }
            else {
                trlwe[i] = 0;
            }
        }
    }
}

#if defined(USE_FFT)
template <class P>
__device__ inline typename P::T CBTorusFromDouble(const double value)
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
#else
template <class P>
__device__ inline typename P::T CBTorusFromNTT(const NTTValue value)
{
    if constexpr (sizeof(typename P::T) == 8) {
        return static_cast<typename P::T>(ntt_mod_to_torus64<P::n>(value));
    }
    else {
        return static_cast<typename P::T>(ntt_mod_to_torus32<P::n>(value));
    }
}
#endif

#if defined(USE_FFT)
template <class P>
__device__ inline void ExternalProductTRLWE_TRGSWFFT(
    typename P::T* const out, const typename P::T* const in,
    const NTTValue* const trgswfft, NTTValue* const sh_acc_ntt,
    const CuNTTHandler<P::n> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t half_n = N / 2;
    constexpr uint32_t num_threads = N / 2;
#ifdef USE_GPU_FFT
    constexpr uint32_t fft_threads = half_n >> 1;
#else
    constexpr uint32_t fft_threads = half_n / (Degree<N>::opt / 2);
#endif
    NTTValue* const sh_fft = &sh_acc_ntt[0];
    NTTValue* const sh_accum = &sh_acc_ntt[half_n];

    for (uint32_t i = tid; i < (P::k + 1) * half_n; i += num_threads)
        sh_accum[i] = {0.0, 0.0};
    __syncthreads();

    for (uint32_t part = 0; part <= P::k; part++) {
        const bool nonce = part < P::k;
        const uint32_t levels = nonce ? P::lₐ : P::l;
        const uint32_t bgbit = nonce ? P::Bgₐbit : P::Bgbit;
        const typename P::T offset =
            nonce ? ExternalProductOffset<P, true>()
                  : ExternalProductOffset<P, false>();
        const int remaining_bits =
            std::numeric_limits<typename P::T>::digits - levels * bgbit;
        const typename P::T roundoffset =
            remaining_bits > 0
                ? (static_cast<typename P::T>(1) << (remaining_bits - 1))
                : static_cast<typename P::T>(0);
        const typename P::T decomp_mask =
            (static_cast<typename P::T>(1) << bgbit) - 1;
        const typename P::T decomp_half =
            static_cast<typename P::T>(1) << (bgbit - 1);

        for (uint32_t digit = 0; digit < levels; digit++) {
            if (tid < half_n) {
                typename P::T temp_re =
                    in[part * N + tid] + offset + roundoffset;
                typename P::T temp_im =
                    in[part * N + tid + half_n] + offset + roundoffset;
                const int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::T>::digits -
                       (digit + 1) * bgbit)) &
                     decomp_mask) -
                    decomp_half);
                const int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::T>::digits -
                       (digit + 1) * bgbit)) &
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
                GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
#else
                NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
            }
            else {
#ifdef USE_GPU_FFT
                for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                    __syncthreads();
#else
                for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
                    __syncthreads();
#endif
            }

            if (tid < half_n) {
                const uint32_t row =
                    nonce ? part * P::lₐ + digit : P::k * P::lₐ + digit;
                const NTTValue fft_val = sh_fft[tid];
                for (uint32_t out_k = 0; out_k <= P::k; out_k++) {
                    const size_t key_offset =
                        (static_cast<size_t>(row) * (P::k + 1) + out_k) *
                            half_n +
                        tid;
                    const NTTValue key_val = __ldg(&trgswfft[key_offset]);
                    sh_accum[out_k * half_n + tid] += fft_val * key_val;
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t k_idx = 0; k_idx <= P::k; k_idx++) {
        NTTValue* const sh_inv = &sh_accum[k_idx * half_n];
        if (tid < fft_threads) {
#ifdef USE_GPU_FFT
            GPUFFTInverse<N>(sh_inv, ntt.inverse_root_, tid);
#else
            NSMFFT_inverse<HalfDegree<Degree<N>>>(sh_inv);
#endif
        }
        else {
#ifdef USE_GPU_FFT
            for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                __syncthreads();
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
            out[k_idx * N + tid] = CBTorusFromDouble<P>(val.x);
            out[k_idx * N + tid + half_n] =
                CBTorusFromDouble<P>(val.y);
        }
        __syncthreads();
    }
}
#else
template <class P>
__device__ inline void ExternalProductTRLWE_TRGSWNTT(
    typename P::T* const out, const typename P::T* const in,
    const NTTValue* const trgswntt, NTTValue* const sh_acc_ntt,
    const CuNTTHandler<P::n> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();
    constexpr uint32_t N = P::n;
    constexpr uint32_t num_threads = N / 2;
    NTTValue* const sh_work = &sh_acc_ntt[0];
    NTTValue* const sh_accum = &sh_acc_ntt[N];

    for (uint32_t i = tid; i < (P::k + 1) * N; i += num_threads)
        sh_accum[i] = 0;
    __syncthreads();

    for (uint32_t part = 0; part <= P::k; part++) {
        const bool nonce = part < P::k;
        const uint32_t levels = nonce ? P::lₐ : P::l;
        const uint32_t bgbit = nonce ? P::Bgₐbit : P::Bgbit;
        const typename P::T offset =
            nonce ? ExternalProductOffset<P, true>()
                  : ExternalProductOffset<P, false>();
        const int remaining_bits =
            std::numeric_limits<typename P::T>::digits - levels * bgbit;
        const typename P::T roundoffset =
            remaining_bits > 0
                ? (static_cast<typename P::T>(1) << (remaining_bits - 1))
                : static_cast<typename P::T>(0);
        const typename P::T decomp_mask =
            (static_cast<typename P::T>(1) << bgbit) - 1;
        const typename P::T decomp_half =
            static_cast<typename P::T>(1) << (bgbit - 1);

        for (uint32_t digit = 0; digit < levels; digit++) {
            if (tid < num_threads) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const uint32_t i = tid + e * num_threads;
                    typename P::T temp =
                        in[part * N + i] + offset + roundoffset;
                    const int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename P::T>::digits -
                           (digit + 1) * bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = signed_int_to_ntt_mod<N>(digit_val);
                }
            }
            __syncthreads();

            if (tid < num_threads) {
                SmallForwardNTT<P::nbit>(sh_work, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                    __syncthreads();
            }

            if (tid < num_threads) {
                const uint32_t row =
                    nonce ? part * P::lₐ + digit : P::k * P::lₐ + digit;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const uint32_t i = tid + e * num_threads;
                    const NTTValue ntt_val = sh_work[i];
                    for (uint32_t out_k = 0; out_k <= P::k; out_k++) {
                        const size_t key_offset =
                            (static_cast<size_t>(row) * (P::k + 1) + out_k) *
                                N +
                            i;
                        sh_accum[out_k * N + i] = small_mod_add<N>(
                            sh_accum[out_k * N + i],
                            small_mod_mult<N>(ntt_val,
                                              __ldg(&trgswntt[key_offset])));
                    }
                }
            }
            __syncthreads();
        }
    }

    for (uint32_t k_idx = 0; k_idx <= P::k; k_idx++) {
        NTTValue* const sh_inv = &sh_accum[k_idx * N];
        if (tid < num_threads) {
            SmallInverseNTT<P::nbit>(sh_inv, ntt.inverse_root_, ntt.n_inverse_,
                                     tid);
        }
        else {
            for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        if (tid < num_threads) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const uint32_t i = tid + e * num_threads;
                out[k_idx * N + i] = CBTorusFromNTT<P>(sh_inv[i]);
            }
        }
        __syncthreads();
    }
}
#endif  // USE_FFT

template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<P>)
void __CBExternalProductKernel__(typename P::T* const out,
                                 const typename P::T* const in,
                                 const NTTValue* const cbsk,
                                 const CuNTTHandler<P::n> ntt)
{
    extern __shared__ NTTValue sh_acc_ntt[];
#if defined(USE_FFT)
    ExternalProductTRLWE_TRGSWFFT<P>(out, in, cbsk, sh_acc_ntt, ntt);
#else
    ExternalProductTRLWE_TRGSWNTT<P>(out, in, cbsk, sh_acc_ntt, ntt);
#endif
}

template <class P, uint32_t num_out, uint32_t target_l_a>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<P>)
void __CBExternalProductBatchKernel__(typename P::T* const trgsw,
                                      const size_t trgsw_stride,
                                      const NTTValue* const cbsk,
                                      const CuNTTHandler<P::n> ntt,
                                      const size_t batch_count)
{
    const uint32_t row = blockIdx.x;
    const size_t batch = blockIdx.y;
    if (batch >= batch_count) return;

    constexpr size_t trlwe_elems = TRLWEElements<P>();
    constexpr uint32_t main_row_offset = P::k * target_l_a;
    constexpr size_t cbsk_one_key_elems = TRGSWFFTElements<P>();
    const uint32_t k = row / num_out;
    const uint32_t i = row % num_out;
    typename P::T* const batch_trgsw = trgsw + batch * trgsw_stride;

    extern __shared__ NTTValue sh_acc_ntt[];
#if defined(USE_FFT)
    ExternalProductTRLWE_TRGSWFFT<P>(
        batch_trgsw +
            static_cast<size_t>(k * target_l_a + i) * trlwe_elems,
        batch_trgsw +
            static_cast<size_t>(main_row_offset + i) * trlwe_elems,
        cbsk + static_cast<size_t>(k) * cbsk_one_key_elems, sh_acc_ntt, ntt);
#else
    ExternalProductTRLWE_TRGSWNTT<P>(
        batch_trgsw +
            static_cast<size_t>(k * target_l_a + i) * trlwe_elems,
        batch_trgsw +
            static_cast<size_t>(main_row_offset + i) * trlwe_elems,
        cbsk + static_cast<size_t>(k) * cbsk_one_key_elems, sh_acc_ntt, ntt);
#endif
}

template <class iksP>
__global__ void __IdentityKeySwitchKernel__(
    typename iksP::targetP::T* const out,
    const typename iksP::domainP::T* const in,
    const typename iksP::targetP::T* const ksk)
{
    KeySwitchFromTLWE<iksP>(out, in, ksk);
}

template <class P>
void CBswitchingKeyPolynomialGen(CBswitchingKeyPolynomial<P>& cbsk,
                                 const TFHEpp::Key<P>& key)
{
    for (uint32_t key_idx = 0; key_idx < P::k; key_idx++) {
        TFHEpp::Polynomial<P> partkey{};
        for (uint32_t i = 0; i < P::n; i++)
            partkey[i] = -key[key_idx * P::n + i];
        TFHEpp::trgswSymEncrypt<P>(cbsk[key_idx], partkey, key);
    }
}

template <class P>
void CBswitchingKeyPolynomialGen(CBswitchingKeyPolynomial<P>& cbsk,
                                 const TFHEpp::SecretKey& sk)
{
    CBswitchingKeyPolynomialGen<P>(cbsk, sk.key.get<P>());
}

template <class P>
void CBswitchingKeyPolynomialToDevice(
    const CBswitchingKeyPolynomial<P>& cbsk, const int gpuNum)
{
    static_assert(P::k == 1,
                  "CUDA CB switching currently supports GLWE dimension 1");
    static_assert(P::l̅ == 1 && P::l̅ₐ == 1,
                  "CUDA CB switching currently supports standard decomposition");

    auto& storage = CBswitchingKeyStorage<P>();
    constexpr uint32_t rows = P::k * TRGSWRows<P>() * (P::k + 1);
    constexpr size_t poly_elems = static_cast<size_t>(rows) * P::n;
    const size_t poly_bytes = poly_elems * sizeof(typename P::T);
    const size_t fft_bytes = CBswitchingKeyElements<P>() * sizeof(NTTValue);

    std::vector<typename P::T> packed(poly_elems);
    size_t row = 0;
    for (uint32_t key_idx = 0; key_idx < P::k; key_idx++) {
        for (uint32_t trgsw_row = 0; trgsw_row < TRGSWRows<P>();
             trgsw_row++) {
            for (uint32_t out_k = 0; out_k <= P::k; out_k++) {
                const auto& poly = cbsk[key_idx][trgsw_row][out_k];
                for (uint32_t i = 0; i < P::n; i++)
                    packed[row * P::n + i] = poly[i];
                row++;
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
        __TRGSWPolynomialToFFT__<P><<<rows, P::n / 2>>>(
            storage[i], d_poly, *RingHandler<P>(i));
#else
        __TRGSWPolynomialToNTT__<P><<<rows, P::n / 2>>>(
            storage[i], d_poly, *RingHandler<P>(i));
#endif
        CuCheckError();
        CuSafeCall(cudaFree(d_poly));
    }
}

template <class P>
void DeleteCBswitchingKey(const int gpuNum)
{
    auto& storage = CBswitchingKeyStorage<P>();
    for (size_t i = 0; i < storage.size(); i++) {
        cudaSetDevice(i);
        cudaFree(storage[i]);
    }
    storage.clear();
}

template <class brP, class ahP>
void AnnihilateCircuitBootstrappingWithWorkspace(
    typename brP::targetP::T* const out,
    const typename brP::domainP::T* const in,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temptrlwe, const cudaStream_t st,
    const int gpuNum)
{
    using targetP = typename brP::targetP;
    static_assert(targetP::k == ahP::k,
                  "brP::targetP::k must match ahP::k");
    static_assert(targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "brP::targetP and ahP must share the same torus ring");
    static_assert(targetP::l == targetP::lₐ,
                  "CUDA Annihilate CB currently expects target l == l_a");

    cudaSetDevice(gpuNum);
    constexpr uint32_t num_out = targetP::l;
    constexpr size_t trlwe_elems = TRLWEElements<targetP>();

    static bool blindrotate_attribute_set = false;
    if (!blindrotate_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __BlindRotateCBKernel__<brP, num_out>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            MEM4HOMGATE<targetP>));
        blindrotate_attribute_set = true;
    }
    __BlindRotateCBKernel__<brP, num_out>
        <<<1, NUM_THREAD4HOMGATE<targetP>, MEM4HOMGATE<targetP>, st>>>(
            acc, in, BootstrappingKeyStorage<brP>(gpuNum),
#ifdef USE_KEY_BUNDLE
            OneTRGSWStorage<targetP>(gpuNum), XaiStorage<targetP>(gpuNum),
#endif
            *RingHandler<targetP>(gpuNum));

    __InvSampleExtractCBKernel__<targetP, num_out>
        <<<num_out, NUM_THREAD4HOMGATE<targetP>, 0, st>>>(temptrlwe, acc);

    constexpr uint32_t main_row_offset = targetP::k * targetP::lₐ;
    for (uint32_t i = 0; i < num_out; i++) {
        typename targetP::T* const extracted =
            temptrlwe + static_cast<size_t>(i) * trlwe_elems;
        AnnihilateKeySwitchingWithWorkspace<ahP>(
            out + static_cast<size_t>(main_row_offset + i) * trlwe_elems,
            extracted, extracted, st, gpuNum);
    }

    constexpr size_t cbsk_one_key_elems = TRGSWFFTElements<ahP>();
    const NTTValue* const cbsk = CBswitchingKeyStorage<ahP>()[gpuNum];
    static bool external_product_attribute_set = false;
    if (!external_product_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __CBExternalProductKernel__<ahP>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, MEM4HOMGATE<ahP>));
        external_product_attribute_set = true;
    }
    for (uint32_t k = 0; k < targetP::k; k++) {
        for (uint32_t i = 0; i < num_out; i++) {
            __CBExternalProductKernel__<ahP>
                <<<1, NUM_THREAD4HOMGATE<ahP>, MEM4HOMGATE<ahP>, st>>>(
                    out + static_cast<size_t>(k * targetP::lₐ + i) *
                              trlwe_elems,
                    out + static_cast<size_t>(main_row_offset + i) *
                              trlwe_elems,
                    cbsk + static_cast<size_t>(k) * cbsk_one_key_elems,
                    *RingHandler<ahP>(gpuNum));
        }
    }

    CuCheckError();
}

template <class brP, class ahP>
void AnnihilateCircuitBootstrappingBatchWithWorkspace(
    typename brP::targetP::T* const out, const size_t out_stride,
    const typename brP::domainP::T* const in, const size_t in_stride,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temptrlwe,
    const size_t batch_count, const cudaStream_t st, const int gpuNum)
{
    using targetP = typename brP::targetP;
    static_assert(targetP::k == ahP::k,
                  "brP::targetP::k must match ahP::k");
    static_assert(targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "brP::targetP and ahP must share the same torus ring");
    static_assert(targetP::l == targetP::lₐ,
                  "CUDA Annihilate CB currently expects target l == l_a");
    if (batch_count == 0) return;

    cudaSetDevice(gpuNum);
    constexpr uint32_t num_out = targetP::l;
    constexpr size_t trlwe_elems = TRLWEElements<targetP>();
    constexpr size_t temptrlwe_stride =
        static_cast<size_t>(num_out) * trlwe_elems;

    static bool blindrotate_batch_attribute_set = false;
    if (!blindrotate_batch_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __BlindRotateCBBatchKernel__<brP, num_out>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            MEM4HOMGATE<targetP>));
        blindrotate_batch_attribute_set = true;
    }
    __BlindRotateCBBatchKernel__<brP, num_out>
        <<<batch_count, NUM_THREAD4HOMGATE<targetP>, MEM4HOMGATE<targetP>,
           st>>>(acc, trlwe_elems, in, in_stride,
                 BootstrappingKeyStorage<brP>(gpuNum),
#ifdef USE_KEY_BUNDLE
                 OneTRGSWStorage<targetP>(gpuNum), XaiStorage<targetP>(gpuNum),
#endif
                 *RingHandler<targetP>(gpuNum), batch_count);

    const dim3 extract_grid(num_out, batch_count);
    __InvSampleExtractCBBatchKernel__<targetP, num_out>
        <<<extract_grid, NUM_THREAD4HOMGATE<targetP>, 0, st>>>(
            temptrlwe, temptrlwe_stride, acc, trlwe_elems, batch_count);

    constexpr uint32_t main_row_offset = targetP::k * targetP::lₐ;
    for (uint32_t i = 0; i < num_out; i++) {
        typename targetP::T* const extracted =
            temptrlwe + static_cast<size_t>(i) * trlwe_elems;
        AnnihilateKeySwitchingBatchWithWorkspace<ahP>(
            out + static_cast<size_t>(main_row_offset + i) * trlwe_elems,
            out_stride, extracted, temptrlwe_stride, extracted,
            temptrlwe_stride, batch_count, st, gpuNum);
    }

    const NTTValue* const cbsk = CBswitchingKeyStorage<ahP>()[gpuNum];
    static bool external_product_batch_attribute_set = false;
    if (!external_product_batch_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __CBExternalProductBatchKernel__<ahP, num_out, targetP::lₐ>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, MEM4HOMGATE<ahP>));
        external_product_batch_attribute_set = true;
    }
    const dim3 external_grid(targetP::k * num_out, batch_count);
    __CBExternalProductBatchKernel__<ahP, num_out, targetP::lₐ>
        <<<external_grid, NUM_THREAD4HOMGATE<ahP>, MEM4HOMGATE<ahP>, st>>>(
            out, out_stride, cbsk, *RingHandler<ahP>(gpuNum), batch_count);

    CuCheckError();
}

template <class brP, class ahP>
void AnnihilateCircuitBootstrapping(typename brP::targetP::T* const out,
                                    const typename brP::domainP::T* const in,
                                    const cudaStream_t st, const int gpuNum)
{
    using targetP = typename brP::targetP;
    constexpr uint32_t num_out = targetP::l;
    constexpr size_t trlwe_elems = TRLWEElements<targetP>();
    constexpr size_t trlwe_bytes = trlwe_elems * sizeof(typename targetP::T);
    typename targetP::T* acc = nullptr;
    typename targetP::T* temptrlwe = nullptr;
    CuSafeCall(cudaMalloc(&acc, trlwe_bytes));
    CuSafeCall(cudaMalloc(&temptrlwe, num_out * trlwe_bytes));
    AnnihilateCircuitBootstrappingWithWorkspace<brP, ahP>(
        out, in, acc, temptrlwe, st, gpuNum);
    CuSafeCall(cudaFree(temptrlwe));
    CuSafeCall(cudaFree(acc));
}

template <class iksP, class brP, class ahP>
void AnnihilateCircuitBootstrapping(typename brP::targetP::T* const out,
                                    const typename iksP::domainP::T* const in,
                                    const cudaStream_t st, const int gpuNum)
{
    static_assert(std::is_same_v<typename iksP::targetP, typename brP::domainP>,
                  "iksP::targetP must match brP::domainP");
    cudaSetDevice(gpuNum);
    constexpr size_t tlwe0_bytes =
        (brP::domainP::k * brP::domainP::n + 1) *
        sizeof(typename brP::domainP::T);
    typename brP::domainP::T* tlwe0 = nullptr;
    CuSafeCall(cudaMalloc(&tlwe0, tlwe0_bytes));
    __IdentityKeySwitchKernel__<iksP>
        <<<1, iksP::targetP::k * iksP::targetP::n + 1, 0, st>>>(
            tlwe0, in, KeySwitchingKeyStorage<iksP>(gpuNum));
    AnnihilateCircuitBootstrapping<brP, ahP>(out, tlwe0, st, gpuNum);
    CuSafeCall(cudaFree(tlwe0));
}

#define INST_KEY(P)                                                          \
    template void CBswitchingKeyPolynomialGen<P>(                            \
        CBswitchingKeyPolynomial<P>&, const TFHEpp::Key<P>&);                \
    template void CBswitchingKeyPolynomialGen<P>(                            \
        CBswitchingKeyPolynomial<P>&, const TFHEpp::SecretKey&);             \
    template void CBswitchingKeyPolynomialToDevice<P>(                       \
        const CBswitchingKeyPolynomial<P>&, const int);                      \
    template void DeleteCBswitchingKey<P>(const int)

INST_KEY(TFHEpp::AHlvl1param);
INST_KEY(TFHEpp::AHlvl2param);

#undef INST_KEY

#define INST_CB(brP, ahP)                                                     \
    template void AnnihilateCircuitBootstrappingWithWorkspace<brP, ahP>(      \
        typename brP::targetP::T* const, const typename brP::domainP::T* const, \
        typename brP::targetP::T* const, typename brP::targetP::T* const,      \
        const cudaStream_t, const int);                                       \
    template void AnnihilateCircuitBootstrappingBatchWithWorkspace<brP, ahP>( \
        typename brP::targetP::T* const, const size_t,                         \
        const typename brP::domainP::T* const, const size_t,                   \
        typename brP::targetP::T* const, typename brP::targetP::T* const,      \
        const size_t, const cudaStream_t, const int);                         \
    template void AnnihilateCircuitBootstrapping<brP, ahP>(                  \
        typename brP::targetP::T* const, const typename brP::domainP::T* const, \
        const cudaStream_t, const int)

INST_CB(TFHEpp::lvl01param, TFHEpp::AHlvl1param);
INST_CB(TFHEpp::lvl02param, TFHEpp::AHlvl2param);
INST_CB(TFHEpp::lvlh2param, TFHEpp::AHlvl2param);

#undef INST_CB

#define INST_CB_IKS(iksP, brP, ahP)                                           \
    template void AnnihilateCircuitBootstrapping<iksP, brP, ahP>(            \
        typename brP::targetP::T* const, const typename iksP::domainP::T* const, \
        const cudaStream_t, const int)

INST_CB_IKS(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::AHlvl1param);
INST_CB_IKS(TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::AHlvl2param);

#undef INST_CB_IKS

}  // namespace cufhe
