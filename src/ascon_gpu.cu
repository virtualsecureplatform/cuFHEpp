#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <include/annihilate_gpu.cuh>
#include <include/ascon_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <include/utils_gpu.cuh>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace cufhe {

extern std::vector<CuNTTHandler<>*> ntt_handlers;
extern std::vector<CuNTTHandler<TFHEpp::lvl2param::n>*> ntt_handlers_lvl02;

namespace {

template <class P>
constexpr bool is_lvl1_ring_v =
    P::n == TFHEpp::lvl1param::n &&
    sizeof(typename P::T) == sizeof(typename TFHEpp::lvl1param::T);

template <class P>
constexpr bool is_lvl2_ring_v =
    P::n == TFHEpp::lvl2param::n &&
    sizeof(typename P::T) == sizeof(typename TFHEpp::lvl2param::T);

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
                      "Unsupported ASCON target ring");
    }
}

template <class P>
__host__ __device__ constexpr uint32_t TLWEElements()
{
    return P::k * P::n + 1;
}

template <class P>
__host__ __device__ constexpr uint32_t TRLWEElements()
{
    return (P::k + 1) * P::n;
}

template <class P>
__host__ __device__ constexpr uint32_t TRGSWRows()
{
    return P::k * P::lₐ * P::l̅ₐ + P::l * P::l̅;
}

template <class P>
__host__ __device__ constexpr size_t TRGSWElements()
{
    return static_cast<size_t>(TRGSWRows<P>()) * TRLWEElements<P>();
}

template <class P>
__host__ __device__ constexpr size_t TRGSWFFTElements()
{
    return static_cast<size_t>(TRGSWRows<P>()) * (P::k + 1) * (P::n / 2);
}

template <class P>
__host__ __device__ constexpr typename P::T BitMu()
{
    return static_cast<typename P::T>(1)
           << (std::numeric_limits<typename P::T>::digits - 2);
}

__host__ __device__ constexpr size_t ASCONBitIndex(const size_t word,
                                                   const size_t bit)
{
    return word * TFHEpp::ascon_word_bits + bit;
}

__host__ __device__ constexpr size_t ASCONRateByteBitIndex(const size_t byte,
                                                           const size_t bit)
{
    return ASCONBitIndex(byte / 8, 8 * (byte % 8) + bit);
}

template <class iksP>
__global__ void __IdentityKeySwitchBatchKernel__(
    typename iksP::targetP::T* const out, const size_t out_stride,
    const typename iksP::domainP::T* const in, const size_t in_stride,
    const typename iksP::targetP::T* const ksk, const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    KeySwitchFromTLWE<iksP>(out + batch * out_stride, in + batch * in_stride,
                            ksk);
}

template <class P>
__global__ void __TRGSWToFFTKernel__(NTTValue* const out,
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

    __shared__ NTTValue sh_fft[half_n];

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
        const double re =
            static_cast<double>(static_cast<std::make_signed_t<typename P::T>>(
                in[in_index + tid])) *
            norm;
        const double im =
            static_cast<double>(static_cast<std::make_signed_t<typename P::T>>(
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
        for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++) __syncthreads();
#endif
    }

    if (tid < half_n) out[out_index + tid] = sh_fft[tid];
}

#if defined(USE_FFT)
template <class P>
__device__ inline typename P::T TorusFromDouble(const double value)
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
                   << (std::numeric_limits<typename P::T>::digits - i * bgbit));
    return offset;
}

template <class P>
__device__ inline void ExternalProductTRLWE_TRGSWFFT_ASCON(
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
        const typename P::T offset = nonce ? ExternalProductOffset<P, true>()
                                           : ExternalProductOffset<P, false>();
        const int remaining_bits =
            std::numeric_limits<typename P::T>::digits - levels * bgbit;
        const typename P::T roundoffset =
            remaining_bits > 0
                ? (static_cast<typename P::T>(1) << (remaining_bits - 1))
                : static_cast<typename P::T>(0);
        const typename P::T decomp_mask =
            (static_cast<typename P::T>(1) << bgbit) - 1;
        const typename P::T decomp_half = static_cast<typename P::T>(1)
                                          << (bgbit - 1);

        for (uint32_t digit = 0; digit < levels; digit++) {
            if (tid < half_n) {
                typename P::T temp_re =
                    in[part * N + tid] + offset + roundoffset;
                typename P::T temp_im =
                    in[part * N + tid + half_n] + offset + roundoffset;
                const int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >> (std::numeric_limits<typename P::T>::digits -
                                  (digit + 1) * bgbit)) &
                     decomp_mask) -
                    decomp_half);
                const int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >> (std::numeric_limits<typename P::T>::digits -
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
                    sh_accum[out_k * half_n + tid] +=
                        fft_val * __ldg(&trgswfft[key_offset]);
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
            out[k_idx * N + tid] = TorusFromDouble<P>(val.x);
            out[k_idx * N + tid + half_n] = TorusFromDouble<P>(val.y);
        }
        __syncthreads();
    }
}

template <class P>
__global__ __launch_bounds__(
    NUM_THREAD4HOMGATE<
        P>) void __ASCONExternalProductBatchKernel__(typename P::T* const out,
                                                     const size_t out_stride,
                                                     const typename P::T* const
                                                         in,
                                                     const size_t in_stride,
                                                     const NTTValue* const
                                                         address,
                                                     const size_t
                                                         address_stride,
                                                     const uint32_t address_bit,
                                                     const CuNTTHandler<P::n>
                                                         ntt,
                                                     const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    extern __shared__ NTTValue sh_acc_ntt[];
    ExternalProductTRLWE_TRGSWFFT_ASCON<P>(
        out + batch * out_stride, in + batch * in_stride,
        address + batch * address_stride +
            static_cast<size_t>(address_bit) * TRGSWFFTElements<P>(),
        sh_acc_ntt, ntt);
}
#endif  // USE_FFT

template <class P>
__global__ void __PolynomialMulByXaiMinusOneTRLWEBatchKernel__(
    typename P::T* const out, const size_t out_stride,
    const typename P::T* const in, const size_t in_stride,
    const uint32_t exponent, const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t N = P::n;
    constexpr uint32_t total = TRLWEElements<P>();
    const bool small = exponent < N;
    const uint32_t a = small ? exponent : exponent - N;
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_in = in + batch * in_stride;

    for (uint32_t index = tid; index < total; index += bdim) {
        const uint32_t poly = index >> P::nbit;
        const uint32_t i = index & (N - 1);
        const typename P::T* const src = batch_in + poly * N;
        typename P::T rotated;
        if (small)
            rotated = i < a ? -src[i - a + N] : src[i - a];
        else
            rotated = i < a ? src[i - a + N] : -src[i - a];
        batch_out[index] = rotated - src[i];
    }
}

template <class P>
__global__ void __TRLWEAddBatchKernel__(typename P::T* const out,
                                        const size_t out_stride,
                                        const typename P::T* const a,
                                        const size_t a_stride,
                                        const typename P::T* const b,
                                        const size_t b_stride,
                                        const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_a = a + batch * a_stride;
    const typename P::T* const batch_b = b + batch * b_stride;
    for (uint32_t i = tid; i < TRLWEElements<P>(); i += bdim)
        batch_out[i] = batch_a[i] + batch_b[i];
}

template <class P>
__global__ void __TRLWEAddROMBatchKernel__(typename P::T* const out,
                                           const size_t out_stride,
                                           const typename P::T* const a,
                                           const size_t a_stride,
                                           const typename P::T* const rom,
                                           const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_a = a + batch * a_stride;
    for (uint32_t i = tid; i < TRLWEElements<P>(); i += bdim)
        batch_out[i] = batch_a[i] + rom[i];
}

template <class P>
__global__ void __TRLWEAddInPlaceBatchKernel__(
    typename P::T* const out, const size_t out_stride,
    const typename P::T* const addend, const size_t addend_stride,
    const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const batch_out = out + batch * out_stride;
    const typename P::T* const batch_addend = addend + batch * addend_stride;
    for (uint32_t i = tid; i < TRLWEElements<P>(); i += bdim)
        batch_out[i] += batch_addend[i];
}

template <class P>
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

template <class P, uint32_t num_tlwe>
__global__ void __SampleExtractManyBatchKernel__(typename P::T* const out,
                                                 const typename P::T* const acc,
                                                 const size_t batch_count)
{
    const uint32_t out_idx = blockIdx.x;
    const size_t batch = blockIdx.y;
    if (batch >= batch_count) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t tlwe_elems = TLWEElements<P>();
    constexpr size_t trlwe_elems = TRLWEElements<P>();
    typename P::T* const tlwe = out + (batch * num_tlwe + out_idx) * tlwe_elems;
    const typename P::T* const batch_acc = acc + batch * trlwe_elems;
    for (uint32_t i = tid; i < tlwe_elems; i += bdim)
        tlwe[i] = SampleExtractValue<P>(batch_acc, out_idx, i);
}

template <class P>
__global__ void __ASCONRoundConstantAndPreSboxKernel__(
    typename P::T* const state, const uint8_t constant)
{
    const uint32_t bit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    typename P::T* const x0 = state + ASCONBitIndex(0, bit) * elems;
    typename P::T* const x2 = state + ASCONBitIndex(2, bit) * elems;
    typename P::T* const x4 = state + ASCONBitIndex(4, bit) * elems;
    const typename P::T* const x1 = state + ASCONBitIndex(1, bit) * elems;
    const typename P::T* const x3 = state + ASCONBitIndex(3, bit) * elems;

    for (uint32_t i = tid; i < elems; i += bdim) {
        if (bit < 8 && ((constant >> bit) & 1U)) x2[i] = -x2[i];
        x0[i] += x4[i];
        x4[i] += x3[i];
        x2[i] += x1[i];
        if (i == P::k * P::n) {
            x0[i] += BitMu<P>();
            x4[i] += BitMu<P>();
            x2[i] += BitMu<P>();
        }
    }
}

template <class P>
__global__ void __ASCONGatherSboxInputsKernel__(
    typename P::T* const out, const typename P::T* const state)
{
    const uint32_t packed = blockIdx.x;
    const uint32_t word = packed % TFHEpp::ascon_words;
    const uint32_t bit = packed / TFHEpp::ascon_words;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    const typename P::T* const src = state + ASCONBitIndex(word, bit) * elems;
    typename P::T* const dst = out + static_cast<size_t>(packed) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) dst[i] = src[i];
}

template <class P>
__global__ void __ASCONScatterSboxOutputsKernel__(
    typename P::T* const t, const typename P::T* const sbox_out)
{
    const uint32_t packed = blockIdx.x;
    const uint32_t word = packed % TFHEpp::ascon_words;
    const uint32_t bit = packed / TFHEpp::ascon_words;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    typename P::T* const dst = t + ASCONBitIndex(word, bit) * elems;
    const typename P::T* const src =
        sbox_out + static_cast<size_t>(packed) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) dst[i] = src[i];
}

template <class P>
__global__ void __ASCONPostSboxXor13Kernel__(typename P::T* const t)
{
    const uint32_t bit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    typename P::T* const x1 = t + ASCONBitIndex(1, bit) * elems;
    typename P::T* const x3 = t + ASCONBitIndex(3, bit) * elems;
    const typename P::T* const x0 = t + ASCONBitIndex(0, bit) * elems;
    const typename P::T* const x2 = t + ASCONBitIndex(2, bit) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) {
        x1[i] += x0[i];
        x3[i] += x2[i];
        if (i == P::k * P::n) {
            x1[i] += BitMu<P>();
            x3[i] += BitMu<P>();
        }
    }
}

template <class P>
__global__ void __ASCONPostSboxXor0Not2Kernel__(typename P::T* const t)
{
    const uint32_t bit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    typename P::T* const x0 = t + ASCONBitIndex(0, bit) * elems;
    typename P::T* const x2 = t + ASCONBitIndex(2, bit) * elems;
    const typename P::T* const x4 = t + ASCONBitIndex(4, bit) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) {
        x0[i] += x4[i];
        x2[i] = -x2[i];
        if (i == P::k * P::n) x0[i] += BitMu<P>();
    }
}

template <class P>
__global__ void __ASCONLinearDiffusionKernel__(typename P::T* const state,
                                               const typename P::T* const t)
{
    constexpr uint32_t rot0[TFHEpp::ascon_words] = {19, 61, 1, 10, 7};
    constexpr uint32_t rot1[TFHEpp::ascon_words] = {28, 39, 6, 17, 41};
    const uint32_t state_bit = blockIdx.x;
    const uint32_t word = state_bit / TFHEpp::ascon_word_bits;
    const uint32_t bit = state_bit & 63;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    const typename P::T* const a = t + ASCONBitIndex(word, bit) * elems;
    const typename P::T* const b =
        t + ASCONBitIndex(word, (bit + rot0[word]) & 63) * elems;
    const typename P::T* const c =
        t + ASCONBitIndex(word, (bit + rot1[word]) & 63) * elems;
    typename P::T* const out = state + static_cast<size_t>(state_bit) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) {
        typename P::T value = a[i] + b[i] + c[i];
        if (i == P::k * P::n)
            value += static_cast<typename P::T>(2) * BitMu<P>();
        out[i] = value;
    }
}

template <class P>
__global__ void __ASCONXORRateInputKernel__(typename P::T* const state,
                                            const typename P::T* const input,
                                            const size_t input_offset_bits,
                                            const size_t byte_count)
{
    const size_t bit_index = blockIdx.x;
    if (bit_index >= byte_count * 8) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    const size_t byte = bit_index / 8;
    const size_t bit = bit_index & 7;
    typename P::T* const dst = state + ASCONRateByteBitIndex(byte, bit) * elems;
    const typename P::T* const src =
        input + (input_offset_bits + bit_index) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) {
        dst[i] += src[i];
        if (i == P::k * P::n) dst[i] += BitMu<P>();
    }
}

template <class P>
__global__ void __ASCONNotRatePadKernel__(typename P::T* const state,
                                          const size_t byte)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    typename P::T* const dst = state + ASCONRateByteBitIndex(byte, 0) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) dst[i] = -dst[i];
}

template <class P>
__global__ void __ASCONCopyRateOutputKernel__(typename P::T* const output,
                                              const typename P::T* const state,
                                              const size_t output_offset_bits,
                                              const size_t byte_count)
{
    const size_t bit_index = blockIdx.x;
    if (bit_index >= byte_count * 8) return;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr size_t elems = TLWEElements<P>();
    const size_t byte = bit_index / 8;
    const size_t bit = bit_index & 7;
    typename P::T* const dst =
        output + (output_offset_bits + bit_index) * elems;
    const typename P::T* const src =
        state + ASCONRateByteBitIndex(byte, bit) * elems;
    for (uint32_t i = tid; i < elems; i += bdim) dst[i] = src[i];
}

template <class P, uint32_t address_bit, uint32_t width_bit, uint32_t num_tlwe>
void DeviceLROMUXBatch(typename P::T* const out, const NTTValue* const address,
                       const typename P::T* const data,
                       typename P::T* const acc, typename P::T* const temp,
                       typename P::T* const product, const size_t batch_count,
                       const cudaStream_t st, const int gpuNum)
{
    static_assert(address_bit == width_bit);
    static_assert(num_tlwe <= (1U << (P::nbit - width_bit)));
    constexpr uint32_t threads = NUM_THREAD4HOMGATE<P>;
    constexpr size_t shmem = MEM4HOMGATE<P>;
    constexpr size_t trlwe_elems = TRLWEElements<P>();
    constexpr size_t trgswfft_elems = TRGSWFFTElements<P>();
    constexpr size_t address_stride =
        static_cast<size_t>(address_bit) * trgswfft_elems;

    static bool external_product_attribute_set = false;
    if (!external_product_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __ASCONExternalProductBatchKernel__<P>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        external_product_attribute_set = true;
    }

    __PolynomialMulByXaiMinusOneTRLWEBatchKernel__<P>
        <<<batch_count, threads, 0, st>>>(temp, trlwe_elems, data, 0,
                                          2 * P::n - (P::n >> 1), batch_count);
    __ASCONExternalProductBatchKernel__<P><<<batch_count, threads, shmem, st>>>(
        product, trlwe_elems, temp, trlwe_elems, address, address_stride,
        width_bit - 1, *RingHandler<P>(gpuNum), batch_count);
    __TRLWEAddROMBatchKernel__<P><<<batch_count, threads, 0, st>>>(
        acc, trlwe_elems, product, trlwe_elems, data, batch_count);

    for (uint32_t bit = 2; bit <= width_bit; bit++) {
        __PolynomialMulByXaiMinusOneTRLWEBatchKernel__<P>
            <<<batch_count, threads, 0, st>>>(
                temp, trlwe_elems, acc, trlwe_elems, 2 * P::n - (P::n >> bit),
                batch_count);
        __ASCONExternalProductBatchKernel__<P>
            <<<batch_count, threads, shmem, st>>>(
                product, trlwe_elems, temp, trlwe_elems, address,
                address_stride, width_bit - bit, *RingHandler<P>(gpuNum),
                batch_count);
        __TRLWEAddInPlaceBatchKernel__<P><<<batch_count, threads, 0, st>>>(
            acc, trlwe_elems, product, trlwe_elems, batch_count);
    }

    const dim3 extract_grid(num_tlwe, batch_count);
    __SampleExtractManyBatchKernel__<P, num_tlwe>
        <<<extract_grid, threads, 0, st>>>(out, acc, batch_count);
    CuCheckError();
}

template <class P>
void UploadASCONSboxROM(typename P::T* const d_rom, const cudaStream_t st)
{
    auto rom = std::make_unique<TFHEpp::TRLWE<P>>();
    *rom = {};
    (*rom)[P::k] = TFHEpp::ASCONSboxROMPoly<P>();
    CuSafeCall(cudaMemcpyAsync(d_rom, rom->data(), sizeof(*rom),
                               cudaMemcpyHostToDevice, st));
}

template <class brP>
void InitializeBRKey(const TFHEpp::EvalKey& ek)
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl2param::n) {
        InitializeNTThandlers_lvl02(_gpuNum);
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        InitializeXaiNTT_lvl02(_gpuNum);
#endif
#ifdef USE_KEY_BUNDLE
        InitializeOneTRGSWNTT_lvl02(_gpuNum);
#endif
    }
    else {
        InitializeNTThandlers(_gpuNum);
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        InitializeXaiNTT(_gpuNum);
#endif
#ifdef USE_KEY_BUNDLE
        InitializeOneTRGSWNTT(_gpuNum);
#endif
    }

#ifdef USE_KEY_BUNDLE
    BootstrappingKeyBundleToNTT<brP>(ek.getbk<brP>(), _gpuNum);
#else
    BootstrappingKeyToNTT<brP>(ek.getbk<brP>(), _gpuNum);
#endif
}

template <class brP>
void DeleteBRKey()
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl2param::n) {
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        DeleteXaiNTT_lvl02();
#endif
#ifdef USE_KEY_BUNDLE
        DeleteOneTRGSWNTT_lvl02();
#endif
        DeleteBootstrappingKeyNTT_lvl02(_gpuNum);
    }
    else {
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        DeleteXaiNTT();
#endif
#ifdef USE_KEY_BUNDLE
        DeleteOneTRGSWNTT();
#endif
        DeleteBootstrappingKeyNTT(_gpuNum);
    }
}

template <class iksP, class brP, class ahP>
void DeviceASCONSboxLayer(typename brP::targetP::T* const state,
                          typename brP::targetP::T* const t,
                          const typename iksP::targetP::T* const ksk,
                          const typename brP::targetP::T* const rom,
                          typename brP::targetP::T* const sbox_input,
                          typename brP::targetP::T* const sbox_output,
                          typename brP::targetP::T* const address_poly,
                          NTTValue* const address_fft,
                          typename brP::targetP::T* const acc,
                          typename brP::targetP::T* const temp,
                          typename brP::targetP::T* const product,
                          typename brP::domainP::T* const domain_tlwe,
                          typename brP::targetP::T* const cb_acc,
                          typename brP::targetP::T* const cb_temptrlwe,
                          const cudaStream_t st, const int gpuNum)
{
    using targetP = typename brP::targetP;
    using domainP = typename brP::domainP;
    using inputP = typename iksP::domainP;
    static_assert(inputP::k == targetP::k && inputP::n == targetP::n &&
                  sizeof(typename inputP::T) == sizeof(typename targetP::T));
    static_assert(std::is_same_v<typename iksP::targetP, domainP>);
    static_assert(targetP::k == ahP::k && targetP::n == ahP::n &&
                  sizeof(typename targetP::T) == sizeof(typename ahP::T));

    constexpr uint32_t columns = TFHEpp::ascon_word_bits;
    constexpr uint32_t address_bit = TFHEpp::ascon_words;
    constexpr uint32_t batch_count = columns * address_bit;
    constexpr size_t tlwe_elems = TLWEElements<targetP>();
    constexpr size_t trgsw_elems = TRGSWElements<targetP>();
    constexpr size_t trgswfft_elems = TRGSWFFTElements<targetP>();
    constexpr size_t trgsw_rows = TRGSWRows<targetP>() * (targetP::k + 1);
    constexpr size_t domain_tlwe_elems = TLWEElements<domainP>();

    __ASCONGatherSboxInputsKernel__<targetP>
        <<<batch_count, 256, 0, st>>>(sbox_input, state);
    const uint32_t ks_threads = std::min<uint32_t>(1024, domain_tlwe_elems);
    __IdentityKeySwitchBatchKernel__<iksP><<<batch_count, ks_threads, 0, st>>>(
        domain_tlwe, domain_tlwe_elems, sbox_input, tlwe_elems, ksk,
        batch_count);

    AnnihilateCircuitBootstrappingBatchWithWorkspace<brP, ahP>(
        address_poly, trgsw_elems, domain_tlwe, domain_tlwe_elems, cb_acc,
        cb_temptrlwe, batch_count, st, gpuNum);

    __TRGSWToFFTKernel__<targetP>
        <<<static_cast<size_t>(batch_count) * trgsw_rows, targetP::n / 2, 0,
           st>>>(address_fft, address_poly, *RingHandler<targetP>(gpuNum));

    DeviceLROMUXBatch<targetP, address_bit, address_bit, address_bit>(
        sbox_output, address_fft, rom, acc, temp, product, columns, st, gpuNum);
    __ASCONScatterSboxOutputsKernel__<targetP>
        <<<batch_count, 256, 0, st>>>(t, sbox_output);
    CuCheckError();
}

template <class iksP, class brP, class ahP>
void DeviceASCONRound(
    typename brP::targetP::T* const state, typename brP::targetP::T* const t,
    const typename iksP::targetP::T* const ksk,
    const typename brP::targetP::T* const rom,
    typename brP::targetP::T* const sbox_input,
    typename brP::targetP::T* const sbox_output,
    typename brP::targetP::T* const address_poly, NTTValue* const address_fft,
    typename brP::targetP::T* const acc, typename brP::targetP::T* const temp,
    typename brP::targetP::T* const product,
    typename brP::domainP::T* const domain_tlwe,
    typename brP::targetP::T* const cb_acc,
    typename brP::targetP::T* const cb_temptrlwe, const uint8_t constant,
    const cudaStream_t st, const int gpuNum)
{
    using targetP = typename brP::targetP;
    __ASCONRoundConstantAndPreSboxKernel__<targetP>
        <<<TFHEpp::ascon_word_bits, 256, 0, st>>>(state, constant);
    DeviceASCONSboxLayer<iksP, brP, ahP>(
        state, t, ksk, rom, sbox_input, sbox_output, address_poly, address_fft,
        acc, temp, product, domain_tlwe, cb_acc, cb_temptrlwe, st, gpuNum);
    __ASCONPostSboxXor13Kernel__<targetP>
        <<<TFHEpp::ascon_word_bits, 256, 0, st>>>(t);
    __ASCONPostSboxXor0Not2Kernel__<targetP>
        <<<TFHEpp::ascon_word_bits, 256, 0, st>>>(t);
    __ASCONLinearDiffusionKernel__<targetP>
        <<<TFHEpp::ascon_state_bits, 256, 0, st>>>(state, t);
    CuCheckError();
}

template <class iksP, class brP, class ahP>
void DeviceASCONPermute(
    typename brP::targetP::T* const state, typename brP::targetP::T* const t,
    const typename iksP::targetP::T* const ksk,
    const typename brP::targetP::T* const rom,
    typename brP::targetP::T* const sbox_input,
    typename brP::targetP::T* const sbox_output,
    typename brP::targetP::T* const address_poly, NTTValue* const address_fft,
    typename brP::targetP::T* const acc, typename brP::targetP::T* const temp,
    typename brP::targetP::T* const product,
    typename brP::domainP::T* const domain_tlwe,
    typename brP::targetP::T* const cb_acc,
    typename brP::targetP::T* const cb_temptrlwe, const std::size_t rounds,
    const cudaStream_t st, const int gpuNum)
{
    const std::size_t begin = TFHEpp::ascon_round_constants.size() - rounds;
    for (std::size_t i = begin; i < TFHEpp::ascon_round_constants.size(); i++) {
        DeviceASCONRound<iksP, brP, ahP>(
            state, t, ksk, rom, sbox_input, sbox_output, address_poly,
            address_fft, acc, temp, product, domain_tlwe, cb_acc, cb_temptrlwe,
            TFHEpp::ascon_round_constants[i], st, gpuNum);
    }
}

template <class P>
typename P::T* DeviceCopyTLWEsToDevice(std::span<const TFHEpp::TLWE<P>> tlwes,
                                       const cudaStream_t st)
{
    if (tlwes.empty()) return nullptr;
    typename P::T* device = nullptr;
    const size_t bytes =
        tlwes.size() * TLWEElements<P>() * sizeof(typename P::T);
    CuSafeCall(cudaMalloc((void**)&device, bytes));
    CuSafeCall(cudaMemcpyAsync(device, tlwes.data(), bytes,
                               cudaMemcpyHostToDevice, st));
    return device;
}

template <class P>
typename P::T* DeviceAllocateTLWEs(const size_t count)
{
    if (count == 0) return nullptr;
    typename P::T* device = nullptr;
    const size_t bytes = count * TLWEElements<P>() * sizeof(typename P::T);
    CuSafeCall(cudaMalloc((void**)&device, bytes));
    return device;
}

template <class P>
void DeviceCopyTLWEsToHost(std::span<TFHEpp::TLWE<P>> tlwes,
                           const typename P::T* const device,
                           const cudaStream_t st)
{
    if (tlwes.empty()) return;
    const size_t bytes =
        tlwes.size() * TLWEElements<P>() * sizeof(typename P::T);
    CuSafeCall(cudaMemcpyAsync(tlwes.data(), device, bytes,
                               cudaMemcpyDeviceToHost, st));
}

template <class iksP, class brP, class ahP>
struct ASCONDeviceWorkspace {
    using targetP = typename brP::targetP;
    using domainP = typename brP::domainP;
    static_assert(iksP::domainP::k == targetP::k &&
                  iksP::domainP::n == targetP::n &&
                  sizeof(typename iksP::domainP::T) ==
                      sizeof(typename targetP::T));
    static_assert(std::is_same_v<typename iksP::targetP, domainP>);
    static_assert(targetP::k == ahP::k && targetP::n == ahP::n &&
                  sizeof(typename targetP::T) == sizeof(typename ahP::T));

    static constexpr size_t state_bits = TFHEpp::ascon_state_bits;
    static constexpr size_t sbox_columns = TFHEpp::ascon_word_bits;
    static constexpr size_t sbox_address_bit = TFHEpp::ascon_words;
    static constexpr size_t sbox_input_count = sbox_columns * sbox_address_bit;
    static constexpr size_t tlwe_elems = TLWEElements<targetP>();
    static constexpr size_t trlwe_elems = TRLWEElements<targetP>();
    static constexpr size_t state_bytes =
        state_bits * tlwe_elems * sizeof(typename targetP::T);
    static constexpr size_t sbox_tlwe_bytes =
        sbox_input_count * tlwe_elems * sizeof(typename targetP::T);
    static constexpr size_t trlwe_bytes =
        trlwe_elems * sizeof(typename targetP::T);
    static constexpr size_t sbox_trlwe_workspace_bytes =
        sbox_columns * trlwe_bytes;
    static constexpr size_t trgsw_bytes = sbox_input_count *
                                          TRGSWElements<targetP>() *
                                          sizeof(typename targetP::T);
    static constexpr size_t trgswfft_bytes =
        sbox_input_count * TRGSWFFTElements<targetP>() * sizeof(NTTValue);
    static constexpr size_t domain_tlwe_bytes =
        TLWEElements<domainP>() * sizeof(typename domainP::T);
    static constexpr size_t domain_tlwe_workspace_bytes =
        sbox_input_count * domain_tlwe_bytes;
    static constexpr size_t cb_acc_workspace_bytes =
        sbox_input_count * trlwe_bytes;
    static constexpr size_t cb_temptrlwe_workspace_bytes =
        sbox_input_count * CircuitBootstrapLUTCount<targetP> * trlwe_bytes;
    static constexpr size_t ksk_bytes = sizeof(TFHEpp::KeySwitchingKey<iksP>);

    typename targetP::T* state = nullptr;
    typename targetP::T* t = nullptr;
    typename targetP::T* rom = nullptr;
    typename targetP::T* sbox_input = nullptr;
    typename targetP::T* sbox_output = nullptr;
    typename targetP::T* address_poly = nullptr;
    NTTValue* address_fft = nullptr;
    typename targetP::T* acc = nullptr;
    typename targetP::T* temp = nullptr;
    typename targetP::T* product = nullptr;
    typename domainP::T* domain_tlwe = nullptr;
    typename targetP::T* cb_acc = nullptr;
    typename targetP::T* cb_temptrlwe = nullptr;
    typename iksP::targetP::T* ksk = nullptr;

    ASCONDeviceWorkspace(const TFHEpp::EvalKey& ek, const cudaStream_t st)
    {
        CuSafeCall(cudaMalloc((void**)&state, state_bytes));
        CuSafeCall(cudaMalloc((void**)&t, state_bytes));
        CuSafeCall(cudaMalloc((void**)&rom, trlwe_bytes));
        CuSafeCall(cudaMalloc((void**)&sbox_input, sbox_tlwe_bytes));
        CuSafeCall(cudaMalloc((void**)&sbox_output, sbox_tlwe_bytes));
        CuSafeCall(cudaMalloc((void**)&address_poly, trgsw_bytes));
        CuSafeCall(cudaMalloc((void**)&address_fft, trgswfft_bytes));
        CuSafeCall(cudaMalloc((void**)&acc, sbox_trlwe_workspace_bytes));
        CuSafeCall(cudaMalloc((void**)&temp, sbox_trlwe_workspace_bytes));
        CuSafeCall(cudaMalloc((void**)&product, sbox_trlwe_workspace_bytes));
        CuSafeCall(
            cudaMalloc((void**)&domain_tlwe, domain_tlwe_workspace_bytes));
        CuSafeCall(cudaMalloc((void**)&cb_acc, cb_acc_workspace_bytes));
        CuSafeCall(
            cudaMalloc((void**)&cb_temptrlwe, cb_temptrlwe_workspace_bytes));
        CuSafeCall(cudaMalloc((void**)&ksk, ksk_bytes));

        CuSafeCall(cudaMemcpyAsync(ksk, ek.getiksk<iksP>().data(), ksk_bytes,
                                   cudaMemcpyHostToDevice, st));
        UploadASCONSboxROM<targetP>(rom, st);
    }

    ASCONDeviceWorkspace(const ASCONDeviceWorkspace&) = delete;
    ASCONDeviceWorkspace& operator=(const ASCONDeviceWorkspace&) = delete;

    ~ASCONDeviceWorkspace()
    {
        cudaFree(ksk);
        cudaFree(cb_temptrlwe);
        cudaFree(cb_acc);
        cudaFree(domain_tlwe);
        cudaFree(product);
        cudaFree(temp);
        cudaFree(acc);
        cudaFree(address_fft);
        cudaFree(address_poly);
        cudaFree(sbox_output);
        cudaFree(sbox_input);
        cudaFree(rom);
        cudaFree(t);
        cudaFree(state);
    }

    void CopyStateToDevice(const TFHEpp::ASCONState<targetP>& host_state,
                           const cudaStream_t st)
    {
        CuSafeCall(cudaMemcpyAsync(state, host_state.data(), state_bytes,
                                   cudaMemcpyHostToDevice, st));
    }

    void CopyStateToHost(TFHEpp::ASCONState<targetP>& host_state,
                         const cudaStream_t st)
    {
        CuSafeCall(cudaMemcpyAsync(host_state.data(), state, state_bytes,
                                   cudaMemcpyDeviceToHost, st));
    }

    void PermuteP12(const cudaStream_t st, const int gpuNum)
    {
        DeviceASCONPermute<iksP, brP, ahP>(
            state, t, ksk, rom, sbox_input, sbox_output, address_poly,
            address_fft, acc, temp, product, domain_tlwe, cb_acc, cb_temptrlwe,
            12, st, gpuNum);
    }
};

template <class iksP, class brP, class ahP>
void DeviceASCONXOFInitialize(ASCONDeviceWorkspace<iksP, brP, ahP>& workspace,
                              const cudaStream_t st, const int gpuNum)
{
    using targetP = typename brP::targetP;
    auto state = std::make_unique<TFHEpp::ASCONState<targetP>>();
    TFHEpp::ASCONSetXOFInitialState<targetP>(*state);
    workspace.CopyStateToDevice(*state, st);
    workspace.PermuteP12(st, gpuNum);
}

template <class iksP, class brP, class ahP>
void DeviceASCONXOFAbsorb(ASCONDeviceWorkspace<iksP, brP, ahP>& workspace,
                          const typename brP::targetP::T* const input,
                          const size_t input_bits, const cudaStream_t st,
                          const int gpuNum)
{
    using targetP = typename brP::targetP;
    assert(input_bits % 8 == 0);

    std::size_t byte_len = input_bits / 8;
    std::size_t offset_bits = 0;
    while (byte_len >= TFHEpp::ascon_xof_rate_bytes) {
        __ASCONXORRateInputKernel__<targetP>
            <<<TFHEpp::ascon_xof_rate_bits, 256, 0, st>>>(
                workspace.state, input, offset_bits,
                TFHEpp::ascon_xof_rate_bytes);
        workspace.PermuteP12(st, gpuNum);
        offset_bits += TFHEpp::ascon_xof_rate_bits;
        byte_len -= TFHEpp::ascon_xof_rate_bytes;
    }
    if (byte_len > 0) {
        __ASCONXORRateInputKernel__<targetP><<<byte_len * 8, 256, 0, st>>>(
            workspace.state, input, offset_bits, byte_len);
    }
    __ASCONNotRatePadKernel__<targetP>
        <<<1, 256, 0, st>>>(workspace.state, byte_len);
    workspace.PermuteP12(st, gpuNum);
}

template <class iksP, class brP, class ahP>
void DeviceASCONXOFSqueeze(ASCONDeviceWorkspace<iksP, brP, ahP>& workspace,
                           typename brP::targetP::T* const output,
                           const size_t output_bits, const cudaStream_t st,
                           const int gpuNum)
{
    using targetP = typename brP::targetP;
    assert(output_bits % 8 == 0);

    std::size_t byte_len = output_bits / 8;
    std::size_t offset_bits = 0;
    while (byte_len > TFHEpp::ascon_xof_rate_bytes) {
        __ASCONCopyRateOutputKernel__<targetP>
            <<<TFHEpp::ascon_xof_rate_bits, 256, 0, st>>>(
                output, workspace.state, offset_bits,
                TFHEpp::ascon_xof_rate_bytes);
        workspace.PermuteP12(st, gpuNum);
        offset_bits += TFHEpp::ascon_xof_rate_bits;
        byte_len -= TFHEpp::ascon_xof_rate_bytes;
    }
    if (byte_len > 0) {
        __ASCONCopyRateOutputKernel__<targetP><<<byte_len * 8, 256, 0, st>>>(
            output, workspace.state, offset_bits, byte_len);
    }
}

}  // namespace

template <class brP, class ahP>
void InitializeASCON(const TFHEpp::EvalKey& ek, const TFHEpp::SecretKey& sk)
{
    using targetP = typename brP::targetP;
    static_assert(targetP::k == ahP::k && targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "ahP must share the brP::targetP torus ring");
    InitializeBRKey<brP>(ek);

    auto ahk = std::make_unique<AnnihilateKeyPolynomial<ahP>>();
    AnnihilateKeyPolynomialGen<ahP>(*ahk, sk);
    AnnihilateKeyPolynomialToDevice<ahP>(*ahk, _gpuNum);

    auto cbsk = std::make_unique<CBswitchingKeyPolynomial<ahP>>();
    CBswitchingKeyPolynomialGen<ahP>(*cbsk, sk);
    CBswitchingKeyPolynomialToDevice<ahP>(*cbsk, _gpuNum);
}

template <class brP, class ahP>
void CleanUpASCON()
{
    DeleteCBswitchingKey<ahP>(_gpuNum);
    DeleteAnnihilateKey<ahP>(_gpuNum);
    DeleteBRKey<brP>();
}

template <class iksP, class brP, class ahP>
void ASCONXOFInitialize(TFHEpp::ASCONState<typename brP::targetP>& state,
                        const TFHEpp::EvalKey& ek, Stream st)
{
    using targetP = typename brP::targetP;
    using inputP = typename iksP::domainP;
    static_assert(inputP::k == targetP::k && inputP::n == targetP::n &&
                      sizeof(typename inputP::T) == sizeof(typename targetP::T),
                  "ASCON state and blind-rotation target rings must match");

    cudaSetDevice(st.device_id());

    ASCONDeviceWorkspace<iksP, brP, ahP> workspace(ek, st.st());
    DeviceASCONXOFInitialize<iksP, brP, ahP>(workspace, st.st(),
                                             st.device_id());
    workspace.CopyStateToHost(state, st.st());
    CuSafeCall(cudaStreamSynchronize(st.st()));
}

template <class iksP, class brP, class ahP>
void ASCONXOFAbsorb(TFHEpp::ASCONState<typename brP::targetP>& state,
                    std::span<const TFHEpp::TLWE<typename brP::targetP>> input,
                    const TFHEpp::EvalKey& ek, Stream st)
{
    using targetP = typename brP::targetP;
    using inputP = typename iksP::domainP;
    static_assert(inputP::k == targetP::k && inputP::n == targetP::n &&
                      sizeof(typename inputP::T) == sizeof(typename targetP::T),
                  "ASCON input and blind-rotation target rings must match");
    assert(input.size() % 8 == 0);

    cudaSetDevice(st.device_id());

    ASCONDeviceWorkspace<iksP, brP, ahP> workspace(ek, st.st());
    workspace.CopyStateToDevice(state, st.st());
    typename targetP::T* const d_input =
        DeviceCopyTLWEsToDevice<targetP>(input, st.st());

    DeviceASCONXOFAbsorb<iksP, brP, ahP>(workspace, d_input, input.size(),
                                         st.st(), st.device_id());
    workspace.CopyStateToHost(state, st.st());
    CuSafeCall(cudaStreamSynchronize(st.st()));
    if (d_input != nullptr) cudaFree(d_input);
}

template <class iksP, class brP, class ahP>
void ASCONXOFSqueeze(TFHEpp::ASCONState<typename brP::targetP>& state,
                     std::span<TFHEpp::TLWE<typename brP::targetP>> output,
                     const TFHEpp::EvalKey& ek, Stream st)
{
    using targetP = typename brP::targetP;
    using inputP = typename iksP::domainP;
    static_assert(inputP::k == targetP::k && inputP::n == targetP::n &&
                      sizeof(typename inputP::T) == sizeof(typename targetP::T),
                  "ASCON state and blind-rotation target rings must match");
    assert(output.size() % 8 == 0);

    cudaSetDevice(st.device_id());

    ASCONDeviceWorkspace<iksP, brP, ahP> workspace(ek, st.st());
    workspace.CopyStateToDevice(state, st.st());
    typename targetP::T* const d_output =
        DeviceAllocateTLWEs<targetP>(output.size());

    DeviceASCONXOFSqueeze<iksP, brP, ahP>(workspace, d_output, output.size(),
                                          st.st(), st.device_id());
    DeviceCopyTLWEsToHost<targetP>(output, d_output, st.st());
    workspace.CopyStateToHost(state, st.st());
    CuSafeCall(cudaStreamSynchronize(st.st()));
    if (d_output != nullptr) cudaFree(d_output);
}

template <class iksP, class brP, class ahP>
void ASCONXOF(std::span<TFHEpp::TLWE<typename brP::targetP>> output,
              std::span<const TFHEpp::TLWE<typename brP::targetP>> input,
              const TFHEpp::EvalKey& ek, Stream st)
{
    using targetP = typename brP::targetP;
    using inputP = typename iksP::domainP;
    static_assert(inputP::k == targetP::k && inputP::n == targetP::n &&
                      sizeof(typename inputP::T) == sizeof(typename targetP::T),
                  "ASCON input and blind-rotation target rings must match");
    assert(input.size() % 8 == 0);
    assert(output.size() % 8 == 0);

    cudaSetDevice(st.device_id());

    ASCONDeviceWorkspace<iksP, brP, ahP> workspace(ek, st.st());
    typename targetP::T* const d_input =
        DeviceCopyTLWEsToDevice<targetP>(input, st.st());
    typename targetP::T* const d_output =
        DeviceAllocateTLWEs<targetP>(output.size());

    DeviceASCONXOFInitialize<iksP, brP, ahP>(workspace, st.st(),
                                             st.device_id());
    DeviceASCONXOFAbsorb<iksP, brP, ahP>(workspace, d_input, input.size(),
                                         st.st(), st.device_id());
    DeviceASCONXOFSqueeze<iksP, brP, ahP>(workspace, d_output, output.size(),
                                          st.st(), st.device_id());
    DeviceCopyTLWEsToHost<targetP>(output, d_output, st.st());
    CuSafeCall(cudaStreamSynchronize(st.st()));

    if (d_output != nullptr) cudaFree(d_output);
    if (d_input != nullptr) cudaFree(d_input);
}

template void InitializeASCON<TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);
template void InitializeASCON<TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);

template void CleanUpASCON<TFHEpp::lvl02param, TFHEpp::AHlvl2param>();
template void CleanUpASCON<TFHEpp::lvlh2param, TFHEpp::AHlvl2param>();

template void
ASCONXOFInitialize<TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&, const TFHEpp::EvalKey&, Stream);
template void
ASCONXOFInitialize<TFHEpp::lvl2hparam, TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&, const TFHEpp::EvalKey&, Stream);

template void
ASCONXOFAbsorb<TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&,
    std::span<const TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&,
    Stream);
template void
ASCONXOFAbsorb<TFHEpp::lvl2hparam, TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&,
    std::span<const TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&,
    Stream);

template void
ASCONXOFSqueeze<TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&,
    std::span<TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&, Stream);
template void
ASCONXOFSqueeze<TFHEpp::lvl2hparam, TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::lvl2param>&,
    std::span<TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&, Stream);

template void
ASCONXOF<TFHEpp::lvl20param, TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    std::span<TFHEpp::TLWE<TFHEpp::lvl2param>>,
    std::span<const TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&,
    Stream);
template void
ASCONXOF<TFHEpp::lvl2hparam, TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    std::span<TFHEpp::TLWE<TFHEpp::lvl2param>>,
    std::span<const TFHEpp::TLWE<TFHEpp::lvl2param>>, const TFHEpp::EvalKey&,
    Stream);

#if defined(USE_DIFFERENT_BR_PARAM) && defined(USE_DIFFERENT_AH_PARAM)
template void InitializeASCON<TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);
template void InitializeASCON<TFHEpp::cblvlh2param, TFHEpp::cbAHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);

template void CleanUpASCON<TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>();
template void CleanUpASCON<TFHEpp::cblvlh2param, TFHEpp::cbAHlvl2param>();

template void ASCONXOFInitialize<TFHEpp::lvl20param, TFHEpp::cblvl02param,
                                 TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&, const TFHEpp::EvalKey&, Stream);
template void ASCONXOFInitialize<TFHEpp::lvl2hparam, TFHEpp::cblvlh2param,
                                 TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&, const TFHEpp::EvalKey&, Stream);

template void
ASCONXOFAbsorb<TFHEpp::lvl20param, TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&,
    std::span<const TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);
template void
ASCONXOFAbsorb<TFHEpp::lvl2hparam, TFHEpp::cblvlh2param, TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&,
    std::span<const TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);

template void ASCONXOFSqueeze<TFHEpp::lvl20param, TFHEpp::cblvl02param,
                              TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&,
    std::span<TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);
template void ASCONXOFSqueeze<TFHEpp::lvl2hparam, TFHEpp::cblvlh2param,
                              TFHEpp::cbAHlvl2param>(
    TFHEpp::ASCONState<TFHEpp::cblvl2param>&,
    std::span<TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);

template void
ASCONXOF<TFHEpp::lvl20param, TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>(
    std::span<TFHEpp::TLWE<TFHEpp::cblvl2param>>,
    std::span<const TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);
template void
ASCONXOF<TFHEpp::lvl2hparam, TFHEpp::cblvlh2param, TFHEpp::cbAHlvl2param>(
    std::span<TFHEpp::TLWE<TFHEpp::cblvl2param>>,
    std::span<const TFHEpp::TLWE<TFHEpp::cblvl2param>>, const TFHEpp::EvalKey&,
    Stream);
#endif

}  // namespace cufhe
