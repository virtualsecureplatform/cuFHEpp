#include <include/aes_gpu.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <include/utils_gpu.cuh>

namespace cufhe {

extern std::vector<CuNTTHandler<>*> ntt_handlers;
extern std::vector<CuNTTHandler<TFHEpp::lvl2param::n>*> ntt_handlers_lvl02;

namespace {

__device__ __constant__ uint32_t kInvMixColumnMasks[32] = {
    0x2161a1e0, 0x62a2e321, 0xc445c643, 0xa9eb2d67,
    0x72b6fa2e, 0xe46cf45c, 0xc8d8e8b8, 0x90b0d070,
    0x61a1e021, 0xa2e32162, 0x45c643c4, 0xeb2d67a9,
    0xb6fa2e72, 0x6cf45ce4, 0xd8e8b8c8, 0xb0d07090,
    0xa1e02161, 0xe32162a2, 0xc643c445, 0x2d67a9eb,
    0xfa2e72b6, 0xf45ce46c, 0xe8b8c8d8, 0xd07090b0,
    0xe02161a1, 0x2162a2e3, 0x43c445c6, 0x67a9eb2d,
    0x2e72b6fa, 0x5ce46cf4, 0xb8c8d8e8, 0x7090b0d0};

template <class P>
constexpr bool is_lvl1_ring_v = P::n == TFHEpp::lvl1param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl1param::T);

template <class P>
constexpr bool is_lvl2_ring_v = P::n == TFHEpp::lvl2param::n &&
                                sizeof(typename P::T) ==
                                    sizeof(typename TFHEpp::lvl2param::T);

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
                      "Unsupported AES target ring");
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

template <class iksP>
__global__ void __IdentityKeySwitchLocalKernel__(
    typename iksP::targetP::T* const out,
    const typename iksP::domainP::T* const in,
    const typename iksP::targetP::T* const ksk)
{
    KeySwitchFromTLWE<iksP>(out, in, ksk);
}

template <class iksP>
__global__ void __IdentityKeySwitchBatchLocalKernel__(
    typename iksP::targetP::T* const out, const size_t out_stride,
    const typename iksP::domainP::T* const in, const size_t in_stride,
    const typename iksP::targetP::T* const ksk,
    const size_t batch_count)
{
    const size_t batch = blockIdx.x;
    if (batch >= batch_count) return;
    KeySwitchFromTLWE<iksP>(out + batch * out_stride,
                            in + batch * in_stride, ksk);
}

template <class P>
__global__ void __NegateTLWEKernel__(typename P::T* const out,
                                     const typename P::T* const in)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < TLWEElements<P>(); i += bdim) out[i] = -in[i];
}

#if defined(USE_FFT)
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
        if constexpr (N == 1024)
            GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
        else if constexpr (N == 2048)
            GPUFFTForward1024(sh_fft, ntt.forward_root_, tid);
#else
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
    }
    else {
#ifdef USE_GPU_FFT
        for (int s = 0; s < 3; s++) __syncthreads();
#else
        for (int s = 0; s < Degree<N>::depth - 1; s++) __syncthreads();
#endif
    }

    if (tid < half_n) out[out_index + tid] = sh_fft[tid];
}

template <class P>
__device__ inline typename P::T AESTorusFromDouble(const double value)
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
                   << (std::numeric_limits<typename P::T>::digits -
                       i * bgbit));
    return offset;
}

template <class P>
__device__ inline void ExternalProductTRLWE_TRGSWFFT_AES(
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
                if constexpr (N == 1024)
                    GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
                else if constexpr (N == 2048)
                    GPUFFTForward1024(sh_fft, ntt.forward_root_, tid);
#else
                NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
#endif
            }
            else {
#ifdef USE_GPU_FFT
                for (int s = 0; s < 3; s++) __syncthreads();
#else
                for (int s = 0; s < Degree<N>::depth - 1; s++)
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
            if constexpr (N == 1024)
                GPUFFTInverse512(sh_inv, ntt.inverse_root_, tid);
            else if constexpr (N == 2048)
                GPUFFTInverse1024(sh_inv, ntt.inverse_root_, tid);
#else
            NSMFFT_inverse<HalfDegree<Degree<N>>>(sh_inv);
#endif
        }
        else {
#ifdef USE_GPU_FFT
            for (int s = 0; s < 3; s++) __syncthreads();
#else
            for (int s = 0; s < Degree<N>::depth - 1; s++) __syncthreads();
#endif
        }

        if (tid < half_n) {
            NTTValue val = sh_inv[tid];
#ifdef USE_GPU_FFT
            val *= __ldg(&ntt.untwist_[tid]);
#endif
            out[k_idx * N + tid] = AESTorusFromDouble<P>(val.x);
            out[k_idx * N + tid + half_n] = AESTorusFromDouble<P>(val.y);
        }
        __syncthreads();
    }
}

template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<P>)
void __AESExternalProductKernel__(typename P::T* const out,
                                  const typename P::T* const in,
                                  const NTTValue* const trgswfft,
                                  const CuNTTHandler<P::n> ntt)
{
    extern __shared__ NTTValue sh_acc_ntt[];
    ExternalProductTRLWE_TRGSWFFT_AES<P>(out, in, trgswfft, sh_acc_ntt, ntt);
}
#endif  // USE_FFT

template <class P>
__global__ void __PolynomialMulByXaiMinusOneTRLWEKernel__(
    typename P::T* const out, const typename P::T* const in,
    const uint32_t exponent)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t N = P::n;
    constexpr uint32_t total = TRLWEElements<P>();
    const bool small = exponent < N;
    const uint32_t a = small ? exponent : exponent - N;

    for (uint32_t index = tid; index < total; index += bdim) {
        const uint32_t poly = index >> P::nbit;
        const uint32_t i = index & (N - 1);
        const typename P::T* const src = in + poly * N;
        typename P::T rotated;
        if (small) {
            rotated = i < a ? -src[i - a + N] : src[i - a];
        }
        else {
            rotated = i < a ? src[i - a + N] : -src[i - a];
        }
        out[index] = rotated - src[i];
    }
}

template <class P>
__global__ void __TRLWEAddKernel__(typename P::T* const out,
                                   const typename P::T* const a,
                                   const typename P::T* const b)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < TRLWEElements<P>(); i += bdim)
        out[i] = a[i] + b[i];
}

template <class P>
__global__ void __TRLWEAddInPlaceKernel__(typename P::T* const out,
                                          const typename P::T* const addend)
{
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (uint32_t i = tid; i < TRLWEElements<P>(); i += bdim) out[i] += addend[i];
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
__global__ void __SampleExtractManyKernel__(typename P::T* const out,
                                            const typename P::T* const acc)
{
    const uint32_t out_idx = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    typename P::T* const tlwe = out + out_idx * TLWEElements<P>();
    for (uint32_t i = tid; i < TLWEElements<P>(); i += bdim)
        tlwe[i] = SampleExtractValue<P>(acc, out_idx, i);
}

template <class P>
__global__ void __AddRoundKeyKernel__(
    typename P::T* const state,
    const typename P::T* const expandedkey_round)
{
    const uint32_t bit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t elems = TLWEElements<P>();
    const uint32_t row = bit / 32;
    const uint32_t rem = bit & 31;
    const uint32_t col = rem / 8;
    const uint32_t bit_in_byte = rem & 7;
    const uint32_t key_bit = col * 32 + row * 8 + bit_in_byte;
    typename P::T* const out = state + bit * elems;
    const typename P::T* const key = expandedkey_round + key_bit * elems;

    for (uint32_t i = tid; i < elems; i += bdim) {
        typename P::T value = out[i] + key[i];
        if (i == P::k * P::n) value += BitMu<P>();
        out[i] = value;
    }
}

template <class P>
__global__ void __InvShiftRowsKernel__(typename P::T* const out,
                                       const typename P::T* const in)
{
    const uint32_t bit = blockIdx.x;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t elems = TLWEElements<P>();
    const uint32_t row = bit / 32;
    const uint32_t rem = bit & 31;
    const uint32_t col = rem / 8;
    const uint32_t bit_in_byte = rem & 7;
    const uint32_t shift = row == 0 ? 0 : 4 - row;
    const uint32_t src_col = (col + shift) & 3;
    const uint32_t src_bit = row * 32 + src_col * 8 + bit_in_byte;

    for (uint32_t i = tid; i < elems; i += bdim)
        out[bit * elems + i] = in[src_bit * elems + i];
}

template <class P>
__global__ void __InvMixColumnsKernel__(typename P::T* const out,
                                        const typename P::T* const in)
{
    const uint32_t col = blockIdx.x;
    const uint32_t y = blockIdx.y;
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint32_t elems = TLWEElements<P>();
    const uint32_t mask = kInvMixColumnMasks[y];
    const uint32_t weight = __popc(mask);
    const uint32_t dst_bit = (y / 8) * 32 + col * 8 + (y & 7);

    for (uint32_t coeff = tid; coeff < elems; coeff += bdim) {
        typename P::T value = 0;
        for (uint32_t src = 0; src < 32; src++) {
            if ((mask >> src) & 1U) {
                const uint32_t src_bit = (src / 8) * 32 + col * 8 + (src & 7);
                value += in[src_bit * elems + coeff];
            }
        }
        if (coeff == P::k * P::n)
            value += static_cast<typename P::T>(weight - 1) * BitMu<P>();
        out[dst_bit * elems + coeff] = value;
    }
}

template <class P>
void DeviceAddRoundKey(typename P::T* const state,
                       const typename P::T* const roundkey,
                       const cudaStream_t st)
{
    __AddRoundKeyKernel__<P><<<128, 256, 0, st>>>(state, roundkey);
    CuCheckError();
}

template <class P>
void DeviceInvShiftRows(typename P::T* const state, typename P::T* const scratch,
                        const cudaStream_t st)
{
    constexpr size_t bytes =
        static_cast<size_t>(128) * TLWEElements<P>() * sizeof(typename P::T);
    __InvShiftRowsKernel__<P><<<128, 256, 0, st>>>(scratch, state);
    CuCheckError();
    CuSafeCall(cudaMemcpyAsync(state, scratch, bytes, cudaMemcpyDeviceToDevice,
                               st));
}

template <class P>
void DeviceInvMixColumns(typename P::T* const state,
                         typename P::T* const scratch,
                         const cudaStream_t st)
{
    constexpr size_t bytes =
        static_cast<size_t>(128) * TLWEElements<P>() * sizeof(typename P::T);
    dim3 grid(4, 32);
    __InvMixColumnsKernel__<P><<<grid, 256, 0, st>>>(scratch, state);
    CuCheckError();
    CuSafeCall(cudaMemcpyAsync(state, scratch, bytes, cudaMemcpyDeviceToDevice,
                               st));
}

template <class P, uint32_t address_bit, uint32_t width_bit, uint32_t num_tlwe>
void DeviceLROMUX(typename P::T* const out, const NTTValue* const address,
                  const typename P::T* const data, typename P::T* const acc,
                  typename P::T* const temp, typename P::T* const product,
                  const cudaStream_t st, const int gpuNum)
{
    static_assert(address_bit == width_bit,
                  "CUDA AES LROMUX currently supports one TRLWE ROM tables");
    static_assert(num_tlwe <= (1U << (P::nbit - width_bit)));
    constexpr uint32_t threads = NUM_THREAD4HOMGATE<P>;
    constexpr size_t shmem = MEM4HOMGATE<P>;
    constexpr size_t trgswfft_elems = TRGSWFFTElements<P>();

    static bool external_product_attribute_set = false;
    if (!external_product_attribute_set) {
        CuSafeCall(cudaFuncSetAttribute(
            __AESExternalProductKernel__<P>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmem));
        external_product_attribute_set = true;
    }

    __PolynomialMulByXaiMinusOneTRLWEKernel__<P><<<1, threads, 0, st>>>(
        temp, data, 2 * P::n - (P::n >> 1));
    __AESExternalProductKernel__<P><<<1, threads, shmem, st>>>(
        product, temp, address + (width_bit - 1) * trgswfft_elems,
        *RingHandler<P>(gpuNum));
    __TRLWEAddKernel__<P><<<1, threads, 0, st>>>(acc, product, data);

    for (uint32_t bit = 2; bit <= width_bit; bit++) {
        __PolynomialMulByXaiMinusOneTRLWEKernel__<P><<<1, threads, 0, st>>>(
            temp, acc, 2 * P::n - (P::n >> bit));
        __AESExternalProductKernel__<P><<<1, threads, shmem, st>>>(
            product, temp, address + (width_bit - bit) * trgswfft_elems,
            *RingHandler<P>(gpuNum));
        __TRLWEAddInPlaceKernel__<P><<<1, threads, 0, st>>>(acc, product);
    }

    __SampleExtractManyKernel__<P, num_tlwe>
        <<<num_tlwe, threads, 0, st>>>(out, acc);
    CuCheckError();
}

template <class iksP, class brP, class ahP>
void DeviceAESInvSboxROM(
    typename brP::targetP::T* const out,
    const typename iksP::domainP::T* const in,
    const typename iksP::targetP::T* const ksk,
    const typename brP::targetP::T* const rom,
    typename brP::targetP::T* const address_poly,
    NTTValue* const address_fft, typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temp,
    typename brP::targetP::T* const product,
    typename brP::domainP::T* const domain_tlwe,
    typename iksP::domainP::T* const shifted,
    typename brP::targetP::T* const cb_acc,
    typename brP::targetP::T* const cb_temptrlwe, const cudaStream_t st,
    const int gpuNum)
{
    using targetP = typename brP::targetP;
    using domainP = typename brP::domainP;
    using inputP = typename iksP::domainP;
    static_assert(std::is_same_v<typename iksP::targetP, domainP>,
                  "iksP target must match blind-rotation domain");
    static_assert(std::is_same_v<inputP, targetP>,
                  "AES state ciphertexts must be in brP::targetP");
    static_assert(targetP::k == ahP::k && targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "ahP must share the brP::targetP torus ring");

    constexpr uint32_t address_bit = 8;
    constexpr uint32_t words_bit = 3;
    constexpr uint32_t width_bit = targetP::nbit - words_bit;
    static_assert(address_bit == width_bit,
                  "CUDA AES currently supports target rings with nbit == 11");

    constexpr uint32_t input_tlwe_elems = TLWEElements<inputP>();
    constexpr uint32_t domain_tlwe_elems = TLWEElements<domainP>();
    constexpr size_t trgsw_elems = TRGSWElements<targetP>();
    constexpr size_t trgsw_rows = TRGSWRows<targetP>() * (targetP::k + 1);

    for (uint32_t bit = 0; bit < address_bit; bit++) {
        const typename inputP::T* bit_in = in + bit * input_tlwe_elems;
        if constexpr (width_bit < address_bit) {
            if (bit >= width_bit) {
                __NegateTLWEKernel__<inputP><<<1, 256, 0, st>>>(shifted, bit_in);
                bit_in = shifted;
            }
        }

        const uint32_t ks_threads =
            std::min<uint32_t>(1024, domain_tlwe_elems);
        __IdentityKeySwitchLocalKernel__<iksP>
            <<<1, ks_threads, 0, st>>>(domain_tlwe, bit_in, ksk);

        AnnihilateCircuitBootstrappingWithWorkspace<brP, ahP>(
            address_poly + static_cast<size_t>(bit) * trgsw_elems,
            domain_tlwe, cb_acc, cb_temptrlwe, st, gpuNum);
    }

    __TRGSWToFFTKernel__<targetP>
        <<<address_bit * trgsw_rows, targetP::n / 2, 0, st>>>(
            address_fft, address_poly, *RingHandler<targetP>(gpuNum));
    CuCheckError();

    DeviceLROMUX<targetP, address_bit, width_bit, 8>(
        out, address_fft, rom, acc, temp, product, st, gpuNum);
}

template <class iksP, class brP, class ahP>
void DeviceAESInvSubBytesParallel(
    typename brP::targetP::T* const state,
    const typename iksP::targetP::T* const ksk,
    const typename brP::targetP::T* const rom,
    typename brP::targetP::T* const address_poly, NTTValue* const address_fft,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temp,
    typename brP::targetP::T* const product,
    typename brP::domainP::T* const domain_tlwe,
    typename iksP::domainP::T* const shifted,
    typename brP::targetP::T* const cb_acc,
    typename brP::targetP::T* const cb_temptrlwe,
    const std::array<cudaStream_t, 16>& byte_streams,
    const cudaEvent_t ready_event,
    const std::array<cudaEvent_t, 16>& done_events,
    const cudaStream_t main_stream, const int gpuNum)
{
    using targetP = typename brP::targetP;
    using domainP = typename brP::domainP;
    using inputP = typename iksP::domainP;
    constexpr uint32_t aes_bytes = 16;
    constexpr uint32_t address_bit = 8;
    constexpr size_t tlwe_elems = TLWEElements<targetP>();
    constexpr size_t trlwe_elems = TRLWEElements<targetP>();
    constexpr size_t trgsw_elems = TRGSWElements<targetP>();
    constexpr size_t trgswfft_elems = TRGSWFFTElements<targetP>();
    constexpr size_t domain_tlwe_elems = TLWEElements<domainP>();
    constexpr size_t input_tlwe_elems = TLWEElements<inputP>();
    constexpr size_t cb_temptrlwe_elems =
        static_cast<size_t>(targetP::l) * trlwe_elems;

    CuSafeCall(cudaEventRecord(ready_event, main_stream));
    for (uint32_t byte = 0; byte < aes_bytes; byte++) {
        const cudaStream_t byte_stream = byte_streams[byte];
        CuSafeCall(cudaStreamWaitEvent(byte_stream, ready_event, 0));
        DeviceAESInvSboxROM<iksP, brP, ahP>(
            state + static_cast<size_t>(byte) * 8 * tlwe_elems,
            state + static_cast<size_t>(byte) * 8 * tlwe_elems, ksk, rom,
            address_poly + static_cast<size_t>(byte) * address_bit * trgsw_elems,
            address_fft +
                static_cast<size_t>(byte) * address_bit * trgswfft_elems,
            acc + static_cast<size_t>(byte) * trlwe_elems,
            temp + static_cast<size_t>(byte) * trlwe_elems,
            product + static_cast<size_t>(byte) * trlwe_elems,
            domain_tlwe + static_cast<size_t>(byte) * domain_tlwe_elems,
            shifted + static_cast<size_t>(byte) * input_tlwe_elems,
            cb_acc + static_cast<size_t>(byte) * trlwe_elems,
            cb_temptrlwe + static_cast<size_t>(byte) * cb_temptrlwe_elems,
            byte_stream, gpuNum);
        CuSafeCall(cudaEventRecord(done_events[byte], byte_stream));
    }
    for (uint32_t byte = 0; byte < aes_bytes; byte++)
        CuSafeCall(cudaStreamWaitEvent(main_stream, done_events[byte], 0));
}

template <class iksP, class brP, class ahP>
void DeviceAESInvSubBytesBatched(
    typename brP::targetP::T* const state,
    const typename iksP::targetP::T* const ksk,
    const typename brP::targetP::T* const rom,
    typename brP::targetP::T* const address_poly, NTTValue* const address_fft,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temp,
    typename brP::targetP::T* const product,
    typename brP::domainP::T* const domain_tlwe,
    typename brP::targetP::T* const cb_acc,
    typename brP::targetP::T* const cb_temptrlwe,
    const std::array<cudaStream_t, 16>& byte_streams,
    const cudaEvent_t ready_event,
    const std::array<cudaEvent_t, 16>& done_events,
    const cudaStream_t main_stream, const int gpuNum)
{
    using targetP = typename brP::targetP;
    using domainP = typename brP::domainP;
    using inputP = typename iksP::domainP;
    static_assert(std::is_same_v<typename iksP::targetP, domainP>,
                  "iksP target must match blind-rotation domain");
    static_assert(std::is_same_v<inputP, targetP>,
                  "AES state ciphertexts must be in brP::targetP");
    static_assert(targetP::k == ahP::k && targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "ahP must share the brP::targetP torus ring");

    constexpr uint32_t aes_bytes = 16;
    constexpr uint32_t address_bit = 8;
    constexpr uint32_t words_bit = 3;
    constexpr uint32_t width_bit = targetP::nbit - words_bit;
    static_assert(address_bit == width_bit,
                  "CUDA AES currently supports target rings with nbit == 11");

    constexpr size_t batch_count =
        static_cast<size_t>(aes_bytes) * address_bit;
    constexpr size_t tlwe_elems = TLWEElements<targetP>();
    constexpr size_t trlwe_elems = TRLWEElements<targetP>();
    constexpr size_t trgsw_elems = TRGSWElements<targetP>();
    constexpr size_t trgswfft_elems = TRGSWFFTElements<targetP>();
    constexpr size_t trgsw_rows = TRGSWRows<targetP>() * (targetP::k + 1);
    constexpr size_t domain_tlwe_elems = TLWEElements<domainP>();
    constexpr size_t input_tlwe_elems = TLWEElements<inputP>();

    const uint32_t ks_threads = std::min<uint32_t>(1024, domain_tlwe_elems);
    __IdentityKeySwitchBatchLocalKernel__<iksP>
        <<<batch_count, ks_threads, 0, main_stream>>>(
            domain_tlwe, domain_tlwe_elems, state, input_tlwe_elems, ksk,
            batch_count);

    AnnihilateCircuitBootstrappingBatchWithWorkspace<brP, ahP>(
        address_poly, trgsw_elems, domain_tlwe, domain_tlwe_elems, cb_acc,
        cb_temptrlwe, batch_count, main_stream, gpuNum);

    __TRGSWToFFTKernel__<targetP>
        <<<batch_count * trgsw_rows, targetP::n / 2, 0, main_stream>>>(
            address_fft, address_poly, *RingHandler<targetP>(gpuNum));
    CuCheckError();

    CuSafeCall(cudaEventRecord(ready_event, main_stream));
    for (uint32_t byte = 0; byte < aes_bytes; byte++) {
        const cudaStream_t byte_stream = byte_streams[byte];
        CuSafeCall(cudaStreamWaitEvent(byte_stream, ready_event, 0));
        DeviceLROMUX<targetP, address_bit, width_bit, 8>(
            state + static_cast<size_t>(byte) * 8 * tlwe_elems,
            address_fft + static_cast<size_t>(byte) * address_bit *
                              trgswfft_elems,
            rom, acc + static_cast<size_t>(byte) * trlwe_elems,
            temp + static_cast<size_t>(byte) * trlwe_elems,
            product + static_cast<size_t>(byte) * trlwe_elems, byte_stream,
            gpuNum);
        CuSafeCall(cudaEventRecord(done_events[byte], byte_stream));
    }
    for (uint32_t byte = 0; byte < aes_bytes; byte++)
        CuSafeCall(cudaStreamWaitEvent(main_stream, done_events[byte], 0));
}

template <class P>
void UploadAESInvSboxROM(typename P::T* const d_rom, const cudaStream_t st)
{
    auto rom = std::make_unique<TFHEpp::TRLWE<P>>();
    *rom = {};
    (*rom)[P::k] = TFHEpp::AESInvSboxROMPoly<P>()[0];
    CuSafeCall(cudaMemcpyAsync(d_rom, rom->data(), sizeof(*rom),
                               cudaMemcpyHostToDevice, st));
}

template <class brP>
void InitializeBRKey(const TFHEpp::EvalKey& ek)
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl2param::n) {
        InitializeNTThandlers_lvl02(_gpuNum);
#ifdef USE_KEY_BUNDLE
        InitializeXaiNTT_lvl02(_gpuNum);
        InitializeOneTRGSWNTT_lvl02(_gpuNum);
#endif
    }
    else {
        InitializeNTThandlers(_gpuNum);
#ifdef USE_KEY_BUNDLE
        InitializeXaiNTT(_gpuNum);
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
#ifdef USE_KEY_BUNDLE
        DeleteXaiNTT_lvl02();
        DeleteOneTRGSWNTT_lvl02();
#endif
        DeleteBootstrappingKeyNTT_lvl02(_gpuNum);
    }
    else {
#ifdef USE_KEY_BUNDLE
        DeleteXaiNTT();
        DeleteOneTRGSWNTT();
#endif
        DeleteBootstrappingKeyNTT(_gpuNum);
    }
}

}  // namespace

template <class brP, class ahP>
void InitializeAES(const TFHEpp::EvalKey& ek, const TFHEpp::SecretKey& sk)
{
    static_assert(std::is_same_v<typename brP::targetP, typename ahP::baseP> ||
                      std::is_same_v<typename brP::targetP, ahP>,
                  "ahP must be compatible with brP::targetP");
    InitializeBRKey<brP>(ek);

    auto ahk = std::make_unique<AnnihilateKeyPolynomial<ahP>>();
    AnnihilateKeyPolynomialGen<ahP>(*ahk, sk);
    AnnihilateKeyPolynomialToDevice<ahP>(*ahk, _gpuNum);

    auto cbsk = std::make_unique<CBswitchingKeyPolynomial<ahP>>();
    CBswitchingKeyPolynomialGen<ahP>(*cbsk, sk);
    CBswitchingKeyPolynomialToDevice<ahP>(*cbsk, _gpuNum);
}

template <class brP, class ahP>
void CleanUpAES()
{
    DeleteCBswitchingKey<ahP>(_gpuNum);
    DeleteAnnihilateKey<ahP>(_gpuNum);
    DeleteBRKey<brP>();
}

template <class iksP, class brP, class ahP>
void AESDec(
    std::array<TFHEpp::TLWE<typename brP::targetP>, 128>& plain,
    const std::array<TFHEpp::TLWE<typename iksP::domainP>, 128>& cipher,
    const std::array<std::array<TFHEpp::TLWE<typename brP::targetP>, 128>,
                     TFHEpp::Nr + 1>& expandedkey,
    const TFHEpp::EvalKey& ek, Stream st)
{
    using targetP = typename brP::targetP;
    using inputP = typename iksP::domainP;
    static_assert(std::is_same_v<inputP, targetP>,
                  "AESDec expects ciphertexts in brP::targetP");

    cudaSetDevice(st.device_id());

    constexpr uint32_t aes_bytes = 16;
    constexpr uint32_t sbox_address_bit = 8;
    constexpr size_t sbox_batch_count =
        static_cast<size_t>(aes_bytes) * sbox_address_bit;
    constexpr uint32_t tlwe_elems = TLWEElements<targetP>();
    constexpr size_t state_bytes =
        static_cast<size_t>(128) * tlwe_elems * sizeof(typename targetP::T);
    constexpr size_t roundkeys_bytes =
        static_cast<size_t>(TFHEpp::Nr + 1) * state_bytes;
    constexpr size_t trlwe_bytes =
        TRLWEElements<targetP>() * sizeof(typename targetP::T);
    constexpr size_t sbox_trlwe_workspace_bytes =
        static_cast<size_t>(aes_bytes) * trlwe_bytes;
    constexpr size_t sbox_address_workspace_bytes =
        sbox_batch_count * trlwe_bytes;
    constexpr size_t trgsw_bytes = static_cast<size_t>(aes_bytes) *
                                   sbox_address_bit *
                                   TRGSWElements<targetP>() *
                                   sizeof(typename targetP::T);
    constexpr size_t trgswfft_bytes = static_cast<size_t>(aes_bytes) *
                                      sbox_address_bit *
                                      TRGSWFFTElements<targetP>() *
                                      sizeof(NTTValue);
    constexpr size_t domain_tlwe_bytes =
        TLWEElements<typename brP::domainP>() *
        sizeof(typename brP::domainP::T);
    constexpr size_t domain_tlwe_workspace_bytes =
        sbox_batch_count * domain_tlwe_bytes;
    constexpr size_t cb_temptrlwe_workspace_bytes =
        sbox_batch_count * targetP::l * trlwe_bytes;
    constexpr size_t ksk_bytes = sizeof(TFHEpp::KeySwitchingKey<iksP>);

    typename targetP::T* d_state = nullptr;
    typename targetP::T* d_scratch = nullptr;
    typename targetP::T* d_roundkeys = nullptr;
    typename targetP::T* d_rom = nullptr;
    typename targetP::T* d_address_poly = nullptr;
    NTTValue* d_address_fft = nullptr;
    typename targetP::T* d_acc = nullptr;
    typename targetP::T* d_temp = nullptr;
    typename targetP::T* d_product = nullptr;
    typename brP::domainP::T* d_domain_tlwe = nullptr;
    typename targetP::T* d_cb_acc = nullptr;
    typename targetP::T* d_cb_temptrlwe = nullptr;
    typename iksP::targetP::T* d_ksk = nullptr;

    CuSafeCall(cudaMalloc((void**)&d_state, state_bytes));
    CuSafeCall(cudaMalloc((void**)&d_scratch, state_bytes));
    CuSafeCall(cudaMalloc((void**)&d_roundkeys, roundkeys_bytes));
    CuSafeCall(cudaMalloc((void**)&d_rom, trlwe_bytes));
    CuSafeCall(cudaMalloc((void**)&d_address_poly, trgsw_bytes));
    CuSafeCall(cudaMalloc((void**)&d_address_fft, trgswfft_bytes));
    CuSafeCall(cudaMalloc((void**)&d_acc, sbox_trlwe_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_temp, sbox_trlwe_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_product, sbox_trlwe_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_domain_tlwe,
                          domain_tlwe_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_cb_acc, sbox_address_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_cb_temptrlwe,
                          cb_temptrlwe_workspace_bytes));
    CuSafeCall(cudaMalloc((void**)&d_ksk, ksk_bytes));

    std::array<cudaStream_t, aes_bytes> sbox_streams{};
    std::array<cudaEvent_t, aes_bytes> sbox_done_events{};
    cudaEvent_t sbox_ready_event = nullptr;
    CuSafeCall(cudaEventCreateWithFlags(&sbox_ready_event,
                                        cudaEventDisableTiming));
    for (uint32_t byte = 0; byte < aes_bytes; byte++) {
        CuSafeCall(cudaStreamCreateWithFlags(&sbox_streams[byte],
                                             cudaStreamNonBlocking));
        CuSafeCall(cudaEventCreateWithFlags(&sbox_done_events[byte],
                                            cudaEventDisableTiming));
    }

    auto state = std::make_unique<std::array<TFHEpp::TLWE<targetP>, 128>>();
    for (uint32_t i = 0; i < 4; i++)
        for (uint32_t j = 0; j < TFHEpp::Nb; j++)
            for (uint32_t k = 0; k < 8; k++)
                (*state)[i * TFHEpp::Nb * 8 + j * 8 + k] =
                    cipher[j * 4 * 8 + i * 8 + k];

    CuSafeCall(cudaMemcpyAsync(d_state, state->data(), state_bytes,
                               cudaMemcpyHostToDevice, st.st()));
    CuSafeCall(cudaMemcpyAsync(d_roundkeys, expandedkey.data(),
                               roundkeys_bytes, cudaMemcpyHostToDevice,
                               st.st()));
    CuSafeCall(cudaMemcpyAsync(d_ksk, ek.getiksk<iksP>().data(), ksk_bytes,
                               cudaMemcpyHostToDevice, st.st()));
    UploadAESInvSboxROM<targetP>(d_rom, st.st());

    DeviceAddRoundKey<targetP>(
        d_state, d_roundkeys + static_cast<size_t>(TFHEpp::Nr) * 128 * tlwe_elems,
        st.st());
    for (int round = TFHEpp::Nr - 1; round > 0; round--) {
        DeviceInvShiftRows<targetP>(d_state, d_scratch, st.st());
        DeviceAESInvSubBytesBatched<iksP, brP, ahP>(
            d_state, d_ksk, d_rom, d_address_poly, d_address_fft, d_acc,
            d_temp, d_product, d_domain_tlwe, d_cb_acc, d_cb_temptrlwe,
            sbox_streams, sbox_ready_event, sbox_done_events, st.st(),
            st.device_id());
        DeviceAddRoundKey<targetP>(
            d_state,
            d_roundkeys + static_cast<size_t>(round) * 128 * tlwe_elems,
            st.st());
        DeviceInvMixColumns<targetP>(d_state, d_scratch, st.st());
    }
    DeviceInvShiftRows<targetP>(d_state, d_scratch, st.st());
    DeviceAESInvSubBytesBatched<iksP, brP, ahP>(
        d_state, d_ksk, d_rom, d_address_poly, d_address_fft, d_acc, d_temp,
        d_product, d_domain_tlwe, d_cb_acc, d_cb_temptrlwe, sbox_streams,
        sbox_ready_event, sbox_done_events, st.st(), st.device_id());
    DeviceAddRoundKey<targetP>(d_state, d_roundkeys, st.st());

    CuSafeCall(cudaMemcpyAsync(state->data(), d_state, state_bytes,
                               cudaMemcpyDeviceToHost, st.st()));
    CuSafeCall(cudaStreamSynchronize(st.st()));

    for (uint32_t i = 0; i < 4; i++)
        for (uint32_t j = 0; j < TFHEpp::Nb; j++)
            for (uint32_t k = 0; k < 8; k++)
                plain[j * 4 * 8 + i * 8 + k] =
                    (*state)[i * TFHEpp::Nb * 8 + j * 8 + k];

    for (uint32_t byte = 0; byte < aes_bytes; byte++) {
        CuSafeCall(cudaEventDestroy(sbox_done_events[byte]));
        CuSafeCall(cudaStreamDestroy(sbox_streams[byte]));
    }
    CuSafeCall(cudaEventDestroy(sbox_ready_event));

    cudaFree(d_ksk);
    cudaFree(d_cb_temptrlwe);
    cudaFree(d_cb_acc);
    cudaFree(d_domain_tlwe);
    cudaFree(d_product);
    cudaFree(d_temp);
    cudaFree(d_acc);
    cudaFree(d_address_fft);
    cudaFree(d_address_poly);
    cudaFree(d_rom);
    cudaFree(d_roundkeys);
    cudaFree(d_scratch);
    cudaFree(d_state);
}

template void InitializeAES<TFHEpp::lvl02param, TFHEpp::AHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);
template void InitializeAES<TFHEpp::lvlh2param, TFHEpp::AHlvl2param>(
    const TFHEpp::EvalKey&, const TFHEpp::SecretKey&);

template void CleanUpAES<TFHEpp::lvl02param, TFHEpp::AHlvl2param>();
template void CleanUpAES<TFHEpp::lvlh2param, TFHEpp::AHlvl2param>();

template void AESDec<TFHEpp::lvl20param, TFHEpp::lvl02param,
                     TFHEpp::AHlvl2param>(
    std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>&,
    const std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>&,
    const std::array<std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>,
                     TFHEpp::Nr + 1>&,
    const TFHEpp::EvalKey&, Stream);

template void AESDec<TFHEpp::lvl2hparam, TFHEpp::lvlh2param,
                     TFHEpp::AHlvl2param>(
    std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>&,
    const std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>&,
    const std::array<std::array<TFHEpp::TLWE<TFHEpp::lvl2param>, 128>,
                     TFHEpp::Nr + 1>&,
    const TFHEpp::EvalKey&, Stream);

}  // namespace cufhe
