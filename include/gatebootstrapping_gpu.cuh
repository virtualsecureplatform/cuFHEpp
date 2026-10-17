#pragma once

#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#if !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
#include <include/ntt_pair64.cuh>
#endif
#include <include/utils_gpu.cuh>
#if defined(USE_CUDA_ASYNC_MONOMIALS) && !defined(CUFHE_USE_HIP)
#include <cuda_pipeline.h>
#endif
#if defined(USE_CUDA_ASYNC_MONOMIALS) && defined(USE_CUDA_PAIRED_NTT_MADD4) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CUFHE_USE_HIP)
#define CUFHE_ASYNC_MONOMIALS_DEVICE 1
#endif

namespace cufhe {

template <class P>
__device__ inline uint32_t modSwitchFromTorus(
    const typename P::domainP::T phase)
{
    constexpr uint32_t Mbit = P::targetP::nbit + 1;
    static_assert(32 >= Mbit, "Undefined modSwitchFromTorus!");
    return (phase >> (std::numeric_limits<typename P::domainP::T>::digits - 1 -
                      P::targetP::nbit));
}

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
// Capture before the caller's correction barrier; publish only after it.
// The existing test-vector barrier makes the output metadata ready to consume.
template <class P, int Sign0, int Sign1, int Threads>
__device__ __forceinline__ uint32_t CapturePublicOutputExponents(
    const typename P::domainP::T* in0, const typename P::domainP::T* in1)
{
    static_assert(P::targetP::nbit + 1 <= 16 && P::domainP::n <= 2 * Threads);
    constexpr typename P::domainP::T roundoffset =
        static_cast<typename P::domainP::T>(1)
        << (std::numeric_limits<typename P::domainP::T>::digits - 2 - P::targetP::nbit);
    uint32_t saved = 0;
#pragma unroll
    for (int slot = 0; slot < (P::domainP::n + Threads - 1) / Threads; ++slot) {
        const int i = ThisThreadRankInBlock() + slot * Threads;
        if (i < P::domainP::n) {
            const typename P::domainP::T phase = Sign0 * in0[i] + Sign1 * in1[i];
            saved |= modSwitchFromTorus<P>(phase + roundoffset) << (16 * slot);
        }
    }
    return saved;
}

template <class P, int Threads>
__device__ __forceinline__ void PublishPublicOutputExponents(
    typename P::domainP::T* out, uint32_t saved)
{
#pragma unroll
    for (int slot = 0; slot < (P::domainP::n + Threads - 1) / Threads; ++slot) {
        const int i = ThisThreadRankInBlock() + slot * Threads;
        if (i < P::domainP::n) out[i] = (saved >> (16 * slot)) & 0xffffU;
    }
}
#endif

// Sum exactly the same torus rounding residuals used by block-binary blind
// rotation. For the 16-bit input domain the centered residual sum cannot wrap;
// its low-bit expression therefore permits an exact warp reduction.
template <class P, int Sign0 = 1, int Sign1 = 0>
__device__ inline std::make_signed_t<typename P::domainP::T>
BlockBinaryModSwitchCorrection(const typename P::domainP::T* in0,
                              const typename P::domainP::T* in1 = nullptr)
{
    using T = typename P::domainP::T;
    using Signed = std::make_signed_t<T>;
    constexpr uint32_t count = P::domainP::k * P::domainP::n;
    constexpr uint32_t shift = std::numeric_limits<T>::digits - 1 - P::targetP::nbit;
    constexpr T rounding = T{1} << (shift - 1);
    __shared__ Signed correction;
#if defined(USE_CUDA_WARP_MODSWITCH_CORRECTION) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800 && !defined(CUFHE_USE_HIP)
    if constexpr (sizeof(T) == 2 && uint64_t(count) * rounding <= ((uint64_t{1} << (std::numeric_limits<T>::digits - 1)) - 1)) {
        if (ThisThreadRankInBlock() < 32) {
            int32_t sum = 0;
#pragma unroll
            for (uint32_t i = ThisThreadRankInBlock(); i < count; i += 32) {
                T phase = Sign0 * in0[i];
                if constexpr (Sign1 != 0) phase += Sign1 * in1[i];
                sum += int32_t((uint32_t(phase) + rounding) & ((1U << shift) - 1)) - rounding;
            }
            const int32_t total = __reduce_add_sync(0xffffffffU, sum);
            if (ThisThreadRankInBlock() == 0) correction = Signed(total);
        }
    }
    else
#endif
    {
        if (ThisThreadRankInBlock() == 0) {
            correction = 0;
            for (uint32_t i = 0; i < count; ++i) {
                T phase = Sign0 * in0[i];
                if constexpr (Sign1 != 0) phase += Sign1 * in1[i];
                const uint32_t moded = modSwitchFromTorus<P>(phase + rounding);
                correction += phase - (T(moded) << shift);
            }
        }
    }
    __syncthreads();
    return correction;
}

template <class P>
__device__ constexpr typename P::T offsetgen()
{
    typename P::T offset = 0;
    for (int i = 1; i <= P::l; i++)
        offset += P::Bg / 2 *
                  (1ULL << (std::numeric_limits<typename P::T>::digits -
                            i * P::Bgbit));
    return offset;
}

template <class P>
__device__ inline void RotatedTestVector(typename P::T* tlwe, const int32_t bar,
                                         const typename P::T μ)
{
    // volatile is needed to make register usage of Mux to 128.
    // Reference
    // https://devtalk.nvidia.com/default/topic/466758/cuda-programming-and-performance/tricks-to-fight-register-pressure-or-how-i-got-down-from-29-to-15-registers-/
    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
#pragma unroll
    for (int i = tid; i < P::n; i += bdim) {
#pragma unroll
        for (int k = 0; k < P::k; k++) tlwe[i + k * P::n] = 0;  // part a
        if (bar == 2 * P::n)
            tlwe[i + P::k * P::n] = μ;
        else {
            tlwe[i + P::k * P::n] =
                ((i < (bar & (P::n - 1))) ^ (bar >> P::nbit)) ? -μ
                                                              : μ;  // part b
        }
    }
    __syncthreads();
}

#ifdef USE_FFT
#ifdef USE_GPU_FFT

/**
 * GPU-FFT Accumulate function
 * Uses 512 threads total, 256 active during FFT, all 512 during
 * decomposition/multiply
 *
 * Differs from tfhe-rs version:
 *   - Decompose + fold + twist (instead of simple packing)
 *   - Forward FFT via GPUFFTForward512 (5 syncs instead of 11)
 *   - Inverse FFT via GPUFFTInverse512 (5 syncs instead of 11)
 *   - Untwist + unfold (instead of simple unpacking)
 */
template <class P>
__device__ inline void Accumulate(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t a_bar,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const tgsw_fft,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;
    constexpr bool LOW_LDS = USE_LOW_LDS_BOOTSTRAP<typename P::targetP>;

    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];
    double2 local_accum[P::targetP::k + 1];

    // Initialize accumulated results to zero
    if constexpr (LOW_LDS) {
#pragma unroll
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++)
            local_accum[k_idx] = {0.0, 0.0};
    }
    else {
        for (int i = tid; i < (P::targetP::k + 1) * HALF_N; i += NUM_THREADS) {
            sh_accum[i] = {0.0, 0.0};
        }
    }
    __syncthreads();

    for (int j = 0; j <= P::targetP::k; j++) {
        const bool nonce = j < P::targetP::k;
        const uint32_t digits = nonce ? P::targetP::lₐ : P::targetP::l;
        const uint32_t basebit = nonce ? P::targetP::Bgₐbit : P::targetP::Bgbit;
        const uint32_t base = nonce ? P::targetP::Bgₐ : P::targetP::Bg;
        const uint32_t decomp_mask = base - 1;
        const int32_t decomp_half = base >> 1;
        typename P::targetP::T decomp_offset = 0;
        for (uint32_t digit = 1; digit <= digits; digit++)
            decomp_offset +=
                static_cast<typename P::targetP::T>(decomp_half)
                << (std::numeric_limits<typename P::targetP::T>::digits -
                    digit * basebit);
        const typename P::targetP::T roundoffset =
            static_cast<typename P::targetP::T>(1)
            << (std::numeric_limits<typename P::targetP::T>::digits -
                digits * basebit - 1);

        for (uint32_t digit = 0; digit < digits; digit++) {
            // Step 1: DECOMPOSE + FOLD + TWIST
            // Fold: Complex[i] = {Poly[i], Poly[i + N/2]}
            // Twist: Complex[i] *= twist[i]
            if (tid < HALF_N) {
                const uint32_t idx_re = tid;
                typename P::targetP::T temp_re =
                    trlwe[j * N + ((idx_re - a_bar) & (N - 1))];
                temp_re =
                    ((idx_re < (a_bar & (N - 1)) ^ (a_bar >> P::targetP::nbit)))
                        ? -temp_re
                        : temp_re;
                temp_re -= trlwe[j * N + idx_re];
                temp_re += decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * basebit)) &
                     decomp_mask) -
                    decomp_half);

                const uint32_t idx_im = tid + HALF_N;
                typename P::targetP::T temp_im =
                    trlwe[j * N + ((idx_im - a_bar) & (N - 1))];
                temp_im =
                    ((idx_im < (a_bar & (N - 1)) ^ (a_bar >> P::targetP::nbit)))
                        ? -temp_im
                        : temp_im;
                temp_im -= trlwe[j * N + idx_im];
                temp_im += decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * basebit)) &
                     decomp_mask) -
                    decomp_half);

                // Fold + twist
                double2 folded = {static_cast<double>(digit_re),
                                  static_cast<double>(digit_im)};
                double2 tw = __ldg(&ntt.twist_[tid]);
                sh_fft[tid] = folded * tw;
            }
            __syncthreads();

            // Step 2: Forward FFT
            if (tid < FFT_THREADS) {
                GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                    __syncthreads();
            }

            // Step 3: Multiply-accumulate with BSK
            const uint32_t digit_linear =
                nonce ? j * P::targetP::lₐ + digit
                      : P::targetP::k * P::targetP::lₐ + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    double2 bk_val = __ldg(
                        &tgsw_fft[((P::targetP::k + 1) * digit_linear + out_k) *
                                      HALF_N +
                                  tid]);
                    if constexpr (LOW_LDS)
                        local_accum[out_k] += fft_val * bk_val;
                    else
                        sh_accum[out_k * HALF_N + tid] += fft_val * bk_val;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT + untwist + unfold + add to trlwe
    constexpr double denorm = (sizeof(typename P::targetP::T) == 4)
                                  ? 4294967296.0
                                  : 18446744073709551616.0;
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* sh_inv;
        if constexpr (LOW_LDS) {
            if (tid < HALF_N) sh_fft[tid] = local_accum[k_idx];
            __syncthreads();
            sh_inv = sh_fft;
        }
        else {
            sh_inv = &sh_accum[k_idx * HALF_N];
        }

        if (tid < FFT_THREADS) {
            GPUFFTInverse<N>(sh_inv, ntt.inverse_root_, tid);
        }
        else {
            for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                __syncthreads();
        }

        // Untwist + unfold + denormalize
        // (n_inverse is folded into the untwist table)
        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            if constexpr (sizeof(typename P::targetP::T) == 4) {
                trlwe[k_idx * N + tid] += static_cast<typename P::targetP::T>(
                    static_cast<int32_t>(llrint(val.x * denorm)));
                trlwe[k_idx * N + tid + HALF_N] +=
                    static_cast<typename P::targetP::T>(
                        static_cast<int32_t>(llrint(val.y * denorm)));
            }
            else {
                trlwe[k_idx * N + tid] += static_cast<typename P::targetP::T>(
                    double_to_torus64(val.x * denorm));
                trlwe[k_idx * N + tid + HALF_N] +=
                    static_cast<typename P::targetP::T>(
                        double_to_torus64(val.y * denorm));
            }
        }
        __syncthreads();
    }
}

#ifdef USE_BLOCK_BINARY
template <class P>
__device__ inline void AccumulateBlockBinary(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t block,
    const uint32_t* const bara, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    using targetP = typename P::targetP;
    using T = typename targetP::T;
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;
    constexpr uint32_t ROWS = BootstrappingTRGSWRows<targetP>;
    constexpr size_t TRGSW_SIZE = ROWS * (targetP::k + 1) * HALF_N;
    constexpr bool LOW_LDS = USE_LOW_LDS_BOOTSTRAP<targetP>;
    static_assert(P::domainP::ell > 1);
    static_assert(
        targetP::l̅ == 1 && targetP::l̅ₐ == 1,
        "Block-binary GPU bootstrapping requires single decomposition");

    const uint32_t tid = ThisThreadRankInBlock();
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_fft =
        sizeof(T) == 4 ? block_xai_fft : block_xai_fft_lvl02;
    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];
    double2 local_accum[targetP::k + 1];
    if constexpr (LOW_LDS) {
#pragma unroll
        for (int component = 0; component <= targetP::k; component++)
            local_accum[component] = {0.0, 0.0};
    }
    else {
        for (int i = tid; i < (targetP::k + 1) * HALF_N; i += NUM_THREADS)
            sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    for (int component = 0; component <= targetP::k; component++) {
        const bool nonce = component < targetP::k;
        const uint32_t digits = nonce ? targetP::lₐ : targetP::l;
        const uint32_t basebit = nonce ? targetP::Bgₐbit : targetP::Bgbit;
        const uint32_t base = nonce ? targetP::Bgₐ : targetP::Bg;
        const uint32_t mask = base - 1;
        const int32_t halfbase = base >> 1;
        T decomp_offset = 0;
        for (uint32_t digit = 1; digit <= digits; digit++)
            decomp_offset +=
                static_cast<T>(halfbase)
                << (std::numeric_limits<T>::digits - digit * basebit);
        const T roundoffset =
            static_cast<T>(1)
            << (std::numeric_limits<T>::digits - digits * basebit - 1);

        for (uint32_t digit = 0; digit < digits; digit++) {
            if (tid < HALF_N) {
                T re = trlwe[component * N + tid] + decomp_offset + roundoffset;
                T im = trlwe[component * N + tid + HALF_N] + decomp_offset +
                       roundoffset;
                const int32_t digit_re =
                    static_cast<int32_t>(
                        (re >> (std::numeric_limits<T>::digits -
                                (digit + 1) * basebit)) &
                        mask) -
                    halfbase;
                const int32_t digit_im =
                    static_cast<int32_t>(
                        (im >> (std::numeric_limits<T>::digits -
                                (digit + 1) * basebit)) &
                        mask) -
                    halfbase;
                const double2 folded = {static_cast<double>(digit_re),
                                        static_cast<double>(digit_im)};
                sh_fft[tid] = folded * __ldg(&ntt.twist_[tid]);
            }
            __syncthreads();

            if (tid < FFT_THREADS)
                GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
            else
                for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                    __syncthreads();

            const uint32_t row = nonce ? component * targetP::lₐ + digit
                                       : targetP::k * targetP::lₐ + digit;
            if (tid < HALF_N) {
#pragma unroll
                for (int out_component = 0; out_component <= targetP::k;
                     out_component++) {
                    double2 block_product = {0.0, 0.0};
#pragma unroll
                    for (uint32_t offset = 0; offset < P::domainP::ell;
                         offset++) {
                        const size_t key_index =
                            block * P::domainP::ell + offset;
                        const size_t poly_index =
                            (row * (targetP::k + 1) + out_component) * HALF_N +
                            tid;
                        const double2 bk_value =
                            __ldg(&bk[key_index * TRGSW_SIZE + poly_index]);
                        const double2 xai =
                            __ldg(&xai_fft[bara[offset] * HALF_N + tid]);
                        complex_madd(block_product, bk_value, xai);
                    }
                    if constexpr (LOW_LDS)
                        complex_madd(local_accum[out_component], sh_fft[tid],
                                     block_product);
                    else
                        complex_madd(sh_accum[out_component * HALF_N + tid],
                                     sh_fft[tid], block_product);
                }
            }
            __syncthreads();
        }
    }

    constexpr double denorm =
        sizeof(T) == 4 ? 4294967296.0 : 18446744073709551616.0;
    for (int component = 0; component <= targetP::k; component++) {
        double2* const sh_inverse =
            LOW_LDS ? sh_fft : &sh_accum[component * HALF_N];
        if constexpr (LOW_LDS) {
            if (tid < HALF_N) sh_fft[tid] = local_accum[component];
            __syncthreads();
        }
        if (tid < FFT_THREADS)
            GPUFFTInverse<N>(sh_inverse, ntt.inverse_root_, tid);
        else
            for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                __syncthreads();

        if (tid < HALF_N) {
            double2 value = sh_inverse[tid] * __ldg(&ntt.untwist_[tid]);
            if constexpr (sizeof(T) == 4) {
                trlwe[component * N + tid] += static_cast<T>(
                    static_cast<int32_t>(llrint(value.x * denorm)));
                trlwe[component * N + tid + HALF_N] += static_cast<T>(
                    static_cast<int32_t>(llrint(value.y * denorm)));
            }
            else {
                trlwe[component * N + tid] +=
                    static_cast<T>(double_to_torus64(value.x * denorm));
                trlwe[component * N + tid + HALF_N] +=
                    static_cast<T>(double_to_torus64(value.y * denorm));
            }
        }
        __syncthreads();
    }
}
#endif
#else  // !USE_GPU_FFT (tfhe-rs FFT)
/**
 * FFT-based Accumulate function (tfhe-rs style)
 * Uses 512 threads total, 256 active during FFT, all 512 during
 * decomposition/multiply
 *
 * Shared memory layout (N=1024, k=1):
 *   sh_fft[0..N/2-1]:              FFT working buffer       = 512 × double2 = 8
 * KB sh_accum[0..(k+1)*N/2-1]:     Accumulated FFT results  = 1024 × double2 =
 * 16 KB Total: 24 KB
 *
 * BSK indexing: bk_fft[digit_linear * (k+1) * (N/2) + out_k * (N/2) +
 * complex_idx]
 */
template <class P>
__device__ inline void Accumulate(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t a_bar,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const tgsw_fft,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;       // 512
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512 threads total
    constexpr uint32_t FFT_THREADS =
        HALF_N / (Degree<N>::opt / 2);  // 256 for N=1024

    // Shared memory layout:
    // sh_fft[0..HALF_N-1]: FFT working buffer (N/2 complex = 8KB)
    // sh_accum[0..(k+1)*HALF_N-1]: accumulated results ((k+1)*N/2 complex =
    // 16KB for k=1)
    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];

    // Initialize accumulated results to zero
    for (int i = tid; i < (P::targetP::k + 1) * HALF_N; i += NUM_THREADS) {
        sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Process each TRLWE component and decomposition level
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: DECOMPOSE + PACK into N/2 complex values
            // Half-size FFT packing: Complex[i] = {Poly[i], Poly[i + N/2]}
            // Thread tid handles coefficient tid (first half -> real) and
            // tid + N/2 (second half -> imaginary)
            if (tid < HALF_N) {
                // First-half coefficient (index tid)
                const uint32_t idx_re = tid;
                typename P::targetP::T temp_re =
                    trlwe[j * N + ((idx_re - a_bar) & (N - 1))];
                temp_re =
                    ((idx_re < (a_bar & (N - 1)) ^ (a_bar >> P::targetP::nbit)))
                        ? -temp_re
                        : temp_re;
                temp_re -= trlwe[j * N + idx_re];
                temp_re += decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                // Second-half coefficient (index tid + N/2)
                const uint32_t idx_im = tid + HALF_N;
                typename P::targetP::T temp_im =
                    trlwe[j * N + ((idx_im - a_bar) & (N - 1))];
                temp_im =
                    ((idx_im < (a_bar & (N - 1)) ^ (a_bar >> P::targetP::nbit)))
                        ? -temp_im
                        : temp_im;
                temp_im -= trlwe[j * N + idx_im];
                temp_im += decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                // Pack: real = first-half coeff, imag = second-half coeff
                sh_fft[tid] = {static_cast<double>(digit_re),
                               static_cast<double>(digit_im)};
            }
            __syncthreads();

            // Step 2: Forward FFT (only first FFT_THREADS threads active)
            if (tid < FFT_THREADS) {
                NSMFFT_direct<HalfDegree<Degree<N> > >(sh_fft);
            }
            else {
                // Non-FFT threads must match the syncthreads inside
                // NSMFFT_direct
                for (int s = 0; s < 11; s++) __syncthreads();
            }

            // Step 3: Multiply-accumulate with BSK in Fourier domain
            // All 512 threads participate, each handles one complex element
            int digit_linear = j * P::targetP::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    double2 bk_val = __ldg(
                        &tgsw_fft[((P::targetP::k + 1) * digit_linear + out_k) *
                                      HALF_N +
                                  tid]);
                    // Complex multiply-accumulate
                    sh_accum[out_k * HALF_N + tid] += fft_val * bk_val;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT on accumulated results and add to trlwe
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];

        // Inverse FFT (only first FFT_THREADS threads active)
        if (tid < FFT_THREADS) {
            NSMFFT_inverse<HalfDegree<Degree<N> > >(sh_inv);
        }
        else {
            // Match syncthreads: 1 (entry) + for l=5..8: 2*4=8 + 1 (pre-store)
            // + 1 (post-store) = 11
            for (int s = 0; s < 11; s++) __syncthreads();
        }

        // Step 5: Unpack complex -> real coefficients, add to trlwe
        // Half-size FFT unpacking: Poly[i] = Complex[i].real, Poly[i+N/2] =
        // Complex[i].imag Multiply by 2^32 to undo the BSK normalization (BSK
        // was divided by 2^32 in __TRGSW2FFT__)
        constexpr double denorm = 4294967296.0;  // 2^32
        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            trlwe[k_idx * N + tid] += static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.x * denorm)));
            trlwe[k_idx * N + tid + HALF_N] += static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}
#endif  // USE_GPU_FFT
#else   // !USE_FFT
/**
 * Sequential NTT Accumulate function (HEonGPU-style)
 * Uses 512 threads, processes NTTs one at a time for maximum efficiency
 *
 * Shared memory layout:
 * - sh_acc_ntt[0..N-1]: Working buffer for NTT operations
 * - sh_acc_ntt[N..(k+2)*N-1]: Accumulated products in NTT domain
 *
 * Uses small modulus (~2^31, uint32_t) with Torus discretization switching
 */
template <class P>
__device__ inline void Accumulate(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t a_bar,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const tgsw_ntt,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512 for N=1024
    constexpr bool REGISTER_ACCUM =
        USE_REGISTER_NTT_ACCUM<typename P::targetP>;

    // Aliases for clarity
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work =
        &sh_acc_ntt[0];  // Working buffer for NTT
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_accum =
        &sh_acc_ntt[N];  // Accumulated results (k+1 polynomials)
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> local_accum[P::targetP::k + 1][2];

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
#pragma unroll
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            if constexpr (REGISTER_ACCUM) {
                local_accum[k_idx][0] = 0;
                local_accum[k_idx][1] = 0;
            }
            else {
                sh_accum[k_idx * N + tid] = 0;
                sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
            }
        }
    }

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Process each TRLWE component (k+1 components) and each decomposition
    // level (l levels)
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Compute decomposed polynomial for component j, digit
            // MulByXaiMinusOne and decomposition - each thread handles 2
            // elements
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;

                    // PolynomialMulByXaiMinus
                    typename P::targetP::T temp =
                        trlwe[j * N + ((i - a_bar) & (N - 1))];
                    temp =
                        ((i < (a_bar & (N - 1)) ^ (a_bar >> P::targetP::nbit)))
                            ? -temp
                            : temp;
                    temp -= trlwe[j * N + i];

                    // Decomposition for this digit
                    temp += decomp_offset + roundoffset;
                    int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename P::targetP::T>::digits -
                           (digit + 1) * P::targetP::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = signed_int_to_ntt_mod<N, sizeof(typename P::targetP::T) * 8>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                SmallForwardNTT<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_work, ntt.forward_root_,
                                                  tid);
            }
            else {
                for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                    __syncthreads();
            }

            // Step 3: Multiply with BK and accumulate into sh_accum
            int digit_linear = j * P::targetP::l + digit;
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt_val = sh_work[i];

// Accumulate into each output component
#pragma unroll
                    for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk_val = __ldg(
                            &tgsw_ntt[(((P::targetP::k + 1) * digit_linear +
                                        out_k)
                                       << P::targetP::nbit) +
                                      i]);
                        if constexpr (REGISTER_ACCUM)
                            local_accum[out_k][e] = small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(
                                ntt_val, bk_val, local_accum[out_k][e]);
                        else
                            sh_accum[out_k * N + i] = small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(
                                ntt_val, bk_val, sh_accum[out_k * N + i]);
                    }
                }
            }
        }
    }

    // Accumulation is coefficient-local. Synchronize once before the inverse
    // butterflies begin reading coefficients written by other threads.
    __syncthreads();

    // Step 4: Inverse NTT on accumulated results and add to trlwe
    // Operate directly on sh_accum to avoid copying to sh_work
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* sh_ntt_buf;
        if constexpr (REGISTER_ACCUM) {
            if (tid < NUM_THREADS) {
                sh_work[tid] = local_accum[k_idx][0];
                sh_work[tid + NUM_THREADS] = local_accum[k_idx][1];
            }
            __syncthreads();
            sh_ntt_buf = sh_work;
        }
        else {
            // Inverse NTT directly on the accumulator buffer.
            sh_ntt_buf = &sh_accum[k_idx * N];
        }
        if (tid < NUM_THREADS) {
            SmallInverseNTT<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf, ntt.inverse_root_,
                                              ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        // Convert with modulus switching and add to trlwe
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                if constexpr (std::numeric_limits<
                                  typename P::targetP::T>::digits == 64) {
                    trlwe[k_idx * N + i] += static_cast<typename P::targetP::T>(
                        ntt_mod_to_torus64<N, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf[i]));
                }
                else {
                    trlwe[k_idx * N + i] += static_cast<typename P::targetP::T>(
                        ntt_mod_to_torus32<N, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf[i]));
                }
            }
        }
        if constexpr (REGISTER_ACCUM) __syncthreads();
    }
    __syncthreads();
}

#if defined(USE_CUDA_PAIRED_NTT_WARP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
// Pair independent decomposition rows within one legacy external product.
// Rotation, digit extraction and each torus update keep their original order.
template <class P>
__device__ inline void AccumulateLegacyPairedNTT(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const scratch,
    const uint32_t a_bar,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    using targetP = typename P::targetP;
    using T = typename targetP::T;
    constexpr uint32_t N = targetP::n;
    if constexpr (sizeof(T) != 4 || N != 1024 || targetP::k != 1 ||
                  targetP::l != targetP::lₐ || targetP::Bgbit != targetP::Bgₐbit) {
        Accumulate<P>(trlwe, scratch, a_bar, bk, ntt);
    } else {
        using Value = NTTValueFor<N, 32>;
        constexpr uint32_t Threads = N / 2;
        constexpr uint32_t Rows = (targetP::k + 1) * targetP::l;
#if defined(USE_CUDA_WORD32_REDUCTION)
        // The full-word reducer accepts the lazy forward range [0,2p).
        // Two such products plus a canonical accumulator cannot overflow.
        constexpr uint64_t p = small_ntt31::P;
        static_assert(2 * (2 * p - 1) <= (UINT64_MAX - (p - 1)) / (p - 1));
        constexpr bool CanonicalForward = false;
#else
        constexpr bool CanonicalForward = true;
#endif
        static_assert(Rows % 2 == 0);
        const uint32_t tid = ThisThreadRankInBlock();
        Value* const work0 = scratch;
        Value* const work1 = scratch + N;
        Value accum[2][2] = {};
        constexpr T offset = offsetgen<targetP>();
        constexpr T roundoffset = T{1} << (32 - targetP::l * targetP::Bgbit - 1);
        // No torus coefficient is updated until all digit products are accumulated.
        // Keep the original rotated differences across the decomposition digits.
        T rotated0[2], rotated1[2];
#pragma unroll
        for (uint32_t e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * Threads;
            const uint32_t source = (i - a_bar) & (N - 1);
            const bool negate = (i < (a_bar & (N - 1))) ^ (a_bar >> targetP::nbit);
            T x0 = trlwe[source], x1 = trlwe[N + source];
            x0 = negate ? -x0 : x0;
            x1 = negate ? -x1 : x1;
            rotated0[e] = x0 - trlwe[i] + offset + roundoffset;
            rotated1[e] = x1 - trlwe[N + i] + offset + roundoffset;
        }
#pragma unroll
        for (uint32_t pair = 0; pair < Rows / 2; ++pair) {
#pragma unroll
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const uint32_t row = 2 * pair + slot;
                const uint32_t component = row / targetP::l;
                const uint32_t digit = row % targetP::l;
                Value* const work = slot == 0 ? work0 : work1;
#pragma unroll
                for (uint32_t e = 0; e < 2; ++e) {
                    const uint32_t i = tid + e * Threads;
                    const T temp = component == 0 ? rotated0[e] : rotated1[e];
                    const int32_t value = static_cast<int32_t>(
                        (temp >> (32 - (digit + 1) * targetP::Bgbit)) &
                        (targetP::Bg - 1)) - targetP::Bg / 2;
                    work[i] = signed_int_to_ntt_mod<N, 32>(value);
                }
            }
            __syncthreads();
            SmallForwardNTTPair<targetP::nbit, 32, CanonicalForward>(
                work0, work1, ntt.forward_root_, tid);
#pragma unroll
            for (uint32_t e = 0; e < 2; ++e) {
                const uint32_t i = tid + e * Threads;
                const Value d0 = work0[i], d1 = work1[i];
#pragma unroll
                for (uint32_t out = 0; out < 2; ++out) {
                    const uint32_t index = (4 * pair + out) * N + i;
                    // The proven forward/key ranges and canonical accumulator
                    // fit in uint64_t. Reduce their exact sum once.
                    accum[out][e] = small_mod31_madd3(
                        d0, __ldg(bk + index), d1, __ldg(bk + index + 2 * N),
                        0, 0, accum[out][e]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (uint32_t e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * Threads;
            work0[i] = accum[0][e];
            work1[i] = accum[1][e];
        }
        __syncthreads();
        SmallInverseNTTPair<targetP::nbit, 32, true>(
            work0, work1, ntt.inverse_root_, ntt.n_inverse_, tid, trlwe);
    }
}
#endif

#ifdef USE_BLOCK_BINARY
// Pair adjacent decomposition rows so both halves of the block participate
// in the radix-4 stages. This changes the work schedule, not the external
// product: rows are accumulated in the same order over the same prime.
// The internal consumer context selects measured arithmetic grouping and
// inverse storage schedules, each producing the same canonical field results.
template <class P, bool RegisterMonomials = false, bool KeySwitchAfter31 = false,
          bool ReuseInput31 = false>
__device__ inline void AccumulateBlockBinaryPairedNTT(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const scratch,
    const uint32_t block, const uint32_t* const bara,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    using targetP = typename P::targetP;
    using T = typename targetP::T;
    using Value = NTTValueFor<targetP::n, 32>;
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t THREADS = N / 2;
    constexpr uint32_t ROWS = BootstrappingTRGSWRows<targetP>;
    constexpr size_t TRGSW_SIZE = ROWS * (targetP::k + 1) * N;
    static_assert(sizeof(T) == 4 && targetP::k == 1);
    static_assert(targetP::l == targetP::lₐ &&
                  targetP::Bgbit == targetP::Bgₐbit && ROWS % 2 == 0);
    const uint32_t tid = ThisThreadRankInBlock();
    Value* const work0 = scratch;
    Value* const work1 = scratch + N;
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
    Value* const staged_monomials = scratch + 2 * N;
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && defined(CUFHE_ASYNC_MONOMIALS_DEVICE) && defined(USE_CUDA_PAIRED_NTT_MADD4)
    constexpr bool CacheMonomials = RegisterMonomials && P::domainP::ell == 2;
#else
    constexpr bool CacheMonomials = false;
#endif
    Value cached_x0[2], cached_x1[2];
    if constexpr (CacheMonomials) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            cached_x0[e] = __ldg(block_xai_fft + bara[0] * N + i);
            cached_x1[e] = __ldg(block_xai_fft + bara[1] * N + i);
        }
    }
    Value accum[2][2] = {};
    T cached_input[2][2];
    constexpr T offset = offsetgen<targetP>() +
        (T{1} << (32 - targetP::l * targetP::Bgbit - 1));

#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
    // Every digit pair uses these same monomials. Start their copy before
    // decomposition, keeping the copy's address calculations outside the loop.
    if constexpr (P::domainP::ell == 2 && !CacheMonomials) {
        const uint32_t exponent = tid < N / 4 ? bara[0] : bara[1];
        const uint32_t coefficient = 4 * (tid & (N / 4 - 1));
        __pipeline_memcpy_async(staged_monomials + 4 * tid,
            block_xai_fft + exponent * N + coefficient, 16);
        __pipeline_commit();
    }
#endif
#pragma unroll
    for (uint32_t pair = 0; pair < ROWS / 2; ++pair) {
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const uint32_t row = 2 * pair + slot;
            const uint32_t component = row / targetP::l;
            const uint32_t digit = row % targetP::l;
            Value* const work = slot == 0 ? work0 : work1;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint32_t i = tid + e * THREADS;
                T value;
                if constexpr (CacheMonomials && ReuseInput31) {
                    // The TRLWE is immutable until the inverse consumes all rows.
                    // Reuse each coefficient after its first decomposition digit.
                    if (digit == 0)
                        cached_input[component][e] = trlwe[component * N + i] + offset;
                    value = cached_input[component][e];
                } else {
                    value = trlwe[component * N + i] + offset;
                }
                const int32_t decomposed = static_cast<int32_t>(
                    (value >> (32 - (digit + 1) * targetP::Bgbit)) &
                    (targetP::Bg - 1)) - targetP::Bg / 2;
                work[i] = signed_int_to_ntt_mod<N, 32>(decomposed);
            }
        }
        __syncthreads();
#if defined(USE_CUDA_PAIRED_NTT_MADD4) && defined(USE_CUDA_WORD32_REDUCTION)
        // Native-word products accept the full uint64 range and normalize
        // each digit/monomial product before the four-product sum. This
        // consumer can therefore accept transform values in [0, 2p).
        constexpr bool canonical_transform = P::domainP::ell != 2;
#else
        constexpr bool canonical_transform = true;
#endif
        SmallForwardNTTPair<targetP::nbit, 32, canonical_transform>(
            work0, work1, ntt.forward_root_, tid);
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
        if constexpr (P::domainP::ell == 2 && !CacheMonomials) {
            if (pair == 0) {
                __pipeline_wait_prior(0);
                __syncthreads();
            }
        }
#endif
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
#ifdef USE_CUDA_PAIRED_NTT_MADD4
            if constexpr (P::domainP::ell == 2) {
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
                const Value x0 = CacheMonomials ? cached_x0[e] : staged_monomials[i];
                const Value x1 = CacheMonomials ? cached_x1[e] : staged_monomials[N + i];
#else
                const Value x0 = __ldg(&block_xai_fft[bara[0] * N + i]);
                const Value x1 = __ldg(&block_xai_fft[bara[1] * N + i]);
#endif
                const Value d0 = work0[i], d1 = work1[i];
                if constexpr (CacheMonomials && !KeySwitchAfter31) {
#pragma unroll
                    for (int out = 0; out < 2; ++out) {
                        const size_t index = block * 2 * TRGSW_SIZE +
                                             (4 * pair + out) * N + i;
                        const Value k0 = small_mod31_madd3(
                            x0, __ldg(&bk[index]),
                            x1, __ldg(&bk[index + TRGSW_SIZE]), 0, 0, 0);
                        const Value k1 = small_mod31_madd3(
                            x0, __ldg(&bk[index + 2 * N]),
                            x1, __ldg(&bk[index + TRGSW_SIZE + 2 * N]), 0, 0, 0);
                        // Field distributivity preserves the original canonical
                        // sum; the full-word reducer covers lazy d0/d1 as before.
                        accum[out][e] = small_mod31_madd3(
                            d0, k0, d1, k1, 0, 0, accum[out][e]);
                    }
                } else {
                    const Value a0 = small_mod31_mult(d0, x0);
                    const Value a1 = small_mod31_mult(d0, x1);
                    const Value a2 = small_mod31_mult(d1, x0);
                    const Value a3 = small_mod31_mult(d1, x1);
#pragma unroll
                    for (int out = 0; out < 2; ++out) {
                        const size_t index = block * 2 * TRGSW_SIZE +
                                              (4 * pair + out) * N + i;
                        accum[out][e] = small_mod31_madd4(
                            a0, __ldg(&bk[index]),
                            a1, __ldg(&bk[index + TRGSW_SIZE]),
                            a2, __ldg(&bk[index + 2 * N]),
                            a3, __ldg(&bk[index + TRGSW_SIZE + 2 * N]), accum[out][e]);
                    }
                }
                continue;
            }
#endif
#pragma unroll
            for (int slot = 0; slot < 2; ++slot) {
                const uint32_t row = 2 * pair + slot;
                const Value decomposition = (slot == 0 ? work0 : work1)[i];
#pragma unroll
                for (int out = 0; out < 2; ++out) {
                    Value product = 0;
                    if constexpr (P::domainP::ell == 2) {
                        const size_t index = block * 2 * TRGSW_SIZE +
                                                (row * 2 + out) * N + i;
                        // Two products fit below 2*p^2 in uint64_t. Reuse the
                        // exact three-product reducer with a zero third term,
                        // eliminating one reduction without changing residues.
                        product = small_mod31_madd3(
                            __ldg(&bk[index]),
                            __ldg(&block_xai_fft[bara[0] * N + i]),
                            __ldg(&bk[index + TRGSW_SIZE]),
                            __ldg(&block_xai_fft[bara[1] * N + i]), 0, 0, 0);
                    }
                    else {
#pragma unroll
                        for (uint32_t term = 0; term < P::domainP::ell; ++term) {
                            const size_t index =
                                (block * P::domainP::ell + term) * TRGSW_SIZE +
                                (row * 2 + out) * N + i;
                            product = small_mod_madd<N, 32>(
                                __ldg(&bk[index]),
                                __ldg(&block_xai_fft[bara[term] * N + i]), product);
                        }
                    }
                    accum[out][e] = small_mod_madd<N, 32>(
                        decomposition, product, accum[out][e]);
                }
            }
        }
    }
    __syncthreads();
    work0[tid] = accum[0][0];
    work0[tid + THREADS] = accum[0][1];
    work1[tid] = accum[1][0];
    work1[tid + THREADS] = accum[1][1];
    __syncthreads();
    if constexpr (CacheMonomials && KeySwitchAfter31) {
        SmallInverseNTTPair<targetP::nbit, 32, true>(
            work0, work1, ntt.inverse_root_, ntt.n_inverse_, tid, trlwe);
    } else {
        SmallInverseNTTPair<targetP::nbit, 32>(work0, work1,
                                              ntt.inverse_root_, ntt.n_inverse_, tid);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            trlwe[i] += ntt_mod_to_torus32<N, 32>(work0[i]);
            trlwe[N + i] += ntt_mod_to_torus32<N, 32>(work1[i]);
        }
        __syncthreads();
    }
}

#ifdef CUFHE_HAS_PAIR_NTT64
__device__ __forceinline__ uint64_t LoadPairedMonomial64(const uint64_t* ptr)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    // The monomial table is immutable during bootstrapping. This explicit
    // readonly load gives the measured raw paired path a faster schedule.
    uint64_t value;
    asm("ld.global.nc.u64 %0, [%1];" : "=l"(value) : "l"(ptr));
    return value;
#else
    return __ldg(ptr);
#endif
}
// Two unchanged decomposition rows share roots, monomial loads, and a single
// four-product reduction. All field results remain canonical modulo p64.
template <class P, bool StageMonomials64 = true, int NativeAdd = 0>
__device__ inline void AccumulateBlockBinaryPaired64(
    typename P::targetP::T* const trlwe, uint64_t* const scratch,
    const uint32_t block, const uint32_t* const bara, const uint64_t* const bk,
    const CuNTTHandler<P::targetP::n, 64> ntt)
{
    using targetP = typename P::targetP;
    using T = typename targetP::T;
    constexpr uint32_t N = targetP::n, THREADS = N / 2;
    constexpr uint32_t ROWS = BootstrappingTRGSWRows<targetP>;
    constexpr size_t TRGSW_SIZE = ROWS * (targetP::k + 1) * N;
    static_assert(sizeof(T) == 8 && targetP::k == 2 && P::domainP::ell == 2);
    static_assert(targetP::l == targetP::lₐ && targetP::Bgbit == targetP::Bgₐbit);
    const uint32_t tid = ThisThreadRankInBlock();
    auto* const work0 = scratch;
    auto* const work1 = scratch + N;
    auto* const accum = scratch + 2 * N;
    auto* const monomials = scratch + (targetP::k + 3) * N;
    constexpr T offset = offsetgen<targetP>() +
        (T{1} << (64 - targetP::l * targetP::Bgbit - 1));
    if constexpr (StageMonomials64) {
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
    __pipeline_memcpy_async(monomials + 2 * tid,
        block_xai_fft_lvl02 + bara[0] * N + 2 * tid, 16);
    __pipeline_memcpy_async(monomials + N + 2 * tid,
        block_xai_fft_lvl02 + bara[1] * N + 2 * tid, 16);
    __pipeline_commit();
#else
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const uint32_t i = tid + e * THREADS;
        monomials[i] = __ldg(block_xai_fft_lvl02 + bara[0] * N + i);
        monomials[N + i] = __ldg(block_xai_fft_lvl02 + bara[1] * N + i);
    }
#endif
    }
#pragma unroll
    for (int out = 0; out <= targetP::k; ++out) {
        accum[out * N + tid] = 0;
        accum[out * N + tid + THREADS] = 0;
    }
    // The same monomials serve every decomposition row of this outer block.
    // Raw gates retain four words per thread; staged batches keep shared reads.
    uint64_t cached_x0[2], cached_x1[2];
    if constexpr (!StageMonomials64) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            cached_x0[e] = LoadPairedMonomial64(block_xai_fft_lvl02 + bara[0] * N + i);
            cached_x1[e] = LoadPairedMonomial64(block_xai_fft_lvl02 + bara[1] * N + i);
        }
    }
    for (uint32_t pair = 0; pair < ROWS / 2; ++pair) {
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const uint32_t row = 2 * pair + slot;
            const uint32_t component = row / targetP::l, digit = row % targetP::l;
            auto* const work = slot == 0 ? work0 : work1;
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const uint32_t i = tid + e * THREADS;
                const T value = trlwe[component * N + i] + offset;
                const int32_t decomposition = static_cast<int32_t>(
                    (value >> (64 - (digit + 1) * targetP::Bgbit)) &
                    (targetP::Bg - 1)) - targetP::Bg / 2;
                work[i] = signed_int_to_ntt_mod<N, 64>(decomposition);
            }
        }
        __syncthreads();
        // Uniform constant roots benefit the staged paired schedule.
        SmallForwardNTTPair64<targetP::nbit, StageMonomials64, NativeAdd, StageMonomials64>(work0, work1, ntt.forward_root_, tid);
        if (StageMonomials64 && pair == 0) {
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
            __pipeline_wait_prior(0);
#endif
            __syncthreads();
        }
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            const auto x0 = StageMonomials64 ? monomials[i] : cached_x0[e];
            const auto x1 = StageMonomials64 ? monomials[N + i] : cached_x1[e];
            const auto d0 = work0[i], d1 = work1[i];
            const auto a0 = small_mod64_mult<NativeAdd>(d0, x0), a1 = small_mod64_mult<NativeAdd>(d0, x1);
            const auto a2 = small_mod64_mult<NativeAdd>(d1, x0), a3 = small_mod64_mult<NativeAdd>(d1, x1);
#pragma unroll
            for (int out = 0; out <= targetP::k; ++out) {
                const size_t index = block * 2 * TRGSW_SIZE +
                    (2 * pair * (targetP::k + 1) + out) * N + i;
                accum[out * N + i] = small_mod64_madd4<NativeAdd>(
                    a0, __ldg(bk + index), a1, __ldg(bk + index + TRGSW_SIZE),
                    a2, __ldg(bk + index + (targetP::k + 1) * N),
                    a3, __ldg(bk + index + TRGSW_SIZE + (targetP::k + 1) * N),
                    accum[out * N + i]);
            }
        }
    }
    if constexpr (ROWS % 2 != 0) {
        constexpr uint32_t row = ROWS - 1;
        constexpr uint32_t component = row / targetP::l, digit = row % targetP::l;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            const T value = trlwe[component * N + i] + offset;
            const int32_t decomposition = static_cast<int32_t>(
                (value >> (64 - (digit + 1) * targetP::Bgbit)) &
                (targetP::Bg - 1)) - targetP::Bg / 2;
            work0[i] = signed_int_to_ntt_mod<N, 64>(decomposition);
        }
        __syncthreads();
        SmallForwardNTT<targetP::nbit, 64, false, NativeAdd>(work0, ntt.forward_root_, tid);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            const auto x0 = StageMonomials64 ? monomials[i] : cached_x0[e];
            const auto x1 = StageMonomials64 ? monomials[N + i] : cached_x1[e];
            const auto a0 = small_mod64_mult<NativeAdd>(work0[i], x0);
            const auto a1 = small_mod64_mult<NativeAdd>(work0[i], x1);
#pragma unroll
            for (int out = 0; out <= targetP::k; ++out) {
                const size_t index = block * 2 * TRGSW_SIZE +
                    (row * (targetP::k + 1) + out) * N + i;
                accum[out * N + i] = small_mod64_madd2<NativeAdd>(
                    a0, __ldg(bk + index), a1, __ldg(bk + index + TRGSW_SIZE),
                    accum[out * N + i]);
            }
        }
    }
    __syncthreads();

    // CUDA 13.3 / SM89: this conditional compiler memory clobber retains
    // the measured raw inverse schedule, with no extra device instructions.
#if defined(CUFHE_WARP_TRANSFORM_TAIL) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ == 890 && defined(__CUDACC_VER_MAJOR__) && \
    __CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ == 3
    if constexpr (!StageMonomials64)
        if (tid == 0)
            asm volatile("" ::: "memory");
#endif
    SmallInverseNTTPair64<targetP::nbit, NativeAdd, StageMonomials64>(accum, accum + N,
        ntt.inverse_root_, ntt.n_inverse_, tid);
    SmallInverseNTT<targetP::nbit, 64, NativeAdd>(accum + 2 * N,
        ntt.inverse_root_, ntt.n_inverse_, tid);

#pragma unroll
    for (int out = 0; out <= targetP::k; ++out) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const uint32_t i = tid + e * THREADS;
            trlwe[out * N + i] += ntt_mod_to_torus64<N, 64, true>(accum[out * N + i]);
        }
    }
    __syncthreads();
}
#endif

template <class P, bool StageMonomials64 = false, bool PairUnstaged64 = false,
          bool NativeForwardSubtract = false, int Native64Add = 0,
          bool RegisterMonomials31 = false, bool KeySwitchAfter31 = false,
          bool ReuseInput31 = false, bool AdjacentScalar64 = true>
__device__ inline void AccumulateBlockBinary(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t block,
    const uint32_t* const bara, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    if constexpr (USE_PAIRED_NTT<typename P::targetP>) {
        AccumulateBlockBinaryPairedNTT<P, RegisterMonomials31, KeySwitchAfter31, ReuseInput31>(trlwe, sh_acc_ntt, block, bara, bk, ntt);
        return;
    }
#ifdef CUFHE_HAS_PAIR_NTT64
    if constexpr ((StageMonomials64 || PairUnstaged64) && USE_PAIRED_NTT64<typename P::targetP> && P::domainP::ell == 2) {
        // Carry/compare benefits the unstaged paired raw schedule only.
        constexpr int paired_add = PairUnstaged64 && !StageMonomials64 ? 2 : Native64Add;
        AccumulateBlockBinaryPaired64<P, StageMonomials64, paired_add>(
            trlwe, sh_acc_ntt, block, bara, bk, ntt);
        return;
    }
#endif
    using targetP = typename P::targetP;
    using T = typename targetP::T;
    using Value = NTTValueFor<targetP::n, sizeof(T) * 8>;
    constexpr bool REGISTER_ACCUM = USE_REGISTER_NTT_ACCUM<targetP>;
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t ROWS = BootstrappingTRGSWRows<targetP>;
    constexpr size_t TRGSW_SIZE = ROWS * (targetP::k + 1) * N;
    static_assert(P::domainP::ell > 1);
    static_assert(
        targetP::l̅ == 1 && targetP::l̅ₐ == 1,
        "Block-binary GPU bootstrapping requires single decomposition");

    const uint32_t tid = ThisThreadRankInBlock();
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* xai_ntt;
    if constexpr (sizeof(typename P::targetP::T) == 4)
        xai_ntt = block_xai_fft;
    else
        xai_ntt = block_xai_fft_lvl02;
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work = &sh_acc_ntt[0];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_accum = &sh_acc_ntt[N];

#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
    Value* const staged_monomials = sh_acc_ntt + (targetP::k + 2) * N;
    if constexpr (StageMonomials64 && USE_ASYNC_MONOMIALS64<targetP> && P::domainP::ell == 2) {
        __pipeline_memcpy_async(staged_monomials + 2 * tid,
            xai_ntt + bara[0] * N + 2 * tid, 16);
        __pipeline_memcpy_async(staged_monomials + N + 2 * tid,
            xai_ntt + bara[1] * N + 2 * tid, 16);
        __pipeline_commit();
    }
#endif

    Value local_accum[targetP::k + 1][2];
    if (tid < NUM_THREADS) {
#pragma unroll
        for (int component = 0; component <= targetP::k; component++) {
            if constexpr (REGISTER_ACCUM) {
                local_accum[component][0] = 0;
                local_accum[component][1] = 0;
            }
            else {
                sh_accum[component * N + tid] = 0;
                sh_accum[component * N + tid + NUM_THREADS] = 0;
            }
        }
    }

    // Adjacent NTT values of x^a-1 are equal for even a and add to -2
    // for odd a. Preserve the existing lane cache when no register tail is available.
    constexpr bool CACHE_SCALAR_MONOMIALS64 =
#if !defined(CUFHE_USE_HIP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
        sizeof(T) == 8 && NativeForwardSubtract && !StageMonomials64 &&
        P::domainP::ell == 2 && NUM_THREADS % 32 == 0;
#else
        false;
#endif
    constexpr bool DIRECT_SCALAR_NTT64 =
#if defined(CUFHE_WARP_TRANSFORM_TAIL)
        AdjacentScalar64 && CACHE_SCALAR_MONOMIALS64 && UseWarpNTTTail<64>;
#else
        false;
#endif
    // The forward warp tail owns adjacent coefficients. Cache each polynomial
    // at the even coordinate, then reconstruct its odd partner exactly.
    Value cached_monomial[2];
    bool exponent0_odd, exponent1_odd, peer_exponent_odd;
    if constexpr (DIRECT_SCALAR_NTT64) {
        if (tid < NUM_THREADS) {
            exponent0_odd = (bara[0] & 1) != 0;
            exponent1_odd = (bara[1] & 1) != 0;
            cached_monomial[0] = __ldg(&xai_ntt[bara[0] * N + 2 * tid]);
            cached_monomial[1] = __ldg(&xai_ntt[bara[1] * N + 2 * tid]);
        }
    } else if constexpr (CACHE_SCALAR_MONOMIALS64) {
        if (tid < NUM_THREADS) {
            const bool odd_lane = (tid & 1) != 0;
            const uint32_t own_exponent = odd_lane ? bara[1] : bara[0];
            peer_exponent_odd = ((odd_lane ? bara[0] : bara[1]) & 1) != 0;
#pragma unroll
            for (int e = 0; e < 2; ++e)
                cached_monomial[e] = __ldg(&xai_ntt[own_exponent * N + tid + e * NUM_THREADS]);
        }
    }

    for (int component = 0; component <= targetP::k; component++) {
        const bool nonce = component < targetP::k;
        const uint32_t digits = nonce ? targetP::lₐ : targetP::l;
        const uint32_t basebit = nonce ? targetP::Bgₐbit : targetP::Bgbit;
        const uint32_t base = nonce ? targetP::Bgₐ : targetP::Bg;
        const uint32_t mask = base - 1;
        const int32_t halfbase = base >> 1;
        T decomp_offset = 0;
        for (uint32_t digit = 1; digit <= digits; digit++)
            decomp_offset +=
                static_cast<T>(halfbase)
                << (std::numeric_limits<T>::digits - digit * basebit);
        const T roundoffset =
            static_cast<T>(1)
            << (std::numeric_limits<T>::digits - digits * basebit - 1);

        T cached_input[2];
        if constexpr (CACHE_SCALAR_MONOMIALS64) {
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; ++e)
                    cached_input[e] = trlwe[component * N + tid + e * NUM_THREADS] +
                                      decomp_offset + roundoffset;
            }
        }

        for (uint32_t digit = 0; digit < digits; digit++) {
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int i = tid + e * NUM_THREADS;
                    // The TRLWE is unchanged until all decomposition rows
                    // have accumulated, so each digit can reuse the same word.
                    const T value = CACHE_SCALAR_MONOMIALS64 ? cached_input[e] :
                        trlwe[component * N + i] + decomp_offset + roundoffset;
                    const int32_t decomposed =
                        static_cast<int32_t>(
                            (value >> (std::numeric_limits<T>::digits -
                                       (digit + 1) * basebit)) &
                            mask) -
                        halfbase;
                    sh_work[i] = signed_int_to_ntt_mod<N, sizeof(typename P::targetP::T) * 8>(decomposed);
                }
            }
            __syncthreads();

            Value transformed_pair[2];
            if (tid < NUM_THREADS) {
                SmallForwardNTT<targetP::nbit, sizeof(typename P::targetP::T) * 8,
                    NativeForwardSubtract, Native64Add, DIRECT_SCALAR_NTT64>(
                    sh_work, ntt.forward_root_, tid, transformed_pair);
            }
            else {
                for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                    __syncthreads();
            }

#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
            if constexpr (StageMonomials64 && USE_ASYNC_MONOMIALS64<targetP> && P::domainP::ell == 2) {
                if (component == 0 && digit == 0) {
                    __pipeline_wait_prior(0);
                    __syncthreads();
                }
            }
#endif
            const uint32_t row = nonce ? component * targetP::lₐ + digit
                                       : targetP::k * targetP::lₐ + digit;
            if (tid < NUM_THREADS) {
                if constexpr (DIRECT_SCALAR_NTT64) {
                    // Even coefficient pairs and every row/polynomial offset
                    // preserve the allocation's 16-byte vector alignment.
                    static_assert(N % 2 == 0 && TRGSW_SIZE % 2 == 0);
                    Value prefactor0[2], prefactor1[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        Value m0 = cached_monomial[0], m1 = cached_monomial[1];
                        if (e == 1) {
                            if (exponent0_odd) m0 = small_ntt::P - 2 - m0;
                            if (exponent1_odd) m1 = small_ntt::P - 2 - m1;
                        }
                        prefactor0[e] = small_mod64_mult<Native64Add>(transformed_pair[e], m0);
                        prefactor1[e] = small_mod64_mult<Native64Add>(transformed_pair[e], m1);
                    }
#pragma unroll
                    for (int out_component = 0; out_component <= targetP::k; ++out_component) {
                        const size_t index = (block * P::domainP::ell) * TRGSW_SIZE +
                            (row * (targetP::k + 1) + out_component) * N + 2 * tid;
                        const ulonglong2 keys0 = __ldg(reinterpret_cast<const ulonglong2*>(bk + index));
                        const ulonglong2 keys1 = __ldg(reinterpret_cast<const ulonglong2*>(bk + index + TRGSW_SIZE));
                        if constexpr (REGISTER_ACCUM) {
                            local_accum[out_component][0] = small_mod64_madd2<Native64Add>(
                                keys0.x, prefactor0[0], keys1.x, prefactor1[0], local_accum[out_component][0]);
                            local_accum[out_component][1] = small_mod64_madd2<Native64Add>(
                                keys0.y, prefactor0[1], keys1.y, prefactor1[1], local_accum[out_component][1]);
                        } else {
                            auto* pair = reinterpret_cast<ulonglong2*>(sh_accum + out_component * N + 2 * tid);
                            ulonglong2 sum = *pair;
                            sum.x = small_mod64_madd2<Native64Add>(
                                keys0.x, prefactor0[0], keys1.x, prefactor1[0], sum.x);
                            sum.y = small_mod64_madd2<Native64Add>(
                                keys0.y, prefactor0[1], keys1.y, prefactor1[1], sum.y);
                            *pair = sum;
                        }
                    }
                } else {
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const int i = tid + e * NUM_THREADS;
                        const Value decomposition = sh_work[i];
                        if constexpr (sizeof(T) == 8 && P::domainP::ell == 2) {
                            // Share these products across all output components.
                            Value monomial0, monomial1;
#if defined(CUFHE_ASYNC_MONOMIALS_DEVICE)
                            if constexpr (StageMonomials64 && USE_ASYNC_MONOMIALS64<targetP>) {
                                monomial0 = staged_monomials[i];
                                monomial1 = staged_monomials[N + i];
                            } else
#endif
                            {
                                if constexpr (CACHE_SCALAR_MONOMIALS64) {
                                    const Value own = cached_monomial[e];
                                    Value peer = SmallNTTShuffleXor(own, 1);
                                    if (peer_exponent_odd) peer = small_ntt::P - 2 - peer;
                                    monomial0 = (tid & 1) ? peer : own;
                                    monomial1 = (tid & 1) ? own : peer;
                                } else {
                                    monomial0 = __ldg(&xai_ntt[bara[0] * N + i]);
                                    monomial1 = __ldg(&xai_ntt[bara[1] * N + i]);
                                }
                            }
                            const Value prefactor0 = small_mod64_mult<Native64Add>(decomposition, monomial0);
                            const Value prefactor1 = small_mod64_mult<Native64Add>(decomposition, monomial1);
#pragma unroll
                            for (int out_component = 0; out_component <= targetP::k;
                                 ++out_component) {
                                const size_t poly_index =
                                    (row * (targetP::k + 1) + out_component) * N + i;
                                const Value key0 = __ldg(&bk[
                                    (block * P::domainP::ell) * TRGSW_SIZE + poly_index]);
                                const Value key1 = __ldg(&bk[
                                    (block * P::domainP::ell + 1) * TRGSW_SIZE + poly_index]);
                                if constexpr (REGISTER_ACCUM)
                                    local_accum[out_component][e] = small_mod64_madd2<Native64Add>(
                                        key0, prefactor0, key1, prefactor1,
                                        local_accum[out_component][e]);
                                else
                                    sh_accum[out_component * N + i] = small_mod64_madd2<Native64Add>(
                                        key0, prefactor0, key1, prefactor1,
                                        sh_accum[out_component * N + i]);
                            }
                        } else {
#pragma unroll
                            for (int out_component = 0; out_component <= targetP::k;
                                 out_component++) {
                                NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> block_product = 0;
        #pragma unroll
                                for (uint32_t offset = 0; offset < P::domainP::ell;
                                     offset++) {
                                    const size_t key_index =
                                        block * P::domainP::ell + offset;
                                    const size_t poly_index =
                                        (row * (targetP::k + 1) + out_component) * N +
                                        i;
                                    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk_value =
                                        __ldg(&bk[key_index * TRGSW_SIZE + poly_index]);
                                    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai =
                                        __ldg(&xai_ntt[bara[offset] * N + i]);
                                    block_product =
                                        small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(bk_value, xai, block_product);
                                }
                                if constexpr (REGISTER_ACCUM)
                                    local_accum[out_component][e] =
                                        small_mod_madd<N, sizeof(T) * 8>(
                                            decomposition, block_product,
                                            local_accum[out_component][e]);
                                else
                                    sh_accum[out_component * N + i] =
                                        small_mod_madd<N, sizeof(T) * 8>(
                                            decomposition, block_product,
                                            sh_accum[out_component * N + i]);
                            }
                        }
                    }
                }
            }
        }
    }

    // Accumulation is coefficient-local. Synchronize once before the inverse
    // butterflies begin reading coefficients written by other threads.
    __syncthreads();

    // The next inverse uses a disjoint shared segment while upper threads
    // finish loading the previous component's published half.
    constexpr bool RETURN_INVERSE_PAIR64 = DIRECT_SCALAR_NTT64 && !REGISTER_ACCUM;
    for (int component = 0; component <= targetP::k; component++) {
        Value* const sh_result = REGISTER_ACCUM ? sh_work : &sh_accum[component * N];
        if constexpr (REGISTER_ACCUM) {
            if (tid < NUM_THREADS) {
                const int i0 = DIRECT_SCALAR_NTT64 ? 2 * tid : tid;
                const int i1 = DIRECT_SCALAR_NTT64 ? 2 * tid + 1 : tid + NUM_THREADS;
                sh_result[i0] = local_accum[component][0];
                sh_result[i1] = local_accum[component][1];
            }
            __syncthreads();
        }
        Value inverse_pair[2];
        if (tid < NUM_THREADS) {
#if defined(CUFHE_HAS_PAIR_NTT64)
            if constexpr (RETURN_INVERSE_PAIR64)
                SmallInverseNTTReturnPair64<targetP::nbit, Native64Add>(
                    sh_result, ntt.inverse_root_, ntt.n_inverse_, tid, inverse_pair);
            else
#endif
                SmallInverseNTT<targetP::nbit, sizeof(typename P::targetP::T) * 8, Native64Add>(
                    sh_result, ntt.inverse_root_, ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int i = tid + e * NUM_THREADS;
                const Value coefficient = RETURN_INVERSE_PAIR64 ? inverse_pair[e] : sh_result[i];
                if constexpr (std::numeric_limits<T>::digits == 64)
                    trlwe[component * N + i] +=
                        static_cast<T>(ntt_mod_to_torus64<N, sizeof(typename P::targetP::T) * 8>(coefficient));
                else
                    trlwe[component * N + i] +=
                        static_cast<T>(ntt_mod_to_torus32<N, sizeof(typename P::targetP::T) * 8>(coefficient));
            }
        }
    }
    __syncthreads();
}
#endif
#endif  // USE_FFT

template <class P, bool StageMonomials64 = false, bool PairUnstaged64 = false,
          int Native64Add = 0, bool RegisterMonomials31 = false,
          bool KeySwitchAfter31 = false>
__device__ inline void __BlindRotate__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in,
    const typename P::targetP::T mu, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* const sh_acc_ntt =
        reinterpret_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>*>(dynamic_shared);

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ mod_switch(in.b)
    {
        uint32_t bar;
#ifdef USE_BLOCK_BINARY
        const auto correction = BlockBinaryModSwitchCorrection<P>(in);
        constexpr typename P::domainP::T roundoffset =
            static_cast<typename P::domainP::T>(1)
            << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                P::targetP::nbit);
        const typename P::domainP::T corrected =
            in[P::domainP::k * P::domainP::n] - correction / 2 + roundoffset;
        bar = 2 * P::targetP::n - modSwitchFromTorus<P>(corrected);
#else
        bar = 2 * P::targetP::n -
              modSwitchFromTorus<P>(in[P::domainP::k * P::domainP::n]);
#endif
        RotatedTestVector<typename P::targetP>(out, bar, mu);
    }

    // accumulate
#ifdef USE_BLOCK_BINARY
    constexpr uint32_t blocks = P::domainP::n / P::domainP::ell;
    for (uint32_t block = 0; block < blocks; block++) {
        uint32_t bara[P::domainP::ell];
#pragma unroll
        for (uint32_t offset = 0; offset < P::domainP::ell; offset++) {
            constexpr typename P::domainP::T roundoffset =
                static_cast<typename P::domainP::T>(1)
                << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                    P::targetP::nbit);
            bara[offset] = modSwitchFromTorus<P>(
                in[block * P::domainP::ell + offset] + roundoffset);
        }
#ifdef USE_FFT
        AccumulateBlockBinary<P>(out, sh_acc_ntt, block, bara, bk, ntt);
#else
        AccumulateBlockBinary<P, StageMonomials64, PairUnstaged64, false, Native64Add, RegisterMonomials31, KeySwitchAfter31>(
            out, sh_acc_ntt, block, bara, bk, ntt);
#endif
    }
#else
    for (int i = 0; i < P::domainP::k * P::domainP::n;
         i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(in[i] + roundoffset);
#ifdef USE_FFT
        // FFT BSK stride: each TRGSW has (k+1)*l * (k+1) * (N/2) complex
        // elements
        constexpr size_t trgsw_fft_size =
            BootstrappingTRGSWRows<typename P::targetP> * (P::targetP::k + 1) *
            (P::targetP::n / 2);
        Accumulate<P>(out, sh_acc_ntt, bar, bk + i * trgsw_fft_size, ntt);
#else
        Accumulate<P>(out, sh_acc_ntt, bar,
                      bk + (i << P::targetP::nbit) * (P::targetP::k + 1) *
                               (P::targetP::k + 1) * P::targetP::l,
                      ntt);
#endif
    }
#endif
}

#ifdef USE_KEY_BUNDLE
#ifdef USE_FFT
#ifdef USE_GPU_FFT
template <class P>
__device__ inline void AccumulateKeyBundlePairedFFT(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t bara0,
    const uint32_t bara1, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk0_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk1_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk2_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_fft,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;
    static_assert(N == TFHEpp::lvl1param::n ||
                  N == TFHEpp::lvl2param::n);
    static_assert(P::targetP::k == 1);
    static_assert(((P::targetP::k + 1) * P::targetP::l) % 2 == 0);

    const uint32_t tid = ThisThreadRankInBlock();
    double2* const sh_fft0 = &sh_acc_ntt[0];
    double2* const sh_fft1 = &sh_acc_ntt[HALF_N];
    double2 local_accum[P::targetP::k + 1];
#pragma unroll
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++)
        local_accum[k_idx] = {0.0, 0.0};

    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);
    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    // Flatten the TRLWE components and decomposition digits into independent
    // transforms, then pair across component boundaries.
    constexpr int FFT_PAIRS =
        (P::targetP::k + 1) * P::targetP::l / 2;
#pragma unroll 1
    for (int pair = 0; pair < FFT_PAIRS; pair++) {
        if (tid < HALF_N) {
            const double2 tw = __ldg(&ntt.twist_[tid]);
#pragma unroll
            for (int slot = 0; slot < 2; slot++) {
                const int digit_linear = 2 * pair + slot;
                const int j = digit_linear / P::targetP::l;
                const int digit = digit_linear - j * P::targetP::l;

                typename P::targetP::T temp_re = trlwe[j * N + tid];
                temp_re += decomp_offset + roundoffset;
                const int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                typename P::targetP::T temp_im =
                    trlwe[j * N + tid + HALF_N];
                temp_im += decomp_offset + roundoffset;
                const int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                const double2 folded = {static_cast<double>(digit_re),
                                        static_cast<double>(digit_im)};
                double2* const sh_fft = slot == 0 ? sh_fft0 : sh_fft1;
                sh_fft[tid] = folded * tw;
            }
        }
        __syncthreads();

        double2* const thread_fft = tid < FFT_THREADS ? sh_fft0 : sh_fft1;
        GPUFFTForward<N>(thread_fft, ntt.forward_root_,
                         tid & (FFT_THREADS - 1));

        if (tid < HALF_N) {
            const double2 xai0 = __ldg(&xai_fft[bara0 * HALF_N + tid]);
            const double2 xai1 = __ldg(&xai_fft[bara1 * HALF_N + tid]);
            const double2 xai01 = __ldg(&xai_fft[bara01 * HALF_N + tid]);

#pragma unroll
            for (int slot = 0; slot < 2; slot++) {
                const int digit_linear = 2 * pair + slot;
                const int j = digit_linear / P::targetP::l;
                const int digit = digit_linear - j * P::targetP::l;
                double2* const sh_fft = slot == 0 ? sh_fft0 : sh_fft1;
                const double2 fft_val = sh_fft[tid];
                // The identity TRGSW contains a diagonal delta polynomial.
                // Its normalized FFT is this real constant at every bin.
                const double gadget =
                    1.0 / static_cast<double>(
                              1ULL << ((digit + 1) * P::targetP::Bgbit));

#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    const uint32_t bk_offset =
                        ((P::targetP::k + 1) * digit_linear + out_k) *
                            HALF_N +
                        tid;
                    const double2 one_val =
                        out_k == j ? double2{gadget, 0.0}
                                   : double2{0.0, 0.0};
                    double2 combined = one_val;
                    complex_madd(combined, __ldg(&bk2_fft[bk_offset]), xai1);
                    complex_madd(combined, __ldg(&bk1_fft[bk_offset]), xai0);
                    complex_madd(combined, __ldg(&bk0_fft[bk_offset]), xai01);
                    complex_madd(local_accum[out_k], fft_val, combined);
                }
            }
        }
        __syncthreads();
    }

    if (tid < HALF_N) {
        sh_fft0[tid] = local_accum[0];
        sh_fft1[tid] = local_accum[1];
    }
    __syncthreads();

    double2* const thread_inverse =
        tid < FFT_THREADS ? sh_fft0 : sh_fft1;
    GPUFFTInverse<N>(thread_inverse, ntt.inverse_root_,
                     tid & (FFT_THREADS - 1));

    constexpr double denorm = (sizeof(typename P::targetP::T) == 4)
                                  ? 4294967296.0
                                  : 18446744073709551616.0;
    if (tid < HALF_N) {
        const double2 utw = __ldg(&ntt.untwist_[tid]);
        double2 val0 = sh_fft0[tid] * utw;
        double2 val1 = sh_fft1[tid] * utw;
        if constexpr (sizeof(typename P::targetP::T) == 4) {
            trlwe[tid] = static_cast<typename P::targetP::T>(
                static_cast<int32_t>(llrint(val0.x * denorm)));
            trlwe[tid + HALF_N] = static_cast<typename P::targetP::T>(
                static_cast<int32_t>(llrint(val0.y * denorm)));
            trlwe[N + tid] = static_cast<typename P::targetP::T>(
                static_cast<int32_t>(llrint(val1.x * denorm)));
            trlwe[N + tid + HALF_N] = static_cast<typename P::targetP::T>(
                static_cast<int32_t>(llrint(val1.y * denorm)));
        }
        else {
            trlwe[tid] = static_cast<typename P::targetP::T>(
                double_to_torus64(val0.x * denorm));
            trlwe[tid + HALF_N] = static_cast<typename P::targetP::T>(
                double_to_torus64(val0.y * denorm));
            trlwe[N + tid] = static_cast<typename P::targetP::T>(
                double_to_torus64(val1.x * denorm));
            trlwe[N + tid + HALF_N] = static_cast<typename P::targetP::T>(
                double_to_torus64(val1.y * denorm));
        }
    }
    __syncthreads();
}

/**
 * GPU-FFT Key-bundle ExternalProduct accumulator
 *
 * Same as tfhe-rs version but uses fold+twist, GPU-FFT forward/inverse,
 * untwist+unfold
 */
template <class P>
__device__ inline void AccumulateKeyBundle(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t bara0,
    const uint32_t bara1, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk0_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk1_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk2_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const one_trgsw_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_fft,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    if constexpr (USE_PAIRED_GPU_FFT<typename P::targetP>) {
        AccumulateKeyBundlePairedFFT<P>(
            trlwe, sh_acc_ntt, bara0, bara1, bk0_fft, bk1_fft, bk2_fft,
            xai_fft, ntt);
        return;
    }

    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256
    constexpr bool LOW_LDS = USE_LOW_LDS_BOOTSTRAP<typename P::targetP>;

    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];
    double2 local_accum[P::targetP::k + 1];

    if constexpr (LOW_LDS) {
#pragma unroll
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++)
            local_accum[k_idx] = {0.0, 0.0};
    }
    else {
        for (int i = tid; i < (P::targetP::k + 1) * HALF_N; i += NUM_THREADS) {
            sh_accum[i] = {0.0, 0.0};
        }
    }
    __syncthreads();

    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Decompose + fold + twist
            if (tid < HALF_N) {
                typename P::targetP::T temp_re = trlwe[j * N + tid];
                temp_re += decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                typename P::targetP::T temp_im = trlwe[j * N + tid + HALF_N];
                temp_im += decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                double2 folded = {static_cast<double>(digit_re),
                                  static_cast<double>(digit_im)};
                double2 tw = __ldg(&ntt.twist_[tid]);
                sh_fft[tid] = folded * tw;
            }
            __syncthreads();

            // Step 2: Forward FFT (GPU-FFT)
            if (tid < FFT_THREADS) {
                GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++)
                    __syncthreads();
            }

            // Step 3: Multiply with on-the-fly keybundle and accumulate
            int digit_linear = j * P::targetP::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];

                double2 xai0 = __ldg(&xai_fft[bara0 * HALF_N + tid]);
                double2 xai1 = __ldg(&xai_fft[bara1 * HALF_N + tid]);
                double2 xai01 = __ldg(&xai_fft[bara01 * HALF_N + tid]);

#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    uint32_t bk_offset =
                        ((P::targetP::k + 1) * digit_linear + out_k) * HALF_N +
                        tid;

                    double2 one_val = __ldg(&one_trgsw_fft[bk_offset]);
                    double2 bk0_val = __ldg(&bk0_fft[bk_offset]);
                    double2 bk1_val = __ldg(&bk1_fft[bk_offset]);
                    double2 bk2_val = __ldg(&bk2_fft[bk_offset]);

                    double2 combined = one_val;
                    combined += bk2_val * xai1;
                    combined += bk1_val * xai0;
                    combined += bk0_val * xai01;

                    if constexpr (LOW_LDS)
                        local_accum[out_k] += fft_val * combined;
                    else
                        sh_accum[out_k * HALF_N + tid] += fft_val * combined;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT + untwist + unfold, REPLACE trlwe
    constexpr double denorm = (sizeof(typename P::targetP::T) == 4)
                                  ? 4294967296.0
                                  : 18446744073709551616.0;
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* sh_inv;
        if constexpr (LOW_LDS) {
            if (tid < HALF_N) sh_fft[tid] = local_accum[k_idx];
            __syncthreads();
            sh_inv = sh_fft;
        }
        else {
            sh_inv = &sh_accum[k_idx * HALF_N];
        }
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
            if constexpr (sizeof(typename P::targetP::T) == 4) {
                trlwe[k_idx * N + tid] = static_cast<typename P::targetP::T>(
                    static_cast<int32_t>(llrint(val.x * denorm)));
                trlwe[k_idx * N + tid + HALF_N] =
                    static_cast<typename P::targetP::T>(
                        static_cast<int32_t>(llrint(val.y * denorm)));
            }
            else {
                trlwe[k_idx * N + tid] = static_cast<typename P::targetP::T>(
                    double_to_torus64(val.x * denorm));
                trlwe[k_idx * N + tid + HALF_N] =
                    static_cast<typename P::targetP::T>(
                        double_to_torus64(val.y * denorm));
            }
        }
        __syncthreads();
    }
}
#else  // !USE_GPU_FFT (tfhe-rs FFT)
/**
 * FFT Key-bundle ExternalProduct accumulator (tfhe-rs style)
 *
 * Decomposes acc directly, FFTs decomposed polynomial, multiplies with
 * on-the-fly keybundle in Fourier domain, IFFTs and REPLACEs acc.
 *
 * combined = one_fft + bk2_fft*xai_fft[a1] + bk1_fft*xai_fft[a0] +
 * bk0_fft*xai_fft[a0+a1]
 */
template <class P>
__device__ inline void AccumulateKeyBundle(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t bara0,
    const uint32_t bara1,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk0_fft,  // Enc(s0*s1)
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk1_fft,  // Enc(s0*(1-s1))
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk2_fft,  // Enc((1-s0)*s1)
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const one_trgsw_fft,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_fft,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];

    // Initialize accumulated results to zero
    for (int i = tid; i < (P::targetP::k + 1) * HALF_N; i += NUM_THREADS) {
        sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Precompute combined bara for bk0 (a0+a1 mod 2N)
    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Decompose trlwe[j] directly (no MulByXaiMinus) and pack
            if (tid < HALF_N) {
                // First-half coefficient (index tid)
                typename P::targetP::T temp_re = trlwe[j * N + tid];
                temp_re += decomp_offset + roundoffset;
                int32_t digit_re = static_cast<int32_t>(
                    ((temp_re >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                // Second-half coefficient (index tid + N/2)
                typename P::targetP::T temp_im = trlwe[j * N + tid + HALF_N];
                temp_im += decomp_offset + roundoffset;
                int32_t digit_im = static_cast<int32_t>(
                    ((temp_im >>
                      (std::numeric_limits<typename P::targetP::T>::digits -
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                sh_fft[tid] = {static_cast<double>(digit_re),
                               static_cast<double>(digit_im)};
            }
            __syncthreads();

            // Step 2: Forward FFT
            if (tid < FFT_THREADS) {
                NSMFFT_direct<HalfDegree<Degree<N> > >(sh_fft);
            }
            else {
                for (int s = 0; s < 11; s++) __syncthreads();
            }

            // Step 3: Multiply with on-the-fly keybundle and accumulate
            int digit_linear = j * P::targetP::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];

                // Load xai FFT values
                double2 xai0 = __ldg(&xai_fft[bara0 * HALF_N + tid]);
                double2 xai1 = __ldg(&xai_fft[bara1 * HALF_N + tid]);
                double2 xai01 = __ldg(&xai_fft[bara01 * HALF_N + tid]);

#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    uint32_t bk_offset =
                        ((P::targetP::k + 1) * digit_linear + out_k) * HALF_N +
                        tid;

                    double2 one_val = __ldg(&one_trgsw_fft[bk_offset]);
                    double2 bk0_val = __ldg(&bk0_fft[bk_offset]);
                    double2 bk1_val = __ldg(&bk1_fft[bk_offset]);
                    double2 bk2_val = __ldg(&bk2_fft[bk_offset]);

                    // combined = one + bk2*xai1 + bk1*xai0 + bk0*xai01
                    double2 combined = one_val;
                    combined += bk2_val * xai1;
                    combined += bk1_val * xai0;
                    combined += bk0_val * xai01;

                    sh_accum[out_k * HALF_N + tid] += fft_val * combined;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT and REPLACE trlwe (not add)
    constexpr double denorm = 4294967296.0;  // 2^32
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];
        if (tid < FFT_THREADS) {
            NSMFFT_inverse<HalfDegree<Degree<N> > >(sh_inv);
        }
        else {
            for (int s = 0; s < 11; s++) __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            trlwe[k_idx * N + tid] = static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.x * denorm)));
            trlwe[k_idx * N + tid + HALF_N] = static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}
#endif  // USE_GPU_FFT
#else   // !USE_FFT

/**
 * NTT Key-bundle accumulator with paired transforms
 *
 * Flattens the (component, digit) loop into pairs and transforms two
 * decomposed polynomials per SmallForwardNTTPair pass, so the radix-4 stages
 * occupy the whole block.  The two output components share one
 * SmallInverseNTTPair pass at the end.
 */
template <class P>
__device__ inline void AccumulateKeyBundlePairedNTT(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t bara0,
    const uint32_t bara1,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk0_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk1_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk2_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_ntt,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;
    static_assert(P::targetP::k == 1);
    static_assert(((P::targetP::k + 1) * P::targetP::l) % 2 == 0);

    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work0 = &sh_acc_ntt[0];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work1 = &sh_acc_ntt[N];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> local_accum[P::targetP::k + 1][2];

    if (tid < NUM_THREADS) {
#pragma unroll
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            local_accum[k_idx][0] = 0;
            local_accum[k_idx][1] = 0;
        }
    }

    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    constexpr int NTT_PAIRS = (P::targetP::k + 1) * P::targetP::l / 2;
    for (int pair = 0; pair < NTT_PAIRS; pair++) {
        // Step 1: Decompose both slots' digits
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int slot = 0; slot < 2; slot++) {
                const int digit_linear = 2 * pair + slot;
                const int j = digit_linear / P::targetP::l;
                const int digit = digit_linear - j * P::targetP::l;
                NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work =
                    slot == 0 ? sh_work0 : sh_work1;
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    const int i = tid + e * NUM_THREADS;
                    typename P::targetP::T temp = trlwe[j * N + i];
                    temp += decomp_offset + roundoffset;
                    const int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<
                               typename P::targetP::T>::digits -
                           (digit + 1) * P::targetP::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = signed_int_to_ntt_mod<N, sizeof(typename P::targetP::T) * 8>(digit_val);
                }
            }
        }
        __syncthreads();

        // Step 2: One paired forward NTT for both digits
        if (tid < NUM_THREADS) {
            SmallForwardNTTPair<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_work0, sh_work1,
                                                  ntt.forward_root_, tid);
        }
        else {
            for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        // Step 3: Multiply with on-the-fly keybundle and accumulate.  The
        // xai coefficients are shared by both slots.
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int i = tid + e * NUM_THREADS;
                const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai0 =
                    __ldg(&xai_ntt[bara0 * N + i]);
                const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai1 =
                    __ldg(&xai_ntt[bara1 * N + i]);
                const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai01 =
                    __ldg(&xai_ntt[bara01 * N + i]);

#pragma unroll
                for (int slot = 0; slot < 2; slot++) {
                    const int digit_linear = 2 * pair + slot;
                    const int j = digit_linear / P::targetP::l;
                    const int digit = digit_linear - j * P::targetP::l;
                    const int gadget_shift = (digit + 1) * P::targetP::Bgbit;
                    const SmallNTTValue gadget_quotient =
                        SmallNTTModulus<N, sizeof(typename P::targetP::T) * 8>::P >> gadget_shift;
                    const SmallNTTValue gadget_round =
                        (SmallNTTModulus<N, sizeof(typename P::targetP::T) * 8>::P >> (gadget_shift - 1)) & 1;
                    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> gadget =
                        static_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> >(
                            gadget_quotient + gadget_round);
                    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt_val =
                        (slot == 0 ? sh_work0 : sh_work1)[i];

#pragma unroll
                    for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                        const uint32_t bk_offset =
                            (((P::targetP::k + 1) * digit_linear + out_k)
                             << P::targetP::nbit) +
                            i;
                        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> one_val =
                            out_k == j ? gadget : 0;
                        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk0_val =
                            __ldg(&bk0_ntt[bk_offset]);
                        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk1_val =
                            __ldg(&bk1_ntt[bk_offset]);
                        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk2_val =
                            __ldg(&bk2_ntt[bk_offset]);

                        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> combined =
                            small_mod_madd3<N, sizeof(typename P::targetP::T) * 8>(bk2_val, xai1, bk1_val, xai0,
                                               bk0_val, xai01, one_val);
                        local_accum[out_k][e] = small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(
                            ntt_val, combined, local_accum[out_k][e]);
                    }
                }
            }
        }
    }

    __syncthreads();

    // Step 4: One paired inverse NTT covers both output components.
    if (tid < NUM_THREADS) {
        sh_work0[tid] = local_accum[0][0];
        sh_work0[tid + NUM_THREADS] = local_accum[0][1];
        sh_work1[tid] = local_accum[1][0];
        sh_work1[tid + NUM_THREADS] = local_accum[1][1];
    }
    __syncthreads();
    if (tid < NUM_THREADS) {
        SmallInverseNTTPair<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_work0, sh_work1,
                                              ntt.inverse_root_,
                                              ntt.n_inverse_, tid);
    }
    else {
        for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
            __syncthreads();
    }

    if (tid < NUM_THREADS) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int i = tid + e * NUM_THREADS;
            if constexpr (std::numeric_limits<
                              typename P::targetP::T>::digits == 64) {
                trlwe[i] = static_cast<typename P::targetP::T>(
                    ntt_mod_to_torus64<N, sizeof(typename P::targetP::T) * 8>(sh_work0[i]));
                trlwe[N + i] = static_cast<typename P::targetP::T>(
                    ntt_mod_to_torus64<N, sizeof(typename P::targetP::T) * 8>(sh_work1[i]));
            }
            else {
                trlwe[i] = static_cast<typename P::targetP::T>(
                    ntt_mod_to_torus32<N, sizeof(typename P::targetP::T) * 8>(sh_work0[i]));
                trlwe[N + i] = static_cast<typename P::targetP::T>(
                    ntt_mod_to_torus32<N, sizeof(typename P::targetP::T) * 8>(sh_work1[i]));
            }
        }
    }
    __syncthreads();
}

/**
 * NTT Key-bundle ExternalProduct accumulator
 *
 * Instead of computing (X^bar - 1) * acc and adding to acc (CMUX),
 * this decomposes acc directly and replaces it with the ExternalProduct result.
 * The keybundle TRGSW is computed on-the-fly from 3 BK elements and the xai
 * table.
 *
 * combined[digit_linear][out_k][i] = one_ntt[...] + bk2[...]*xai[a1] +
 * bk1[...]*xai[a0] + bk0[...]*xai[a01]
 */
template <class P>
__device__ inline void AccumulateKeyBundle(
    typename P::targetP::T* const trlwe,
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_acc_ntt, const uint32_t bara0,
    const uint32_t bara1,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk0_ntt,  // Enc(s0*s1)
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk1_ntt,  // Enc(s0*(1-s1))
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk2_ntt,  // Enc((1-s0)*s1)
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const one_trgsw_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_ntt,
    const CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    (void)one_trgsw_ntt;
    if constexpr (USE_PAIRED_NTT<typename P::targetP>) {
        AccumulateKeyBundlePairedNTT<P>(trlwe, sh_acc_ntt, bara0, bara1,
                                        bk0_ntt, bk1_ntt, bk2_ntt, xai_ntt,
                                        ntt);
        return;
    }

    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr bool REGISTER_ACCUM =
        USE_REGISTER_NTT_ACCUM<typename P::targetP>;

    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_work = &sh_acc_ntt[0];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const sh_accum = &sh_acc_ntt[N];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> local_accum[P::targetP::k + 1][2];

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
#pragma unroll
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            if constexpr (REGISTER_ACCUM) {
                local_accum[k_idx][0] = 0;
                local_accum[k_idx][1] = 0;
            }
            else {
                sh_accum[k_idx * N + tid] = 0;
                sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
            }
        }
    }

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr typename P::targetP::T decomp_offset =
        offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Precompute combined bara for bk0 (a0+a1 mod 2N)
    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    // xai is common to every row and decomposition digit.  On lvl02, keep the
    // six coefficients owned by each thread in registers for the complete
    // external product.  lvl01 keeps its shorter live ranges.
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> cached_xai0[2];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> cached_xai1[2];
    NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> cached_xai01[2];
    if constexpr (sizeof(typename P::targetP::T) == 8) {
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int i = tid + e * NUM_THREADS;
                cached_xai0[e] = __ldg(&xai_ntt[bara0 * N + i]);
                cached_xai1[e] = __ldg(&xai_ntt[bara1 * N + i]);
                cached_xai01[e] = __ldg(&xai_ntt[bara01 * N + i]);
            }
        }
    }

    constexpr int REUSE_DIGITS = N == TFHEpp::lvl2param::n ? 2 : 1;
    static_assert(P::targetP::l % REUSE_DIGITS == 0);
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit_group = 0; digit_group < P::targetP::l / REUSE_DIGITS;
             digit_group++) {
#pragma unroll
            for (int slot = 0; slot < REUSE_DIGITS; slot++) {
                const int digit = REUSE_DIGITS * digit_group + slot;
                const int digit_linear = j * P::targetP::l + digit;

                // Step 1: Decompose trlwe[j] directly (no MulByXaiMinus)
                if (tid < NUM_THREADS) {
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        int i = tid + e * NUM_THREADS;
                        typename P::targetP::T temp = trlwe[j * N + i];
                        temp += decomp_offset + roundoffset;
                        int32_t digit_val = static_cast<int32_t>(
                            ((temp >> (std::numeric_limits<
                                           typename P::targetP::T>::digits -
                                       (digit + 1) * P::targetP::Bgbit)) &
                             decomp_mask) -
                            decomp_half);
                        sh_work[i] = signed_int_to_ntt_mod<N, sizeof(typename P::targetP::T) * 8>(digit_val);
                    }
                }
                __syncthreads();

                // Step 2: Forward NTT on decomposed polynomial
                if (tid < NUM_THREADS) {
                    SmallForwardNTT<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_work,
                                                      ntt.forward_root_, tid);
                }
                else {
                    for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
                        __syncthreads();
                }

                // Step 3: Multiply with on-the-fly keybundle and accumulate
                const int gadget_shift = (digit + 1) * P::targetP::Bgbit;
                const SmallNTTValue gadget_quotient =
                    SmallNTTModulus<N, sizeof(typename P::targetP::T) * 8>::P >> gadget_shift;
                const SmallNTTValue gadget_round =
                    (SmallNTTModulus<N, sizeof(typename P::targetP::T) * 8>::P >> (gadget_shift - 1)) & 1;
                const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> gadget =
                    static_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> >(gadget_quotient +
                                                             gadget_round);
                if (tid < NUM_THREADS) {
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        int i = tid + e * NUM_THREADS;
                        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt_val = sh_work[i];
                        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai0;
                        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai1;
                        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> xai01;
                        if constexpr (sizeof(typename P::targetP::T) == 8) {
                            xai0 = cached_xai0[e];
                            xai1 = cached_xai1[e];
                            xai01 = cached_xai01[e];
                        }
                        else {
                            xai0 = __ldg(&xai_ntt[bara0 * N + i]);
                            xai1 = __ldg(&xai_ntt[bara1 * N + i]);
                            xai01 = __ldg(&xai_ntt[bara01 * N + i]);
                        }

#pragma unroll
                        for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                            uint32_t bk_offset =
                                (((P::targetP::k + 1) * digit_linear + out_k)
                                 << P::targetP::nbit) +
                                i;

                            // NTT(delta * h[digit]) is the same gadget scalar
                            // at every frequency, on the matching diagonal
                            // only.
                            NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> one_val =
                                out_k == j ? gadget : 0;
                            NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk0_val =
                                __ldg(&bk0_ntt[bk_offset]);
                            NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk1_val =
                                __ldg(&bk1_ntt[bk_offset]);
                            NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> bk2_val =
                                __ldg(&bk2_ntt[bk_offset]);

                            // combined = one + bk2*xai1 + bk1*xai0 + bk0*xai01
                            NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8> combined =
                                small_mod_madd3<N, sizeof(typename P::targetP::T) * 8>(bk2_val, xai1, bk1_val, xai0,
                                                   bk0_val, xai01, one_val);

                            // Accumulate: decomp_ntt * combined
                            if constexpr (REGISTER_ACCUM)
                                local_accum[out_k][e] = small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(
                                    ntt_val, combined, local_accum[out_k][e]);
                            else
                                sh_accum[out_k * N + i] = small_mod_madd<N, sizeof(typename P::targetP::T) * 8>(
                                    ntt_val, combined, sh_accum[out_k * N + i]);
                        }
                    }
                }
            }
        }
    }

    // Accumulation is coefficient-local. Synchronize once before the inverse
    // butterflies begin reading coefficients written by other threads.
    __syncthreads();

    // Step 4: Inverse NTT and REPLACE trlwe (not add)
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* sh_ntt_buf;
        if constexpr (REGISTER_ACCUM) {
            if (tid < NUM_THREADS) {
                sh_work[tid] = local_accum[k_idx][0];
                sh_work[tid + NUM_THREADS] = local_accum[k_idx][1];
            }
            __syncthreads();
            sh_ntt_buf = sh_work;
        }
        else {
            sh_ntt_buf = &sh_accum[k_idx * N];
        }
        if (tid < NUM_THREADS) {
            SmallInverseNTT<P::targetP::nbit, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf, ntt.inverse_root_,
                                              ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < SmallInverseNTTSyncCount<N>(); s++)
                __syncthreads();
        }

        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                if constexpr (std::numeric_limits<
                                  typename P::targetP::T>::digits == 64) {
                    trlwe[k_idx * N + i] = static_cast<typename P::targetP::T>(
                        ntt_mod_to_torus64<N, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf[i]));
                }
                else {
                    trlwe[k_idx * N + i] = static_cast<typename P::targetP::T>(
                        ntt_mod_to_torus32<N, sizeof(typename P::targetP::T) * 8>(sh_ntt_buf[i]));
                }
            }
        }
        if constexpr (REGISTER_ACCUM) __syncthreads();
    }
    __syncthreads();
}
#endif  // USE_FFT

template <class P>
__device__ inline void __BlindRotateKeyBundle__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in,
    const typename P::targetP::T mu, const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const one_trgsw_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_ntt,
    CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* const sh_acc_ntt =
        reinterpret_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>*>(dynamic_shared);

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(in[P::domainP::k * P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, mu);
    }

    // accumulate in pairs
    constexpr uint32_t n = P::domainP::k * P::domainP::n;
    constexpr uint32_t num_pairs = n / P::Addends;
#ifdef USE_FFT
    constexpr uint32_t trgsw_size = (P::targetP::k + 1) * (P::targetP::k + 1) *
                                    P::targetP::l * (P::targetP::n / 2);
#else
    constexpr uint32_t trgsw_size = (P::targetP::k + 1) * (P::targetP::k + 1) *
                                    P::targetP::l * P::targetP::n;
#endif

    for (uint32_t i = 0; i < num_pairs; i++) {
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bara0 = modSwitchFromTorus<P>(in[2 * i] + roundoffset);
        const uint32_t bara1 =
            modSwitchFromTorus<P>(in[2 * i + 1] + roundoffset);

        // Each key bundle group has 3 TRGSW elements: bk[0], bk[1], bk[2]
        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk0 = bk + i * 3 * trgsw_size;
        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk2 = bk + (i * 3 + 2) * trgsw_size;

        AccumulateKeyBundle<P>(out, sh_acc_ntt, bara0, bara1, bk0, bk1, bk2,
                               one_trgsw_ntt, xai_ntt, ntt);
    }
}

template <class P, int casign, int cbsign, auto offset>
__device__ inline void __BlindRotatePreAddKeyBundle__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in0,
    const typename P::domainP::T* const in1,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const one_trgsw_ntt,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const xai_ntt,
    CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt)
{
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* const sh_acc_ntt =
        reinterpret_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>*>(dynamic_shared);

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(offset +
                                  casign * in0[P::domainP::k * P::domainP::n] +
                                  cbsign * in1[P::domainP::k * P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, P::targetP::μ);
    }

    // accumulate in pairs
    constexpr uint32_t n = P::domainP::k * P::domainP::n;
    constexpr uint32_t num_pairs = n / P::Addends;
#ifdef USE_FFT
    constexpr uint32_t trgsw_size = (P::targetP::k + 1) * (P::targetP::k + 1) *
                                    P::targetP::l * (P::targetP::n / 2);
#else
    constexpr uint32_t trgsw_size = (P::targetP::k + 1) * (P::targetP::k + 1) *
                                    P::targetP::l * P::targetP::n;
#endif

    for (uint32_t i = 0; i < num_pairs; i++) {
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bara0 = modSwitchFromTorus<P>(
            casign * in0[2 * i] + cbsign * in1[2 * i] + roundoffset);
        const uint32_t bara1 = modSwitchFromTorus<P>(
            casign * in0[2 * i + 1] + cbsign * in1[2 * i + 1] + roundoffset);

        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk0 = bk + i * 3 * trgsw_size;
        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* bk2 = bk + (i * 3 + 2) * trgsw_size;

        AccumulateKeyBundle<P>(out, sh_acc_ntt, bara0, bara1, bk0, bk1, bk2,
                               one_trgsw_ntt, xai_ntt, ntt);
    }
}
#endif  // USE_KEY_BUNDLE

template <class P, int casign, int cbsign, auto offset,
          bool NativeForwardSubtract = false, bool RegisterMonomials31 = false,
          bool OutputExponentCache = false,
          bool AdjacentScalar64 = OutputExponentCache>
__device__ inline void __BlindRotatePreAdd__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in0,
    const typename P::domainP::T* const in1,
    const NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>* const bk, CuNTTHandler<P::targetP::n, sizeof(typename P::targetP::T) * 8> ntt,
    typename P::domainP::T* exponent_workspace = nullptr)
{
    extern __shared__ __align__(16) unsigned char dynamic_shared[];
    auto* const sh_acc_ntt =
        reinterpret_cast<NTTValueFor<P::targetP::n, sizeof(typename P::targetP::T) * 8>*>(dynamic_shared);

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP) && \
    defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    constexpr bool CacheOutputExponents = OutputExponentCache &&
        std::is_same_v<P, TFHEpp::lvl02param>;
#else
    constexpr bool CacheOutputExponents = false;
#endif
    uint32_t saved_exponents = 0;
    typename P::domainP::T saved_body = 0;
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
    if constexpr (CacheOutputExponents) {
        saved_exponents = CapturePublicOutputExponents<P, casign, cbsign,
            NUM_THREAD4HOMGATE<typename P::targetP>>(in0, in1);
        // A forward overlap can cover the original body, so capture it before
        // the correction barrier permits any output metadata stores.
        saved_body = offset + casign * in0[P::domainP::k * P::domainP::n] +
            cbsign * in1[P::domainP::k * P::domainP::n];
    }
#endif

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ mod_switch(in.b)
    {
        uint32_t bar;
#ifdef USE_BLOCK_BINARY
        const auto correction = BlockBinaryModSwitchCorrection<P, casign, cbsign>(in0, in1);
        constexpr typename P::domainP::T roundoffset =
            static_cast<typename P::domainP::T>(1)
            << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                P::targetP::nbit);
        const typename P::domainP::T body = CacheOutputExponents ? saved_body :
            offset + casign * in0[P::domainP::k * P::domainP::n] +
            cbsign * in1[P::domainP::k * P::domainP::n];
        bar = 2 * P::targetP::n -
              modSwitchFromTorus<P>(body - correction / 2 + roundoffset);
#else
        bar = 2 * P::targetP::n -
              modSwitchFromTorus<P>(
                  offset + casign * in0[P::domainP::k * P::domainP::n] +
                  cbsign * in1[P::domainP::k * P::domainP::n]);
#endif
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
        // The correction barrier has captured all original mask/body data;
        // the output workspace may overlap either original input.
        if constexpr (CacheOutputExponents)
            PublishPublicOutputExponents<P, NUM_THREAD4HOMGATE<typename P::targetP>>(
                exponent_workspace, saved_exponents);
#endif
        RotatedTestVector<typename P::targetP>(out, bar, P::targetP::μ);
    }

    // accumulate
#ifdef USE_BLOCK_BINARY
    constexpr uint32_t blocks = P::domainP::n / P::domainP::ell;
    for (uint32_t block = 0; block < blocks; block++) {
        uint32_t bara[P::domainP::ell];
#pragma unroll
        for (uint32_t block_offset = 0; block_offset < P::domainP::ell;
             block_offset++) {
            constexpr typename P::domainP::T roundoffset =
                static_cast<typename P::domainP::T>(1)
                << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                    P::targetP::nbit);
            const uint32_t index = block * P::domainP::ell + block_offset;
            if constexpr (CacheOutputExponents) {
                // Ordinary coherent global reads follow the publication barrier.
                bara[block_offset] = exponent_workspace[index];
            } else {
                const typename P::domainP::T phase =
                    casign * in0[index] + cbsign * in1[index];
                bara[block_offset] = modSwitchFromTorus<P>(phase + roundoffset);
            }
        }
#ifdef USE_FFT
        AccumulateBlockBinary<P>(out, sh_acc_ntt, block, bara, bk, ntt);
#else
        // Keep the multi-sample batch schedule unchanged. Scalar callers can
        // select the measured register/vector schedule independently of metadata.
        AccumulateBlockBinary<P, false, false, NativeForwardSubtract, 0,
                              RegisterMonomials31, true, true, AdjacentScalar64>(
            out, sh_acc_ntt, block, bara, bk, ntt);
#endif
    }
#else
    for (int i = 0; i < P::domainP::k * P::domainP::n; i++) {
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(
            0 + casign * in0[i] + cbsign * in1[i] + roundoffset);
#ifdef USE_FFT
        constexpr size_t trgsw_fft_size =
            BootstrappingTRGSWRows<typename P::targetP> * (P::targetP::k + 1) *
            (P::targetP::n / 2);
        Accumulate<P>(out, sh_acc_ntt, bar, bk + i * trgsw_fft_size, ntt);
#else
        Accumulate<P>(out, sh_acc_ntt, bar,
                      bk + (i << P::targetP::nbit) * (P::targetP::k + 1) *
                               (P::targetP::k + 1) * P::targetP::l,
                      ntt);
#endif
    }
#endif
}
}  // namespace cufhe
