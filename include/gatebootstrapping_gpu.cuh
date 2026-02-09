#pragma once

#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <include/utils_gpu.cuh>

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
__device__ inline void Accumulate(typename P::targetP::T* const trlwe,
                                  NTTValue* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const NTTValue* const tgsw_fft,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;            // 512
    constexpr uint32_t NUM_THREADS = N >> 1;       // 512 threads total
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256 for GPU-FFT

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
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
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
                       (digit + 1) * P::targetP::Bgbit)) &
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
                       (digit + 1) * P::targetP::Bgbit)) &
                     decomp_mask) -
                    decomp_half);

                // Fold + twist
                double2 folded = {static_cast<double>(digit_re),
                                  static_cast<double>(digit_im)};
                double2 tw = __ldg(&ntt.twist_[tid]);
                sh_fft[tid] = folded * tw;
            }
            __syncthreads();

            // Step 2: Forward FFT (GPU-FFT: 256 threads, 5 syncs)
            if (tid < FFT_THREADS) {
                GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < 5; s++) __syncthreads();
            }

            // Step 3: Multiply-accumulate with BSK
            int digit_linear = j * P::targetP::l + digit;
            if (tid < HALF_N) {
                double2 fft_val = sh_fft[tid];
#pragma unroll
                for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                    double2 bk_val = __ldg(
                        &tgsw_fft[((P::targetP::k + 1) * digit_linear + out_k) *
                                      HALF_N +
                                  tid]);
                    sh_accum[out_k * HALF_N + tid] += fft_val * bk_val;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT + untwist + unfold + add to trlwe
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];

        if (tid < FFT_THREADS) {
            GPUFFTInverse512(sh_inv, ntt.inverse_root_, ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Untwist + unfold + denormalize
        constexpr double denorm = 4294967296.0;  // 2^32
        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            trlwe[k_idx * N + tid] += static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.x * denorm)));
            trlwe[k_idx * N + tid + HALF_N] += static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.y * denorm)));
        }
        __syncthreads();
    }
}
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
__device__ inline void Accumulate(typename P::targetP::T* const trlwe,
                                  NTTValue* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const NTTValue* const tgsw_fft,
                                  const CuNTTHandler<> ntt)
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
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
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
__device__ inline void Accumulate(typename P::targetP::T* const trlwe,
                                  NTTValue* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const NTTValue* const tgsw_ntt,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512 for N=1024

    // Aliases for clarity
    NTTValue* const sh_work = &sh_acc_ntt[0];  // Working buffer for NTT
    NTTValue* const sh_accum =
        &sh_acc_ntt[N];  // Accumulated results (k+1 polynomials)

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            sh_accum[k_idx * N + tid] = 0;
            sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
        }
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
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
                    sh_work[i] = (digit_val < 0)
                                     ? (small_ntt::P + digit_val)
                                     : static_cast<uint32_t>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                if constexpr (N == 1024) {
                    SmallForwardNTT32_1024(sh_work, ntt.forward_root_, tid);
                }
            }
            else {
                for (int s = 0; s < 5; s++) __syncthreads();
            }

            // Step 3: Multiply with BK and accumulate into sh_accum
            int digit_linear = j * P::targetP::l + digit;
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    NTTValue ntt_val = sh_work[i];

// Accumulate into each output component
#pragma unroll
                    for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                        NTTValue bk_val = __ldg(
                            &tgsw_ntt[(((P::targetP::k + 1) * digit_linear +
                                        out_k)
                                       << P::targetP::nbit) +
                                      i]);
                        sh_accum[out_k * N + i] =
                            small_mod_add(sh_accum[out_k * N + i],
                                          small_mod_mult(ntt_val, bk_val));
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT on accumulated results and add to trlwe
    // Operate directly on sh_accum to avoid copying to sh_work
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        // Inverse NTT directly on the accumulator buffer
        NTTValue* const sh_ntt_buf = &sh_accum[k_idx * N];
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallInverseNTT32_1024(sh_ntt_buf, ntt.inverse_root_,
                                       ntt.n_inverse_, tid);
            }
        }
        else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Convert with modulus switching and add to trlwe
        constexpr uint32_t half_mod = small_ntt::P / 2;
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                uint32_t val = sh_ntt_buf[i];
                int32_t signed_val =
                    (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P)
                                     : static_cast<int32_t>(val);
                trlwe[k_idx * N + i] += ntt_mod_to_torus32(signed_val);
            }
        }
        __syncthreads();
    }
}
#endif  // USE_FFT

template <class P>
__device__ inline void __BlindRotate__(typename P::targetP::T* const out,
                                       const typename P::domainP::T* const in,
                                       const typename P::targetP::T mu,
                                       const NTTValue* const bk,
                                       CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(in[P::domainP::k * P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, mu);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k * P::domainP::n;
         i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(in[i] + roundoffset);
#ifdef USE_FFT
        // FFT BSK stride: each TRGSW has (k+1)*l * (k+1) * (N/2) complex
        // elements
        constexpr size_t trgsw_fft_size = (P::targetP::k + 1) * P::targetP::l *
                                          (P::targetP::k + 1) *
                                          (P::targetP::n / 2);
        Accumulate<P>(out, sh_acc_ntt, bar, bk + i * trgsw_fft_size, ntt);
#else
        Accumulate<P>(out, sh_acc_ntt, bar,
                      bk + (i << P::targetP::nbit) * (P::targetP::k + 1) *
                               (P::targetP::k + 1) * P::targetP::l,
                      ntt);
#endif
    }
}

#ifdef USE_KEY_BUNDLE
#ifdef USE_FFT
#ifdef USE_GPU_FFT
/**
 * GPU-FFT Key-bundle ExternalProduct accumulator
 *
 * Same as tfhe-rs version but uses fold+twist, GPU-FFT forward/inverse,
 * untwist+unfold
 */
template <class P>
__device__ inline void AccumulateKeyBundle(
    typename P::targetP::T* const trlwe, NTTValue* const sh_acc_ntt,
    const uint32_t bara0, const uint32_t bara1, const NTTValue* const bk0_fft,
    const NTTValue* const bk1_fft, const NTTValue* const bk2_fft,
    const NTTValue* const one_trgsw_fft, const NTTValue* const xai_fft,
    const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256

    double2* const sh_fft = &sh_acc_ntt[0];
    double2* const sh_accum = &sh_acc_ntt[HALF_N];

    for (int i = tid; i < (P::targetP::k + 1) * HALF_N; i += NUM_THREADS) {
        sh_accum[i] = {0.0, 0.0};
    }
    __syncthreads();

    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
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
                GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
            }
            else {
                for (int s = 0; s < 5; s++) __syncthreads();
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

                    sh_accum[out_k * HALF_N + tid] += fft_val * combined;
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse FFT + untwist + unfold, REPLACE trlwe
    constexpr double denorm = 4294967296.0;  // 2^32
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        double2* const sh_inv = &sh_accum[k_idx * HALF_N];
        if (tid < FFT_THREADS) {
            GPUFFTInverse512(sh_inv, ntt.inverse_root_, ntt.n_inverse_, tid);
        }
        else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        if (tid < HALF_N) {
            double2 val = sh_inv[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            trlwe[k_idx * N + tid] = static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.x * denorm)));
            trlwe[k_idx * N + tid + HALF_N] = static_cast<uint32_t>(
                static_cast<int64_t>(llrint(val.y * denorm)));
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
    typename P::targetP::T* const trlwe, NTTValue* const sh_acc_ntt,
    const uint32_t bara0, const uint32_t bara1,
    const NTTValue* const bk0_fft,  // Enc(s0*s1)
    const NTTValue* const bk1_fft,  // Enc(s0*(1-s1))
    const NTTValue* const bk2_fft,  // Enc((1-s0)*s1)
    const NTTValue* const one_trgsw_fft, const NTTValue* const xai_fft,
    const CuNTTHandler<> ntt)
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
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
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
    typename P::targetP::T* const trlwe, NTTValue* const sh_acc_ntt,
    const uint32_t bara0, const uint32_t bara1,
    const NTTValue* const bk0_ntt,  // Enc(s0*s1)
    const NTTValue* const bk1_ntt,  // Enc(s0*(1-s1))
    const NTTValue* const bk2_ntt,  // Enc((1-s0)*s1)
    const NTTValue* const one_trgsw_ntt, const NTTValue* const xai_ntt,
    const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;

    NTTValue* const sh_work = &sh_acc_ntt[0];
    NTTValue* const sh_accum = &sh_acc_ntt[N];

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
        for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
            sh_accum[k_idx * N + tid] = 0;
            sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
        }
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << P::targetP::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (P::targetP::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<typename P::targetP>();
    constexpr typename P::targetP::T roundoffset =
        1ULL << (std::numeric_limits<typename P::targetP::T>::digits -
                 P::targetP::l * P::targetP::Bgbit - 1);

    // Precompute combined bara for bk0 (a0+a1 mod 2N)
    const uint32_t bara01 = (bara0 + bara1) & (2 * N - 1);

    // Process each TRLWE component and decomposition level
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Decompose trlwe[j] directly (no MulByXaiMinus)
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    typename P::targetP::T temp = trlwe[j * N + i];
                    temp += decomp_offset + roundoffset;
                    int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename P::targetP::T>::digits -
                           (digit + 1) * P::targetP::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = (digit_val < 0)
                                     ? (small_ntt::P + digit_val)
                                     : static_cast<uint32_t>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                if constexpr (N == 1024) {
                    SmallForwardNTT32_1024(sh_work, ntt.forward_root_, tid);
                }
            }
            else {
                for (int s = 0; s < 5; s++) __syncthreads();
            }

            // Step 3: Multiply with on-the-fly keybundle and accumulate
            int digit_linear = j * P::targetP::l + digit;
            if (tid < NUM_THREADS) {
#pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    NTTValue ntt_val = sh_work[i];

                    // Load xai values for this NTT coefficient
                    NTTValue xai0 = __ldg(&xai_ntt[bara0 * N + i]);
                    NTTValue xai1 = __ldg(&xai_ntt[bara1 * N + i]);
                    NTTValue xai01 = __ldg(&xai_ntt[bara01 * N + i]);

#pragma unroll
                    for (int out_k = 0; out_k <= P::targetP::k; out_k++) {
                        uint32_t bk_offset =
                            (((P::targetP::k + 1) * digit_linear + out_k)
                             << P::targetP::nbit) +
                            i;

                        // Load BK values and one_ntt value
                        NTTValue one_val = __ldg(&one_trgsw_ntt[bk_offset]);
                        NTTValue bk0_val = __ldg(&bk0_ntt[bk_offset]);
                        NTTValue bk1_val = __ldg(&bk1_ntt[bk_offset]);
                        NTTValue bk2_val = __ldg(&bk2_ntt[bk_offset]);

                        // combined = one + bk2*xai1 + bk1*xai0 + bk0*xai01
                        NTTValue combined = one_val;
                        combined = small_mod_add(combined,
                                                 small_mod_mult(bk2_val, xai1));
                        combined = small_mod_add(combined,
                                                 small_mod_mult(bk1_val, xai0));
                        combined = small_mod_add(
                            combined, small_mod_mult(bk0_val, xai01));

                        // Accumulate: decomp_ntt * combined
                        sh_accum[out_k * N + i] =
                            small_mod_add(sh_accum[out_k * N + i],
                                          small_mod_mult(ntt_val, combined));
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT and REPLACE trlwe (not add)
    for (int k_idx = 0; k_idx <= P::targetP::k; k_idx++) {
        NTTValue* const sh_ntt_buf = &sh_accum[k_idx * N];
        if (tid < NUM_THREADS) {
            if constexpr (N == 1024) {
                SmallInverseNTT32_1024(sh_ntt_buf, ntt.inverse_root_,
                                       ntt.n_inverse_, tid);
            }
        }
        else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        constexpr uint32_t half_mod = small_ntt::P / 2;
        if (tid < NUM_THREADS) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                uint32_t val = sh_ntt_buf[i];
                int32_t signed_val =
                    (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P)
                                     : static_cast<int32_t>(val);
                trlwe[k_idx * N + i] =
                    ntt_mod_to_torus32(signed_val);  // REPLACE, not add
            }
        }
        __syncthreads();
    }
}
#endif  // USE_FFT

template <class P>
__device__ inline void __BlindRotateKeyBundle__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in,
    const typename P::targetP::T mu, const NTTValue* const bk,
    const NTTValue* const one_trgsw_ntt, const NTTValue* const xai_ntt,
    CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

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
        const NTTValue* bk0 = bk + i * 3 * trgsw_size;
        const NTTValue* bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValue* bk2 = bk + (i * 3 + 2) * trgsw_size;

        AccumulateKeyBundle<P>(out, sh_acc_ntt, bara0, bara1, bk0, bk1, bk2,
                               one_trgsw_ntt, xai_ntt, ntt);
    }
}

template <class P, int casign, int cbsign,
          std::make_signed_t<typename P::domainP::T> offset>
__device__ inline void __BlindRotatePreAddKeyBundle__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in0,
    const typename P::domainP::T* const in1, const NTTValue* const bk,
    const NTTValue* const one_trgsw_ntt, const NTTValue* const xai_ntt,
    CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

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

        const NTTValue* bk0 = bk + i * 3 * trgsw_size;
        const NTTValue* bk1 = bk + (i * 3 + 1) * trgsw_size;
        const NTTValue* bk2 = bk + (i * 3 + 2) * trgsw_size;

        AccumulateKeyBundle<P>(out, sh_acc_ntt, bara0, bara1, bk0, bk1, bk2,
                               one_trgsw_ntt, xai_ntt, ntt);
    }
}
#endif  // USE_KEY_BUNDLE

template <class P, int casign, int cbsign,
          std::make_signed_t<typename P::domainP::T> offset>
__device__ inline void __BlindRotatePreAdd__(
    typename P::targetP::T* const out, const typename P::domainP::T* const in0,
    const typename P::domainP::T* const in1, const NTTValue* const bk,
    CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(offset +
                                  casign * in0[P::domainP::k * P::domainP::n] +
                                  cbsign * in1[P::domainP::k * P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, P::targetP::μ);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k * P::domainP::n;
         i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
            1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                     P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(
            0 + casign * in0[i] + cbsign * in1[i] + roundoffset);
#ifdef USE_FFT
        constexpr size_t trgsw_fft_size = (P::targetP::k + 1) * P::targetP::l *
                                          (P::targetP::k + 1) *
                                          (P::targetP::n / 2);
        Accumulate<P>(out, sh_acc_ntt, bar, bk + i * trgsw_fft_size, ntt);
#else
        Accumulate<P>(out, sh_acc_ntt, bar,
                      bk + (i << P::targetP::nbit) * (P::targetP::k + 1) *
                               (P::targetP::k + 1) * P::targetP::l,
                      ntt);
#endif
    }
}
}  // namespace cufhe