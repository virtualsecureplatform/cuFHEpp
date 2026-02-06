#pragma once

#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/utils_gpu.cuh>
#include <include/ntt_small_modulus.cuh>

namespace cufhe{

template <class P>
__device__ inline uint32_t modSwitchFromTorus(const typename P::domainP::T phase)
{
    constexpr uint32_t Mbit = P::targetP::nbit + 1;
    static_assert(32 >= Mbit, "Undefined modSwitchFromTorus!");
    return (phase >> (std::numeric_limits<typename P::domainP::T>::digits - 1 - P::targetP::nbit));
}

template <class P>
__device__ constexpr typename P::T offsetgen()
{
    typename P::T offset = 0;
    for (int i = 1; i <= P::l; i++)
        offset +=
            P::Bg / 2 *
            (1ULL << (std::numeric_limits<typename P::T>::digits - i * P::Bgbit));
    return offset;
}

template <class P>
__device__ inline void RotatedTestVector(typename P::T* tlwe,
                                         const int32_t bar,
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
        for(int k = 0; k < P::k; k++) tlwe[i + k*P::n] = 0;  // part a
        if (bar == 2 * P::n)
            tlwe[i + P::k*P::n] = μ;
        else {
            tlwe[i + P::k*P::n] = ((i < (bar & (P::n - 1))) ^ (bar >> P::nbit))
                                 ? -μ
                                 : μ;  // part b
        }
    }
    __syncthreads();
}

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
template<class P>
__device__ inline void Accumulate(typename P::targetP::T* const trlwe, NTTValue* const sh_acc_ntt,
                                  const uint32_t a_bar,
                                  const NTTValue* const tgsw_ntt,
                                  const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512 for N=1024

    // Aliases for clarity
    NTTValue* const sh_work = &sh_acc_ntt[0];              // Working buffer for NTT
    NTTValue* const sh_accum = &sh_acc_ntt[N];             // Accumulated results (k+1 polynomials)

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

    // Process each TRLWE component (k+1 components) and each decomposition level (l levels)
    for (int j = 0; j <= P::targetP::k; j++) {
        for (int digit = 0; digit < P::targetP::l; digit++) {
            // Step 1: Compute decomposed polynomial for component j, digit
            // MulByXaiMinusOne and decomposition - each thread handles 2 elements
            if (tid < NUM_THREADS) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;

                    // PolynomialMulByXaiMinus
                    typename P::targetP::T temp =
                        trlwe[j * N + ((i - a_bar) & (N - 1))];
                    temp = ((i < (a_bar & (N - 1)) ^
                             (a_bar >> P::targetP::nbit)))
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
                    sh_work[i] = (digit_val < 0) ? (small_ntt::P + digit_val) : static_cast<uint32_t>(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                if constexpr (N == 1024) {
                    SmallForwardNTT32_1024(sh_work, ntt.forward_root_, tid);
                }
            } else {
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
                        NTTValue bk_val = __ldg(&tgsw_ntt[(((P::targetP::k + 1) * digit_linear + out_k) << P::targetP::nbit) + i]);
                        sh_accum[out_k * N + i] = small_mod_add(sh_accum[out_k * N + i], small_mod_mult(ntt_val, bk_val));
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
                SmallInverseNTT32_1024(sh_ntt_buf, ntt.inverse_root_, ntt.n_inverse_, tid);
            }
        } else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Convert with modulus switching and add to trlwe
        constexpr uint32_t half_mod = small_ntt::P / 2;
        if (tid < NUM_THREADS) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                uint32_t val = sh_ntt_buf[i];
                int32_t signed_val = (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P) : static_cast<int32_t>(val);
                trlwe[k_idx * N + i] += ntt_mod_to_torus32(signed_val);
            }
        }
        __syncthreads();
    }
}

template<class P>
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
            modSwitchFromTorus<P>(in[P::domainP::k*P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, mu);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k*P::domainP::n; i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
                1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                        P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(in[i]+roundoffset);
        Accumulate<P>(out, sh_acc_ntt, bar,
                   bk + (i << P::targetP::nbit) * (P::targetP::k+1) * (P::targetP::k+1) * P::targetP::l, ntt);
    }
}

template <class P, int casign, int cbsign, std::make_signed_t<typename P::domainP::T> offset>
__device__ inline void __BlindRotatePreAdd__(typename P::targetP::T* const out,
                                   const typename P::domainP::T* const in0,
                                   const typename P::domainP::T* const in1,
                                   const NTTValue* const bk,
                                   CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

    // test vector: acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    {
        const uint32_t bar =
            2 * P::targetP::n -
            modSwitchFromTorus<P>(offset + casign * in0[P::domainP::k*P::domainP::n] +
                                          cbsign * in1[P::domainP::k*P::domainP::n]);
        RotatedTestVector<typename P::targetP>(out, bar, P::targetP::μ);
    }

    // accumulate
    for (int i = 0; i < P::domainP::k*P::domainP::n; i++) {  // lvl0param::n iterations
        constexpr typename P::domainP::T roundoffset =
                1ULL << (std::numeric_limits<typename P::domainP::T>::digits - 2 -
                        P::targetP::nbit);
        const uint32_t bar = modSwitchFromTorus<P>(0 + casign * in0[i] +
                                                           cbsign * in1[i] + roundoffset);
        Accumulate<P>(out, sh_acc_ntt, bar,
                   bk + (i << P::targetP::nbit) * (P::targetP::k+1) * (P::targetP::k+1) * P::targetP::l, ntt);
    }
}
}