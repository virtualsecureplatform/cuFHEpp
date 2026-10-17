#pragma once

#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/utils_gpu.cuh>
#include <type_traits>

namespace cufhe {

extern std::vector<TFHEpp::lvl0param::T*> ksk_devs;
extern std::vector<TFHEpp::lvl0param::T*> ksk_devs_lvl20;

template <class P>
__device__ constexpr typename P::domainP::T iksoffsetgen()
{
    typename P::domainP::T offset = 0;
    for (int i = 1; i <= P::t; i++)
        offset +=
            (1ULL << P::basebit) / 2 *
            (1ULL << (std::numeric_limits<typename P::domainP::T>::digits -
                      i * P::basebit));
    return offset;
}

template <class P>
inline constexpr bool use_subset_key_switch =
#ifdef USE_SUBSET_KEY
    std::is_same_v<P, TFHEpp::lvl10param>;
#else
    false;
#endif

template <class P>
__device__ inline typename P::targetP::T KeySwitchTorusConvert(
    const typename P::domainP::T value)
{
    constexpr uint32_t domain_digit =
        std::numeric_limits<typename P::domainP::T>::digits;
    constexpr uint32_t target_digit =
        std::numeric_limits<typename P::targetP::T>::digits;
    if constexpr (domain_digit == target_digit)
        return value;
    else if constexpr (domain_digit > target_digit)
        return (value + (typename P::domainP::T{1}
                         << (domain_digit - target_digit - 1))) >>
               (domain_digit - target_digit);
    else
        return static_cast<typename P::targetP::T>(value)
               << (target_digit - domain_digit);
}

template <class P>
__device__ inline typename P::domainP::T SampleExtractCoefficient(
    const typename P::domainP::T* const trlwe, const int index)
{
    constexpr int polynomial_size = P::domainP::n;
    const int component = index / polynomial_size;
    const int coefficient = index % polynomial_size;
    return coefficient == 0
               ? trlwe[component * polynomial_size]
               : -trlwe[(component + 1) * polynomial_size - coefficient];
}

template <class P>
__device__ inline void KeySwitch(typename P::targetP::T* const lwe,
                                 const typename P::domainP::T* const tlwe,
                                 const typename P::targetP::T* const ksk)
{
    constexpr uint domain_digit =
        std::numeric_limits<typename P::domainP::T>::digits;
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < domain_digit
            ? 1ULL << (domain_digit - (1 + P::basebit * P::t))
            : 0;
    constexpr typename P::domainP::T decompoffset = iksoffsetgen<P>();
    constexpr typename P::domainP::T mask = (1ULL << P::basebit) - 1;
    constexpr typename P::domainP::T halfbase = 1ULL << (P::basebit - 1);
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (int i = tid; i <= P::targetP::k * P::targetP::n; i += bdim) {
        typename P::targetP::T res = 0;
        if (i == P::targetP::k * P::targetP::n) {
            res = KeySwitchTorusConvert<P>(tlwe[P::domainP::k * P::domainP::n]);
        }
        else if constexpr (use_subset_key_switch<P>) {
            res =
                KeySwitchTorusConvert<P>(SampleExtractCoefficient<P>(tlwe, i));
        }
        constexpr int first_switched =
            use_subset_key_switch<P> ? P::targetP::k * P::targetP::n : 0;
        constexpr int switched_count =
            P::domainP::k * P::domainP::n - first_switched;
        for (int key_index = 0; key_index < switched_count; key_index++) {
            const int j = first_switched + key_index;
            typename P::domainP::T tmp = SampleExtractCoefficient<P>(tlwe, j);
            tmp += decompoffset + roundoffset;
            for (int k = 0; k < P::t; k++) {
                const int32_t val =
                    ((tmp >>
                      (std::numeric_limits<typename P::domainP::T>::digits -
                       (k + 1) * P::basebit)) &
                     mask) -
                    halfbase;
                constexpr int numbase = 1 << (P::basebit - 1);
                if (val != 0) {
                    const typename P::targetP::T kskelem = __ldg(
                        &ksk[key_index * (P::t * numbase *
                                          (P::targetP::k * P::targetP::n + 1)) +
                             k * (numbase *
                                  (P::targetP::k * P::targetP::n + 1)) +
                             (abs(val) - 1) *
                                 (P::targetP::k * P::targetP::n + 1) +
                             i]);
                    if (val > 0)
                        res -= kskelem;
                    else if (val < 0)
                        res += kskelem;
                }
            }
        }
        lwe[i] = res;
    }
}

// KeySwitch variant that takes an already-extracted TLWE (not TRLWE)
// Use this when sample extraction has been done separately via
// __SampleExtractIndex__
template <class P>
__device__ inline void KeySwitchFromTLWE(
    typename P::targetP::T* const lwe, const typename P::domainP::T* const tlwe,
    const typename P::targetP::T* const ksk)
{
    constexpr uint domain_digit =
        std::numeric_limits<typename P::domainP::T>::digits;
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < domain_digit
            ? 1ULL << (domain_digit - (1 + P::basebit * P::t))
            : 0;
    constexpr typename P::domainP::T decompoffset = iksoffsetgen<P>();
    constexpr typename P::domainP::T mask = (1ULL << P::basebit) - 1;
    constexpr typename P::domainP::T halfbase = 1ULL << (P::basebit - 1);
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (int i = tid; i <= P::targetP::k * P::targetP::n; i += bdim) {
        typename P::targetP::T res = 0;
        if (i == P::targetP::k * P::targetP::n) {
            res = KeySwitchTorusConvert<P>(tlwe[P::domainP::k * P::domainP::n]);
        }
        else if constexpr (use_subset_key_switch<P>)
            res = KeySwitchTorusConvert<P>(tlwe[i]);
        // Direct access to already-extracted TLWE elements (no sample
        // extraction needed)
        constexpr int first_switched =
            use_subset_key_switch<P> ? P::targetP::k * P::targetP::n : 0;
        constexpr int switched_count =
            P::domainP::k * P::domainP::n - first_switched;
        for (int key_index = 0; key_index < switched_count; key_index++) {
            const int j = first_switched + key_index;
            typename P::domainP::T tmp = tlwe[j];
            tmp += decompoffset + roundoffset;
            for (int k = 0; k < P::t; k++) {
                const int32_t val =
                    ((tmp >>
                      (std::numeric_limits<typename P::domainP::T>::digits -
                       (k + 1) * P::basebit)) &
                     mask) -
                    halfbase;
                constexpr int numbase = 1 << (P::basebit - 1);
                if (val != 0) {
                    const typename P::targetP::T kskelem = __ldg(
                        &ksk[key_index * (P::t * numbase *
                                          (P::targetP::k * P::targetP::n + 1)) +
                             k * (numbase *
                                  (P::targetP::k * P::targetP::n + 1)) +
                             (abs(val) - 1) *
                                 (P::targetP::k * P::targetP::n + 1) +
                             i]);
                    if (val > 0)
                        res -= kskelem;
                    else if (val < 0)
                        res += kskelem;
                }
            }
        }
        lwe[i] = res;
    }
}

template <class P, int casign, int cbsign, auto offset>
__device__ inline void IdentityKeySwitchPreAdd(
    typename P::targetP::T* const lwe, const typename P::domainP::T* const ina,
    const typename P::domainP::T* const inb,
    const typename P::targetP::T* const ksk)
{
    constexpr uint domain_digit =
        std::numeric_limits<typename P::domainP::T>::digits;
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < domain_digit
            ? 1ULL << (domain_digit - (1 + P::basebit * P::t))
            : 0;
    constexpr typename P::domainP::T decompoffset = iksoffsetgen<P>();
    constexpr typename P::domainP::T mask = (1ULL << P::basebit) - 1;
    constexpr typename P::domainP::T halfbase = 1ULL << (P::basebit - 1);
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    for (int i = tid; i <= P::targetP::k * P::targetP::n; i += bdim) {
        typename P::targetP::T res = 0;
        if (i == P::targetP::k * P::targetP::n) {
            const typename P::domainP::T added =
                casign * ina[P::domainP::k * P::domainP::n] +
                cbsign * inb[P::domainP::k * P::domainP::n] + offset;
            res = KeySwitchTorusConvert<P>(added);
        }
        else if constexpr (use_subset_key_switch<P>) {
            const typename P::domainP::T added =
                casign * ina[i] + cbsign * inb[i];
            res = KeySwitchTorusConvert<P>(added);
        }
        constexpr int first_switched =
            use_subset_key_switch<P> ? P::targetP::k * P::targetP::n : 0;
        constexpr int switched_count =
            P::domainP::k * P::domainP::n - first_switched;
        for (int key_index = 0; key_index < switched_count; key_index++) {
            const int j = first_switched + key_index;
            typename P::domainP::T tmp;
            tmp = casign * ina[j] + cbsign * inb[j] + 0 + decompoffset +
                  roundoffset;
            for (int k = 0; k < P::t; k++) {
                const int32_t val =
                    ((tmp >>
                      (std::numeric_limits<typename P::domainP::T>::digits -
                       (k + 1) * P::basebit)) &
                     mask) -
                    halfbase;
                constexpr int numbase = 1 << (P::basebit - 1);
                if (val > 0)
                    res -= __ldg(
                        &ksk[key_index * (P::t * numbase *
                                          (P::targetP::k * P::targetP::n + 1)) +
                             k * (numbase *
                                  (P::targetP::k * P::targetP::n + 1)) +
                             (val - 1) * (P::targetP::k * P::targetP::n + 1) +
                             i]);
                else if (val < 0)
                    res += __ldg(
                        &ksk[key_index * (P::t * numbase *
                                          (P::targetP::k * P::targetP::n + 1)) +
                             k * (numbase *
                                  (P::targetP::k * P::targetP::n + 1)) +
                             (-val - 1) * (P::targetP::k * P::targetP::n + 1) +
                             i]);
            }
        }
        lwe[i] = res;
    }
}

// Prepare the existing signed-digit table addresses once per ciphertext.
// All CTA threads must participate. rows needs count * P::t uint32_t words
// disjoint from inputs, output and key; a caller barrier is required before
// reusing it. The arithmetic, rounding and key representation are unchanged.
template <class P, int casign, int cbsign, auto offset>
__device__ __forceinline__ void IdentityKeySwitchPreAddPrepared(
    typename P::targetP::T* lwe, const typename P::domainP::T* ina,
    const typename P::domainP::T* inb, const typename P::targetP::T* ksk,
    uint32_t* rows)
{
    constexpr uint32_t digits = std::numeric_limits<typename P::domainP::T>::digits;
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < digits ? 1ULL << (digits - (1 + P::basebit * P::t)) : 0;
    constexpr auto decompoffset = iksoffsetgen<P>();
    constexpr uint32_t mask = (1U << P::basebit) - 1;
    constexpr uint32_t halfbase = 1U << (P::basebit - 1);
    constexpr int output_size = P::targetP::k * P::targetP::n + 1;
    constexpr int first = use_subset_key_switch<P> ? output_size - 1 : 0;
    constexpr int count = P::domainP::k * P::domainP::n - first;
    static_assert(uint64_t(count) * P::t * halfbase * output_size < (1ULL << 31));
    const uint32_t tid = threadIdx.x;
    for (int d = tid; d < count * P::t; d += blockDim.x) {
        const int j = d / P::t + first, k = d % P::t;
        const typename P::domainP::T tmp = casign * ina[j] + cbsign * inb[j] + decompoffset + roundoffset;
        const int val = int((tmp >> (digits - (k + 1) * P::basebit)) & mask) - int(halfbase);
        rows[d] = val == 0 ? UINT32_MAX :
            uint32_t((d * halfbase + abs(val) - 1) * output_size) |
            (val < 0 ? 0x80000000U : 0U);
    }
    __syncthreads();
    for (int i = tid; i < output_size; i += blockDim.x) {
        typename P::targetP::T res = 0;
        if (i == output_size - 1) {
            const typename P::domainP::T added = casign * ina[P::domainP::k * P::domainP::n] +
                cbsign * inb[P::domainP::k * P::domainP::n] + offset;
            res = KeySwitchTorusConvert<P>(added);
        } else if constexpr (use_subset_key_switch<P>) {
            const typename P::domainP::T added = casign * ina[i] + cbsign * inb[i];
            res = KeySwitchTorusConvert<P>(added);
        }
        for (int key_index = 0; key_index < count; ++key_index) {
#pragma unroll
            for (int k = 0; k < P::t; ++k) {
                const uint32_t row = rows[key_index * P::t + k];
                if (row != UINT32_MAX) {
                    const auto value = __ldg(ksk + (row & 0x7fffffffU) + i);
                    res = row & 0x80000000U ? res + value : res - value;
                }
            }
        }
        lwe[i] = res;
    }
}

// Compact signed row numbers for the existing lvl20 key layout.
// All CTA threads must participate. rows needs count * P::t uint16_t words
// disjoint from inputs, output and key; a caller barrier is required before
// reusing it. The arithmetic, rounding and key representation are unchanged.
template <class P, int casign, int cbsign, auto offset>
__device__ __forceinline__ void IdentityKeySwitchPreAddPrepared16(
    typename P::targetP::T* lwe, const typename P::domainP::T* ina,
    const typename P::domainP::T* inb, const typename P::targetP::T* ksk,
    uint16_t* rows)
{
    constexpr uint32_t digits = std::numeric_limits<typename P::domainP::T>::digits;
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < digits ? 1ULL << (digits - (1 + P::basebit * P::t)) : 0;
    constexpr auto decompoffset = iksoffsetgen<P>();
    constexpr uint32_t mask = (1U << P::basebit) - 1;
    constexpr uint32_t halfbase = 1U << (P::basebit - 1);
    constexpr int output_size = P::targetP::k * P::targetP::n + 1;
    constexpr int first = use_subset_key_switch<P> ? output_size - 1 : 0;
    constexpr int count = P::domainP::k * P::domainP::n - first;
    static_assert(uint64_t(count) * P::t * halfbase < (1ULL << 15));
    const uint32_t tid = threadIdx.x;
    for (int d = tid; d < count * P::t; d += blockDim.x) {
        const int j = d / P::t + first, k = d % P::t;
        const typename P::domainP::T tmp = casign * ina[j] + cbsign * inb[j] + decompoffset + roundoffset;
        const int val = int((tmp >> (digits - (k + 1) * P::basebit)) & mask) - int(halfbase);
        rows[d] = val == 0 ? UINT16_MAX :
            uint16_t(d * halfbase + abs(val) - 1) |
            (val < 0 ? 0x8000U : 0U);
    }
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(CUFHE_USE_HIP)
    if constexpr (std::is_same_v<P, TFHEpp::lvl20param> &&
                  !use_subset_key_switch<P> && output_size > 512 && output_size <= 1024) {
        if (blockDim.x == 512) {
            // Match the after-BR schedule: handle the short second output
            // group alongside the first, with independent native-word sums.
            const int other = tid + 512 < output_size ? tid + 512 : tid;
            uint32_t sum0 = 0, sum1 = 0;
            if (other == output_size - 1) {
                const typename P::domainP::T added =
                    casign * ina[P::domainP::k * P::domainP::n] +
                    cbsign * inb[P::domainP::k * P::domainP::n] + offset;
                sum1 = KeySwitchTorusConvert<P>(added);
            }
            static_assert(count * P::t % 32 == 0);
            static_assert(uint64_t(count) * P::t * halfbase * output_size * sizeof(*ksk) < (1ULL << 31));
            for (int d = 0; d < count * P::t; d += 32) {
                // Each lane prepares one offset, then broadcasts it in digit order.
                const uint32_t packed = rows[d + (tid & 31)];
                const uint32_t lane_row = packed == UINT16_MAX ? UINT32_MAX :
                    ((packed & 0x7fffU) * output_size * sizeof(*ksk)) |
                    ((packed & 0x8000U) << 16);
#pragma unroll
                for (int step = 0; step < 32; ++step) {
                    const uint32_t row = __shfl_sync(0xffffffff, lane_row, step);
                    const uintptr_t address = reinterpret_cast<uintptr_t>(ksk) +
                        uintptr_t(row & 0x7fffffffU);
                    uint32_t value0, value1;
                    asm("{ .reg .pred valid, second;\n\t"
                        "setp.ne.u32 valid, %4, 0xffffffff;\n\t"
                        "setp.ne.u32 second, %5, 0;\n\t"
                        "and.pred second, second, valid;\n\t"
                        "mov.u32 %0, 0;\n\t"
                        "mov.u32 %1, 0;\n\t"
                        "@valid ld.global.nc.u16 %0, [%2];\n\t"
                        "@second ld.global.nc.u16 %1, [%3];\n\t}"
                        : "=&r"(value0), "=&r"(value1)
                        : "l"(address + tid * sizeof(*ksk)),
                          "l"(address + other * sizeof(*ksk)), "r"(row),
                          "r"(uint32_t(tid + 512 < output_size)));
                    sum0 = row & 0x80000000U ? sum0 + value0 : sum0 - value0;
                    sum1 = row & 0x80000000U ? sum1 + value1 : sum1 - value1;
                }
            }
            // Final 16-bit stores preserve all original modular additions.
            lwe[tid] = sum0;
            if (tid + 512 < output_size) lwe[tid + 512] = sum1;
            return;
        }
    }
#endif
    for (int i = tid; i < output_size; i += blockDim.x) {
        typename P::targetP::T res = 0;
        if (i == output_size - 1) {
            const typename P::domainP::T added = casign * ina[P::domainP::k * P::domainP::n] +
                cbsign * inb[P::domainP::k * P::domainP::n] + offset;
            res = KeySwitchTorusConvert<P>(added);
        } else if constexpr (use_subset_key_switch<P>) {
            const typename P::domainP::T added = casign * ina[i] + cbsign * inb[i];
            res = KeySwitchTorusConvert<P>(added);
        }
        for (int key_index = 0; key_index < count; ++key_index) {
#pragma unroll
            for (int k = 0; k < P::t; ++k) {
                const uint32_t row = rows[key_index * P::t + k];
                if (row != UINT16_MAX) {
                    const auto value = __ldg(ksk + (row & 0x7fffU) * output_size + i);
                    res = row & 0x8000U ? res + value : res - value;
                }
            }
        }
        lwe[i] = res;
    }
}


// Exact lvl10 key switching with prepared addresses and optional index-zero
// extraction. Scratch may overlap the consumed input: every mask coefficient
// and the body are captured before the first barrier permits descriptor writes.
// rows needs (switched_count * P::t + 1) uint32_t words; output/key are disjoint.
template <class P, int Threads, bool ExtractFromTRLWE>
__device__ __forceinline__ void KeySwitchPrepared32(
    typename P::targetP::T* lwe, const typename P::domainP::T* input,
    const typename P::targetP::T* ksk, uint32_t* rows)
{
    static_assert(std::is_same_v<P, TFHEpp::lvl10param>);
    constexpr int domain_size = P::domainP::k * P::domainP::n;
    constexpr int output_size = P::targetP::k * P::targetP::n + 1;
    constexpr int first = use_subset_key_switch<P> ? output_size - 1 : 0;
    constexpr int count = domain_size - first;
    constexpr int digits = std::numeric_limits<typename P::domainP::T>::digits;
    constexpr uint32_t halfbase = 1U << (P::basebit - 1);
    constexpr uint32_t mask = (1U << P::basebit) - 1;
    constexpr auto decompoffset = iksoffsetgen<P>();
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < digits ? 1ULL << (digits - 1 - P::basebit * P::t) : 0;
    static_assert(domain_size % Threads == 0);
    static_assert(uint64_t(count) * P::t * halfbase * output_size < (1ULL << 31));
    const int tid = threadIdx.x;
    typename P::domainP::T saved[domain_size / Threads];
#pragma unroll
    for (int slot = 0; slot < domain_size / Threads; ++slot) {
        const int j = tid + slot * Threads;
        if constexpr (ExtractFromTRLWE) saved[slot] = SampleExtractCoefficient<P>(input, j);
        else saved[slot] = input[j];
    }
    const auto body = tid == 0 ? KeySwitchTorusConvert<P>(input[domain_size]) : 0;
    __syncthreads();
#pragma unroll
    for (int slot = 0; slot < domain_size / Threads; ++slot) {
        const int j = tid + slot * Threads;
        if (j >= first) {
            const typename P::domainP::T tmp = saved[slot] + decompoffset + roundoffset;
#pragma unroll
            for (int k = 0; k < P::t; ++k) {
                const int d = (j - first) * P::t + k;
                const int val = int((tmp >> (digits - (k + 1) * P::basebit)) & mask) - int(halfbase);
                rows[d] = val == 0 ? UINT32_MAX :
                    uint32_t((d * halfbase + abs(val) - 1) * output_size) |
                    (val < 0 ? 0x80000000U : 0U);
            }
        }
    }
    if (tid == 0) rows[count * P::t] = body;
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(CUFHE_USE_HIP)
    if constexpr (!ExtractFromTRLWE && Threads == 512 &&
                  output_size > Threads && output_size <= 2 * Threads) {
        static_assert(std::is_same_v<typename P::targetP::T, uint16_t>);
        const int other = tid + Threads;
        uint32_t sum0 = 0, sum1 = 0;
        if constexpr (use_subset_key_switch<P>) sum0 = KeySwitchTorusConvert<P>(saved[0]);
        if (other < output_size) {
            if (other == output_size - 1) sum1 = rows[count * P::t];
            else if constexpr (use_subset_key_switch<P>) sum1 = KeySwitchTorusConvert<P>(saved[1]);
        }
        // Traverse descriptors once for both outputs. Native-word sums retain
        // the exact original uint16 modular result when stored.
        for (int key_index = 0; key_index < count; ++key_index) {
#pragma unroll
            for (int k = 0; k < P::t; ++k) {
                const uint32_t row = rows[key_index * P::t + k];
                if (row != UINT32_MAX) {
                    const auto* key_row = ksk + (row & 0x7fffffffU);
                    const auto value0 = __ldg(key_row + tid);
                    sum0 = row & 0x80000000U ? sum0 + value0 : sum0 - value0;
                    if (other < output_size) {
                        const auto value1 = __ldg(key_row + other);
                        sum1 = row & 0x80000000U ? sum1 + value1 : sum1 - value1;
                    }
                }
            }
        }
        lwe[tid] = sum0;
        if (other < output_size) lwe[other] = sum1;
        return;
    }
#endif
#pragma unroll
    for (int slot = 0; slot < (output_size + Threads - 1) / Threads; ++slot) {
        const int i = tid + slot * Threads;
        if (i < output_size) {
            typename P::targetP::T res = 0;
            if (i == output_size - 1) res = rows[count * P::t];
            else if constexpr (use_subset_key_switch<P>) res = KeySwitchTorusConvert<P>(saved[slot]);
            for (int key_index = 0; key_index < count; ++key_index) {
#pragma unroll
                for (int k = 0; k < P::t; ++k) {
                    const uint32_t row = rows[key_index * P::t + k];
                    if (row != UINT32_MAX) {
                        const auto value = __ldg(ksk + (row & 0x7fffffffU) + i);
                        res = row & 0x80000000U ? res + value : res - value;
                    }
                }
            }
            lwe[i] = res;
        }
    }
}

// Exact lvl20 key switching with optional index-zero sample extraction.
// Launch with Threads threads. Scratch needs (domain_size * P::t + 1) uint16_t
// words and may overlap the consumed TRLWE: all coefficients are captured in
// registers before the first barrier permits scratch writes. Output and key
// must be disjoint from scratch. No TRLWE data is needed after this call.
template <class P, int Threads, bool ExtractFromTRLWE>
__device__ __forceinline__ void KeySwitchPrepared16(
    typename P::targetP::T* lwe, const typename P::domainP::T* trlwe,
    const typename P::targetP::T* ksk, uint16_t* rows)
{
    static_assert(std::is_same_v<P, TFHEpp::lvl20param>);
    static_assert(std::is_same_v<typename P::targetP::T, uint16_t>);
    constexpr int domain_size = P::domainP::k * P::domainP::n;
    constexpr int output_size = P::targetP::k * P::targetP::n + 1;
    constexpr int digits = std::numeric_limits<typename P::domainP::T>::digits;
    constexpr int halfbase = 1 << (P::basebit - 1);
    constexpr int mask = (1 << P::basebit) - 1;
    constexpr auto decompoffset = iksoffsetgen<P>();
    constexpr typename P::domainP::T roundoffset =
        (P::basebit * P::t) < digits ? 1ULL << (digits - 1 - P::basebit * P::t) : 0;
    static_assert(domain_size % Threads == 0);
    static_assert(uint64_t(domain_size) * P::t * halfbase < (1ULL << 15));
    const int tid = threadIdx.x;
    typename P::domainP::T saved[domain_size / Threads];
#pragma unroll
    for (int slot = 0; slot < domain_size / Threads; ++slot)
        if constexpr (ExtractFromTRLWE)
            saved[slot] = SampleExtractCoefficient<P>(trlwe, tid + slot * Threads);
        else
            saved[slot] = trlwe[tid + slot * Threads];
    const auto body = tid == 0 ? KeySwitchTorusConvert<P>(trlwe[domain_size]) : 0;
    __syncthreads();
#pragma unroll
    for (int slot = 0; slot < domain_size / Threads; ++slot) {
        const int j = tid + slot * Threads;
        const typename P::domainP::T tmp = saved[slot] + decompoffset + roundoffset;
#pragma unroll
        for (int k = 0; k < P::t; ++k) {
            const int d = j * P::t + k;
            const int val = int((tmp >> (digits - (k + 1) * P::basebit)) & mask) - halfbase;
            rows[d] = val == 0 ? UINT16_MAX :
                uint16_t(d * halfbase + abs(val) - 1) | (val < 0 ? 0x8000U : 0U);
        }
    }
    if (tid == 0) rows[domain_size * P::t] = body;
    __syncthreads();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && !defined(CUFHE_USE_HIP)
    if constexpr (Threads == 512 && output_size > Threads && output_size <= 2 * Threads) {
        if constexpr (!ExtractFromTRLWE) {
            // The already-combined MUX/NMux input benefits from five outputs
            // per thread. Binary gates retain the measured two-output path.
            constexpr int OutputThreads = (((output_size + 4) / 5 + 31) / 32) * 32;
            if (tid < OutputThreads) {
                const int first = 5 * tid;
                uint32_t sum[5] = {};
    #pragma unroll
                for (int e = 0; e < 5; ++e)
                    if (first + e == output_size - 1) sum[e] = rows[domain_size * P::t];
                static_assert(domain_size * P::t % 32 == 0);
                static_assert(uint64_t(domain_size) * P::t * halfbase * output_size * sizeof(*ksk) < (1ULL << 31));
                for (int d = 0; d < domain_size * P::t; d += 32) {
                    const uint32_t packed = rows[d + (tid & 31)];
                    const uint32_t lane_row = packed == UINT16_MAX ? UINT32_MAX :
                        ((packed & 0x7fffU) * output_size * sizeof(*ksk)) |
                        ((packed & 0x8000U) << 16);
    #pragma unroll
                    for (int step = 0; step < 32; ++step) {
                        const uint32_t row = __shfl_sync(0xffffffff, lane_row, step);
                        const uintptr_t address = reinterpret_cast<uintptr_t>(ksk) +
                            uintptr_t(row & 0x7fffffffU);
                        uint32_t v0, v1, v2, v3, v4;
                        asm("{ .reg .pred p0, p1, p2, p3, p4;\n\t"
                            "setp.ne.u32 p0, %10, 0xffffffff;\n\t"
                            "setp.ne.u32 p1, %11, 0;\n\tand.pred p1, p1, p0;\n\t"
                            "setp.ne.u32 p2, %12, 0;\n\tand.pred p2, p2, p0;\n\t"
                            "setp.ne.u32 p3, %13, 0;\n\tand.pred p3, p3, p0;\n\t"
                            "setp.ne.u32 p4, %14, 0;\n\tand.pred p4, p4, p0;\n\t"
                            "mov.u32 %0, 0;\n\tmov.u32 %1, 0;\n\t"
                            "mov.u32 %2, 0;\n\tmov.u32 %3, 0;\n\tmov.u32 %4, 0;\n\t"
                            "@p0 ld.global.nc.u16 %0, [%5];\n\t"
                            "@p1 ld.global.nc.u16 %1, [%6];\n\t"
                            "@p2 ld.global.nc.u16 %2, [%7];\n\t"
                            "@p3 ld.global.nc.u16 %3, [%8];\n\t"
                            "@p4 ld.global.nc.u16 %4, [%9];\n\t}"
                            : "=&r"(v0), "=&r"(v1), "=&r"(v2), "=&r"(v3), "=&r"(v4)
                            : "l"(address + first * sizeof(*ksk)),
                              "l"(address + (first + 1) * sizeof(*ksk)),
                              "l"(address + (first + 2) * sizeof(*ksk)),
                              "l"(address + (first + 3) * sizeof(*ksk)),
                              "l"(address + (first + 4) * sizeof(*ksk)),
                              "r"(first < output_size ? row : UINT32_MAX),
                              "r"(uint32_t(first + 1 < output_size)),
                              "r"(uint32_t(first + 2 < output_size)),
                              "r"(uint32_t(first + 3 < output_size)),
                              "r"(uint32_t(first + 4 < output_size)));
                        sum[0] = row & 0x80000000U ? sum[0] + v0 : sum[0] - v0;
                        sum[1] = row & 0x80000000U ? sum[1] + v1 : sum[1] - v1;
                        sum[2] = row & 0x80000000U ? sum[2] + v2 : sum[2] - v2;
                        sum[3] = row & 0x80000000U ? sum[3] + v3 : sum[3] - v3;
                        sum[4] = row & 0x80000000U ? sum[4] + v4 : sum[4] - v4;
                    }
                }
    #pragma unroll
                for (int e = 0; e < 5; ++e)
                    if (first + e < output_size) lwe[first + e] = sum[e];
            }
            return;
        }
        // Adjacent output pairs share one descriptor traversal. Keep the
        // active thread range warp-aligned so every shuffle has all 32 lanes.
        // The partial final pair skips invalid loads and stores.
        constexpr int OutputThreads = (((output_size + 1) / 2 + 31) / 32) * 32;
        if (tid < OutputThreads) {
            const int first = 2 * tid, other = first + 1;
            uint32_t sum0 = first == output_size - 1 ? rows[domain_size * P::t] : 0;
            uint32_t sum1 = other == output_size - 1 ? rows[domain_size * P::t] : 0;
            static_assert(domain_size * P::t % 32 == 0);
            static_assert(uint64_t(domain_size) * P::t * halfbase * output_size * sizeof(*ksk) < (1ULL << 31));
            for (int d = 0; d < domain_size * P::t; d += 32) {
                // Each lane prepares one offset, then broadcasts it in digit order.
                const uint32_t packed = rows[d + (tid & 31)];
                const uint32_t lane_row = packed == UINT16_MAX ? UINT32_MAX :
                    ((packed & 0x7fffU) * output_size * sizeof(*ksk)) |
                    ((packed & 0x8000U) << 16);
    #pragma unroll
                for (int step = 0; step < 32; ++step) {
                    const uint32_t row = __shfl_sync(0xffffffff, lane_row, step);
                    // Integer addresses allow the inactive sentinel row to be masked
                    // before any memory access, without forming an out-of-range pointer.
                    const uintptr_t address = reinterpret_cast<uintptr_t>(ksk) +
                        uintptr_t(row & 0x7fffffffU);
                    uint32_t value0, value1;
                    asm("{ .reg .pred valid, second;\n\t"
                        "setp.ne.u32 valid, %4, 0xffffffff;\n\t"
                        "setp.ne.u32 second, %5, 0;\n\t"
                        "and.pred second, second, valid;\n\t"
                        "mov.u32 %0, 0;\n\t"
                        "mov.u32 %1, 0;\n\t"
                        "@valid ld.global.nc.u16 %0, [%2];\n\t"
                        "@second ld.global.nc.u16 %1, [%3];\n\t}"
                        : "=&r"(value0), "=&r"(value1)
                        : "l"(address + first * sizeof(*ksk)),
                          "l"(address + other * sizeof(*ksk)),
                          "r"(first < output_size ? row : UINT32_MAX),
                          "r"(uint32_t(other < output_size)));
                    sum0 = row & 0x80000000U ? sum0 + value0 : sum0 - value0;
                    sum1 = row & 0x80000000U ? sum1 + value1 : sum1 - value1;
                }
            }
            // Reduction from unsigned 32-bit wraparound to the existing 16-bit
            // torus commutes with every addition/subtraction above.
            if (first < output_size) lwe[first] = sum0;
            if (other < output_size) lwe[other] = sum1;
        }
        return;
    }
#endif
    for (int i = tid; i < output_size; i += Threads) {
        typename P::targetP::T res =
            i == output_size - 1 ? rows[domain_size * P::t] : 0;
        for (int j = 0; j < domain_size; ++j) {
#pragma unroll
            for (int k = 0; k < P::t; ++k) {
                const uint32_t row = rows[j * P::t + k];
                if (row != UINT16_MAX) {
                    const auto value = __ldg(ksk + (row & 0x7fffU) * output_size + i);
                    res = row & 0x8000U ? res + value : res - value;
                }
            }
        }
        lwe[i] = res;
    }
}

void KeySwitchingKeyToDevice(
#ifdef USE_SUBSET_KEY
    const TFHEpp::SubsetKeySwitchingKey<TFHEpp::lvl10param>& ksk,
#else
    const TFHEpp::KeySwitchingKey<TFHEpp::lvl10param>& ksk,
#endif
    const int gpuNum);

void DeleteKeySwitchingKey(const int gpuNum);

void KeySwitchingKeyToDevice_lvl20(
    const TFHEpp::KeySwitchingKey<TFHEpp::lvl20param>& ksk, const int gpuNum);

void DeleteKeySwitchingKey_lvl20(const int gpuNum);

void SEIandKS(TFHEpp::lvl0param::T* const out,
              const TFHEpp::lvl1param::T* const in, const cudaStream_t& st,
              const int gpuNum);
}  // namespace cufhe
