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
