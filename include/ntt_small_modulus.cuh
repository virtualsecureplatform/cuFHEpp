/**
 * Small-modulus NTT implementation for cuFHE
 *
 * The modulus is selected by ring size:
 * - lvl1/N=1024 uses the original 31-bit prime for faster gate evaluation
 * - lvl2/N=2048 uses the Goldilocks prime to preserve Torus64 precision
 */

#pragma once

#include "gpu_runtime.cuh"

#include <cstdint>
#include <include/utils_gpu.cuh>
#include <limits>
#include <params.hpp>
#include <type_traits>
#include <vector>

#ifdef USE_FFT
#ifdef USE_GPU_FFT
// GPU-FFT: no fft_negacyclic.cuh needed; tables stored in handler
#else
#include <fft_negacyclic.cuh>
#endif
#endif

namespace cufhe {

template <class P>
inline constexpr uint32_t BootstrappingTRGSWRows =
    P::k * P::lₐ * P::l̅ₐ + P::l * P::l̅;

//=============================================================================
// NTT Constants
//=============================================================================

namespace small_ntt31 {

using Value = uint32_t;

constexpr Value K = 1048571;
constexpr Value SHIFTAMOUNT = 11;
constexpr Value WORDBITS = 32;
constexpr Value P = (K << SHIFTAMOUNT) + 1;
constexpr Value P_MINUS_ONE = P - 1;
constexpr Value HALF_P = P / 2;

static_assert(P == 2147473409U, "Unexpected lvl1 NTT prime value");
static_assert(P < 3037000500ULL,
              "Modulus too large for safe 64-bit multiplication");

constexpr Value FOLD_BITS = 31;
constexpr Value FOLD_MASK = (1U << FOLD_BITS) - 1;
constexpr Value FOLD_FACTOR = (1U << FOLD_BITS) - P;
constexpr uint64_t MAX_SECOND_FOLD =
    static_cast<uint64_t>(FOLD_MASK) +
    static_cast<uint64_t>(FOLD_FACTOR) * FOLD_FACTOR;
constexpr uint64_t MAX_PRODUCT =
    static_cast<uint64_t>(P_MINUS_ONE) * P_MINUS_ONE;
static_assert(MAX_PRODUCT <=
                  (std::numeric_limits<uint64_t>::max() - P_MINUS_ONE) / 3ULL,
              "Three lvl1 products and one addend must fit in uint64_t");
constexpr uint64_t MAX_MADD3_SUM =
    3ULL * MAX_PRODUCT + P_MINUS_ONE;
constexpr uint64_t MAX_MADD3_FIRST_FOLD =
    static_cast<uint64_t>(FOLD_MASK) +
    (MAX_MADD3_SUM >> FOLD_BITS) * FOLD_FACTOR;
constexpr uint64_t MAX_MADD3_SECOND_FOLD =
    static_cast<uint64_t>(FOLD_MASK) +
    (MAX_MADD3_FIRST_FOLD >> FOLD_BITS) * FOLD_FACTOR;
constexpr uint64_t INV_MODSWITCH_MUL = (1ULL << 63) / P;

static_assert(FOLD_FACTOR == 10239U, "Unexpected lvl1 NTT fold factor");
static_assert(MAX_SECOND_FOLD < 2ULL * P,
              "Two folds must leave at most one conditional subtraction");
static_assert(MAX_SECOND_FOLD < (1ULL << 32),
              "Second fold must fit in a uint32_t");
static_assert(MAX_MADD3_SECOND_FOLD < 2ULL * P,
              "Fused madd fold must need at most one subtraction");
static_assert(MAX_MADD3_SECOND_FOLD < (1ULL << 32),
              "Fused madd fold must fit in a uint32_t");

}  // namespace small_ntt31

namespace small_ntt {

using Value = uint64_t;

constexpr Value P = 0xFFFFFFFF00000001ULL;
constexpr Value P_MINUS_ONE = P - 1;
constexpr Value HALF_P = P / 2;

// Same primitive 2^32-th root used by TFHEpp's integer NTT.
constexpr Value ROOT_2_32 = 12037493425763644479ULL;
constexpr uint32_t WORDBITS = 64;

static_assert(P == 18446744069414584321ULL,
              "Unexpected Goldilocks prime value");

}  // namespace small_ntt

using SmallNTTValue = uint64_t;

template <uint32_t N>
using SmallNTTValueFor =
    std::conditional_t<N == TFHEpp::lvl1param::n, uint32_t, uint64_t>;

template <uint32_t N>
struct SmallNTTModulus {
    static_assert(N == TFHEpp::lvl1param::n || N == TFHEpp::lvl2param::n,
                  "Unsupported small NTT length");
};

template <>
struct SmallNTTModulus<TFHEpp::lvl1param::n> {
    static constexpr SmallNTTValue P = small_ntt31::P;
    static constexpr SmallNTTValue P_MINUS_ONE = small_ntt31::P_MINUS_ONE;
    static constexpr SmallNTTValue HALF_P = small_ntt31::HALF_P;
    static constexpr uint32_t WORDBITS = small_ntt31::WORDBITS;
};

template <>
struct SmallNTTModulus<TFHEpp::lvl2param::n> {
    static constexpr SmallNTTValue P = small_ntt::P;
    static constexpr SmallNTTValue P_MINUS_ONE = small_ntt::P_MINUS_ONE;
    static constexpr SmallNTTValue HALF_P = small_ntt::HALF_P;
    static constexpr uint32_t WORDBITS = small_ntt::WORDBITS;
};

template <uint32_t N>
__host__ __device__ constexpr uint32_t SmallLog2()
{
    uint32_t n = N;
    uint32_t log = 0;
    while (n > 1) {
        n >>= 1;
        ++log;
    }
    return log;
}

//=============================================================================
// 64-bit Goldilocks Finite Field Element
//=============================================================================

class FFP64 {
   private:
    SmallNTTValue val_;

   public:
    __host__ __device__ inline FFP64() : val_(0) {}
    __host__ __device__ inline FFP64(SmallNTTValue a)
        : val_(a >= small_ntt::P ? a + static_cast<uint32_t>(-1) : a)
    {
    }
    __host__ __device__ inline FFP64(int32_t a)
    {
        if (a < 0) {
            val_ = small_ntt::P - static_cast<SmallNTTValue>(-a);
            if (val_ == small_ntt::P) val_ = 0;
        }
        else {
            val_ = static_cast<SmallNTTValue>(a);
        }
    }

    __host__ __device__ inline SmallNTTValue& val() { return val_; }
    __host__ __device__ inline const SmallNTTValue& val() const { return val_; }
    __host__ __device__ inline static constexpr SmallNTTValue kModulus()
    {
        return small_ntt::P;
    }

    __host__ __device__ inline explicit operator SmallNTTValue() const
    {
        return val_;
    }
};

//=============================================================================
// Length-selected Modulus Operations
//=============================================================================

__host__ __device__ __forceinline__ SmallNTTValue
small_mod64_normalize(SmallNTTValue a)
{
    return a + static_cast<uint32_t>(-(a >= small_ntt::P));
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod64_add(SmallNTTValue a, SmallNTTValue b)
{
    SmallNTTValue tmp = a + b;
    return tmp + static_cast<uint32_t>(-(tmp < b || tmp >= small_ntt::P));
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod64_sub(SmallNTTValue a, SmallNTTValue b)
{
    SmallNTTValue tmp = a - b;
    return tmp - static_cast<uint32_t>(-(tmp > a));
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod64_mult(SmallNTTValue a, SmallNTTValue b)
{
    unsigned __int128 prod = static_cast<unsigned __int128>(a) * b;
    const SmallNTTValue lo = static_cast<SmallNTTValue>(prod);

    const uint32_t limb0 = static_cast<uint32_t>(prod);
    prod >>= 32;
    const uint32_t limb1 = static_cast<uint32_t>(prod);
    prod >>= 32;
    const uint32_t limb2 = static_cast<uint32_t>(prod);
    prod >>= 32;
    const uint32_t limb3 = static_cast<uint32_t>(prod);

    SmallNTTValue res = ((static_cast<SmallNTTValue>(limb1) + limb2) << 32) +
                        limb0 - limb3 - limb2;
    res -= static_cast<uint32_t>(-((res > lo) && (limb2 == 0)));
    res += static_cast<uint32_t>(-((res < lo) && (limb2 != 0)));
    return small_mod64_normalize(res);
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod64_madd(SmallNTTValue a, SmallNTTValue b, SmallNTTValue c)
{
    unsigned __int128 sum = static_cast<unsigned __int128>(a) * b + c;
    const SmallNTTValue lo = static_cast<SmallNTTValue>(sum);

    const uint32_t limb0 = static_cast<uint32_t>(sum);
    sum >>= 32;
    const uint32_t limb1 = static_cast<uint32_t>(sum);
    sum >>= 32;
    const uint32_t limb2 = static_cast<uint32_t>(sum);
    sum >>= 32;
    const uint32_t limb3 = static_cast<uint32_t>(sum);

    SmallNTTValue res = ((static_cast<SmallNTTValue>(limb1) + limb2) << 32) +
                        limb0 - limb3 - limb2;
    res -= static_cast<uint32_t>(-((res > lo) && (limb2 == 0)));
    res += static_cast<uint32_t>(-((res < lo) && (limb2 != 0)));
    return small_mod64_normalize(res);
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod31_normalize(SmallNTTValue a)
{
    if (a >= small_ntt31::P) a %= small_ntt31::P;
    return static_cast<uint32_t>(a);
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod31_add(SmallNTTValue a, SmallNTTValue b)
{
    uint32_t sum = static_cast<uint32_t>(a) + static_cast<uint32_t>(b);
    return (sum >= small_ntt31::P) ? (sum - small_ntt31::P) : sum;
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod31_sub(SmallNTTValue a, SmallNTTValue b)
{
    uint32_t diff =
        static_cast<uint32_t>(a) + small_ntt31::P - static_cast<uint32_t>(b);
    return (diff >= small_ntt31::P) ? (diff - small_ntt31::P) : diff;
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod31_mult(SmallNTTValue a, SmallNTTValue b)
{
    constexpr uint32_t p = small_ntt31::P;
    const uint64_t z = static_cast<uint64_t>(static_cast<uint32_t>(a)) *
                       static_cast<uint32_t>(b);
    constexpr uint32_t mask = small_ntt31::FOLD_MASK;
    constexpr uint32_t factor = small_ntt31::FOLD_FACTOR;

    // p = 2^31 - 10239, so 2^31 is congruent to 10239 modulo p.
    // For z < p^2, two folds leave a value below 2p.
    const uint32_t lo = static_cast<uint32_t>(z) & mask;
    const uint32_t hi = static_cast<uint32_t>(z >> 31);
    const uint64_t folded = lo + static_cast<uint64_t>(hi) * factor;
    const uint32_t result = (static_cast<uint32_t>(folded) & mask) +
                            static_cast<uint32_t>(folded >> 31) * factor;
    return (result >= p) ? (result - p) : result;
}

__host__ __device__ __forceinline__ SmallNTTValue
small_mod31_madd(SmallNTTValue a, SmallNTTValue b, SmallNTTValue c)
{
    constexpr uint32_t p = small_ntt31::P;
    const uint64_t z = static_cast<uint64_t>(static_cast<uint32_t>(a)) *
                           static_cast<uint32_t>(b) +
                       static_cast<uint32_t>(c);
    constexpr uint32_t mask = small_ntt31::FOLD_MASK;
    constexpr uint32_t factor = small_ntt31::FOLD_FACTOR;

    const uint32_t lo = static_cast<uint32_t>(z) & mask;
    const uint32_t hi = static_cast<uint32_t>(z >> 31);
    const uint64_t folded = lo + static_cast<uint64_t>(hi) * factor;
    const uint32_t result = (static_cast<uint32_t>(folded) & mask) +
                            static_cast<uint32_t>(folded >> 31) * factor;
    return (result >= p) ? (result - p) : result;
}

// Reduce three products and an addend together.  For the 31-bit modulus the
// unreduced sum still fits in uint64_t, saving two pseudo-Mersenne reductions
// in the key-bundle inner loop.
__host__ __device__ __forceinline__ SmallNTTValue small_mod31_madd3(
    SmallNTTValue a0, SmallNTTValue b0, SmallNTTValue a1, SmallNTTValue b1,
    SmallNTTValue a2, SmallNTTValue b2, SmallNTTValue c)
{
    constexpr uint32_t p = small_ntt31::P;
    constexpr uint32_t mask = small_ntt31::FOLD_MASK;
    constexpr uint32_t factor = small_ntt31::FOLD_FACTOR;
    const uint64_t z =
        static_cast<uint64_t>(static_cast<uint32_t>(a0)) *
            static_cast<uint32_t>(b0) +
        static_cast<uint64_t>(static_cast<uint32_t>(a1)) *
            static_cast<uint32_t>(b1) +
        static_cast<uint64_t>(static_cast<uint32_t>(a2)) *
            static_cast<uint32_t>(b2) +
        static_cast<uint32_t>(c);

    const uint64_t folded =
        (z & mask) + static_cast<uint64_t>(z >> 31) * factor;
    const uint32_t result =
        static_cast<uint32_t>(folded & mask) +
        static_cast<uint32_t>(folded >> 31) * factor;
    return (result >= p) ? (result - p) : result;
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue
small_mod_normalize(SmallNTTValue a)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_normalize(a);
    }
    else {
        return small_mod64_normalize(a);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue small_mod_add(SmallNTTValue a,
                                                                SmallNTTValue b)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_add(a, b);
    }
    else {
        return small_mod64_add(a, b);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue small_mod_sub(SmallNTTValue a,
                                                                SmallNTTValue b)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_sub(a, b);
    }
    else {
        return small_mod64_sub(a, b);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue
small_mod_mult(SmallNTTValue a, SmallNTTValue b)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_mult(a, b);
    }
    else {
        return small_mod64_mult(a, b);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue
small_mod_madd(SmallNTTValue a, SmallNTTValue b, SmallNTTValue c)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_madd(a, b, c);
    }
    else {
        return small_mod64_madd(a, b, c);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue small_mod_madd3(
    SmallNTTValue a0, SmallNTTValue b0, SmallNTTValue a1, SmallNTTValue b1,
    SmallNTTValue a2, SmallNTTValue b2, SmallNTTValue c)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        return small_mod31_madd3(a0, b0, a1, b1, a2, b2, c);
    }
    else {
        SmallNTTValue result = small_mod64_madd(a0, b0, c);
        result = small_mod64_madd(a1, b1, result);
        return small_mod64_madd(a2, b2, result);
    }
}

template <uint32_t N, typename TorusT>
__host__ __device__ __forceinline__ SmallNTTValue
torus_to_ntt_mod(TorusT torus_val)
{
    using UnsignedT = std::make_unsigned_t<TorusT>;
    constexpr int bits = std::numeric_limits<UnsignedT>::digits;
    const auto a = static_cast<UnsignedT>(torus_val);
    unsigned __int128 prod =
        static_cast<unsigned __int128>(a) * SmallNTTModulus<N>::P;
    prod += static_cast<unsigned __int128>(1) << (bits - 1);
    return static_cast<SmallNTTValue>(prod >> bits);
}

template <uint32_t N>
__host__ __device__ __forceinline__ SmallNTTValue
signed_int_to_ntt_mod(int32_t val)
{
    if (val < 0) {
        return small_mod_sub<N>(
            0, static_cast<SmallNTTValue>(-static_cast<int64_t>(val)));
    }
    return static_cast<SmallNTTValue>(val);
}

__host__ __device__ __forceinline__ uint64_t
ntt_abs_to_torus64_goldilocks(SmallNTTValue val)
{
    const unsigned __int128 mul = val;
    return static_cast<uint64_t>(((mul << 64) + (mul << 32) - mul +
                                  (static_cast<unsigned __int128>(1) << 63)) >>
                                 64);
}

template <uint32_t N>
__host__ __device__ __forceinline__ uint64_t
ntt_mod_to_torus64(SmallNTTValue val)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        const bool neg = val > small_ntt31::HALF_P;
        const SmallNTTValue mag = neg ? (small_ntt31::P - val) : val;
        const unsigned __int128 scaled =
            (static_cast<unsigned __int128>(mag) << 64) + small_ntt31::HALF_P;
        const uint64_t torus64 = static_cast<uint64_t>(scaled / small_ntt31::P);
        return neg ? static_cast<uint64_t>(-torus64) : torus64;
    }
    else {
        if (val > small_ntt::HALF_P) {
            const uint64_t mag =
                ntt_abs_to_torus64_goldilocks(small_ntt::P - val);
            return static_cast<uint64_t>(-mag);
        }
        return ntt_abs_to_torus64_goldilocks(val);
    }
}

template <uint32_t N>
__host__ __device__ __forceinline__ uint32_t
ntt_mod_to_torus32(SmallNTTValue val)
{
    if constexpr (N == TFHEpp::lvl1param::n) {
        const bool neg = val > small_ntt31::HALF_P;
        const uint32_t mag =
            static_cast<uint32_t>(neg ? (small_ntt31::P - val) : val);
        uint64_t temp =
            static_cast<uint64_t>(mag) * small_ntt31::INV_MODSWITCH_MUL;
        uint32_t torus32 = static_cast<uint32_t>((temp + (1ULL << 30)) >> 31);
        return neg ? static_cast<uint32_t>(-torus32) : torus32;
    }
    else {
        const bool neg = val > small_ntt::HALF_P;
        const uint64_t torus64 =
            ntt_abs_to_torus64_goldilocks(neg ? (small_ntt::P - val) : val);
        uint32_t torus32 = static_cast<uint32_t>(torus64 >> 32);
        torus32 +=
            static_cast<uint32_t>((torus64 & 0xFFFFFFFFULL) >= 0x80000000ULL);
        return neg ? static_cast<uint32_t>(-torus32) : torus32;
    }
}

#ifdef CUFHE_GPU_DEVICE_COMPILER

extern __constant__ uint32_t d_const_forward_root_31[TFHEpp::lvl1param::n];
extern __constant__ uint32_t d_const_inverse_root_31[TFHEpp::lvl1param::n];
extern __constant__ uint64_t d_const_forward_root_64[TFHEpp::lvl2param::n];
extern __constant__ uint64_t d_const_inverse_root_64[TFHEpp::lvl2param::n];

template <class T>
__device__ __forceinline__ T SmallNTTShuffleXor(const T value,
                                                const int lane_mask)
{
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return static_cast<T>(__shfl_xor_sync(
            0xFFFFFFFFULL, static_cast<uint32_t>(value), lane_mask));
    }
    else {
        static_assert(sizeof(T) == sizeof(uint64_t),
                      "Unsupported NTT shuffle value width");
        uint32_t lo = static_cast<uint32_t>(value);
        uint32_t hi = static_cast<uint32_t>(value >> 32);
        lo = __shfl_xor_sync(0xFFFFFFFFULL, lo, lane_mask);
        hi = __shfl_xor_sync(0xFFFFFFFFULL, hi, lane_mask);
        return static_cast<T>((static_cast<uint64_t>(hi) << 32) | lo);
    }
}

// Cooley-Tukey butterfly for forward NTT
template <int N_POWER>
__device__ __forceinline__ void SmallCooleyTukeyUnit(
    SmallNTTValueFor<1U << N_POWER>& U, SmallNTTValueFor<1U << N_POWER>& V,
    SmallNTTValueFor<1U << N_POWER> root)
{
    constexpr uint32_t N = 1U << N_POWER;
    SmallNTTValue u = U;
    SmallNTTValue v = small_mod_mult<N>(V, root);
    U = small_mod_add<N>(u, v);
    V = small_mod_sub<N>(u, v);
}

// Gentleman-Sande butterfly for inverse NTT
template <int N_POWER>
__device__ __forceinline__ void SmallGentlemanSandeUnit(
    SmallNTTValueFor<1U << N_POWER>& U, SmallNTTValueFor<1U << N_POWER>& V,
    SmallNTTValueFor<1U << N_POWER> root)
{
    constexpr uint32_t N = 1U << N_POWER;
    SmallNTTValue u = U;
    SmallNTTValue v = V;
    U = small_mod_add<N>(u, v);
    V = small_mod_mult<N>(small_mod_sub<N>(u, v), root);
}

template <int N_POWER>
__device__ __forceinline__ void SmallCooleyTukeyRadix4Unit(
    SmallNTTValueFor<1U << N_POWER>& a,
    SmallNTTValueFor<1U << N_POWER>& b,
    SmallNTTValueFor<1U << N_POWER>& c,
    SmallNTTValueFor<1U << N_POWER>& d,
    const SmallNTTValueFor<1U << N_POWER> root_stage0,
    const SmallNTTValueFor<1U << N_POWER> root_stage1_lo,
    const SmallNTTValueFor<1U << N_POWER> root_stage1_hi)
{
    SmallCooleyTukeyUnit<N_POWER>(a, c, root_stage0);
    SmallCooleyTukeyUnit<N_POWER>(b, d, root_stage0);
    SmallCooleyTukeyUnit<N_POWER>(a, b, root_stage1_lo);
    SmallCooleyTukeyUnit<N_POWER>(c, d, root_stage1_hi);
}

template <int N_POWER>
__device__ __forceinline__ void SmallGentlemanSandeRadix4Unit(
    SmallNTTValueFor<1U << N_POWER>& a,
    SmallNTTValueFor<1U << N_POWER>& b,
    SmallNTTValueFor<1U << N_POWER>& c,
    SmallNTTValueFor<1U << N_POWER>& d,
    const SmallNTTValueFor<1U << N_POWER> root_stage0_lo,
    const SmallNTTValueFor<1U << N_POWER> root_stage0_hi,
    const SmallNTTValueFor<1U << N_POWER> root_stage1)
{
    SmallGentlemanSandeUnit<N_POWER>(a, b, root_stage0_lo);
    SmallGentlemanSandeUnit<N_POWER>(c, d, root_stage0_hi);
    SmallGentlemanSandeUnit<N_POWER>(a, c, root_stage1);
    SmallGentlemanSandeUnit<N_POWER>(b, d, root_stage1);
}

template <int N_POWER>
__device__ __forceinline__ void SmallForwardNTT(
    SmallNTTValueFor<1U << N_POWER>* sh,
    const SmallNTTValueFor<1U << N_POWER>* root_table, int tid)
{
    static_assert(N_POWER >= 6, "NTT length must be at least 64");
#if defined(CUFHE_USE_HIP)
    constexpr uint32_t N = 1U << N_POWER;
#endif

    int t_2 = N_POWER - 1;
    int t_ = N_POWER - 1;
    int m = 1;
    int t = 1 << t_;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

#pragma unroll
    for (int lp = 0; lp < (N_POWER - 6) / 2; lp++) {
        if (tid < (1 << (N_POWER - 2))) {
            const int radix_t = t >> 1;
            const int group = tid >> (t_ - 1);
            const int offset = tid & (radix_t - 1);
            const int address = (group << (t_ + 1)) + offset;

            const int root0_index = m + group;
            const int root1_lo_index = (m << 1) + (group << 1);
            const int root1_hi_index = root1_lo_index + 1;
            SmallNTTValueFor<1U << N_POWER> root0 =
                __ldg(&root_table[root0_index]);
            SmallNTTValueFor<1U << N_POWER> root1_lo =
                __ldg(&root_table[root1_lo_index]);
            SmallNTTValueFor<1U << N_POWER> root1_hi =
                __ldg(&root_table[root1_hi_index]);
            if constexpr ((1U << N_POWER) == TFHEpp::lvl1param::n) {
                root0 = d_const_forward_root_31[root0_index];
                root1_lo = d_const_forward_root_31[root1_lo_index];
                root1_hi = d_const_forward_root_31[root1_hi_index];
            }
            else if constexpr ((1U << N_POWER) == TFHEpp::lvl2param::n) {
                root0 = d_const_forward_root_64[root0_index];
                root1_lo = d_const_forward_root_64[root1_lo_index];
                root1_hi = d_const_forward_root_64[root1_hi_index];
            }

            SmallNTTValueFor<1U << N_POWER> a = sh[address];
            SmallNTTValueFor<1U << N_POWER> b = sh[address + radix_t];
            SmallNTTValueFor<1U << N_POWER> c = sh[address + t];
            SmallNTTValueFor<1U << N_POWER> d =
                sh[address + t + radix_t];
            SmallCooleyTukeyRadix4Unit<N_POWER>(a, b, c, d, root0,
                                                 root1_lo, root1_hi);
            sh[address] = a;
            sh[address + radix_t] = b;
            sh[address + t] = c;
            sh[address + t + radix_t] = d;
        }

        t >>= 2;
        t_2 -= 2;
        t_ -= 2;
        m <<= 2;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    if constexpr ((N_POWER - 6) % 2 != 0) {
        current_root_index = m + (tid >> t_2);
        SmallNTTValueFor<1U << N_POWER> root =
            __ldg(&root_table[current_root_index]);
        if constexpr ((1U << N_POWER) == TFHEpp::lvl1param::n) {
            root = d_const_forward_root_31[current_root_index];
        }
        else if constexpr ((1U << N_POWER) == TFHEpp::lvl2param::n) {
            root = d_const_forward_root_64[current_root_index];
        }
        SmallCooleyTukeyUnit<N_POWER>(sh[in_shared_address],
                                      sh[in_shared_address + t], root);
        t >>= 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

#if defined(CUFHE_USE_HIP)
    if constexpr (N == TFHEpp::lvl1param::n) {
        // Stride 32 is the boundary between shared-memory and wave-local
        // stages on wave32.  Keep its outputs in registers and shuffle the
        // stride-16 pairs into place instead of round-tripping through LDS.
        current_root_index = m + (tid >> t_2);
        auto boundary_u = sh[in_shared_address];
        auto boundary_v = sh[in_shared_address + t];
        SmallCooleyTukeyUnit<N_POWER>(
            boundary_u, boundary_v,
            __ldg(&root_table[current_root_index]));

        const auto peer_u = SmallNTTShuffleXor(boundary_u, 16);
        const auto peer_v = SmallNTTShuffleXor(boundary_v, 16);
        const bool upper_half = (tid & 16) != 0;
        auto reg_u = upper_half ? peer_v : boundary_u;
        auto reg_v = upper_half ? boundary_v : peer_u;

        t >>= 1;
        --t_2;
        --t_;
        m <<= 1;

        current_root_index = m + (tid >> t_2);
        SmallCooleyTukeyUnit<N_POWER>(
            reg_u, reg_v, __ldg(&root_table[current_root_index]));

        t >>= 1;
        --t_2;
        --t_;
        m <<= 1;

#pragma unroll
        for (int xor_mask = 8; xor_mask >= 1; xor_mask >>= 1) {
            const bool is_upper = (tid & xor_mask) != 0;
            const auto sent = is_upper ? reg_u : reg_v;
            const auto received = SmallNTTShuffleXor(sent, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            current_root_index = m + (tid >> t_2);
            SmallCooleyTukeyUnit<N_POWER>(
                reg_u, reg_v, __ldg(&root_table[current_root_index]));
            t >>= 1;
            --t_2;
            --t_;
            m <<= 1;
        }

        sh[2 * tid] = reg_u;
        sh[2 * tid + 1] = reg_v;
    }
    else
#endif
    {
#pragma unroll 1
        for (int lp = 0; lp < 6; lp++) {
            current_root_index = m + (tid >> t_2);
            SmallCooleyTukeyUnit<N_POWER>(
                sh[in_shared_address], sh[in_shared_address + t],
                __ldg(&root_table[current_root_index]));

            t >>= 1;
            --t_2;
            --t_;
            m <<= 1;
            in_shared_address = ((tid >> t_) << t_) + tid;
        }
    }
    __syncthreads();
}

template <int N_POWER>
__device__ __forceinline__ void SmallInverseNTT(
    SmallNTTValueFor<1U << N_POWER>* sh,
    const SmallNTTValueFor<1U << N_POWER>* root_table,
    SmallNTTValueFor<1U << N_POWER> n_inverse, int tid)
{
    static_assert(N_POWER >= 6, "NTT length must be at least 64");
    constexpr uint32_t N = 1U << N_POWER;
    constexpr int NUM_THREADS = 1 << (N_POWER - 1);

    int t_2 = 0;
    int t_ = 0;
    int m = 1 << (N_POWER - 1);
    int t = 1;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

#if defined(CUFHE_USE_HIP)
    if constexpr (N == TFHEpp::lvl1param::n) {
        auto reg_u = sh[2 * tid];
        auto reg_v = sh[2 * tid + 1];
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit<N_POWER>(
            reg_u, reg_v, __ldg(&root_table[current_root_index]));

        t <<= 1;
        ++t_2;
        ++t_;
        m >>= 1;

#pragma unroll
        for (int xor_mask = 1; xor_mask <= 8; xor_mask <<= 1) {
            const bool is_upper = (tid & xor_mask) != 0;
            const auto sent = is_upper ? reg_u : reg_v;
            const auto received = SmallNTTShuffleXor(sent, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            current_root_index = m + (tid >> t_2);
            SmallGentlemanSandeUnit<N_POWER>(
                reg_u, reg_v, __ldg(&root_table[current_root_index]));
            t <<= 1;
            ++t_2;
            ++t_;
            m >>= 1;
        }

        // Reassemble the stride-32 butterfly pairs directly from registers.
        // This avoids the stride-16 write/read handoff through LDS.
        const auto peer_u = SmallNTTShuffleXor(reg_u, 16);
        const auto peer_v = SmallNTTShuffleXor(reg_v, 16);
        const bool upper_half = (tid & 16) != 0;
        auto boundary_u = upper_half ? peer_v : reg_u;
        auto boundary_v = upper_half ? reg_v : peer_u;

        in_shared_address = ((tid >> t_) << t_) + tid;
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit<N_POWER>(
            boundary_u, boundary_v,
            __ldg(&root_table[current_root_index]));
        sh[in_shared_address] = boundary_u;
        sh[in_shared_address + t] = boundary_v;
        t <<= 1;
        ++t_2;
        ++t_;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
    }
    else
#endif
    {
#pragma unroll 1
        for (int lp = 0; lp < 6; lp++) {
            current_root_index = m + (tid >> t_2);
            SmallGentlemanSandeUnit<N_POWER>(
                sh[in_shared_address], sh[in_shared_address + t],
                __ldg(&root_table[current_root_index]));

            t <<= 1;
            ++t_2;
            ++t_;
            m >>= 1;
            in_shared_address = ((tid >> t_) << t_) + tid;
        }
    }
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < (N_POWER - 6) / 2; lp++) {
        if (tid < (1 << (N_POWER - 2))) {
            const int group = tid >> t_;
            const int offset = tid & (t - 1);
            const int address = (group << (t_ + 2)) + offset;

            const int root0_lo_index = m + (group << 1);
            const int root0_hi_index = root0_lo_index + 1;
            const int root1_index = (m >> 1) + group;
            SmallNTTValueFor<1U << N_POWER> root0_lo =
                __ldg(&root_table[root0_lo_index]);
            SmallNTTValueFor<1U << N_POWER> root0_hi =
                __ldg(&root_table[root0_hi_index]);
            SmallNTTValueFor<1U << N_POWER> root1 =
                __ldg(&root_table[root1_index]);
            if constexpr ((1U << N_POWER) == TFHEpp::lvl1param::n) {
                root0_lo = d_const_inverse_root_31[root0_lo_index];
                root0_hi = d_const_inverse_root_31[root0_hi_index];
                root1 = d_const_inverse_root_31[root1_index];
            }
            else if constexpr ((1U << N_POWER) == TFHEpp::lvl2param::n) {
                root0_lo = d_const_inverse_root_64[root0_lo_index];
                root0_hi = d_const_inverse_root_64[root0_hi_index];
                root1 = d_const_inverse_root_64[root1_index];
            }

            SmallNTTValueFor<1U << N_POWER> a = sh[address];
            SmallNTTValueFor<1U << N_POWER> b = sh[address + t];
            SmallNTTValueFor<1U << N_POWER> c = sh[address + 2 * t];
            SmallNTTValueFor<1U << N_POWER> d = sh[address + 3 * t];
            SmallGentlemanSandeRadix4Unit<N_POWER>(
                a, b, c, d, root0_lo, root0_hi, root1);
            sh[address] = a;
            sh[address + t] = b;
            sh[address + 2 * t] = c;
            sh[address + 3 * t] = d;
        }

        t <<= 2;
        t_2 += 2;
        t_ += 2;
        m >>= 2;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    if constexpr ((N_POWER - 6) % 2 != 0) {
        current_root_index = m + (tid >> t_2);
        SmallNTTValueFor<1U << N_POWER> root =
            __ldg(&root_table[current_root_index]);
        if constexpr ((1U << N_POWER) == TFHEpp::lvl1param::n) {
            root = d_const_inverse_root_31[current_root_index];
        }
        else if constexpr ((1U << N_POWER) == TFHEpp::lvl2param::n) {
            root = d_const_inverse_root_64[current_root_index];
        }
        SmallGentlemanSandeUnit<N_POWER>(sh[in_shared_address],
                                         sh[in_shared_address + t], root);
        t <<= 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    // Scaling is coefficient-local. Callers synchronize only when the shared
    // buffer is handed to another phase.
    sh[tid] = small_mod_mult<N>(sh[tid], n_inverse);
    sh[tid + NUM_THREADS] = small_mod_mult<N>(sh[tid + NUM_THREADS], n_inverse);
}

template <uint32_t N>
__host__ __device__ constexpr int SmallForwardNTTSyncCount()
{
    return (static_cast<int>(SmallLog2<N>()) - 5) / 2 + 1;
}

template <uint32_t N>
__host__ __device__ constexpr int SmallInverseNTTSyncCount()
{
    return (static_cast<int>(SmallLog2<N>()) - 5) / 2 + 1;
}

#endif  // CUFHE_GPU_DEVICE_COMPILER

//=============================================================================
// Small Modulus NTT Handler
//=============================================================================

// Host-side storage for small NTT parameters
template <uint32_t length>
struct SmallNTTParams {
    SmallNTTValueFor<length>* forward_root;
    SmallNTTValueFor<length>* inverse_root;
    SmallNTTValueFor<length> n_inverse;
    bool initialized;
};

extern std::vector<SmallNTTParams<TFHEpp::lvl1param::n>> g_small_ntt_params;
extern std::vector<SmallNTTParams<TFHEpp::lvl2param::n>>
    g_small_ntt_params_lvl02;

/**
 * Length-selected small-modulus NTT handler for cuFHE
 *
 * Before NTT, coefficients are modulus-switched from the torus to P. After
 * INTT, coefficients are centered and switched back to Torus32/Torus64.
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuSmallNTTHandler {
   public:
    using Value = SmallNTTValueFor<length>;
    static constexpr uint32_t kLength = length;
    static constexpr uint32_t kLogLength = []() constexpr {
        uint32_t n = length, log = 0;
        while (n > 1) {
            n >>= 1;
            ++log;
        }
        return log;
    }();

    Value* forward_root_;
    Value* inverse_root_;
    Value n_inverse_;

    __host__ __device__ CuSmallNTTHandler()
        : forward_root_(nullptr), inverse_root_(nullptr), n_inverse_(0)
    {
    }
    __host__ __device__ ~CuSmallNTTHandler() {}

    __host__ static void Create();
    __host__ static void CreateConstant() {
    }  // No-op for small modulus (tables already set in Create())
    __host__ static void Destroy();
    __host__ void SetDevicePointers(int device_id);

#ifdef CUFHE_GPU_DEVICE_COMPILER
    /**
     * Forward NTT with modulus switching (torus -> NTT domain)
     *
     * This performs:
     * 1. Modulus switch: Convert input from 2^32 to P discretization
     * 2. Forward NTT in modulus P
     */
    template <typename TorusT>
    __device__ inline void NTTWithModSwitch(Value* const out,
                                            const TorusT* const in,
                                            Value* const sh_temp,
                                            uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;  // 512 for N=1024

        // Load and modulus switch: Torus32 -> NTT modulus
        if (tid < NUM_THREADS) {
            sh_temp[tid] = torus_to_ntt_mod<N>(in[tid]);
            sh_temp[tid + NUM_THREADS] =
                torus_to_ntt_mod<N>(in[tid + NUM_THREADS]);
        }
        __syncthreads();

        // Forward NTT
        if (tid < NUM_THREADS) {
            SmallForwardNTT<kLogLength>(sh_temp, forward_root_, tid);
        }
        else {
            for (int i = 0; i < SmallForwardNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Copy to output
        if (tid < NUM_THREADS) {
            out[tid] = sh_temp[tid];
            out[tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
        }
        __syncthreads();
    }

    /**
     * Forward NTT without modulus switching (for integer polynomials)
     *
     * Used for decomposed polynomials that are already integers
     */
    __device__ inline void NTT(Value* const out, const int32_t* const in,
                               Value* const sh_temp,
                               uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;

        // Load integer values and reduce to [0, P)
        if (tid < NUM_THREADS) {
            int32_t v0 = in[tid];
            int32_t v1 = in[tid + NUM_THREADS];
            sh_temp[tid] = signed_int_to_ntt_mod<N>(v0);
            sh_temp[tid + NUM_THREADS] = signed_int_to_ntt_mod<N>(v1);
        }
        __syncthreads();

        // Forward NTT
        if (tid < NUM_THREADS) {
            SmallForwardNTT<kLogLength>(sh_temp, forward_root_, tid);
        }
        else {
            for (int i = 0; i < SmallForwardNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Copy to output
        if (tid < NUM_THREADS) {
            out[tid] = sh_temp[tid];
            out[tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
        }
        __syncthreads();
    }

    /**
     * Inverse NTT with modulus switching (NTT domain -> Torus32)
     *
     * This performs:
     * 1. Inverse NTT in modulus P
     * 2. Modulus switch: Convert from P to 2^32 discretization
     */
    __device__ inline void NTTInvWithModSwitch(
        uint32_t* const out, const Value* const in, Value* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;
        // Load to shared
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT
        if (tid < NUM_THREADS) {
            SmallInverseNTT<kLogLength>(sh_temp, inverse_root_, n_inverse_,
                                        tid);
        }
        else {
            for (int i = 0; i < SmallInverseNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Convert to signed and apply inverse modulus switch
        if (tid < NUM_THREADS) {
            out[tid] = ntt_mod_to_torus32<N>(sh_temp[tid]);
            out[tid + NUM_THREADS] =
                ntt_mod_to_torus32<N>(sh_temp[tid + NUM_THREADS]);
        }
        __syncthreads();
    }

    /**
     * Inverse NTT with modulus switching and addition
     */
    __device__ inline void NTTInvAddWithModSwitch(
        uint32_t* const out, const Value* const in, Value* const sh_temp,
        uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;
        // Load to shared
        if (tid < NUM_THREADS) {
            sh_temp[tid] = in[tid];
            sh_temp[tid + NUM_THREADS] = in[tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT
        if (tid < NUM_THREADS) {
            SmallInverseNTT<kLogLength>(sh_temp, inverse_root_, n_inverse_,
                                        tid);
        }
        else {
            for (int i = 0; i < SmallInverseNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Convert and ADD to output
        if (tid < NUM_THREADS) {
            out[tid] += ntt_mod_to_torus32<N>(sh_temp[tid]);
            out[tid + NUM_THREADS] +=
                ntt_mod_to_torus32<N>(sh_temp[tid + NUM_THREADS]);
        }
        __syncthreads();
    }
#endif  // CUFHE_GPU_DEVICE_COMPILER
};

// gfx1201 exposes 64 KiB of LDS per workgroup.  A lvl02 bootstrap using the
// traditional layout needs 80 KiB (112 KiB for Mux), even though the
// transform itself only needs 16 KiB.  Lvl01's custom FFT also benefits from
// keeping its accumulators in registers: doing so drops the block below half
// the available LDS and removes the LDS limit on a second resident block.  On
// HIP's custom FFT and NTT paths, reuse the transform area as TLWE scratch
// after blind rotate.  CUDA retains the existing shared-memory fast path; the
// 32-bit-only tfhe-rs-style FFT retains its original layout.
template <class P>
constexpr bool USE_LOW_LDS_BOOTSTRAP =
#if defined(CUFHE_USE_HIP) && !defined(USE_BLOCK_BINARY) && \
    (!defined(USE_FFT) || defined(USE_GPU_FFT))
#if defined(USE_FFT) && defined(USE_GPU_FFT)
    P::n == TFHEpp::lvl1param::n || P::n == TFHEpp::lvl2param::n;
#else
    P::n == TFHEpp::lvl2param::n;
#endif
#else
    false;
#endif

#ifdef USE_FFT

//=============================================================================
// FFT mode: Use negacyclic FFT over double2
//=============================================================================

// NTT value type: double2 complex for FFT
using NTTValue = double2;

template <uint32_t N>
using NTTValueFor = NTTValue;

// Thread configuration: still N/2 = 512 threads per block
// (FFT uses 256 active threads, decomposition uses all 512)
constexpr uint32_t NTT_THREAD_UNITBIT = 1;

// A lvl1 GPU FFT uses only half of the 512-thread gate block.  In the
// KeyBundle path, use the other half for a second transform while retaining
// enough LDS for two resident blocks on gfx1201.
template <class P>
constexpr bool USE_PAIRED_GPU_FFT =
#if defined(CUFHE_USE_HIP) && defined(USE_GPU_FFT) && \
    defined(USE_KEY_BUNDLE) && !defined(USE_BLOCK_BINARY)
    P::n == TFHEpp::lvl1param::n;
#else
    false;
#endif

// Shared memory size per gate:
// sh_fft[N/2] = 512 × double2 = 8 KB (FFT working buffer)
// sh_accum[(k+1) × N/2] = (k+1) × 512 × double2 = 16 KB (for k=1)
// Total: 24 KB for k=1
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_WORK =
    (P::n / 2) * (USE_PAIRED_GPU_FFT<P> ? 2 : 1) * sizeof(double2);

template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_TLWE = (P::k * P::n + 1) * sizeof(typename P::T);

template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE =
    USE_LOW_LDS_BOOTSTRAP<P>
        ? (MEM4HOMGATE_WORK<P> > MEM4HOMGATE_TLWE<P> ? MEM4HOMGATE_WORK<P>
                                                     : MEM4HOMGATE_TLWE<P>)
        : ((P::n / 2) + (P::k + 1) * (P::n / 2)) * sizeof(double2);

// Dynamic shared memory size for regular gates:
// FFT workspace + one TRLWE array placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_DYN =
    MEM4HOMGATE<P> + (P::k + 1) * P::n * sizeof(typename P::T);

// Dynamic shared memory size for Mux/NMux gates.  The low-LDS path reuses one
// TRLWE array sequentially; the regular path keeps two arrays.
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4MUXGATE_DYN =
    MEM4HOMGATE<P> + (USE_LOW_LDS_BOOTSTRAP<P> ? 1 : 2) * (P::k + 1) * P::n *
                         sizeof(typename P::T);

// Number of threads for homomorphic gate (N/2 = 512 for N=1024)
template <class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> 1;

#ifdef USE_GPU_FFT

//=============================================================================
// GPU-FFT mode: Custom shared-memory FFT using GPU-FFT's table generation
//=============================================================================

#ifdef CUFHE_GPU_DEVICE_COMPILER
// double2 operator overloads for GPU-FFT path
// (These match the tfhe-rs operators but are defined here when
// fft_negacyclic.cuh is excluded)

__device__ inline double2 operator+(const double2 a, const double2 b)
{
    return {__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y)};
}

__device__ inline double2 operator-(const double2 a, const double2 b)
{
    return {__dsub_rn(a.x, b.x), __dsub_rn(a.y, b.y)};
}

__device__ inline void operator+=(double2& lh, const double2 rh)
{
    lh.x = __dadd_rn(lh.x, rh.x);
    lh.y = __dadd_rn(lh.y, rh.y);
}

__device__ __forceinline__ void complex_madd(double2& accum, const double2 a,
                                              const double2 b)
{
    accum.x = __fma_rn(a.x, b.x, accum.x);
    accum.x = __fma_rn(-a.y, b.y, accum.x);
    accum.y = __fma_rn(a.x, b.y, accum.y);
    accum.y = __fma_rn(a.y, b.x, accum.y);
}

__device__ inline double2 operator*(const double2 a, const double2 b)
{
    return {__fma_rn(a.x, b.x, -__dmul_rn(a.y, b.y)),
            __fma_rn(a.x, b.y, __dmul_rn(a.y, b.x))};
}

__device__ inline void operator*=(double2& a, const double2 b)
{
    double real = __fma_rn(a.x, b.x, -__dmul_rn(a.y, b.y));
    a.y = __fma_rn(a.x, b.y, __dmul_rn(a.y, b.x));
    a.x = real;
}

__device__ inline double2 operator*(const double2 a, double b)
{
    return {__dmul_rn(a.x, b), __dmul_rn(a.y, b)};
}

// Warp shuffle helpers for register-based warp-local FFT stages.
// Eliminates shared memory bank conflicts (up to 8-way for double2 at stride
// 1).
__device__ __forceinline__ double shfl_xor_d(double val, int mask)
{
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_xor_sync(0xFFFFFFFFULL, lo, mask);
    hi = __shfl_xor_sync(0xFFFFFFFFULL, hi, mask);
    return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double2 shfl_xor_d2(double2 val, int mask)
{
    return {shfl_xor_d(val.x, mask), shfl_xor_d(val.y, mask)};
}

/**
 * GPU-FFT Forward FFT for N/2=256 complex elements.
 * Uses 128 threads and a straightforward shared-memory Cooley-Tukey pass.
 */
__device__ __forceinline__ void GPUFFTForward256(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    if (tid < 64) {
        const double2 a = sh[tid];
        const double2 b = sh[tid + 64];
        const double2 c = sh[tid + 128];
        const double2 d = sh[tid + 192];
        const double2 w0 = __ldg(&root_table[0]);
        const double2 w1 = __ldg(&root_table[1]);

        const double2 cw = c * w0;
        const double2 dw = d * w0;
        const double2 a1 = a + cw;
        const double2 c1 = a - cw;
        const double2 b1 = b + dw;
        const double2 d1 = b - dw;

        const double2 b1w = b1 * w0;
        const double2 d1w = d1 * w1;
        sh[tid] = a1 + b1w;
        sh[tid + 64] = a1 - b1w;
        sh[tid + 128] = c1 + d1w;
        sh[tid + 192] = c1 - d1w;
    }
    __syncthreads();

    {
        const int address = ((tid >> 5) << 5) + tid;
        const double2 root = __ldg(&root_table[tid >> 5]);
        const double2 u = sh[address];
        const double2 v = sh[address + 32] * root;
        sh[address] = u + v;
        sh[address + 32] = u - v;
    }

    {
        int root_shift = 4;
        const int address = ((tid >> 4) << 4) + tid;
        double2 reg_u = sh[address];
        double2 reg_v = sh[address + 16];

        double2 root = __ldg(&root_table[tid >> root_shift]);
        double2 weighted = reg_v * root;
        reg_v = reg_u - weighted;
        reg_u = reg_u + weighted;

#pragma unroll
        for (int xor_mask = 8; xor_mask >= 1; xor_mask >>= 1) {
            root_shift--;
            const bool is_upper = (tid & xor_mask) != 0;
            const double2 sent = is_upper ? reg_u : reg_v;
            const double2 received = shfl_xor_d2(sent, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            root = __ldg(&root_table[tid >> root_shift]);
            weighted = reg_v * root;
            reg_v = reg_u - weighted;
            reg_u = reg_u + weighted;
        }

        sh[2 * tid] = reg_u;
        sh[2 * tid + 1] = reg_v;
    }
    __syncthreads();
}

/**
 * GPU-FFT Inverse FFT for N/2=256 complex elements.
 * The inverse scaling is folded into the untwist table.
 */
__device__ __forceinline__ void GPUFFTInverse256(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    double2 reg_u = sh[2 * tid];
    double2 reg_v = sh[2 * tid + 1];
    int root_shift = 0;
    double2 root = __ldg(&root_table[tid >> root_shift]);
    double2 sum = reg_u + reg_v;
    reg_v = (reg_u - reg_v) * root;
    reg_u = sum;

#pragma unroll
    for (int xor_mask = 1; xor_mask <= 8; xor_mask <<= 1) {
        root_shift++;
        const bool is_upper = (tid & xor_mask) != 0;
        const double2 sent = is_upper ? reg_u : reg_v;
        const double2 received = shfl_xor_d2(sent, xor_mask);
        if (is_upper)
            reg_u = received;
        else
            reg_v = received;

        root = __ldg(&root_table[tid >> root_shift]);
        sum = reg_u + reg_v;
        reg_v = (reg_u - reg_v) * root;
        reg_u = sum;
    }

    const int write_address = ((tid >> 4) << 4) + tid;
    sh[write_address] = reg_u;
    sh[write_address + 16] = reg_v;

    {
        const int address = ((tid >> 5) << 5) + tid;
        root = __ldg(&root_table[tid >> 5]);
        const double2 u = sh[address];
        const double2 v = sh[address + 32];
        sh[address] = u + v;
        sh[address + 32] = (u - v) * root;
    }
    __syncthreads();

    if (tid < 64) {
        const double2 a = sh[tid];
        const double2 b = sh[tid + 64];
        const double2 c = sh[tid + 128];
        const double2 d = sh[tid + 192];
        const double2 w0 = __ldg(&root_table[0]);
        const double2 w1 = __ldg(&root_table[1]);

        const double2 t0 = a + b;
        const double2 t1 = (a - b) * w0;
        const double2 t2 = c + d;
        const double2 t3 = (c - d) * w1;

        sh[tid] = t0 + t2;
        sh[tid + 128] = (t0 - t2) * w0;
        sh[tid + 64] = t1 + t3;
        sh[tid + 192] = (t1 - t3) * w0;
    }
    __syncthreads();
}

/**
 * GPU-FFT Forward FFT for N/2=512 complex elements
 * Uses 256 threads, Cooley-Tukey butterfly, 9 stages
 *
 * Root table: bit-reversed forward roots from FFNT::ReverseRootTable_ffnt()
 * Root indexing: current_root_index = omega_address >> t_2
 *
 * Radix-4 optimization: stages 0+1 merged into a single radix-4 butterfly,
 * eliminating one __syncthreads barrier.
 *
 * Sync pattern: 1 radix-4 (strides 256+128) + 1 radix-2 (stride 64) +
 * boundary (no sync) + warp-local + final sync = 3 total syncs
 */
__device__ __forceinline__ void GPUFFTForward512(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    // Radix-4 merge of stages 0+1 (strides 256+128, CT DIT)
    // 128 active threads, each handles 4 elements
    if (tid < 128) {
        double2 a = sh[tid];
        double2 b = sh[tid + 128];
        double2 c = sh[tid + 256];
        double2 d = sh[tid + 384];

        double2 w1b = __ldg(&root_table[1]);  // fine (stride 128), second pair

        // Step 1: coarse stage (stride 256)
        // root_table[0] is exactly 1 + 0i.
        double2 a1 = a + c;
        double2 c1 = a - c;
        double2 b1 = b + d;
        double2 d1 = b - d;

        // Step 2: fine stage (stride 128)
        double2 b1w = b1;
        double2 d1w = d1 * w1b;
        sh[tid] = a1 + b1w;
        sh[tid + 128] = a1 - b1w;
        sh[tid + 256] = c1 + d1w;
        sh[tid + 384] = c1 - d1w;
    }
    __syncthreads();

    // Stage 2 (stride 64): all 256 threads, normal radix-2
    {
        int in_shared_address = ((tid >> 6) << 6) + tid;
        double2 root = __ldg(&root_table[tid >> 6]);
        double2 U = sh[in_shared_address];
        double2 V = sh[in_shared_address + 64] * root;
        sh[in_shared_address] = U + V;
        sh[in_shared_address + 64] = U - V;
    }
    __syncthreads();

    // Boundary stage (stride 32): no sync needed after — stride-16 reads
    // are warp-local
    {
        int in_shared_address = ((tid >> 5) << 5) + tid;
        double2 root = __ldg(&root_table[tid >> 5]);
        double2 U = sh[in_shared_address];
        double2 V = sh[in_shared_address + 32] * root;
        sh[in_shared_address] = U + V;
        sh[in_shared_address + 32] = U - V;
    }

    // Warp-local stages (stride 16..1): register-based with warp shuffle.
    {
        int t_2 = 4;
        int in_shared_address = ((tid >> 4) << 4) + tid;

        double2 reg_u = sh[in_shared_address];
        double2 reg_v = sh[in_shared_address + 16];

        // Stride 16: butterfly pair already in registers
        {
            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 Vw = reg_v * root;
            reg_v = reg_u - Vw;
            reg_u = reg_u + Vw;
        }

        // Strides 8, 4, 2, 1: shuffle exchange then butterfly
#pragma unroll
        for (int xor_mask = 8; xor_mask >= 1; xor_mask >>= 1) {
            t_2 -= 1;
            bool is_upper = (tid & xor_mask) != 0;
            double2 to_send = is_upper ? reg_u : reg_v;
            double2 received = shfl_xor_d2(to_send, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 Vw = reg_v * root;
            reg_v = reg_u - Vw;
            reg_u = reg_u + Vw;
        }

        sh[2 * tid] = reg_u;
        sh[2 * tid + 1] = reg_v;
    }
    __syncthreads();
}

/**
 * GPU-FFT Inverse FFT for N/2=512 complex elements
 * Uses 256 threads, Gentleman-Sande butterfly, 9 stages
 *
 * Root table: bit-reversed inverse roots from
 * FFNT::InverseReverseRootTable_ffnt() Root indexing: current_root_index = m +
 * (tid >> t_2)
 *
 * n_inverse (1/512) is folded into the untwist table, so no separate scaling
 * pass is needed here.
 *
 * Radix-4 optimization: stages 7+8 (strides 128+256) merged into a single
 * GS radix-4 butterfly, eliminating one __syncthreads barrier.
 *
 * Sync pattern: warp-local + boundary [sync] + stride-64 [sync] +
 * radix-4 strides 128+256 [sync] = 3 total syncs
 */
__device__ __forceinline__ void GPUFFTInverse512(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    int t_2 = 0;
    int t = 1;

    int in_shared_address = ((tid >> 0) << 0) + tid;

    // Warp-local stages (stride 1..16): register-based with warp shuffle.
    {
        double2 reg_u = sh[in_shared_address];
        double2 reg_v = sh[in_shared_address + t];

        // Stride 1: butterfly pair already in registers
        {
            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 sum = reg_u + reg_v;
            reg_v = (reg_u - reg_v) * root;
            reg_u = sum;
        }

        // Strides 2, 4, 8, 16: shuffle exchange then butterfly
#pragma unroll
        for (int xor_mask = 1; xor_mask <= 8; xor_mask <<= 1) {
            t_2 += 1;
            bool is_upper = (tid & xor_mask) != 0;
            double2 to_send = is_upper ? reg_u : reg_v;
            double2 received = shfl_xor_d2(to_send, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 sum = reg_u + reg_v;
            reg_v = (reg_u - reg_v) * root;
            reg_u = sum;
        }

        // Write back (stride-16 addressing for boundary stage)
        int wb_addr = ((tid >> 4) << 4) + tid;
        sh[wb_addr] = reg_u;
        sh[wb_addr + 16] = reg_v;
    }

    // Stage 5 (stride 32): boundary stage. No sync needed before because
    // stride-32 reads only access data written by same-warp threads.
    {
        int in_shared_address = ((tid >> 5) << 5) + tid;
        double2 root = __ldg(&root_table[tid >> 5]);
        double2 u = sh[in_shared_address];
        double2 v = sh[in_shared_address + 32];
        sh[in_shared_address] = u + v;
        sh[in_shared_address + 32] = (u - v) * root;
    }
    __syncthreads();

    // Stage 6 (stride 64): all 256 threads, normal GS radix-2
    {
        int in_shared_address = ((tid >> 6) << 6) + tid;
        double2 root = __ldg(&root_table[tid >> 6]);
        double2 u = sh[in_shared_address];
        double2 v = sh[in_shared_address + 64];
        sh[in_shared_address] = u + v;
        sh[in_shared_address + 64] = (u - v) * root;
    }
    __syncthreads();

    // Radix-4 merge of stages 7+8 (strides 128+256, GS DIF)
    // 128 active threads, each handles 4 elements
    if (tid < 128) {
        double2 a = sh[tid];
        double2 b = sh[tid + 128];
        double2 c = sh[tid + 256];
        double2 d = sh[tid + 384];

        double2 w_s2 = __ldg(&root_table[1]);  // stride 128, second pair

        // Step 1: GS at stride 128
        double2 t0 = a + b;
        double2 t1 = a - b;  // root_table[0] is exactly 1 + 0i
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 256
        sh[tid] = t0 + t2;
        sh[tid + 256] = t0 - t2;
        sh[tid + 128] = t1 + t3;
        sh[tid + 384] = t1 - t3;
    }
    __syncthreads();
}

/**
 * GPU-FFT Forward FFT for N/2=1024 complex elements
 * Uses 512 threads, Cooley-Tukey butterfly, 10 stages (log2(1024)=10)
 *
 * Radix-4 optimization: stages 0+1 and 2+3 each merged into radix-4
 * butterflies, eliminating two __syncthreads barriers.
 *
 * Sync pattern: 2 radix-4 stages + boundary (no sync) + warp-local +
 * final sync = 3 total syncs
 */
__device__ __forceinline__ void GPUFFTForward1024(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    // Radix-4 merge of stages 0+1 (strides 512+256, CT DIT)
    // 256 active threads out of 512, each handles 4 elements
    if (tid < 256) {
        double2 a = sh[tid];
        double2 b = sh[tid + 256];
        double2 c = sh[tid + 512];
        double2 d = sh[tid + 768];

        double2 w0 =
            __ldg(&root_table[0]);  // coarse (stride 512) + fine first pair
        double2 w1b = __ldg(&root_table[1]);  // fine (stride 256), second pair

        // Step 1: coarse stage (stride 512)
        double2 cw = c * w0;
        double2 dw = d * w0;
        double2 a1 = a + cw;
        double2 c1 = a - cw;
        double2 b1 = b + dw;
        double2 d1 = b - dw;

        // Step 2: fine stage (stride 256)
        double2 b1w = b1 * w0;  // w1a = w0 = root_table[0]
        double2 d1w = d1 * w1b;
        sh[tid] = a1 + b1w;
        sh[tid + 256] = a1 - b1w;
        sh[tid + 512] = c1 + d1w;
        sh[tid + 768] = c1 - d1w;
    }
    __syncthreads();

    // Radix-4 merge of stages 2+3 (strides 128+64, CT DIT)
    // 256 active threads, each handles 4 elements in groups of 256
    if (tid < 256) {
        int group = tid >> 6;  // 0..3
        int local = tid & 63;  // 0..63
        int base = group * 256 + local;

        double2 a = sh[base];
        double2 b = sh[base + 64];
        double2 c = sh[base + 128];
        double2 d = sh[base + 192];

        double2 w0 = __ldg(&root_table[group]);  // coarse (stride 128)
        double2 w1a =
            __ldg(&root_table[2 * group]);  // fine (stride 64), first pair
        double2 w1b =
            __ldg(&root_table[2 * group + 1]);  // fine (stride 64), second pair

        // Step 1: coarse stage (stride 128)
        double2 cw = c * w0;
        double2 dw = d * w0;
        double2 a1 = a + cw;
        double2 c1 = a - cw;
        double2 b1 = b + dw;
        double2 d1 = b - dw;

        // Step 2: fine stage (stride 64)
        double2 b1w = b1 * w1a;
        double2 d1w = d1 * w1b;
        sh[base] = a1 + b1w;
        sh[base + 64] = a1 - b1w;
        sh[base + 128] = c1 + d1w;
        sh[base + 192] = c1 - d1w;
    }
    __syncthreads();

    // Boundary stage (stride 32): all 512 threads, no sync after —
    // stride-16 reads are warp-local
    {
        int in_shared_address = ((tid >> 5) << 5) + tid;
        double2 root = __ldg(&root_table[tid >> 5]);
        double2 U = sh[in_shared_address];
        double2 V = sh[in_shared_address + 32] * root;
        sh[in_shared_address] = U + V;
        sh[in_shared_address + 32] = U - V;
    }

    // Warp-local stages (stride 16..1): register-based with warp shuffle.
    {
        int t_2 = 4;
        int in_shared_address = ((tid >> 4) << 4) + tid;

        double2 reg_u = sh[in_shared_address];
        double2 reg_v = sh[in_shared_address + 16];

        // Stride 16: butterfly pair already in registers
        {
            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 Vw = reg_v * root;
            reg_v = reg_u - Vw;
            reg_u = reg_u + Vw;
        }

        // Strides 8, 4, 2, 1: shuffle exchange then butterfly
#pragma unroll
        for (int xor_mask = 8; xor_mask >= 1; xor_mask >>= 1) {
            t_2 -= 1;
            bool is_upper = (tid & xor_mask) != 0;
            double2 to_send = is_upper ? reg_u : reg_v;
            double2 received = shfl_xor_d2(to_send, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 Vw = reg_v * root;
            reg_v = reg_u - Vw;
            reg_u = reg_u + Vw;
        }

        sh[2 * tid] = reg_u;
        sh[2 * tid + 1] = reg_v;
    }
    __syncthreads();
}

/**
 * GPU-FFT Inverse FFT for N/2=1024 complex elements
 * Uses 512 threads, Gentleman-Sande butterfly, 10 stages
 *
 * n_inverse (1/1024) is folded into the untwist table.
 *
 * Radix-4 optimization: stages 6+7 and 8+9 each merged into GS radix-4
 * butterflies, eliminating two __syncthreads barriers.
 *
 * Sync pattern: warp-local + boundary [sync] + radix-4 strides 64+128
 * [sync] + radix-4 strides 256+512 [sync] = 3 total syncs
 */
__device__ __forceinline__ void GPUFFTInverse1024(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    int t_2 = 0;
    int t = 1;

    int in_shared_address = ((tid >> 0) << 0) + tid;

    // Warp-local stages (stride 1..16): register-based with warp shuffle.
    {
        double2 reg_u = sh[in_shared_address];
        double2 reg_v = sh[in_shared_address + t];

        // Stride 1: butterfly pair already in registers
        {
            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 sum = reg_u + reg_v;
            reg_v = (reg_u - reg_v) * root;
            reg_u = sum;
        }

        // Strides 2, 4, 8, 16: shuffle exchange then butterfly
#pragma unroll
        for (int xor_mask = 1; xor_mask <= 8; xor_mask <<= 1) {
            t_2 += 1;
            bool is_upper = (tid & xor_mask) != 0;
            double2 to_send = is_upper ? reg_u : reg_v;
            double2 received = shfl_xor_d2(to_send, xor_mask);
            if (is_upper)
                reg_u = received;
            else
                reg_v = received;

            double2 root = __ldg(&root_table[tid >> t_2]);
            double2 sum = reg_u + reg_v;
            reg_v = (reg_u - reg_v) * root;
            reg_u = sum;
        }

        int wb_addr = ((tid >> 4) << 4) + tid;
        sh[wb_addr] = reg_u;
        sh[wb_addr + 16] = reg_v;
    }

    // Stage 5 (stride 32): boundary stage. No sync needed before —
    // stride-32 reads only access data written by same-warp threads.
    {
        int in_shared_address = ((tid >> 5) << 5) + tid;
        double2 root = __ldg(&root_table[tid >> 5]);
        double2 u = sh[in_shared_address];
        double2 v = sh[in_shared_address + 32];
        sh[in_shared_address] = u + v;
        sh[in_shared_address + 32] = (u - v) * root;
    }
    __syncthreads();

    // Radix-4 merge of stages 6+7 (strides 64+128, GS DIF)
    // 256 active threads out of 512, each handles 4 elements in groups of 256
    if (tid < 256) {
        int group = tid >> 6;  // 0..3
        int local = tid & 63;  // 0..63
        int base = group * 256 + local;

        double2 a = sh[base];
        double2 b = sh[base + 64];
        double2 c = sh[base + 128];
        double2 d = sh[base + 192];

        double2 w_s1 = __ldg(&root_table[2 * group]);  // stride 64, first pair
        double2 w_s2 =
            __ldg(&root_table[2 * group + 1]);     // stride 64, second pair
        double2 w_2s = __ldg(&root_table[group]);  // stride 128

        // Step 1: GS at stride 64
        double2 t0 = a + b;
        double2 t1 = (a - b) * w_s1;
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 128
        sh[base] = t0 + t2;
        sh[base + 128] = (t0 - t2) * w_2s;
        sh[base + 64] = t1 + t3;
        sh[base + 192] = (t1 - t3) * w_2s;
    }
    __syncthreads();

    // Radix-4 merge of stages 8+9 (strides 256+512, GS DIF)
    // 256 active threads, each handles 4 elements
    if (tid < 256) {
        double2 a = sh[tid];
        double2 b = sh[tid + 256];
        double2 c = sh[tid + 512];
        double2 d = sh[tid + 768];

        double2 w_s =
            __ldg(&root_table[0]);  // stride 256, first pair + stride 512
        double2 w_s2 = __ldg(&root_table[1]);  // stride 256, second pair

        // Step 1: GS at stride 256
        double2 t0 = a + b;
        double2 t1 = (a - b) * w_s;  // w_s1 = root_table[0]
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 512
        sh[tid] = t0 + t2;
        sh[tid + 512] = (t0 - t2) * w_s;  // w_2s = root_table[0]
        sh[tid + 256] = t1 + t3;
        sh[tid + 768] = (t1 - t3) * w_s;  // w_2s = root_table[0]
    }
    __syncthreads();
}

template <uint32_t N>
__host__ __device__ constexpr int GPUFFTSharedSyncCount()
{
    static_assert(N == 512 || N == 1024 || N == 2048,
                  "Unsupported GPU-FFT polynomial degree");
    return N == 512 ? 2 : 3;
}

template <uint32_t N>
__device__ __forceinline__ void GPUFFTForward(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    if constexpr (N == 512)
        GPUFFTForward256(sh, root_table, tid);
    else if constexpr (N == 1024)
        GPUFFTForward512(sh, root_table, tid);
    else if constexpr (N == 2048)
        GPUFFTForward1024(sh, root_table, tid);
    else
        static_assert(N == 512 || N == 1024 || N == 2048,
                      "Unsupported GPU-FFT polynomial degree");
}

template <uint32_t N>
__device__ __forceinline__ void GPUFFTInverse(
    double2* sh, const double2* __restrict__ root_table, int tid)
{
    if constexpr (N == 512)
        GPUFFTInverse256(sh, root_table, tid);
    else if constexpr (N == 1024)
        GPUFFTInverse512(sh, root_table, tid);
    else if constexpr (N == 2048)
        GPUFFTInverse1024(sh, root_table, tid);
    else
        static_assert(N == 512 || N == 1024 || N == 2048,
                      "Unsupported GPU-FFT polynomial degree");
}

#endif  // CUFHE_GPU_DEVICE_COMPILER

/**
 * CuGPUFFTHandler - handler for GPU-FFT library FFT
 *
 * Stores forward/inverse root tables and twist/untwist tables generated by
 * gpufft::FFNT<Float64>. Each table has N/2 = 512 complex (double2) entries.
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuGPUFFTHandler {
   public:
    static constexpr uint32_t kLength = length;
    static constexpr uint32_t kHalfLength = length >> 1;

    double2* forward_root_;  // N/2 forward roots (bit-reversed)
    double2* inverse_root_;  // N/2 inverse roots (bit-reversed)
    double2* twist_;         // N/2 twist factors
    double2* untwist_;       // N/2 untwist factors (scaled by n_inverse)

    __host__ __device__ CuGPUFFTHandler()
        : forward_root_(nullptr),
          inverse_root_(nullptr),
          twist_(nullptr),
          untwist_(nullptr)
    {
    }
    __host__ __device__ ~CuGPUFFTHandler() {}

    __host__ static void Create();
    __host__ static void CreateConstant() {}
    __host__ static void Destroy();
    __host__ void SetDevicePointers(int device_id);
};

template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuGPUFFTHandler<length>;

#else  // !USE_GPU_FFT (tfhe-rs FFT)

//=============================================================================
// tfhe-rs FFT mode
// (fft_negacyclic.cuh included above, outside namespace cufhe)
//=============================================================================

/**
 * CuFFTHandler - replacement for CuNTTHandler when using tfhe-rs negacyclic FFT
 *
 * The twiddle factors are stored in device memory (__device__ negtwiddles[])
 * rather than in handler-specific allocations, so the handler is mostly a
 * no-op placeholder to maintain API compatibility.
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuFFTHandler {
   public:
    static constexpr uint32_t kLength = length;

    __host__ __device__ CuFFTHandler() {}
    __host__ __device__ ~CuFFTHandler() {}

    __host__ static void Create() {}
    __host__ static void CreateConstant() {}
    __host__ static void Destroy() {}
    __host__ void SetDevicePointers(int device_id) {}
};

template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuFFTHandler<length>;

template <uint32_t N>
__host__ __device__ constexpr int TfheRsFFTSharedSyncCount()
{
    return 2 * HalfDegree<Degree<N> >::log2_degree - 7;
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT

//=============================================================================
// NTT mode: use a modulus selected for each supported ring size
//=============================================================================

// Thread configuration for NTT
// N/2 threads, each handles 2 elements (e.g., 512 threads for N=1024)
constexpr uint32_t NTT_THREAD_UNITBIT = 1;

template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuSmallNTTHandler<length>;

// NTT storage follows the selected modulus: 32-bit for lvl1 and 64-bit for
// lvl02's Goldilocks prime. NTTValue remains the lvl1 API shorthand.
template <uint32_t N>
using NTTValueFor = SmallNTTValueFor<N>;
using NTTValue = NTTValueFor<TFHEpp::lvl1param::n>;

// The lvl02 low-LDS path requires register-resident accumulators to fit the
// R9700's per-workgroup LDS limit.  lvl01 does not require them for capacity,
// but benefits from avoiding an LDS read/write for every pointwise MAC.  Keep
// CUDA and block-binary builds on their established shared-memory layout.
template <class P>
constexpr bool USE_REGISTER_NTT_ACCUM =
#if defined(CUFHE_USE_HIP) && !defined(USE_BLOCK_BINARY)
    P::n == TFHEpp::lvl1param::n || USE_LOW_LDS_BOOTSTRAP<P>;
#else
    USE_LOW_LDS_BOOTSTRAP<P>;
#endif

// Shared memory size per gate: (k+2) * N field elements normally.  The
// register-accumulator path needs one N-element transform buffer plus
// extracted-TLWE scratch, with the larger determining the reusable prefix.
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_WORK = P::n * sizeof(NTTValueFor<P::n>);

template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_TLWE = (P::k * P::n + 1) * sizeof(typename P::T);

template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE =
    USE_REGISTER_NTT_ACCUM<P>
        ? (MEM4HOMGATE_WORK<P> > MEM4HOMGATE_TLWE<P> ? MEM4HOMGATE_WORK<P>
                                                     : MEM4HOMGATE_TLWE<P>)
        : (P::k + 2) * P::n * sizeof(NTTValueFor<P::n>);

// Dynamic shared memory size for regular gates:
// NTT workspace + one TRLWE array placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_DYN =
    MEM4HOMGATE<P> + (P::k + 1) * P::n * sizeof(typename P::T);

// Dynamic shared memory size for Mux/NMux gates.  The low-LDS path reuses one
// TRLWE array sequentially; the regular path keeps two arrays.
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4MUXGATE_DYN =
    MEM4HOMGATE<P> + (USE_LOW_LDS_BOOTSTRAP<P> ? 1 : 2) * (P::k + 1) * P::n *
                         sizeof(typename P::T);

// Number of threads for NTT (N/2 = 512 for N=1024)
template <class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> 1;

#endif  // USE_FFT

#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
extern std::vector<NTTValue*> xai_ntt_devs;
extern std::vector<NTTValue*> one_trgsw_ntt_devs;
#ifdef USE_BLOCK_BINARY
extern __device__ NTTValue* block_xai_fft;
extern __device__ NTTValueFor<TFHEpp::lvl2param::n>* block_xai_fft_lvl02;
#endif

void InitializeXaiNTT(const int gpuNum);
void InitializeOneTRGSWNTT(const int gpuNum);
void DeleteXaiNTT();
void DeleteOneTRGSWNTT();

// lvl02 (N=2048) key-bundle tables
extern std::vector<NTTValueFor<TFHEpp::lvl2param::n>*> xai_ntt_devs_lvl02;
extern std::vector<NTTValueFor<TFHEpp::lvl2param::n>*> one_trgsw_ntt_devs_lvl02;

void InitializeXaiNTT_lvl02(const int gpuNum);
void InitializeOneTRGSWNTT_lvl02(const int gpuNum);
void DeleteXaiNTT_lvl02();
void DeleteOneTRGSWNTT_lvl02();
#endif

}  // namespace cufhe
