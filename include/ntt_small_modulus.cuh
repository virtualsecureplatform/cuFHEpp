/**
 * Goldilocks-prime NTT implementation for cuFHE
 *
 * This implements the RAINTT-style approach where:
 * - Torus discretization switches to the NTT modulus before
 * multiplication
 * - After INTT, results are converted back to the torus discretization
 *
 * Current modulus: P = 2^64 - 2^32 + 1
 * - Close to 64 bits, matching lvl2 Torus64 precision much better than the
 *   previous 31-bit modulus
 * - P - 1 is divisible by 2^32, so both lvl1 (N=1024) and lvl2 (N=2048)
 *   negacyclic NTTs have the required primitive 2N-th root of unity
 */

#pragma once

#include <cuda_runtime.h>

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

//=============================================================================
// Goldilocks NTT Constants (RAINTT-style)
//=============================================================================

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

using SmallNTTValue = small_ntt::Value;

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
// Goldilocks Modulus Operations
//=============================================================================

__host__ __device__ __forceinline__ SmallNTTValue small_mod_normalize(
    SmallNTTValue a)
{
    return a + static_cast<uint32_t>(-(a >= small_ntt::P));
}

__host__ __device__ __forceinline__ SmallNTTValue small_mod_add(
    SmallNTTValue a, SmallNTTValue b)
{
    SmallNTTValue tmp = a + b;
    return tmp + static_cast<uint32_t>(-(tmp < b || tmp >= small_ntt::P));
}

__host__ __device__ __forceinline__ SmallNTTValue small_mod_sub(
    SmallNTTValue a, SmallNTTValue b)
{
    SmallNTTValue tmp = a - b;
    return tmp - static_cast<uint32_t>(-(tmp > a));
}

__host__ __device__ __forceinline__ SmallNTTValue small_mod_mult(
    SmallNTTValue a, SmallNTTValue b)
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

    SmallNTTValue res =
        ((static_cast<SmallNTTValue>(limb1) + limb2) << 32) + limb0 - limb3 -
        limb2;
    res -= static_cast<uint32_t>(-((res > lo) && (limb2 == 0)));
    res += static_cast<uint32_t>(-((res < lo) && (limb2 != 0)));
    return small_mod_normalize(res);
}

template <typename TorusT>
__host__ __device__ __forceinline__ SmallNTTValue torus_to_ntt_mod(
    TorusT torus_val)
{
    using UnsignedT = std::make_unsigned_t<TorusT>;
    constexpr int bits = std::numeric_limits<UnsignedT>::digits;
    const auto a = static_cast<UnsignedT>(torus_val);
    unsigned __int128 prod =
        static_cast<unsigned __int128>(a) * small_ntt::P;
    prod += static_cast<unsigned __int128>(1) << (bits - 1);
    return static_cast<SmallNTTValue>(prod >> bits);
}

__host__ __device__ __forceinline__ SmallNTTValue torus32_to_ntt_mod(
    uint32_t torus_val)
{
    return torus_to_ntt_mod<uint32_t>(torus_val);
}

__host__ __device__ __forceinline__ SmallNTTValue torus64_to_ntt_mod(
    uint64_t torus_val)
{
    return torus_to_ntt_mod<uint64_t>(torus_val);
}

__host__ __device__ __forceinline__ SmallNTTValue signed_int_to_ntt_mod(
    int32_t val)
{
    if (val < 0) {
        return small_mod_sub(
            0, static_cast<SmallNTTValue>(-static_cast<int64_t>(val)));
    }
    return static_cast<SmallNTTValue>(val);
}

__host__ __device__ __forceinline__ uint64_t ntt_abs_to_torus64(
    SmallNTTValue val)
{
    const unsigned __int128 mul = val;
    return static_cast<uint64_t>(((mul << 64) + (mul << 32) - mul +
                                  (static_cast<unsigned __int128>(1) << 63)) >>
                                 64);
}

__host__ __device__ __forceinline__ uint64_t ntt_mod_to_torus64(
    SmallNTTValue val)
{
    if (val > small_ntt::HALF_P) {
        const uint64_t mag = ntt_abs_to_torus64(small_ntt::P - val);
        return static_cast<uint64_t>(-mag);
    }
    return ntt_abs_to_torus64(val);
}

__host__ __device__ __forceinline__ uint32_t ntt_mod_to_torus32(
    SmallNTTValue val)
{
    const bool neg = val > small_ntt::HALF_P;
    const uint64_t torus64 =
        ntt_abs_to_torus64(neg ? (small_ntt::P - val) : val);
    uint32_t torus32 = static_cast<uint32_t>(torus64 >> 32);
    torus32 += static_cast<uint32_t>((torus64 & 0xFFFFFFFFULL) >=
                                     0x80000000ULL);
    return neg ? static_cast<uint32_t>(-torus32) : torus32;
}

#ifdef __CUDACC__

// Cooley-Tukey butterfly for forward NTT
__device__ __forceinline__ void SmallCooleyTukeyUnit(SmallNTTValue& U,
                                                     SmallNTTValue& V,
                                                     SmallNTTValue root)
{
    SmallNTTValue u = U;
    SmallNTTValue v = small_mod_mult(V, root);
    U = small_mod_add(u, v);
    V = small_mod_sub(u, v);
}

// Gentleman-Sande butterfly for inverse NTT
__device__ __forceinline__ void SmallGentlemanSandeUnit(SmallNTTValue& U,
                                                        SmallNTTValue& V,
                                                        SmallNTTValue root)
{
    SmallNTTValue u = U;
    SmallNTTValue v = V;
    U = small_mod_add(u, v);
    V = small_mod_mult(small_mod_sub(u, v), root);
}

template <int N_POWER>
__device__ __forceinline__ void SmallForwardNTT(SmallNTTValue* sh,
                                                const SmallNTTValue* root_table,
                                                int tid)
{
    static_assert(N_POWER >= 6, "NTT length must be at least 64");

    int t_2 = N_POWER - 1;
    int t_ = N_POWER - 1;
    int m = 1;
    int t = 1 << t_;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

#pragma unroll
    for (int lp = 0; lp < N_POWER - 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallCooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
                             __ldg(&root_table[current_root_index]));

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

#pragma unroll
    for (int lp = 0; lp < 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallCooleyTukeyUnit(sh[in_shared_address], sh[in_shared_address + t],
                             __ldg(&root_table[current_root_index]));

        t = t >> 1;
        t_2 -= 1;
        t_ -= 1;
        m <<= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
    }
    __syncthreads();
}

template <int N_POWER>
__device__ __forceinline__ void SmallInverseNTT(
    SmallNTTValue* sh, const SmallNTTValue* root_table,
    SmallNTTValue n_inverse, int tid)
{
    static_assert(N_POWER >= 6, "NTT length must be at least 64");
    constexpr int NUM_THREADS = 1 << (N_POWER - 1);

    int t_2 = 0;
    int t_ = 0;
    int m = 1 << (N_POWER - 1);
    int t = 1;

    int in_shared_address = ((tid >> t_) << t_) + tid;
    int current_root_index;

#pragma unroll
    for (int lp = 0; lp < 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit(sh[in_shared_address],
                                sh[in_shared_address + t],
                                __ldg(&root_table[current_root_index]));

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
    }
    __syncthreads();

#pragma unroll
    for (int lp = 0; lp < N_POWER - 6; lp++) {
        current_root_index = m + (tid >> t_2);
        SmallGentlemanSandeUnit(sh[in_shared_address],
                                sh[in_shared_address + t],
                                __ldg(&root_table[current_root_index]));

        t = t << 1;
        t_2 += 1;
        t_ += 1;
        m >>= 1;
        in_shared_address = ((tid >> t_) << t_) + tid;
        __syncthreads();
    }

    sh[tid] = small_mod_mult(sh[tid], n_inverse);
    sh[tid + NUM_THREADS] = small_mod_mult(sh[tid + NUM_THREADS], n_inverse);
    __syncthreads();
}

template <uint32_t N>
__host__ __device__ constexpr int SmallForwardNTTSyncCount()
{
    return static_cast<int>(SmallLog2<N>()) - 5;
}

template <uint32_t N>
__host__ __device__ constexpr int SmallInverseNTTSyncCount()
{
    return static_cast<int>(SmallLog2<N>()) - 4;
}

#endif  // __CUDACC__

//=============================================================================
// Small Modulus NTT Handler
//=============================================================================

// Host-side storage for small NTT parameters
struct SmallNTTParams {
    SmallNTTValue* forward_root;
    SmallNTTValue* inverse_root;
    SmallNTTValue n_inverse;
    bool initialized;
};

extern std::vector<SmallNTTParams> g_small_ntt_params;
extern std::vector<SmallNTTParams> g_small_ntt_params_lvl02;

/**
 * Goldilocks-prime NTT Handler for cuFHE
 *
 * Before NTT, coefficients are modulus-switched from the torus to P. After
 * INTT, coefficients are centered and switched back to Torus32/Torus64.
 */
template <uint32_t length = TFHEpp::lvl1param::n>
class CuSmallNTTHandler {
   public:
    static constexpr uint32_t kLength = length;
    static constexpr uint32_t kLogLength = []() constexpr {
        uint32_t n = length, log = 0;
        while (n > 1) {
            n >>= 1;
            ++log;
        }
        return log;
    }();

    SmallNTTValue* forward_root_;
    SmallNTTValue* inverse_root_;
    SmallNTTValue n_inverse_;

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

#ifdef __CUDACC__
    /**
     * Forward NTT with modulus switching (Torus32 -> NTT domain)
     *
     * This performs:
     * 1. Modulus switch: Convert input from 2^32 to P discretization
     * 2. Forward NTT in modulus P
     */
    template <typename TorusT>
    __device__ inline void NTTWithModSwitch(SmallNTTValue* const out,
                                            const TorusT* const in,
                                            SmallNTTValue* const sh_temp,
                                            uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;  // 512 for N=1024

        // Load and modulus switch: Torus32 -> NTT modulus
        if (tid < NUM_THREADS) {
            sh_temp[tid] = torus_to_ntt_mod(in[tid]);
            sh_temp[tid + NUM_THREADS] =
                torus_to_ntt_mod(in[tid + NUM_THREADS]);
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
    __device__ inline void NTT(SmallNTTValue* const out, const int32_t* const in,
                               SmallNTTValue* const sh_temp,
                               uint32_t leading_thread = 0) const
    {
        const int tid = threadIdx.x - leading_thread;
        constexpr int N = length;
        constexpr int NUM_THREADS = N >> 1;

        // Load integer values and reduce to [0, P)
        if (tid < NUM_THREADS) {
            int32_t v0 = in[tid];
            int32_t v1 = in[tid + NUM_THREADS];
            sh_temp[tid] = signed_int_to_ntt_mod(v0);
            sh_temp[tid + NUM_THREADS] = signed_int_to_ntt_mod(v1);
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
        uint32_t* const out, const SmallNTTValue* const in,
        SmallNTTValue* const sh_temp,
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
            SmallInverseNTT<kLogLength>(sh_temp, inverse_root_, n_inverse_, tid);
        }
        else {
            for (int i = 0; i < SmallInverseNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Convert to signed and apply inverse modulus switch
        if (tid < NUM_THREADS) {
            out[tid] = ntt_mod_to_torus32(sh_temp[tid]);
            out[tid + NUM_THREADS] =
                ntt_mod_to_torus32(sh_temp[tid + NUM_THREADS]);
        }
        __syncthreads();
    }

    /**
     * Inverse NTT with modulus switching and addition
     */
    __device__ inline void NTTInvAddWithModSwitch(
        uint32_t* const out, const SmallNTTValue* const in,
        SmallNTTValue* const sh_temp,
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
            SmallInverseNTT<kLogLength>(sh_temp, inverse_root_, n_inverse_, tid);
        }
        else {
            for (int i = 0; i < SmallInverseNTTSyncCount<N>(); i++)
                __syncthreads();
        }

        // Convert and ADD to output
        if (tid < NUM_THREADS) {
            out[tid] += ntt_mod_to_torus32(sh_temp[tid]);
            out[tid + NUM_THREADS] +=
                ntt_mod_to_torus32(sh_temp[tid + NUM_THREADS]);
        }
        __syncthreads();
    }
#endif  // __CUDACC__
};

#ifdef USE_FFT

//=============================================================================
// FFT mode: Use negacyclic FFT over double2
//=============================================================================

// NTT value type: double2 complex for FFT
using NTTValue = double2;

// Thread configuration: still N/2 = 512 threads per block
// (FFT uses 256 active threads, decomposition uses all 512)
constexpr uint32_t NTT_THREAD_UNITBIT = 1;

// Shared memory size per gate:
// sh_fft[N/2] = 512 × double2 = 8 KB (FFT working buffer)
// sh_accum[(k+1) × N/2] = (k+1) × 512 × double2 = 16 KB (for k=1)
// Total: 24 KB for k=1
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE =
    ((P::n / 2) + (P::k + 1) * (P::n / 2)) * sizeof(double2);

// Dynamic shared memory size for regular gates:
// FFT workspace + one TRLWE array placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_DYN =
    MEM4HOMGATE<P> + (P::k + 1) * P::n * sizeof(typename P::T);

// Dynamic shared memory size for Mux/NMux gates:
// FFT workspace + two TRLWE arrays placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4MUXGATE_DYN =
    MEM4HOMGATE<P> + 2 * (P::k + 1) * P::n * sizeof(typename P::T);

// Number of threads for homomorphic gate (N/2 = 512 for N=1024)
template <class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> 1;

#ifdef USE_GPU_FFT

//=============================================================================
// GPU-FFT mode: Custom shared-memory FFT using GPU-FFT's table generation
//=============================================================================

#ifdef __CUDACC__
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
// Eliminates shared memory bank conflicts (up to 8-way for double2 at stride 1).
__device__ __forceinline__ double shfl_xor_d(double val, int mask)
{
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    lo = __shfl_xor_sync(0xFFFFFFFF, lo, mask);
    hi = __shfl_xor_sync(0xFFFFFFFF, hi, mask);
    return __hiloint2double(hi, lo);
}

__device__ __forceinline__ double2 shfl_xor_d2(double2 val, int mask)
{
    return {shfl_xor_d(val.x, mask), shfl_xor_d(val.y, mask)};
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

        double2 w0 = __ldg(&root_table[0]);   // coarse (stride 256) + fine first pair
        double2 w1b = __ldg(&root_table[1]);   // fine (stride 128), second pair

        // Step 1: coarse stage (stride 256)
        double2 cw = c * w0;
        double2 dw = d * w0;
        double2 a1 = a + cw;
        double2 c1 = a - cw;
        double2 b1 = b + dw;
        double2 d1 = b - dw;

        // Step 2: fine stage (stride 128)
        double2 b1w = b1 * w0;    // w1a = w0 = root_table[0]
        double2 d1w = d1 * w1b;
        sh[tid]       = a1 + b1w;
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

        double2 w_s  = __ldg(&root_table[0]);   // stride 128, first pair + stride 256
        double2 w_s2 = __ldg(&root_table[1]);   // stride 128, second pair

        // Step 1: GS at stride 128
        double2 t0 = a + b;
        double2 t1 = (a - b) * w_s;     // w_s1 = root_table[0]
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 256
        sh[tid]       = t0 + t2;
        sh[tid + 256] = (t0 - t2) * w_s;   // w_2s = root_table[0]
        sh[tid + 128] = t1 + t3;
        sh[tid + 384] = (t1 - t3) * w_s;   // w_2s = root_table[0]
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

        double2 w0 = __ldg(&root_table[0]);   // coarse (stride 512) + fine first pair
        double2 w1b = __ldg(&root_table[1]);   // fine (stride 256), second pair

        // Step 1: coarse stage (stride 512)
        double2 cw = c * w0;
        double2 dw = d * w0;
        double2 a1 = a + cw;
        double2 c1 = a - cw;
        double2 b1 = b + dw;
        double2 d1 = b - dw;

        // Step 2: fine stage (stride 256)
        double2 b1w = b1 * w0;    // w1a = w0 = root_table[0]
        double2 d1w = d1 * w1b;
        sh[tid]       = a1 + b1w;
        sh[tid + 256] = a1 - b1w;
        sh[tid + 512] = c1 + d1w;
        sh[tid + 768] = c1 - d1w;
    }
    __syncthreads();

    // Radix-4 merge of stages 2+3 (strides 128+64, CT DIT)
    // 256 active threads, each handles 4 elements in groups of 256
    if (tid < 256) {
        int group = tid >> 6;        // 0..3
        int local = tid & 63;        // 0..63
        int base = group * 256 + local;

        double2 a = sh[base];
        double2 b = sh[base + 64];
        double2 c = sh[base + 128];
        double2 d = sh[base + 192];

        double2 w0  = __ldg(&root_table[group]);           // coarse (stride 128)
        double2 w1a = __ldg(&root_table[2 * group]);       // fine (stride 64), first pair
        double2 w1b = __ldg(&root_table[2 * group + 1]);   // fine (stride 64), second pair

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
        sh[base]       = a1 + b1w;
        sh[base + 64]  = a1 - b1w;
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
        int group = tid >> 6;        // 0..3
        int local = tid & 63;        // 0..63
        int base = group * 256 + local;

        double2 a = sh[base];
        double2 b = sh[base + 64];
        double2 c = sh[base + 128];
        double2 d = sh[base + 192];

        double2 w_s1 = __ldg(&root_table[2 * group]);       // stride 64, first pair
        double2 w_s2 = __ldg(&root_table[2 * group + 1]);   // stride 64, second pair
        double2 w_2s = __ldg(&root_table[group]);            // stride 128

        // Step 1: GS at stride 64
        double2 t0 = a + b;
        double2 t1 = (a - b) * w_s1;
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 128
        sh[base]       = t0 + t2;
        sh[base + 128] = (t0 - t2) * w_2s;
        sh[base + 64]  = t1 + t3;
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

        double2 w_s  = __ldg(&root_table[0]);   // stride 256, first pair + stride 512
        double2 w_s2 = __ldg(&root_table[1]);   // stride 256, second pair

        // Step 1: GS at stride 256
        double2 t0 = a + b;
        double2 t1 = (a - b) * w_s;     // w_s1 = root_table[0]
        double2 t2 = c + d;
        double2 t3 = (c - d) * w_s2;

        // Step 2: GS at stride 512
        sh[tid]       = t0 + t2;
        sh[tid + 512] = (t0 - t2) * w_s;   // w_2s = root_table[0]
        sh[tid + 256] = t1 + t3;
        sh[tid + 768] = (t1 - t3) * w_s;   // w_2s = root_table[0]
    }
    __syncthreads();
}

#endif  // __CUDACC__

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
    return 2 * HalfDegree<Degree<N>>::log2_degree - 7;
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT

//=============================================================================
// NTT mode: Use Goldilocks-prime NTT
//=============================================================================

// Thread configuration for NTT
// N/2 threads, each handles 2 elements (e.g., 512 threads for N=1024)
constexpr uint32_t NTT_THREAD_UNITBIT = 1;

template <uint32_t length = TFHEpp::lvl1param::n>
using CuNTTHandler = CuSmallNTTHandler<length>;

// NTT value type: 64-bit for the Goldilocks prime.
using NTTValue = SmallNTTValue;

// Shared memory size per gate: (k+2) * N * sizeof(NTTValue)
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE = (P::k + 2) * P::n * sizeof(NTTValue);

// Dynamic shared memory size for regular gates:
// NTT workspace + one TRLWE array placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4HOMGATE_DYN =
    MEM4HOMGATE<P> + (P::k + 1) * P::n * sizeof(typename P::T);

// Dynamic shared memory size for Mux/NMux gates:
// NTT workspace + two TRLWE arrays placed after it
template <class P = TFHEpp::lvl1param>
constexpr uint32_t MEM4MUXGATE_DYN =
    MEM4HOMGATE<P> + 2 * (P::k + 1) * P::n * sizeof(typename P::T);

// Number of threads for NTT (N/2 = 512 for N=1024)
template <class P = TFHEpp::lvl1param>
constexpr uint32_t NUM_THREAD4HOMGATE = P::n >> 1;

#endif  // USE_FFT

#ifdef USE_KEY_BUNDLE
extern std::vector<NTTValue*> xai_ntt_devs;
extern std::vector<NTTValue*> one_trgsw_ntt_devs;

void InitializeXaiNTT(const int gpuNum);
void InitializeOneTRGSWNTT(const int gpuNum);
void DeleteXaiNTT();
void DeleteOneTRGSWNTT();

// lvl02 (N=2048) key-bundle tables
extern std::vector<NTTValue*> xai_ntt_devs_lvl02;
extern std::vector<NTTValue*> one_trgsw_ntt_devs_lvl02;

void InitializeXaiNTT_lvl02(const int gpuNum);
void InitializeOneTRGSWNTT_lvl02(const int gpuNum);
void DeleteXaiNTT_lvl02();
void DeleteOneTRGSWNTT_lvl02();
#endif

}  // namespace cufhe
