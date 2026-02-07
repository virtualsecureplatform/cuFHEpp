/**
 * Negacyclic FFT implementation for cuFHE
 *
 * Adapted from tfhe-rs CUDA backend:
 *   - Complex operations from types/complex/operations.cuh
 *   - FFT kernels from fft/bnsmfft.cuh
 *   - Parameter classes from polynomial/parameters.cuh
 *   - Twiddle declarations from fft/twiddles.cuh
 *
 * Uses double-precision complex (double2) negacyclic FFT instead of
 * modular NTT for polynomial multiplication. The half-size FFT trick
 * packs N real coefficients into N/2 complex values.
 *
 * For N=1024: HalfDegree<Degree<1024>> gives degree=512, opt=2,
 * log2_degree=9. This means BUTTERFLY_DEPTH=1, STRIDE=256, using 256 threads.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

//=============================================================================
// Parameter classes (from tfhe-rs polynomial/parameters.cuh)
//=============================================================================

template <int N> class Degree {
public:
    constexpr static int degree = N;
    constexpr static int opt = (N <= 4096) ? 4 : (N == 8192 ? 8 : 16);
    constexpr static int log2_degree = []() constexpr {
        int n = N, log = 0;
        while (n > 1) { n >>= 1; ++log; }
        return log;
    }();
};

template <class params> class HalfDegree {
public:
    constexpr static int degree = params::degree / 2;
    constexpr static int opt = params::opt / 2;
    constexpr static int log2_degree = params::log2_degree - 1;
};

//=============================================================================
// Twiddle factor declaration (from tfhe-rs fft/twiddles.cuh)
// Defined in src/fft_negacyclic.cu
//=============================================================================

extern __device__ double2 negtwiddles[8192];

//=============================================================================
// Complex number operations (from tfhe-rs types/complex/operations.cuh)
//=============================================================================

#ifdef __CUDACC__

__device__ inline double2 conjugate(const double2 num) {
    return {num.x, -num.y};
}

__device__ inline void operator+=(double2 &lh, const double2 rh) {
    lh.x = __dadd_rn(lh.x, rh.x);
    lh.y = __dadd_rn(lh.y, rh.y);
}

__device__ inline void operator-=(double2 &lh, const double2 rh) {
    lh.x = __dsub_rn(lh.x, rh.x);
    lh.y = __dsub_rn(lh.y, rh.y);
}

__device__ inline double2 operator+(const double2 a, const double2 b) {
    return {__dadd_rn(a.x, b.x), __dadd_rn(a.y, b.y)};
}

__device__ inline double2 operator-(const double2 a, const double2 b) {
    return {__dsub_rn(a.x, b.x), __dsub_rn(a.y, b.y)};
}

__device__ inline double2 operator*(const double2 a, const double2 b) {
    return {
        __fma_rn(a.x, b.x, -__dmul_rn(a.y, b.y)),
        __fma_rn(a.x, b.y, __dmul_rn(a.y, b.x))
    };
}

__device__ inline void operator*=(double2 &a, const double2 b) {
    double real = __fma_rn(a.x, b.x, -__dmul_rn(a.y, b.y));
    a.y = __fma_rn(a.x, b.y, __dmul_rn(a.y, b.x));
    a.x = real;
}

__device__ inline double2 operator*(const double2 a, double b) {
    return {__dmul_rn(a.x, b), __dmul_rn(a.y, b)};
}

__device__ inline void operator*=(double2 &a, const double b) {
    a.x = __dmul_rn(a.x, b);
    a.y = __dmul_rn(a.y, b);
}

__device__ inline void operator/=(double2 &a, const double b) {
    double inv_b = __drcp_rn(b);
    a.x = __dmul_rn(a.x, inv_b);
    a.y = __dmul_rn(a.y, inv_b);
}

__device__ inline double2 operator*(double a, double2 b) {
    return {__dmul_rn(b.x, a), __dmul_rn(b.y, a)};
}

__device__ inline double2 shfl_xor_double2(double2 val, int laneMask,
                                           unsigned mask = 0xFFFFFFFF) {
    double re = __shfl_xor_sync(mask, val.x, laneMask);
    double im = __shfl_xor_sync(mask, val.y, laneMask);
    return make_double2(re, im);
}

//=============================================================================
// Forward negacyclic FFT (from tfhe-rs fft/bnsmfft.cuh NSMFFT_direct)
//
// For N=1024: params = HalfDegree<Degree<1024>>
//   degree=512, opt=2, log2_degree=9
//   BUTTERFLY_DEPTH=1, STRIDE=256, uses 256 threads
//=============================================================================

using Index = unsigned;

template <class params> __device__ void NSMFFT_direct(double2 *A) {

    __syncthreads();
    constexpr Index BUTTERFLY_DEPTH = params::opt >> 1;
    constexpr Index LOG2_DEGREE = params::log2_degree;
    constexpr Index HALF_DEGREE = params::degree >> 1;
    constexpr Index STRIDE = params::degree / params::opt;

    Index tid = threadIdx.x;
    double2 u[BUTTERFLY_DEPTH], v[BUTTERFLY_DEPTH], w;

    // load into registers
#pragma unroll
    for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
        u[i] = A[tid];
        v[i] = A[tid + HALF_DEGREE];
        tid += STRIDE;
    }

    // level 1: special twiddle {sqrt(2)/2, sqrt(2)/2}
#pragma unroll
    for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
        w = v[i] * (double2){0.707106781186547461715008466854,
                             0.707106781186547461715008466854};
        v[i] = u[i] - w;
        u[i] = u[i] + w;
    }

    Index twiddle_shift = 1;
    for (Index l = LOG2_DEGREE - 1; l >= 5; --l) {
        Index lane_mask = 1 << (l - 1);
        Index thread_mask = (1 << l) - 1;
        twiddle_shift <<= 1;

        tid = threadIdx.x;
        __syncthreads();
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            A[tid] = (u_stays_in_register) ? v[i] : u[i];
            tid = tid + STRIDE;
        }
        __syncthreads();

        tid = threadIdx.x;
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            w = A[tid ^ lane_mask];
            u[i] = (u_stays_in_register) ? u[i] : w;
            v[i] = (u_stays_in_register) ? w : v[i];
            w = negtwiddles[tid / lane_mask + twiddle_shift];

            w *= v[i];

            v[i] = u[i] - w;
            u[i] = u[i] + w;
            tid = tid + STRIDE;
        }
    }

    for (Index l = 4; l >= 1; --l) {
        Index lane_mask = 1 << (l - 1);
        Index thread_mask = (1 << l) - 1;
        twiddle_shift <<= 1;

        tid = threadIdx.x;
        __syncwarp();
        double2 reg_A[BUTTERFLY_DEPTH];
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            reg_A[i] = (u_stays_in_register) ? v[i] : u[i];
            tid = tid + STRIDE;
        }
        __syncwarp();

        tid = threadIdx.x;
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            w = shfl_xor_double2(reg_A[i], 1 << (l - 1), 0xFFFFFFFF);
            u[i] = (u_stays_in_register) ? u[i] : w;
            v[i] = (u_stays_in_register) ? w : v[i];
            w = negtwiddles[tid / lane_mask + twiddle_shift];

            w *= v[i];

            v[i] = u[i] - w;
            u[i] = u[i] + w;
            tid = tid + STRIDE;
        }
    }

    __syncthreads();
    // store registers in SM
    tid = threadIdx.x;
#pragma unroll
    for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
        A[tid * 2] = u[i];
        A[tid * 2 + 1] = v[i];
        tid = tid + STRIDE;
    }
    __syncthreads();
}

//=============================================================================
// Inverse negacyclic FFT (from tfhe-rs fft/bnsmfft.cuh NSMFFT_inverse)
//=============================================================================

template <class params> __device__ void NSMFFT_inverse(double2 *A) {

    __syncthreads();
    constexpr Index BUTTERFLY_DEPTH = params::opt >> 1;
    constexpr Index LOG2_DEGREE = params::log2_degree;
    constexpr Index DEGREE = params::degree;
    constexpr Index HALF_DEGREE = params::degree >> 1;
    constexpr Index STRIDE = params::degree / params::opt;

    size_t tid = threadIdx.x;
    double2 u[BUTTERFLY_DEPTH], v[BUTTERFLY_DEPTH], w;

    // load into registers and divide by compressed polynomial size
#pragma unroll
    for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
        u[i] = A[2 * tid];
        v[i] = A[2 * tid + 1];

        u[i] /= DEGREE;
        v[i] /= DEGREE;

        tid += STRIDE;
    }

    Index twiddle_shift = DEGREE;
    for (Index l = 1; l <= 4; ++l) {
        Index lane_mask = 1 << (l - 1);
        Index thread_mask = (1 << l) - 1;
        tid = threadIdx.x;
        twiddle_shift >>= 1;

        tid = threadIdx.x;
        __syncwarp();
        double2 reg_A[BUTTERFLY_DEPTH];
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
            w = (u[i] - v[i]);
            u[i] += v[i];
            v[i] = w * conjugate(negtwiddles[tid / lane_mask + twiddle_shift]);

            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            reg_A[i] = (u_stays_in_register) ? v[i] : u[i];

            tid = tid + STRIDE;
        }
        __syncwarp();

        tid = threadIdx.x;
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            w = shfl_xor_double2(reg_A[i], 1 << (l - 1), 0xFFFFFFFF);
            u[i] = (u_stays_in_register) ? u[i] : w;
            v[i] = (u_stays_in_register) ? w : v[i];

            tid = tid + STRIDE;
        }
    }

    for (Index l = 5; l <= LOG2_DEGREE - 1; ++l) {
        Index lane_mask = 1 << (l - 1);
        Index thread_mask = (1 << l) - 1;
        tid = threadIdx.x;
        twiddle_shift >>= 1;

        tid = threadIdx.x;
        __syncthreads();
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
            w = (u[i] - v[i]);
            u[i] += v[i];
            v[i] = w * conjugate(negtwiddles[tid / lane_mask + twiddle_shift]);

            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            A[tid] = (u_stays_in_register) ? v[i] : u[i];

            tid = tid + STRIDE;
        }
        __syncthreads();

        tid = threadIdx.x;
#pragma unroll
        for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
            Index rank = tid & thread_mask;
            bool u_stays_in_register = rank < lane_mask;
            w = A[tid ^ lane_mask];
            u[i] = (u_stays_in_register) ? u[i] : w;
            v[i] = (u_stays_in_register) ? w : v[i];

            tid = tid + STRIDE;
        }
    }

    // last iteration
    for (Index i = 0; i < BUTTERFLY_DEPTH; ++i) {
        w = (u[i] - v[i]);
        u[i] = u[i] + v[i];
        v[i] = w * (double2){0.707106781186547461715008466854,
                             -0.707106781186547461715008466854};
    }
    __syncthreads();
    // store registers in SM
    tid = threadIdx.x;
#pragma unroll
    for (Index i = 0; i < BUTTERFLY_DEPTH; i++) {
        A[tid] = u[i];
        A[tid + HALF_DEGREE] = v[i];
        tid = tid + STRIDE;
    }
    __syncthreads();
}

#endif // __CUDACC__
