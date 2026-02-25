/**
 * FFT performance benchmark for GPU-FFT kernels.
 *
 * Measures forward FFT, inverse FFT, and full negacyclic-multiply roundtrip
 * (fold+twist+forward+pointwise_mul+inverse+untwist+unfold) throughput.
 *
 * Usage: ./test_fft_perf [num_iterations]
 *   default: 100000 iterations
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <cfloat>

#include <include/ntt_small_modulus.cuh>

using namespace cufhe;

// ---------------------------------------------------------------------------
// Benchmark kernels
// ---------------------------------------------------------------------------

// Forward FFT only (shared mem already loaded)
template <uint32_t N>
__global__ void __BenchForwardFFT__(
    double2* __restrict__ out,
    const double2* __restrict__ in,
    CuGPUFFTHandler<N> ntt,
    int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh[HALF_N];
    const uint32_t tid = threadIdx.x;

    // Load once
    if (tid < HALF_N) sh[tid] = in[tid];
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        // Reset data each iteration to avoid NaN propagation
        if (tid < HALF_N) sh[tid] = in[tid];
        __syncthreads();

        if (tid < FFT_THREADS) {
            if constexpr (N == 1024)
                GPUFFTForward512(sh, ntt.forward_root_, tid);
            else if constexpr (N == 2048)
                GPUFFTForward1024(sh, ntt.forward_root_, tid);
        } else {
            if constexpr (N == 1024)
                for (int s = 0; s < 5; s++) __syncthreads();
            else if constexpr (N == 2048)
                for (int s = 0; s < 6; s++) __syncthreads();
        }
    }

    // Write out to prevent dead-code elimination
    if (tid < HALF_N) out[tid] = sh[tid];
}

// Inverse FFT only
template <uint32_t N>
__global__ void __BenchInverseFFT__(
    double2* __restrict__ out,
    const double2* __restrict__ in,
    CuGPUFFTHandler<N> ntt,
    int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh[HALF_N];
    const uint32_t tid = threadIdx.x;

    if (tid < HALF_N) sh[tid] = in[tid];
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        if (tid < HALF_N) sh[tid] = in[tid];
        __syncthreads();

        if (tid < FFT_THREADS) {
            if constexpr (N == 1024)
                GPUFFTInverse512(sh, ntt.inverse_root_, tid);
            else if constexpr (N == 2048)
                GPUFFTInverse1024(sh, ntt.inverse_root_, tid);
        } else {
            if constexpr (N == 1024)
                for (int s = 0; s < 5; s++) __syncthreads();
            else if constexpr (N == 2048)
                for (int s = 0; s < 6; s++) __syncthreads();
        }
    }

    if (tid < HALF_N) out[tid] = sh[tid];
}

// Full negacyclic polynomial multiply roundtrip:
// fold+twist+forward+pointwise_mul+inverse+untwist+unfold
// This mirrors the hot path in Accumulate()
template <uint32_t N>
__global__ void __launch_bounds__(1024) __BenchFullRoundtrip__(
    double* __restrict__ out,
    const double* __restrict__ poly_a,
    const double2* __restrict__ poly_b_fft,  // pre-FFT'd second operand
    CuGPUFFTHandler<N> ntt,
    int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh[HALF_N];
    const uint32_t tid = threadIdx.x;

    for (int iter = 0; iter < num_iters; iter++) {
        // Step 1: Fold + twist (like decompose step in Accumulate)
        if (tid < HALF_N) {
            double re = poly_a[tid];
            double im = poly_a[tid + HALF_N];
            double2 folded = {re, im};
            double2 tw = __ldg(&ntt.twist_[tid]);
            sh[tid] = folded * tw;
        }
        __syncthreads();

        // Step 2: Forward FFT
        if (tid < FFT_THREADS) {
            if constexpr (N == 1024)
                GPUFFTForward512(sh, ntt.forward_root_, tid);
            else if constexpr (N == 2048)
                GPUFFTForward1024(sh, ntt.forward_root_, tid);
        } else {
            if constexpr (N == 1024)
                for (int s = 0; s < 5; s++) __syncthreads();
            else if constexpr (N == 2048)
                for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Step 3: Pointwise multiply (like multiply-accumulate in Accumulate)
        if (tid < HALF_N) {
            sh[tid] = sh[tid] * __ldg(&poly_b_fft[tid]);
        }
        __syncthreads();

        // Step 4: Inverse FFT
        if (tid < FFT_THREADS) {
            if constexpr (N == 1024)
                GPUFFTInverse512(sh, ntt.inverse_root_, tid);
            else if constexpr (N == 2048)
                GPUFFTInverse1024(sh, ntt.inverse_root_, tid);
        } else {
            if constexpr (N == 1024)
                for (int s = 0; s < 5; s++) __syncthreads();
            else if constexpr (N == 2048)
                for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Step 5: Untwist + unfold
        if (tid < HALF_N) {
            double2 val = sh[tid];
            double2 utw = __ldg(&ntt.untwist_[tid]);
            val = val * utw;
            out[tid] = val.x;
            out[tid + HALF_N] = val.y;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

template <uint32_t N>
void bench_forward(CuGPUFFTHandler<N>& handler, int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = HALF_N;

    // Prepare input data in frequency domain (random complex values)
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double2> h_in(HALF_N);
    for (uint32_t i = 0; i < HALF_N; i++)
        h_in[i] = {dist(rng), dist(rng)};

    double2 *d_in, *d_out;
    cudaMalloc(&d_in, HALF_N * sizeof(double2));
    cudaMalloc(&d_out, HALF_N * sizeof(double2));
    cudaMemcpy(d_in, h_in.data(), HALF_N * sizeof(double2),
               cudaMemcpyHostToDevice);

    // Warmup
    __BenchForwardFFT__<N><<<1, NUM_THREADS>>>(d_out, d_in, handler, 1000);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    __BenchForwardFFT__<N><<<1, NUM_THREADS>>>(d_out, d_in, handler, num_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double us_per_fft = (ms * 1000.0) / num_iters;

    printf("  Forward FFT %4u-pt:  %.3f us/op  (%d iters, %.1f ms total)\n",
           HALF_N, us_per_fft, num_iters, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
}

template <uint32_t N>
void bench_inverse(CuGPUFFTHandler<N>& handler, int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = HALF_N;

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double2> h_in(HALF_N);
    for (uint32_t i = 0; i < HALF_N; i++)
        h_in[i] = {dist(rng), dist(rng)};

    double2 *d_in, *d_out;
    cudaMalloc(&d_in, HALF_N * sizeof(double2));
    cudaMalloc(&d_out, HALF_N * sizeof(double2));
    cudaMemcpy(d_in, h_in.data(), HALF_N * sizeof(double2),
               cudaMemcpyHostToDevice);

    // Warmup
    __BenchInverseFFT__<N><<<1, NUM_THREADS>>>(d_out, d_in, handler, 1000);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    __BenchInverseFFT__<N><<<1, NUM_THREADS>>>(d_out, d_in, handler, num_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double us_per_fft = (ms * 1000.0) / num_iters;

    printf("  Inverse FFT %4u-pt:  %.3f us/op  (%d iters, %.1f ms total)\n",
           HALF_N, us_per_fft, num_iters, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
}

template <uint32_t N>
void bench_full_roundtrip(CuGPUFFTHandler<N>& handler, int num_iters)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t NUM_THREADS = HALF_N;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    // Prepare poly_a (time domain) and poly_b_fft (pre-transformed)
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<double> h_poly_a(N);
    std::vector<double2> h_poly_b_fft(HALF_N);
    for (uint32_t i = 0; i < N; i++) h_poly_a[i] = dist(rng);
    for (uint32_t i = 0; i < HALF_N; i++)
        h_poly_b_fft[i] = {dist(rng), dist(rng)};

    double *d_poly_a, *d_out;
    double2 *d_poly_b_fft;
    cudaMalloc(&d_poly_a, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMalloc(&d_poly_b_fft, HALF_N * sizeof(double2));
    cudaMemcpy(d_poly_a, h_poly_a.data(), N * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_poly_b_fft, h_poly_b_fft.data(), HALF_N * sizeof(double2),
               cudaMemcpyHostToDevice);

    // Warmup
    __BenchFullRoundtrip__<N><<<1, NUM_THREADS>>>(
        d_out, d_poly_a, d_poly_b_fft, handler, 1000);
    cudaDeviceSynchronize();
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Full roundtrip N=%u: warmup CUDA error: %s\n",
                   N, cudaGetErrorString(err));
            cudaFree(d_poly_a);
            cudaFree(d_out);
            cudaFree(d_poly_b_fft);
            return;
        }
    }

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    __BenchFullRoundtrip__<N><<<1, NUM_THREADS>>>(
        d_out, d_poly_a, d_poly_b_fft, handler, num_iters);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double us_per_op = (ms * 1000.0) / num_iters;

    printf("  Full roundtrip N=%u:  %.3f us/op  (%d iters, %.1f ms total)\n",
           N, us_per_op, num_iters, ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_poly_a);
    cudaFree(d_out);
    cudaFree(d_poly_b_fft);
}

// ---------------------------------------------------------------------------
// Correctness verification (quick sanity check before benchmarking)
// ---------------------------------------------------------------------------

template <uint32_t N>
__global__ void __VerifyRoundtrip__(
    double* const out,
    const double* const in,
    CuGPUFFTHandler<N> ntt)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh[HALF_N];
    const uint32_t tid = threadIdx.x;

    // Fold + twist
    if (tid < HALF_N) {
        double2 folded = {in[tid], in[tid + HALF_N]};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh[tid] = folded * tw;
    }
    __syncthreads();

    // Forward
    if (tid < FFT_THREADS) {
        if constexpr (N == 1024)
            GPUFFTForward512(sh, ntt.forward_root_, tid);
        else if constexpr (N == 2048)
            GPUFFTForward1024(sh, ntt.forward_root_, tid);
    } else {
        if constexpr (N == 1024)
            for (int s = 0; s < 5; s++) __syncthreads();
        else if constexpr (N == 2048)
            for (int s = 0; s < 6; s++) __syncthreads();
    }

    // Inverse
    if (tid < FFT_THREADS) {
        if constexpr (N == 1024)
            GPUFFTInverse512(sh, ntt.inverse_root_, tid);
        else if constexpr (N == 2048)
            GPUFFTInverse1024(sh, ntt.inverse_root_, tid);
    } else {
        if constexpr (N == 1024)
            for (int s = 0; s < 5; s++) __syncthreads();
        else if constexpr (N == 2048)
            for (int s = 0; s < 6; s++) __syncthreads();
    }

    // Untwist + unfold
    if (tid < HALF_N) {
        double2 val = sh[tid];
        double2 utw = __ldg(&ntt.untwist_[tid]);
        val = val * utw;
        out[tid] = val.x;
        out[tid + HALF_N] = val.y;
    }
}

template <uint32_t N>
bool verify_correctness(CuGPUFFTHandler<N>& handler)
{
    std::mt19937 rng(789);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    std::vector<double> h_in(N), h_out(N);
    for (uint32_t i = 0; i < N; i++) h_in[i] = dist(rng);

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    __VerifyRoundtrip__<N><<<1, N >> 1>>>(d_out, d_in, handler);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    cudaMemcpy(h_out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (uint32_t i = 0; i < N; i++)
        max_err = std::max(max_err, std::abs(h_in[i] - h_out[i]));

    printf("  Correctness check N=%u: max_err=%.2e %s\n",
           N, max_err, max_err < 1e-6 ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    return max_err < 1e-6;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    int num_iters = 100000;
    if (argc > 1) num_iters = atoi(argv[1]);

    cudaSetDevice(0);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Iterations: %d\n\n", num_iters);

    // --- N=1024 (512-point FFT) ---
    printf("=== N=1024 (512-point complex FFT) ===\n");
    {
        CuGPUFFTHandler<1024>::Create();
        CuGPUFFTHandler<1024> handler;
        handler.SetDevicePointers(0);
        cudaDeviceSynchronize();

        if (!verify_correctness<1024>(handler)) {
            printf("  Correctness failed, skipping benchmarks\n");
        } else {
            bench_forward<1024>(handler, num_iters);
            bench_inverse<1024>(handler, num_iters);
            bench_full_roundtrip<1024>(handler, num_iters);
        }

        CuGPUFFTHandler<1024>::Destroy();
    }

    printf("\n");

    // --- N=2048 (1024-point FFT) ---
    printf("=== N=2048 (1024-point complex FFT) ===\n");
    {
        CuGPUFFTHandler<2048>::Create();
        CuGPUFFTHandler<2048> handler;
        handler.SetDevicePointers(0);
        cudaDeviceSynchronize();

        if (!verify_correctness<2048>(handler)) {
            printf("  Correctness failed, skipping benchmarks\n");
        } else {
            bench_forward<2048>(handler, num_iters);
            bench_inverse<2048>(handler, num_iters);
            bench_full_roundtrip<2048>(handler, num_iters);
        }

        CuGPUFFTHandler<2048>::Destroy();
    }

    printf("\nDone.\n");
    return 0;
}
