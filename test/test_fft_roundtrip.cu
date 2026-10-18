/**
 * Diagnostic test: verify GPU-FFT roundtrip for both 512-point and 1024-point.
 * Tests: fold + twist + forward FFT + inverse FFT + untwist + unfold
 * Should recover the original polynomial (within floating-point error).
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <include/ntt_small_modulus.cuh>

using namespace cufhe;

// Match the handler specializations instantiated by the library. When both
// levels have the same degree, their Fourier tables are identical.
template <uint32_t N>
using FFTTestHandler = CuGPUFFTHandler<N,
    N == TFHEpp::lvl1param::n ? 32 : 64>;

// The same transform must produce identical bits with extra idle warps.
// Arbitrary unit roots isolate this scheduling invariant from table generation.
template <uint32_t N, bool Inverse>
__global__ __launch_bounds__(N / 2) void FFTParticipation(
    double2* out, const double2* input, const double2* roots)
{
    __shared__ double2 work[N / 2];
    for (int i = threadIdx.x; i < N / 2; i += blockDim.x) work[i] = input[i];
    __syncthreads();
    if constexpr (Inverse) GPUFFTInverse<N>(work, roots, threadIdx.x);
    else GPUFFTForward<N>(work, roots, threadIdx.x);
    for (int i = threadIdx.x; i < N / 2; i += blockDim.x) out[i] = work[i];
}

template <uint32_t N>
bool test_participation()
{
    auto check = [](cudaError_t error) {
        if (error != cudaSuccess) {
            std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
            std::exit(2);
        }
    };
    constexpr size_t bytes = N / 2 * sizeof(double2);
    std::vector<double2> input(N / 2), roots(N / 2), ref(N / 2), actual(N / 2);
    const double pi = std::acos(-1.0);
    for (int i = 0; i < N / 2; ++i)
        roots[i] = {std::cos(2 * pi * i / N), std::sin(2 * pi * i / N)};
    double2 *din, *droots, *dref, *dactual;
    check(cudaMalloc(&din, bytes)); check(cudaMalloc(&droots, bytes));
    check(cudaMalloc(&dref, bytes)); check(cudaMalloc(&dactual, bytes));
    check(cudaMemcpy(droots, roots.data(), bytes, cudaMemcpyHostToDevice));
    std::mt19937_64 rng(7100 + N);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    bool pass = true;
    for (int sample = 0; sample < 4; ++sample) {
        for (int i = 0; i < N / 2; ++i)
            input[i] = sample == 0 ? double2{0, 0} :
                       sample == 1 ? double2{i & 1 ? -1.0 : 1.0, 0} :
                       double2{dist(rng), dist(rng)};
        check(cudaMemcpy(din, input.data(), bytes, cudaMemcpyHostToDevice));
        for (bool inverse : {false, true}) {
            if (inverse) {
                FFTParticipation<N, true><<<1, N / 4>>>(dref, din, droots);
                FFTParticipation<N, true><<<1, N / 2>>>(dactual, din, droots);
            } else {
                FFTParticipation<N, false><<<1, N / 4>>>(dref, din, droots);
                FFTParticipation<N, false><<<1, N / 2>>>(dactual, din, droots);
            }
            check(cudaGetLastError());
            check(cudaMemcpy(ref.data(), dref, bytes, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual.data(), dactual, bytes, cudaMemcpyDeviceToHost));
            pass &= std::memcmp(ref.data(), actual.data(), bytes) == 0;
        }
    }
    check(cudaFree(din)); check(cudaFree(droots));
    check(cudaFree(dref)); check(cudaFree(dactual));
    std::printf("N=%u active/full-CTA forward/inverse bit equivalence: %s\n",
                N, pass ? "PASS" : "FAIL");
    return pass;
}

// Test kernel: fold+twist+FFT+IFFT+untwist+unfold for N-coefficient polynomial
template <uint32_t N>
__global__ void __FFTRoundtrip__(
    double* const out,
    const double* const in,
    FFTTestHandler<N> ntt)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t tid = threadIdx.x;

    // Step 1: Fold + twist (same as in __TRGSW2FFT__)
    if (tid < HALF_N) {
        double re = in[tid];
        double im = in[tid + HALF_N];
        double2 folded = {re, im};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    // Step 2: Forward FFT
    if (tid < FFT_THREADS || cufhe::GPUFFTAllThreads) {
        GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    // Step 3: Inverse FFT
    if (tid < FFT_THREADS || cufhe::GPUFFTAllThreads) {
        GPUFFTInverse<N>(sh_fft, ntt.inverse_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    // Step 4: Untwist + unfold
    if (tid < HALF_N) {
        double2 val = sh_fft[tid];
        double2 utw = __ldg(&ntt.untwist_[tid]);
        val = val * utw;
        out[tid] = val.x;
        out[tid + HALF_N] = val.y;
    }
}

// Test kernel: fold+twist+FFT, pointwise square, IFFT+untwist+unfold
// This tests negacyclic polynomial squaring: poly * poly mod (X^N + 1)
template <uint32_t N>
__global__ void __FFTNegacyclicSquare__(
    double* const out,
    const double* const in,
    FFTTestHandler<N> ntt)
{
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t tid = threadIdx.x;

    // Fold + twist
    if (tid < HALF_N) {
        double re = in[tid];
        double im = in[tid + HALF_N];
        double2 folded = {re, im};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    // Forward FFT
    if (tid < FFT_THREADS || cufhe::GPUFFTAllThreads) {
        GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    // Pointwise square (complex multiply: z * z)
    if (tid < HALF_N) {
        double2 z = sh_fft[tid];
        sh_fft[tid] = z * z;
    }
    __syncthreads();

    // Inverse FFT
    if (tid < FFT_THREADS || cufhe::GPUFFTAllThreads) {
        GPUFFTInverse<N>(sh_fft, ntt.inverse_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    // Untwist + unfold
    if (tid < HALF_N) {
        double2 val = sh_fft[tid];
        double2 utw = __ldg(&ntt.untwist_[tid]);
        val = val * utw;
        out[tid] = val.x;
        out[tid + HALF_N] = val.y;
    }
}

// CPU reference: negacyclic polynomial multiply
void negacyclic_poly_mul_cpu(const double* a, const double* b, double* out, int N)
{
    for (int i = 0; i < N; i++) out[i] = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i + j;
            if (idx < N)
                out[idx] += a[i] * b[j];
            else
                out[idx - N] -= a[i] * b[j];  // negacyclic: X^N = -1
        }
    }
}

template <uint32_t N>
bool test_roundtrip(FFTTestHandler<N>& handler)
{
    printf("  Testing FFT roundtrip for N=%u (FFT size %u)...\n", N, N/2);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    std::vector<double> h_in(N), h_out(N);
    for (uint32_t i = 0; i < N; i++) h_in[i] = dist(rng);

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(N >> 1);  // N/2 threads
    __FFTRoundtrip__<N><<<1, block>>>(d_out, d_in, handler);
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
    for (uint32_t i = 0; i < N; i++) {
        double e = std::abs(h_in[i] - h_out[i]);
        max_err = std::max(max_err, e);
    }

    printf("  Max roundtrip error: %.6e\n", max_err);
    bool pass = max_err < 1e-6;
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    return pass;
}

template <uint32_t N>
bool test_negacyclic_square(FFTTestHandler<N>& handler)
{
    printf("  Testing negacyclic polynomial squaring for N=%u...\n", N);

    // Use small integer coefficients for exact comparison
    std::vector<double> h_in(N, 0.0), h_gpu_out(N), h_cpu_out(N);

    // Simple polynomial: 1 + X + X^2 (small to avoid overflow in CPU reference)
    h_in[0] = 1.0;
    h_in[1] = 1.0;
    h_in[2] = 1.0;

    // CPU reference
    negacyclic_poly_mul_cpu(h_in.data(), h_in.data(), h_cpu_out.data(), N);

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(N >> 1);
    __FFTNegacyclicSquare__<N><<<1, block>>>(d_out, d_in, handler);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    cudaMemcpy(h_gpu_out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    int worst_idx = -1;
    for (uint32_t i = 0; i < N; i++) {
        double e = std::abs(h_cpu_out[i] - h_gpu_out[i]);
        if (e > max_err) {
            max_err = e;
            worst_idx = i;
        }
    }

    // Print first few coefficients for comparison
    printf("  First 10 coefficients (CPU vs GPU):\n");
    for (int i = 0; i < 10; i++) {
        printf("    [%d] CPU=%.6f GPU=%.6f\n", i, h_cpu_out[i], h_gpu_out[i]);
    }
    printf("  Max error: %.6e at index %d\n", max_err, worst_idx);
    bool pass = max_err < 1e-6;
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    return pass;
}

template <uint32_t N>
bool test_negacyclic_random(FFTTestHandler<N>& handler)
{
    printf("  Testing negacyclic squaring with random small-int coefficients N=%u...\n", N);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-5, 5);

    std::vector<double> h_in(N), h_gpu_out(N), h_cpu_out(N);
    for (uint32_t i = 0; i < N; i++) h_in[i] = dist(rng);

    negacyclic_poly_mul_cpu(h_in.data(), h_in.data(), h_cpu_out.data(), N);

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block(N >> 1);
    __FFTNegacyclicSquare__<N><<<1, block>>>(d_out, d_in, handler);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return false;
    }

    cudaMemcpy(h_gpu_out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    int worst_idx = -1;
    for (uint32_t i = 0; i < N; i++) {
        double e = std::abs(h_cpu_out[i] - h_gpu_out[i]);
        if (e > max_err) {
            max_err = e;
            worst_idx = i;
        }
    }

    printf("  Max error: %.6e at index %d\n", max_err, worst_idx);
    // With N*max_coeff^2 accumulation, expect some FP error
    bool pass = max_err < 0.5;
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_in);
    cudaFree(d_out);
    return pass;
}

template <uint32_t N>
bool test_degree()
{
    printf("=== N=%u (%u-point FFT) ===\n", N, N / 2);
    FFTTestHandler<N>::Create();
    FFTTestHandler<N> handler;
    handler.SetDevicePointers(0);
    cudaDeviceSynchronize();

    const bool pass = test_roundtrip<N>(handler) &&
                      test_negacyclic_square<N>(handler) &&
                      test_negacyclic_random<N>(handler);
    FFTTestHandler<N>::Destroy();
    return pass;
}

int main()
{
    cudaSetDevice(0);

    bool all_pass = test_participation<512>();
    all_pass &= test_participation<1024>();
    all_pass &= test_participation<2048>();
    all_pass &= test_degree<TFHEpp::lvl1param::n>();
    printf("\n");
    all_pass &= test_degree<TFHEpp::lvl2param::n>();

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return all_pass ? 0 : 1;
}
