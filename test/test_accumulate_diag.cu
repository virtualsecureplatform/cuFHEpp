/**
 * Diagnostic test for the Accumulate (external product) function.
 * Tests:
 * 1. FFT roundtrip through __TRGSW2FFT__ normalization
 * 2. Accumulate with TRGSW=0, a_bar=0 (trivial, should be identity)
 * 3. Accumulate with noiseless TRGSW(1), a_bar=1 on vec(mu) input
 *    Expected: X * vec(mu) = (-mu, mu, mu, ..., mu)
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <type_traits>

#include <include/ntt_small_modulus.cuh>
#include <include/gatebootstrapping_gpu.cuh>

using namespace cufhe;

// Reuse the __TRGSW2FFT__ from bootstrap_gpu.cu to convert raw TRGSW to FFT
template <class P = TFHEpp::lvl1param>
__global__ void __LocalTRGSW2FFT__(NTTValue* const bk_fft,
                                   const typename P::T* const bk,
                                   CuGPUFFTHandler<P::n> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh_fft[HALF_N];

    const int in_index =
        blockIdx.z * ((P::k + 1) * P::l * (P::k + 1) * N) + blockIdx.y * N;
    const int out_index =
        blockIdx.z * ((P::k + 1) * P::l * (P::k + 1) * HALF_N) +
        blockIdx.y * HALF_N;

    const uint32_t tid = threadIdx.x;

    constexpr double norm =
        1.0 / static_cast<double>(1ULL << (std::numeric_limits<typename P::T>::digits / 2)) /
        static_cast<double>(1ULL << (std::numeric_limits<typename P::T>::digits - std::numeric_limits<typename P::T>::digits / 2));
    if (tid < HALF_N) {
        double re = static_cast<double>(
                        static_cast<std::make_signed_t<typename P::T>>(
                            bk[in_index + tid])) * norm;
        double im = static_cast<double>(
                        static_cast<std::make_signed_t<typename P::T>>(
                            bk[in_index + tid + HALF_N])) * norm;
        double2 folded = {re, im};
        double2 tw = __ldg(&ntt.twist_[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        if constexpr (N == 1024) {
            GPUFFTForward512(sh_fft, ntt.forward_root_, tid);
        } else if constexpr (N == 2048) {
            GPUFFTForward1024(sh_fft, ntt.forward_root_, tid);
        }
    }
    else {
        if constexpr (N == 1024) {
            for (int s = 0; s < 3; s++) __syncthreads();
        } else if constexpr (N == 2048) {
            for (int s = 0; s < 3; s++) __syncthreads();
        }
    }

    if (tid < HALF_N) {
        bk_fft[out_index + tid] = sh_fft[tid];
    }
}

// Test kernel: Run Accumulate with given a_bar
template <class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename P::targetP>)
void __TestAccumulate__(
    typename P::targetP::T* const trlwe_out,
    const typename P::targetP::T* const trlwe_in,
    const NTTValue* const tgsw_fft,
    const uint32_t a_bar,
    CuGPUFFTHandler<P::targetP::n> ntt)
{
    constexpr uint32_t N = P::targetP::n;
    constexpr uint32_t NUM_THREADS = N >> 1;

    extern __shared__ NTTValue sh[];
    NTTValue* sh_acc_ntt = &sh[0];

    const uint32_t tid = threadIdx.x;

    // Copy input TRLWE to output
    for (int i = tid; i < (P::targetP::k + 1) * N; i += NUM_THREADS) {
        trlwe_out[i] = trlwe_in[i];
    }
    __syncthreads();

    Accumulate<P>(trlwe_out, sh_acc_ntt, a_bar, tgsw_fft, ntt);
    __syncthreads();
}


// Create noiseless TRGSW encrypting m (on host, in coefficient domain)
template <class P>
std::vector<typename P::T> make_noiseless_trgsw(int m)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t k = P::k;
    constexpr uint32_t l = P::l;
    constexpr uint32_t Bgbit = P::Bgbit;
    constexpr uint32_t num_rows = (k + 1) * l;
    constexpr uint32_t polys_per_row = k + 1;
    constexpr size_t total_elements = num_rows * polys_per_row * N;

    std::vector<typename P::T> trgsw(total_elements, 0);

    // TRGSW(m) = m * H + Enc(0)
    // H is the gadget matrix with entries g_i on the diagonal
    // For noiseless: just set the gadget entries

    for (uint32_t row = 0; row < num_rows; row++) {
        // Which polynomial (j) and which digit (d)?
        uint32_t j = row / l;      // 0..k
        uint32_t d = row % l;      // 0..l-1

        // Gadget value: B^{-(d+1)} in Torus = 2^(digits - (d+1)*Bgbit)
        typename P::T g_val = static_cast<typename P::T>(1ULL <<
            (std::numeric_limits<typename P::T>::digits - (d + 1) * Bgbit));

        // Add m * g_val to polynomial j at coefficient 0
        // TRGSW layout: row * (k+1) polynomials, each of N elements
        // poly[j][0] += m * g_val
        trgsw[row * polys_per_row * N + j * N + 0] += m * g_val;
    }

    return trgsw;
}

template <class brP>
bool test_accumulate_noiseless()
{
    using targetP = typename brP::targetP;
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t num_polys = (targetP::k + 1) * targetP::l * (targetP::k + 1);
    using T = typename targetP::T;

    printf("  Testing Accumulate with noiseless TRGSW(1), a_bar=1, N=%u...\n", N);

    // Create noiseless TRGSW(1) on host
    auto h_trgsw = make_noiseless_trgsw<targetP>(1);

    // Convert to FFT on GPU
    T* d_trgsw;
    cudaMalloc(&d_trgsw, h_trgsw.size() * sizeof(T));
    cudaMemcpy(d_trgsw, h_trgsw.data(), h_trgsw.size() * sizeof(T),
               cudaMemcpyHostToDevice);

    NTTValue* d_tgsw_fft;
    cudaMalloc(&d_tgsw_fft, num_polys * HALF_N * sizeof(NTTValue));

    CuGPUFFTHandler<N> handler;
    handler.SetDevicePointers(0);
    cudaDeviceSynchronize();

    // Convert TRGSW to FFT using same kernel as BSK conversion
    dim3 grid_fft(1, num_polys, 1);  // 1 TRGSW element
    dim3 block_fft(N >> 1);
    __LocalTRGSW2FFT__<targetP><<<grid_fft, block_fft>>>(
        d_tgsw_fft, d_trgsw, handler);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error in TRGSW2FFT: %s\n", cudaGetErrorString(err));
        cudaFree(d_trgsw);
        cudaFree(d_tgsw_fft);
        return false;
    }
    cudaFree(d_trgsw);

    // Create test TRLWE: a=0, b=vec(mu)
    constexpr size_t trlwe_size = (targetP::k + 1) * N;
    std::vector<T> h_trlwe_in(trlwe_size, 0);
    for (uint32_t i = 0; i < N; i++) {
        h_trlwe_in[targetP::k * N + i] = targetP::μ;
    }

    T *d_trlwe_in, *d_trlwe_out;
    cudaMalloc(&d_trlwe_in, trlwe_size * sizeof(T));
    cudaMalloc(&d_trlwe_out, trlwe_size * sizeof(T));
    cudaMemcpy(d_trlwe_in, h_trlwe_in.data(),
               trlwe_size * sizeof(T), cudaMemcpyHostToDevice);

    constexpr size_t shmem = MEM4HOMGATE<targetP>;
    dim3 block(N >> 1);

    cudaFuncSetAttribute(__TestAccumulate__<brP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem);
    __TestAccumulate__<brP><<<1, block, shmem>>>(
        d_trlwe_out, d_trlwe_in, d_tgsw_fft, 1, handler);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error in Accumulate: %s\n", cudaGetErrorString(err));
        cudaFree(d_tgsw_fft);
        cudaFree(d_trlwe_in);
        cudaFree(d_trlwe_out);
        return false;
    }

    std::vector<T> h_out(trlwe_size);
    cudaMemcpy(h_out.data(), d_trlwe_out,
               trlwe_size * sizeof(T), cudaMemcpyDeviceToHost);

    // Expected: TRLWE += TRGSW(1) ⊡ (X^1 - 1) * TRLWE
    // Input: a=0, b=vec(mu) = (mu, mu, ..., mu)
    // (X-1) * vec(mu) for b-poly: (-2mu, 0, 0, ..., 0)
    // (X-1) * 0 for a-poly: (0, 0, ..., 0)
    // External product with TRGSW(1): result = (X-1) * TRLWE (ideally)
    // So final = TRLWE + (X-1)*TRLWE = X * TRLWE
    // X * {a=0, b=vec(mu)} = {a=0, b=X*vec(mu)}
    // X * (mu + mu*X + ... + mu*X^{N-1}) = mu*X + ... + mu*X^{N-1} + mu*X^N
    //   = mu*X + ... + mu*X^{N-1} - mu   (X^N = -1)
    // So expected b = (-mu, mu, mu, ..., mu)

    std::vector<T> h_expected(trlwe_size, 0);
    // a-poly should remain 0
    // b-poly: coeff[0] = -mu, coeff[1..N-1] = mu
    using ST = std::make_signed_t<T>;
    h_expected[targetP::k * N + 0] = static_cast<T>(-static_cast<ST>(targetP::μ));
    for (uint32_t i = 1; i < N; i++) {
        h_expected[targetP::k * N + i] = targetP::μ;
    }

    // Check with tolerance for double precision rounding
    int mismatches = 0;
    T max_err = 0;
    for (uint32_t i = 0; i < trlwe_size; i++) {
        ST diff = static_cast<ST>(h_out[i]) - static_cast<ST>(h_expected[i]);
        T abs_diff = (diff < 0) ? static_cast<T>(-diff) : static_cast<T>(diff);
        if (abs_diff > max_err) max_err = abs_diff;
        // Tolerance: 2^20 for uint64_t (generous for double FFT rounding)
        T tolerance = (sizeof(T) == 4) ? 1 : (1ULL << 20);
        if (abs_diff > tolerance) {
            if (mismatches < 10) {
                printf("    Mismatch at [%u]: expected=%lld got=%lld diff=%lld\n",
                       i, (long long)(ST)h_expected[i], (long long)(ST)h_out[i],
                       (long long)diff);
            }
            mismatches++;
        }
    }

    printf("  Max error: %llu\n", (unsigned long long)max_err);
    printf("  Mismatches (beyond tolerance): %d / %zu\n", mismatches, trlwe_size);
    bool pass = (mismatches == 0);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_tgsw_fft);
    cudaFree(d_trlwe_in);
    cudaFree(d_trlwe_out);
    return pass;
}

// Also test a_bar=0 with TRGSW=0 (trivial case)
template <class brP>
bool test_accumulate_zero_bar()
{
    using targetP = typename brP::targetP;
    constexpr uint32_t N = targetP::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t num_polys = (targetP::k + 1) * targetP::l * (targetP::k + 1);
    using T = typename targetP::T;

    printf("  Testing Accumulate with a_bar=0, TRGSW=0, N=%u...\n", N);

    NTTValue* d_tgsw_fft;
    cudaMalloc(&d_tgsw_fft, num_polys * HALF_N * sizeof(NTTValue));
    cudaMemset(d_tgsw_fft, 0, num_polys * HALF_N * sizeof(NTTValue));

    constexpr size_t trlwe_size = (targetP::k + 1) * N;
    std::vector<T> h_trlwe_in(trlwe_size, 0);
    for (uint32_t i = 0; i < N; i++) {
        h_trlwe_in[targetP::k * N + i] = targetP::μ;
    }

    T *d_trlwe_in, *d_trlwe_out;
    cudaMalloc(&d_trlwe_in, trlwe_size * sizeof(T));
    cudaMalloc(&d_trlwe_out, trlwe_size * sizeof(T));
    cudaMemcpy(d_trlwe_in, h_trlwe_in.data(),
               trlwe_size * sizeof(T), cudaMemcpyHostToDevice);

    CuGPUFFTHandler<N> handler;
    handler.SetDevicePointers(0);
    cudaDeviceSynchronize();

    constexpr size_t shmem = MEM4HOMGATE<targetP>;
    dim3 block(N >> 1);

    cudaFuncSetAttribute(__TestAccumulate__<brP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem);
    __TestAccumulate__<brP><<<1, block, shmem>>>(
        d_trlwe_out, d_trlwe_in, d_tgsw_fft, 0, handler);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        cudaFree(d_tgsw_fft);
        cudaFree(d_trlwe_in);
        cudaFree(d_trlwe_out);
        return false;
    }

    std::vector<T> h_trlwe_out(trlwe_size);
    cudaMemcpy(h_trlwe_out.data(), d_trlwe_out,
               trlwe_size * sizeof(T), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (uint32_t i = 0; i < trlwe_size && mismatches < 20; i++) {
        if (h_trlwe_in[i] != h_trlwe_out[i]) {
            using ST = std::make_signed_t<T>;
            printf("    Mismatch at [%u]: in=%lld out=%lld\n",
                   i, (long long)(ST)h_trlwe_in[i], (long long)(ST)h_trlwe_out[i]);
            mismatches++;
        }
    }
    printf("  Mismatches: %d / %zu\n", mismatches, trlwe_size);
    bool pass = (mismatches == 0);
    printf("  %s\n", pass ? "PASS" : "FAIL");

    cudaFree(d_tgsw_fft);
    cudaFree(d_trlwe_in);
    cudaFree(d_trlwe_out);
    return pass;
}

int main()
{
    cudaSetDevice(0);

    CuGPUFFTHandler<1024>::Create();
    CuGPUFFTHandler<2048>::Create();

    bool all_pass = true;

    printf("=== Test 1: Accumulate a_bar=0, TRGSW=0 ===\n");
    all_pass &= test_accumulate_zero_bar<TFHEpp::lvl01param>();
    all_pass &= test_accumulate_zero_bar<TFHEpp::lvl02param>();

    printf("\n=== Test 2: Accumulate a_bar=1, noiseless TRGSW(1), vec(mu) ===\n");
    all_pass &= test_accumulate_noiseless<TFHEpp::lvl01param>();
    all_pass &= test_accumulate_noiseless<TFHEpp::lvl02param>();

    printf("\n%s\n", all_pass ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    CuGPUFFTHandler<1024>::Destroy();
    CuGPUFFTHandler<2048>::Destroy();

    return all_pass ? 0 : 1;
}
