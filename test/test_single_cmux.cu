/**
 * Definitive test: compare GPU Accumulate vs CPU ExternalProduct
 * using the SAME polynomial BSK (converted to FFT separately).
 *
 * This isolates whether the GPU's FFT conversion + Accumulate produces
 * different results from TFHEpp's TwistIFFT + ExternalProduct.
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>

#include <params.hpp>
#include <tfhe/detwfa.hpp>
#include <tfhe/key.hpp>
#include <tfhe/trgsw.hpp>

#include <include/ntt_small_modulus.cuh>
#include <include/gatebootstrapping_gpu.cuh>

using namespace cufhe;

// BSK poly -> FFT (single element)
template <class P>
__global__ void __TRGSW2FFT_Test__(NTTValue* const bk_fft,
                                    const typename P::T* const bk,
                                    CuGPUFFTHandler<P::n> ntt)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;

    __shared__ double2 sh_fft[HALF_N];

    const int in_index = blockIdx.y * N;
    const int out_index = blockIdx.y * HALF_N;
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
        GPUFFTForward<N>(sh_fft, ntt.forward_root_, tid);
    }
    else {
        for (int s = 0; s < GPUFFTSharedSyncCount<N>(); s++) __syncthreads();
    }

    if (tid < HALF_N) {
        bk_fft[out_index + tid] = sh_fft[tid];
    }
}

// Single Accumulate with trlwe in global memory
template <class P>
__global__ void __TestAccumulate__(
    typename P::targetP::T* const trlwe,
    const NTTValue* const tgsw_fft,
    const uint32_t a_bar,
    CuGPUFFTHandler<P::targetP::n> ntt)
{
    extern __shared__ NTTValue sh[];
    Accumulate<P>(trlwe, &sh[0], a_bar, tgsw_fft, ntt);
}

int main()
{
    using brP = TFHEpp::lvl02param;
    using tgtP = TFHEpp::lvl2param;
    constexpr uint32_t N = tgtP::n;
    constexpr uint32_t HALF_N = N >> 1;
    using T = tgtP::T;

    cudaSetDevice(0);

    printf("=== Definitive test: GPU vs CPU ExternalProduct (same BSK) ===\n\n");

    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;
    ek.emplacebk<brP>(sk);

    CuGPUFFTHandler<N>::Create();
    CuGPUFFTHandler<N> handler;
    handler.SetDevicePointers(0);
    cudaDeviceSynchronize();

    // Get BSK element 0 (polynomial domain)
    const auto& bk = ek.getbk<brP>();
    const auto& bk_elem = bk[0][0];  // TRGSW<lvl2param>

    int s0 = (int)sk.key.get<TFHEpp::lvl0param>()[0];
    printf("s_0 = %d\n\n", s0);

    // Convert to GPU FFT
    constexpr uint32_t num_polys = (tgtP::k + 1) * tgtP::l * (tgtP::k + 1);
    T* d_trgsw_poly;
    NTTValue* d_trgsw_fft;
    cudaMalloc(&d_trgsw_poly, sizeof(bk_elem));
    cudaMalloc(&d_trgsw_fft, num_polys * HALF_N * sizeof(NTTValue));
    cudaMemcpy(d_trgsw_poly, &bk_elem, sizeof(bk_elem), cudaMemcpyHostToDevice);

    dim3 grid_fft(1, num_polys, 1);
    dim3 block_fft(N >> 1);
    __TRGSW2FFT_Test__<tgtP><<<grid_fft, block_fft>>>(
        d_trgsw_fft, d_trgsw_poly, handler);
    cudaDeviceSynchronize();
    cudaFree(d_trgsw_poly);

    // Convert SAME polynomial BSK to CPU FFT using TFHEpp's ApplyFFT2trgsw
    TFHEpp::TRGSWFFT<tgtP> cpu_trgswfft = TFHEpp::ApplyFFT2trgsw<tgtP>(bk_elem);

    // Test TRLWE: a=0, b=vec(mu)
    std::vector<T> h_trlwe_base((tgtP::k + 1) * N, 0);
    for (uint32_t i = 0; i < N; i++)
        h_trlwe_base[tgtP::k * N + i] = tgtP::μ;

    T* d_trlwe;
    cudaMalloc(&d_trlwe, (tgtP::k + 1) * N * sizeof(T));

    constexpr size_t shmem = MEM4HOMGATE<tgtP>;
    cudaFuncSetAttribute(__TestAccumulate__<brP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

    printf("bar  | GPU_phase   | CPU_phase   | diff(phase)  | max_coeff_diff\n");
    printf("-----|-------------|-------------|--------------|---------------\n");

    int test_bars[] = {0, 1, 2, 10, 100, 500, 1000, 2048, 3000, 4095};
    int num_tests = sizeof(test_bars) / sizeof(test_bars[0]);

    for (int t = 0; t < num_tests; t++) {
        uint32_t bar = test_bars[t];

        // ====== GPU ======
        cudaMemcpy(d_trlwe, h_trlwe_base.data(),
                   (tgtP::k + 1) * N * sizeof(T), cudaMemcpyHostToDevice);
        __TestAccumulate__<brP><<<1, N >> 1, shmem>>>(
            d_trlwe, d_trgsw_fft, bar, handler);
        cudaDeviceSynchronize();

        std::vector<T> h_gpu((tgtP::k + 1) * N);
        cudaMemcpy(h_gpu.data(), d_trlwe,
                   (tgtP::k + 1) * N * sizeof(T), cudaMemcpyDeviceToHost);

        // ====== CPU ======
        // Compute (X^bar - 1) * trlwe
        TFHEpp::TRLWE<tgtP> cpu_trlwe;
        for (uint32_t k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++)
                cpu_trlwe[k][i] = h_trlwe_base[k * N + i];

        TFHEpp::TRLWE<tgtP> temp;
        for (int k = 0; k <= tgtP::k; k++)
            TFHEpp::PolynomialMulByXaiMinusOne<tgtP>(temp[k], cpu_trlwe[k], bar);

        // ExternalProduct using the SAME BSK (converted by CPU FFT)
        TFHEpp::ExternalProduct<tgtP>(temp, temp, cpu_trgswfft);

        // Add to original
        TFHEpp::TRLWE<tgtP> cpu_result;
        for (int k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++)
                cpu_result[k][i] = cpu_trlwe[k][i] + temp[k][i];

        // ====== Compare ======
        // Compute phases
        const auto& key = sk.key.get<tgtP>();
        T gpu_phase = h_gpu[tgtP::k * N];
        T cpu_phase = cpu_result[tgtP::k][0];
        for (uint32_t k = 0; k < tgtP::k; k++)
            for (uint32_t j = 0; j < N; j++) {
                gpu_phase -= h_gpu[k * N + j] * static_cast<T>(key[k * N + j]);
                cpu_phase -= cpu_result[k][j] * static_cast<T>(key[k * N + j]);
            }

        auto sgpu = static_cast<std::make_signed_t<T>>(gpu_phase);
        auto scpu = static_cast<std::make_signed_t<T>>(cpu_phase);
        double gpu_frac = (double)sgpu / (double)(1ULL << 63) / 2.0;
        double cpu_frac = (double)scpu / (double)(1ULL << 63) / 2.0;
        double diff_frac = gpu_frac - cpu_frac;

        // Find max coefficient difference
        T max_coeff_diff = 0;
        for (uint32_t k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++) {
                auto d = static_cast<std::make_signed_t<T>>(
                    h_gpu[k * N + i] - cpu_result[k][i]);
                T ad = (d < 0) ? static_cast<T>(-d) : static_cast<T>(d);
                if (ad > max_coeff_diff) max_coeff_diff = ad;
            }

        printf("%4u | %+11.6f | %+11.6f | %+12.6f | 2^%.1f\n",
               bar, gpu_frac, cpu_frac, diff_frac,
               max_coeff_diff > 0 ? log2((double)max_coeff_diff) : 0.0);
    }

    // Also print first few coefficient diffs for bar=1
    printf("\n--- Detailed comparison for bar=1 ---\n");
    {
        uint32_t bar = 1;

        cudaMemcpy(d_trlwe, h_trlwe_base.data(),
                   (tgtP::k + 1) * N * sizeof(T), cudaMemcpyHostToDevice);
        __TestAccumulate__<brP><<<1, N >> 1, shmem>>>(
            d_trlwe, d_trgsw_fft, bar, handler);
        cudaDeviceSynchronize();

        std::vector<T> h_gpu((tgtP::k + 1) * N);
        cudaMemcpy(h_gpu.data(), d_trlwe,
                   (tgtP::k + 1) * N * sizeof(T), cudaMemcpyDeviceToHost);

        TFHEpp::TRLWE<tgtP> cpu_trlwe;
        for (uint32_t k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++)
                cpu_trlwe[k][i] = h_trlwe_base[k * N + i];

        TFHEpp::TRLWE<tgtP> temp;
        for (int k = 0; k <= tgtP::k; k++)
            TFHEpp::PolynomialMulByXaiMinusOne<tgtP>(temp[k], cpu_trlwe[k], bar);
        TFHEpp::ExternalProduct<tgtP>(temp, temp, cpu_trgswfft);

        TFHEpp::TRLWE<tgtP> cpu_result;
        for (int k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++)
                cpu_result[k][i] = cpu_trlwe[k][i] + temp[k][i];

        printf("poly | idx  | GPU                  | CPU                  | diff (signed)\n");
        printf("-----|------|----------------------|----------------------|----\n");
        for (int k = 0; k <= tgtP::k; k++) {
            for (int i = 0; i < 4; i++) {
                auto d = static_cast<std::make_signed_t<T>>(
                    h_gpu[k * N + i] - cpu_result[k][i]);
                printf("  %d  | %4d | 0x%016llx | 0x%016llx | %lld (2^%.1f)\n",
                       k, i,
                       (unsigned long long)h_gpu[k * N + i],
                       (unsigned long long)cpu_result[k][i],
                       (long long)d,
                       d != 0 ? log2(fabs((double)d)) : 0.0);
            }
        }
    }

    cudaFree(d_trgsw_fft);
    cudaFree(d_trlwe);
    CuGPUFFTHandler<N>::Destroy();

    return 0;
}
