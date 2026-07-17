/**
 * Definitive test: compare GPU vs CPU full blind rotation using the SAME BSK.
 *
 * 1. Generate polynomial BSK via emplacebk<lvl02param>
 * 2. Convert all bootstrapping key TRGSWs to CPU FFT using ApplyFFT2trgsw
 * 3. Convert the BSK to GPU FFT using the active TFHEpp key layout
 * 4. Create the combined TLWE for NAND gate (all 4 input combos)
 * 5. Run CPU BlindRotate
 * 6. Run GPU BlindRotatePreAdd (via a custom kernel that exposes TRLWE)
 * 7. Compare TRLWE outputs
 */

#include <cstdio>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>

#include <params.hpp>
#include <tfhe/detwfa.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trgsw.hpp>

#include <include/ntt_small_modulus.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

namespace cufhe {
extern std::vector<NTTValue*> bk_ntts_lvl02;
extern std::vector<CuNTTHandler<TFHEpp::lvl2param::n>*> ntt_handlers_lvl02;
#ifdef USE_KEY_BUNDLE
extern std::vector<NTTValue*> xai_ntt_devs_lvl02;
extern std::vector<NTTValue*> one_trgsw_ntt_devs_lvl02;
#endif
}

using namespace TFHEpp;
using namespace cufhe;

// Kernel: run just BlindRotatePreAdd and copy TRLWE output to global memory
template <class brP, int casign, int cbsign, int offset>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>)
void __TestBlindRotate__(
    typename brP::targetP::T* const trlwe_out,
    const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1,
    const cufhe::NTTValue* const bk,
#ifdef USE_KEY_BUNDLE
    const cufhe::NTTValue* const one_trgsw_ntt,
    const cufhe::NTTValue* const xai_ntt,
#endif
    const cufhe::CuNTTHandler<brP::targetP::n> ntt)
{
    constexpr uint32_t N = brP::targetP::n;

    extern __shared__ char dyn_sh[];
    constexpr size_t fft_bytes = MEM4HOMGATE<typename brP::targetP>;
    typename brP::targetP::T* trlwe =
        reinterpret_cast<typename brP::targetP::T*>(dyn_sh + fft_bytes);

#ifdef USE_KEY_BUNDLE
    cufhe::__BlindRotatePreAddKeyBundle__<brP, casign, cbsign, offset>(
        trlwe, in0, in1, bk, one_trgsw_ntt, xai_ntt, ntt);
#else
    cufhe::__BlindRotatePreAdd__<brP, casign, cbsign, offset>(
        trlwe, in0, in1, bk, ntt);
#endif

    // Copy TRLWE from shared memory to global memory
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    for (uint32_t i = tid; i < (brP::targetP::k + 1) * N; i += bdim) {
        trlwe_out[i] = trlwe[i];
    }
    __threadfence();
}

int main()
{
    using brP = lvl02param;
    using tgtP = lvl2param;
    using domP = lvl0param;
    constexpr uint32_t N = tgtP::n;
    using T = tgtP::T;

    cudaSetDevice(0);
    printf("=== Full Blind Rotation: GPU vs CPU (SAME BSK) ===\n\n");

    // Generate keys
    printf("Generating keys...\n");
    SecretKey sk;
    EvalKey ek;
    ek.emplacebk<brP>(sk);
    printf("Keys generated.\n\n");

    // Convert SAME polynomial BSK to CPU FFT format
    printf("Converting BSK to CPU FFT...\n");
    const auto& bk = ek.getbk<brP>();
    // Build the CPU FFT BSK from the polynomial BSK
    auto cpu_bkfft = std::make_unique<BootstrappingKeyFFT<brP>>();
    for (uint32_t i = 0; i < bk.size(); i++) {
        for (uint32_t j = 0; j < bk[i].size(); j++) {
            (*cpu_bkfft)[i][j] = ApplyFFT2trgsw<tgtP>(bk[i][j]);
        }
    }
    printf("CPU FFT BSK ready.\n");

    // Initialize GPU FFT handler and convert BSK to GPU FFT
    printf("Converting BSK to GPU FFT...\n");
    cufhe::CuGPUFFTHandler<N>::Create();
    cufhe::CuGPUFFTHandler<N> handler;
    handler.SetDevicePointers(0);
    cufhe::InitializeNTThandlers_lvl02(1);
#ifdef USE_KEY_BUNDLE
    cufhe::BootstrappingKeyBundleToNTT<brP>(bk, 1);
    cufhe::InitializeXaiNTT_lvl02(1);
    cufhe::InitializeOneTRGSWNTT_lvl02(1);
#else
    cufhe::BootstrappingKeyToNTT<brP>(bk, 1);
#endif
    cudaDeviceSynchronize();
    printf("GPU FFT BSK ready.\n\n");

    // Get BSK device pointer
    NTTValue* d_bk = cufhe::bk_ntts_lvl02[0];

    // Allocate device memory for input and output
    typename domP::T* d_in0;
    typename domP::T* d_in1;
    T* d_trlwe;
    constexpr size_t in_size = (domP::k * domP::n + 1) * sizeof(typename domP::T);
    constexpr size_t trlwe_size = (tgtP::k + 1) * N * sizeof(T);

    cudaMalloc(&d_in0, in_size);
    cudaMalloc(&d_in1, in_size);
    cudaMalloc(&d_trlwe, trlwe_size);

    constexpr size_t shmem = MEM4HOMGATE_DYN<tgtP>;
    constexpr int nand_offset = domP::μ;
    cudaFuncSetAttribute(
        __TestBlindRotate__<brP, -1, -1, nand_offset>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);

    printf("Testing NAND gate: casign=-1, cbsign=-1, offset=mu=%d\n", (int)domP::μ);
    printf("Shared memory: %zu bytes\n\n", shmem);

    // Test all 4 input combinations
    uint8_t inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    uint8_t nand_expected[4] = {1, 1, 1, 0};
    int failures = 0;

    for (int t = 0; t < 4; t++) {
        uint8_t a = inputs[t][0];
        uint8_t b = inputs[t][1];
        uint8_t expected = nand_expected[t];

        printf("--- NAND(%d, %d) expected=%d ---\n", a, b, expected);

        // Encrypt inputs
        TLWE<domP> ct0, ct1;
        auto mu_a = a ? domP::μ : static_cast<domP::T>(-domP::μ);
        auto mu_b = b ? domP::μ : static_cast<domP::T>(-domP::μ);
        tlweSymEncrypt<domP>(ct0, mu_a, sk.key.get<domP>());
        tlweSymEncrypt<domP>(ct1, mu_b, sk.key.get<domP>());

        // ====== CPU Blind Rotation ======
        // Combine inputs as NAND: res = -ct0 - ct1 + mu
        TLWE<domP> combined;
        for (int i = 0; i <= domP::k * domP::n; i++)
            combined[i] = -ct0[i] - ct1[i];
        combined[domP::k * domP::n] += domP::μ;

        // CPU BlindRotate
        TRLWE<tgtP> cpu_trlwe;
        BlindRotate<brP>(cpu_trlwe, combined, *cpu_bkfft,
                         μpolygen<tgtP, tgtP::μ>());

        // CPU SampleExtract
        TLWE<tgtP> cpu_tlwe;
        SampleExtractIndex<tgtP>(cpu_tlwe, cpu_trlwe, 0);

        // ====== GPU Blind Rotation ======
        cudaMemcpy(d_in0, ct0.data(), in_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_in1, ct1.data(), in_size, cudaMemcpyHostToDevice);
        cudaMemset(d_trlwe, 0, trlwe_size);

        __TestBlindRotate__<brP, -1, -1, nand_offset>
            <<<1, N >> 1, shmem>>>(
                d_trlwe, d_in0, d_in1, d_bk,
#ifdef USE_KEY_BUNDLE
                cufhe::one_trgsw_ntt_devs_lvl02[0],
                cufhe::xai_ntt_devs_lvl02[0],
#endif
                *cufhe::ntt_handlers_lvl02[0]);
        cudaDeviceSynchronize();

        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }

        // Read GPU TRLWE
        std::vector<T> gpu_trlwe_flat((tgtP::k + 1) * N);
        cudaMemcpy(gpu_trlwe_flat.data(), d_trlwe, trlwe_size,
                   cudaMemcpyDeviceToHost);

        TRLWE<tgtP> gpu_trlwe;
        for (uint32_t k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++)
                gpu_trlwe[k][i] = gpu_trlwe_flat[k * N + i];

        TLWE<tgtP> gpu_tlwe;
        SampleExtractIndex<tgtP>(gpu_tlwe, gpu_trlwe, 0);

        // Compare
        const T cpu_phase = tlweSymPhase<tgtP>(cpu_tlwe, sk.key.get<tgtP>());
        const T gpu_phase = tlweSymPhase<tgtP>(gpu_tlwe, sk.key.get<tgtP>());
        auto scpu = static_cast<std::make_signed_t<T>>(cpu_phase);
        auto sgpu = static_cast<std::make_signed_t<T>>(gpu_phase);
        double cpu_frac = (double)scpu / (double)(1ULL << 63) / 2.0;
        double gpu_frac = (double)sgpu / (double)(1ULL << 63) / 2.0;
        double diff_frac = gpu_frac - cpu_frac;

        uint8_t cpu_dec = (scpu > 0) ? 1 : 0;
        uint8_t gpu_dec = (sgpu > 0) ? 1 : 0;
        if (cpu_dec != expected || gpu_dec != expected) failures++;

        // Max coefficient diff across all TRLWE polynomials
        T max_diff = 0;
        int diff_count = 0;
        for (uint32_t k = 0; k <= tgtP::k; k++)
            for (uint32_t i = 0; i < N; i++) {
                T cpu_val = cpu_trlwe[k][i];
                T gpu_val = gpu_trlwe_flat[k * N + i];
                if (cpu_val != gpu_val) diff_count++;
                auto d = static_cast<std::make_signed_t<T>>(gpu_val - cpu_val);
                T ad = (d < 0) ? static_cast<T>(-d) : static_cast<T>(d);
                if (ad > max_diff) max_diff = ad;
            }

        printf("  CPU: phase=%.6f (sign=%c) dec=%d %s\n",
               cpu_frac, scpu > 0 ? '+' : '-', cpu_dec,
               cpu_dec == expected ? "PASS" : "FAIL");
        printf("  GPU: phase=%.6f (sign=%c) dec=%d %s\n",
               gpu_frac, sgpu > 0 ? '+' : '-', gpu_dec,
               gpu_dec == expected ? "PASS" : "FAIL");
        printf("  diff(phase)=%.6f, diff_coeffs=%d/%d, max_diff=2^%.1f\n\n",
               diff_frac, diff_count, (int)((tgtP::k + 1) * N),
               max_diff > 0 ? log2((double)max_diff) : 0.0);
    }

    cudaFree(d_in0);
    cudaFree(d_in1);
    cudaFree(d_trlwe);
#ifdef USE_KEY_BUNDLE
    cufhe::DeleteXaiNTT_lvl02();
    cufhe::DeleteOneTRGSWNTT_lvl02();
#endif
    cufhe::DeleteBootstrappingKeyNTT_lvl02(1);
    cufhe::CuGPUFFTHandler<N>::Destroy();

    printf("Results: %d passed, %d failed\n", 4 - failures, failures);
    return failures == 0 ? 0 : 1;
}
