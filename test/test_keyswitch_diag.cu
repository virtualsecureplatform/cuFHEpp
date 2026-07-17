/**
 * Diagnostic: compare GPU vs CPU key switching for lvl20param.
 * Encrypts under lvl2 key, key-switches to lvl0 on both CPU and GPU.
 */

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>

#include <params.hpp>
#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/tlwe.hpp>

#include <include/keyswitch_gpu.cuh>
#include <include/cufhe_gpu.cuh>

using namespace TFHEpp;

// GPU kernel that runs KeySwitchFromTLWE
template <class P>
__global__ void __TestKeySwitch__(typename P::targetP::T* const out,
                                   const typename P::domainP::T* const in,
                                   const typename P::targetP::T* const ksk)
{
    cufhe::KeySwitchFromTLWE<P>(out, in, ksk);
    __threadfence();
}

int main()
{
    cudaSetDevice(0);
    printf("=== Key switching diagnostic: GPU vs CPU for lvl20param ===\n\n");

    using iksP = lvl20param;
    using domP = typename iksP::domainP;  // lvl2param
    using tgtP = typename iksP::targetP;  // lvl0param

    printf("Domain: n=%u, k=%u, T=%lu bits\n", domP::n, domP::k,
           (unsigned long)(sizeof(typename domP::T) * 8));
    printf("Target: n=%u, k=%u, T=%lu bits\n", tgtP::n, tgtP::k,
           (unsigned long)(sizeof(typename tgtP::T) * 8));
    printf("KS params: t=%u, basebit=%u\n\n", iksP::t, iksP::basebit);

    // Generate keys
    SecretKey sk;
    EvalKey ek;
    ek.emplaceiksk<iksP>(sk);

    // Upload KSK to GPU
    cufhe::KeySwitchingKeyToDevice_lvl20(ek.getiksk<iksP>(), 1);
    cudaDeviceSynchronize();

    constexpr int in_size = domP::k * domP::n + 1;
    constexpr int out_size = tgtP::k * tgtP::n + 1;

    typename domP::T* d_in;
    typename tgtP::T* d_out;
    cudaMalloc(&d_in, in_size * sizeof(typename domP::T));
    cudaMalloc(&d_out, out_size * sizeof(typename tgtP::T));

    int pass = 0, fail = 0;
    for (int trial = 0; trial < 20; trial++) {
        uint8_t pt = trial % 2;
        auto mu = pt ? domP::μ : static_cast<typename domP::T>(-domP::μ);
        TLWE<domP> ct_dom;
        tlweSymEncrypt<domP>(ct_dom, mu, sk.key.get<domP>());

        // === CPU key switch ===
        TLWE<tgtP> cpu_result = {};
        IdentityKeySwitch<iksP>(cpu_result, ct_dom, ek.getiksk<iksP>());

        // === GPU key switch ===
        cudaMemcpy(d_in, ct_dom.data(), in_size * sizeof(typename domP::T),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, out_size * sizeof(typename tgtP::T));
        __TestKeySwitch__<iksP><<<1, 1024>>>(d_out, d_in, cufhe::ksk_devs_lvl20[0]);
        cudaDeviceSynchronize();

        TLWE<tgtP> gpu_result;
        cudaMemcpy(gpu_result.data(), d_out, out_size * sizeof(typename tgtP::T),
                   cudaMemcpyDeviceToHost);

        // Compute phases
        typename tgtP::T cpu_phase = cpu_result[tgtP::k * tgtP::n];
        typename tgtP::T gpu_phase = gpu_result[tgtP::k * tgtP::n];
        for (uint32_t i = 0; i < tgtP::k * tgtP::n; i++) {
            cpu_phase -= cpu_result[i] * static_cast<typename tgtP::T>(
                             sk.key.get<tgtP>()[i]);
            gpu_phase -= gpu_result[i] * static_cast<typename tgtP::T>(
                             sk.key.get<tgtP>()[i]);
        }

        auto scpu = static_cast<std::make_signed_t<typename tgtP::T>>(cpu_phase);
        auto sgpu = static_cast<std::make_signed_t<typename tgtP::T>>(gpu_phase);
        uint8_t cpu_dec = (scpu > 0) ? 1 : 0;
        uint8_t gpu_dec = (sgpu > 0) ? 1 : 0;

        // Max coefficient diff
        typename tgtP::T max_diff = 0;
        int diff_count = 0;
        for (int i = 0; i < out_size; i++) {
            if (cpu_result[i] != gpu_result[i]) diff_count++;
            auto d = static_cast<std::make_signed_t<typename tgtP::T>>(
                gpu_result[i] - cpu_result[i]);
            auto ad = (d < 0) ? static_cast<typename tgtP::T>(-d) : static_cast<typename tgtP::T>(d);
            if (ad > max_diff) max_diff = ad;
        }

        bool ok = (cpu_dec == pt) && (gpu_dec == pt);
        printf("Trial %2d (pt=%d): CPU_phase=%+6d(dec=%d) GPU_phase=%+6d(dec=%d) diff_coeffs=%d/%d max_diff=%u %s\n",
               trial, pt, (int)scpu, cpu_dec, (int)sgpu, gpu_dec,
               diff_count, out_size, (unsigned)max_diff,
               ok ? "PASS" : (gpu_dec != pt ? "FAIL-GPU" : "FAIL-CPU"));

        if (ok) pass++;
        else fail++;
    }

    printf("\nResults: %d pass, %d fail\n", pass, fail);

    cudaFree(d_in);
    cudaFree(d_out);
    cufhe::DeleteKeySwitchingKey_lvl20(1);
    return (fail == 0) ? 0 : 1;
}
