/**
 * Compare CPU vs GPU gate bootstrapping for lvl02param.
 * Runs a single NAND gate on both paths and prints detailed results.
 */

#include <cstdio>
#include <cstdint>
#include <array>
#include <cmath>

#include <params.hpp>
#include <key.hpp>
#include <tlwe.hpp>
#include <gate.hpp>
#include <cloudkey.hpp>

#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/keyswitch_gpu.cuh>

using namespace TFHEpp;

// Print TLWE phase info
template <class P>
void print_tlwe_phase(const char* label, const TLWE<P>& ct, const SecretKey& sk)
{
    // Compute phase = b - a·s
    using T = typename P::T;
    T phase = ct[P::k * P::n];  // b
    for (int i = 0; i < P::k * P::n; i++) {
        phase -= ct[i] * sk.key.get<P>()[i];
    }
    // Interpret as signed for display
    using ST = std::make_signed_t<T>;
    ST sphase = static_cast<ST>(phase);
    double frac = static_cast<double>(sphase) /
                  static_cast<double>(static_cast<T>(1) << (std::numeric_limits<T>::digits - 1)) /
                  2.0;

    printf("  %s: phase=0x%016llx (signed=%lld, frac=%.6f)\n",
           label, (unsigned long long)phase, (long long)sphase, frac);
    printf("    mu=0x%016llx, dist_from_+mu=0x%016llx, dist_from_-mu=0x%016llx\n",
           (unsigned long long)(T)P::μ,
           (unsigned long long)(T)(phase - (T)P::μ),
           (unsigned long long)(T)(phase + (T)P::μ));

    // Decrypt
    uint8_t pt = tlweSymDecrypt<P>(ct, sk.key.get<P>());
    printf("    decrypted=%d\n", pt);
}

int main()
{
    printf("=== CPU vs GPU NAND comparison for lvl02param ===\n\n");

    // Generate keys
    printf("Generating keys...\n");
    SecretKey sk;
    EvalKey ek;

    printf("Generating polynomial BSK for GPU...\n");
    ek.emplacebk<lvl02param>(sk);

    printf("Generating FFT BSK for CPU...\n");
    ek.emplacebkfft<lvl02param>(sk);

    printf("Generating KSK...\n");
    ek.emplaceiksk<lvl20param>(sk);

    printf("Keys generated.\n\n");

    // Initialize GPU
    printf("Initializing GPU...\n");
    cudaSetDevice(0);
    cufhe::Initialize_lvl02(ek, sk);

    // Create stream
    cufhe::Stream st;
    st.Create();

    // Test all 4 input combinations
    uint8_t inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    uint8_t nand_expected[4] = {1, 1, 1, 0};

    int cpu_pass = 0, gpu_pass = 0;

    for (int t = 0; t < 4; t++) {
        uint8_t a = inputs[t][0];
        uint8_t b = inputs[t][1];
        uint8_t expected = nand_expected[t];

        printf("--- NAND(%d, %d) expected=%d ---\n", a, b, expected);

        // Encrypt inputs
        TLWE<lvl0param> ct0, ct1;
        auto mu_a = a ? lvl0param::μ : static_cast<lvl0param::T>(-lvl0param::μ);
        auto mu_b = b ? lvl0param::μ : static_cast<lvl0param::T>(-lvl0param::μ);
        tlweSymEncrypt<lvl0param>(ct0, mu_a, sk.key.get<lvl0param>());
        tlweSymEncrypt<lvl0param>(ct1, mu_b, sk.key.get<lvl0param>());

        printf("  Input ciphertexts:\n");
        print_tlwe_phase<lvl0param>("ct0", ct0, sk);
        print_tlwe_phase<lvl0param>("ct1", ct1, sk);

        // CPU NAND
        TLWE<lvl0param> cpu_result;
        HomNAND<lvl02param, lvl2param::μ, lvl20param>(cpu_result, ct0, ct1, ek);
        printf("\n  CPU NAND result:\n");
        print_tlwe_phase<lvl0param>("cpu", cpu_result, sk);

        uint8_t cpu_pt = tlweSymDecrypt<lvl0param>(cpu_result, sk.key.get<lvl0param>());
        if (cpu_pt == expected) cpu_pass++;

        // GPU NAND
        cufhe::Ctxt<lvl0param> gpu_ct0, gpu_ct1, gpu_result;
        // Copy input to Ctxt
        for (int i = 0; i <= lvl0param::k * lvl0param::n; i++) {
            gpu_ct0.tlwehost[i] = ct0[i];
            gpu_ct1.tlwehost[i] = ct1[i];
        }

        cufhe::Nand_lvl02(gpu_result, gpu_ct0, gpu_ct1, st);
        cudaStreamSynchronize(st.st());

        // Copy result back to TLWE for phase computation
        TLWE<lvl0param> gpu_result_tlwe;
        for (int i = 0; i <= lvl0param::k * lvl0param::n; i++) {
            gpu_result_tlwe[i] = gpu_result.tlwehost[i];
        }
        printf("\n  GPU NAND result:\n");
        print_tlwe_phase<lvl0param>("gpu", gpu_result_tlwe, sk);

        uint8_t gpu_pt = tlweSymDecrypt<lvl0param>(gpu_result_tlwe, sk.key.get<lvl0param>());
        if (gpu_pt == expected) gpu_pass++;

        // Compare
        printf("\n  CPU decrypted=%d (expected=%d) %s\n", cpu_pt, expected,
               cpu_pt == expected ? "PASS" : "FAIL");
        printf("  GPU decrypted=%d (expected=%d) %s\n", gpu_pt, expected,
               gpu_pt == expected ? "PASS" : "FAIL");

        // Compare raw output bytes
        int diff_count = 0;
        for (int i = 0; i <= lvl0param::k * lvl0param::n; i++) {
            if (cpu_result[i] != gpu_result_tlwe[i]) diff_count++;
        }
        printf("  Differing TLWE coefficients: %d / %d\n\n",
               diff_count, lvl0param::k * lvl0param::n + 1);
    }

    printf("=== Summary: CPU %d/4, GPU %d/4 ===\n", cpu_pass, gpu_pass);

    st.Destroy();
    cufhe::CleanUp_lvl02();

    return (cpu_pass == 4 && gpu_pass == 4) ? 0 : 1;
}
