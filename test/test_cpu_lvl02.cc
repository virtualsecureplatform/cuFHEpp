/**
 * CPU-only test: verify TFHEpp gate bootstrapping works for lvl02param.
 * If this fails, the issue is in the parameter definitions, not the GPU code.
 */

#include <cstdio>
#include <cstdint>
#include <array>

#include <params.hpp>
#include <key.hpp>
#include <tlwe.hpp>
#include <gate.hpp>
#include <cloudkey.hpp>

int main()
{
    printf("Generating keys for lvl02param CPU test...\n");
    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek;

    printf("Generating BSK (FFT domain) for lvl02param...\n");
    ek.emplacebkfft<TFHEpp::lvl02param>(sk);

    printf("Generating KSK for lvl20param...\n");
    ek.emplaceiksk<TFHEpp::lvl20param>(sk);

    printf("Keys generated. Running CPU gate tests...\n");

    using domP = TFHEpp::lvl0param;
    using brP = TFHEpp::lvl02param;
    using iksP = TFHEpp::lvl20param;
    constexpr auto mu_tgt = TFHEpp::lvl2param::μ;

    int pass = 0, fail = 0;

    // Test all 4 input combinations for NAND gate
    uint8_t inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    uint8_t nand_expected[4] = {1, 1, 1, 0};

    for (int t = 0; t < 4; t++) {
        uint8_t a = inputs[t][0];
        uint8_t b = inputs[t][1];
        uint8_t expected = nand_expected[t];

        // Encrypt
        TFHEpp::TLWE<domP> ct0, ct1, result;
        auto mu_a = a ? domP::μ : static_cast<typename domP::T>(-domP::μ);
        auto mu_b = b ? domP::μ : static_cast<typename domP::T>(-domP::μ);
        TFHEpp::tlweSymEncrypt<domP>(ct0, mu_a, sk.key.get<domP>());
        TFHEpp::tlweSymEncrypt<domP>(ct1, mu_b, sk.key.get<domP>());

        // Run CPU NAND
        TFHEpp::HomNAND<brP, mu_tgt, iksP>(result, ct0, ct1, ek);

        // Decrypt
        uint8_t pt = TFHEpp::tlweSymDecrypt<domP>(result, sk.key.get<domP>());
        if (pt == expected) {
            pass++;
            printf("  NAND(%d, %d) = %d [expected %d] PASS\n", a, b, pt, expected);
        } else {
            fail++;
            printf("  NAND(%d, %d) = %d [expected %d] FAIL\n", a, b, pt, expected);
        }
    }

    // Test AND gate
    uint8_t and_expected[4] = {0, 0, 0, 1};
    for (int t = 0; t < 4; t++) {
        uint8_t a = inputs[t][0];
        uint8_t b = inputs[t][1];
        uint8_t expected = and_expected[t];

        TFHEpp::TLWE<domP> ct0, ct1, result;
        auto mu_a = a ? domP::μ : static_cast<typename domP::T>(-domP::μ);
        auto mu_b = b ? domP::μ : static_cast<typename domP::T>(-domP::μ);
        TFHEpp::tlweSymEncrypt<domP>(ct0, mu_a, sk.key.get<domP>());
        TFHEpp::tlweSymEncrypt<domP>(ct1, mu_b, sk.key.get<domP>());

        TFHEpp::HomAND<brP, mu_tgt, iksP>(result, ct0, ct1, ek);

        uint8_t pt = TFHEpp::tlweSymDecrypt<domP>(result, sk.key.get<domP>());
        if (pt == expected) {
            pass++;
            printf("  AND(%d, %d) = %d [expected %d] PASS\n", a, b, pt, expected);
        } else {
            fail++;
            printf("  AND(%d, %d) = %d [expected %d] FAIL\n", a, b, pt, expected);
        }
    }

    printf("\nResults: %d passed, %d failed\n", pass, fail);
    printf("%s\n", fail == 0 ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return fail == 0 ? 0 : 1;
}
