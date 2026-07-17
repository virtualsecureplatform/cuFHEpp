#include <cstdint>
#include <include/cufhe_gpu.cuh>
#include <iostream>
#include <string_view>

namespace {

template <class P>
bool TestNandTruthTable(const std::string_view label,
                        const TFHEpp::SecretKey& sk, cufhe::Stream st)
{
    cufhe::Ctxt<P> in0;
    cufhe::Ctxt<P> in1;
    cufhe::Ctxt<P> out;

    bool passed = true;
    for (uint8_t lhs = 0; lhs <= 1; lhs++) {
        for (uint8_t rhs = 0; rhs <= 1; rhs++) {
            TFHEpp::tlweSymEncrypt<P>(in0.tlwehost, lhs ? P::μ : -P::μ,
                                      sk.key.get<P>());
            TFHEpp::tlweSymEncrypt<P>(in1.tlwehost, rhs ? P::μ : -P::μ,
                                      sk.key.get<P>());

            cufhe::Nand<P>(out, in0, in1, st);
            cufhe::Synchronize();

            const bool actual =
                TFHEpp::tlweSymDecrypt<P>(out.tlwehost, sk.key.get<P>());
            const bool expected = !(lhs && rhs);
            if (actual != expected) {
                std::cerr << label << " NAND failed for " << unsigned(lhs)
                          << ", " << unsigned(rhs) << ": expected " << expected
                          << ", got " << actual << '\n';
                passed = false;
            }
        }
    }

    std::cout << label << " NAND: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

template <class P>
bool TestMuxTruthTable(const std::string_view label,
                       const TFHEpp::SecretKey& sk, cufhe::Stream st)
{
    cufhe::Ctxt<P> condition;
    cufhe::Ctxt<P> if_true;
    cufhe::Ctxt<P> if_false;
    cufhe::Ctxt<P> out;

    bool passed = true;
    for (uint8_t select = 0; select <= 1; select++) {
        for (uint8_t true_value = 0; true_value <= 1; true_value++) {
            for (uint8_t false_value = 0; false_value <= 1; false_value++) {
                TFHEpp::tlweSymEncrypt<P>(
                    condition.tlwehost, select ? P::μ : -P::μ, sk.key.get<P>());
                TFHEpp::tlweSymEncrypt<P>(if_true.tlwehost,
                                          true_value ? P::μ : -P::μ,
                                          sk.key.get<P>());
                TFHEpp::tlweSymEncrypt<P>(if_false.tlwehost,
                                          false_value ? P::μ : -P::μ,
                                          sk.key.get<P>());

                cufhe::Mux<P>(out, condition, if_true, if_false, st);
                cufhe::Synchronize();

                const bool actual =
                    TFHEpp::tlweSymDecrypt<P>(out.tlwehost, sk.key.get<P>());
                const bool expected = select ? true_value : false_value;
                if (actual != expected) {
                    std::cerr << label << " MUX failed for " << unsigned(select)
                              << ", " << unsigned(true_value) << ", "
                              << unsigned(false_value) << ": expected "
                              << expected << ", got " << actual << '\n';
                    passed = false;
                }
            }
        }
    }

    std::cout << label << " MUX: " << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool ValidateBlockKey(const TFHEpp::SecretKey& sk)
{
    static_assert(TFHEpp::lvl0param::ell > 1);
    static_assert(TFHEpp::lvl1param::k * TFHEpp::lvl1param::n >=
                  TFHEpp::lvl0param::n);

    const auto& lvl0 = sk.key.get<TFHEpp::lvl0param>();
    const auto& lvl1 = sk.key.get<TFHEpp::lvl1param>();
    for (uint32_t block = 0;
         block < TFHEpp::lvl0param::n / TFHEpp::lvl0param::ell; block++) {
        uint32_t weight = 0;
        for (uint32_t offset = 0; offset < TFHEpp::lvl0param::ell; offset++) {
            const uint32_t index = block * TFHEpp::lvl0param::ell + offset;
            weight += lvl0[index];
            if (lvl0[index] != lvl1[index]) {
                std::cerr << "Secret-key subset prefix mismatch at " << index
                          << '\n';
                return false;
            }
        }
        if (weight > 1) {
            std::cerr << "Block " << block << " has Hamming weight " << weight
                      << '\n';
            return false;
        }
    }
    std::cout << "Block-binary key structure: PASS\n";
    return true;
}

bool TestSubsetSampleExtraction(const TFHEpp::SecretKey& sk, cufhe::Stream st)
{
    cufhe::cuFHETRLWElvl1 input;
    cufhe::Ctxt<TFHEpp::lvl0param> output;

    bool passed = true;
    for (uint8_t message = 0; message <= 1; message++) {
        TFHEpp::Polynomial<TFHEpp::lvl1param> encoded{};
        encoded[0] = message ? TFHEpp::lvl1param::μ : -TFHEpp::lvl1param::μ;
        TFHEpp::trlweSymEncrypt<TFHEpp::lvl1param>(
            input.trlwehost, encoded, sk.key.get<TFHEpp::lvl1param>());

        cufhe::SampleExtractAndKeySwitch(output, input, st);
        cufhe::Synchronize();
        const bool actual = TFHEpp::tlweSymDecrypt<TFHEpp::lvl0param>(
            output.tlwehost, sk.key.get<TFHEpp::lvl0param>());
        if (actual != message) {
            std::cerr << "Subset sample extraction failed for "
                      << unsigned(message) << ": got " << actual << '\n';
            passed = false;
        }
    }

    std::cout << "Subset sample extraction: " << (passed ? "PASS" : "FAIL")
              << '\n';
    return passed;
}

}  // namespace

int main()
{
    cudaSetDevice(0);
    cufhe::SetGPUNum(1);

    TFHEpp::SecretKey sk;
    if (!ValidateBlockKey(sk)) return 1;

    TFHEpp::EvalKey ek(sk);
    ek.emplacebk<TFHEpp::lvl01param>(sk);
    ek.emplacesubiksk<TFHEpp::lvl10param>(sk);
    cufhe::Initialize(ek);

    cufhe::Stream st(0);
    st.Create();
    const bool passed = TestSubsetSampleExtraction(sk, st) &&
                        TestNandTruthTable<TFHEpp::lvl0param>("lvl0", sk, st) &&
                        TestNandTruthTable<TFHEpp::lvl1param>("lvl1", sk, st) &&
                        TestMuxTruthTable<TFHEpp::lvl0param>("lvl0", sk, st) &&
                        TestMuxTruthTable<TFHEpp::lvl1param>("lvl1", sk, st);
    st.Destroy();
    cufhe::CleanUp();

    return passed ? 0 : 1;
}
