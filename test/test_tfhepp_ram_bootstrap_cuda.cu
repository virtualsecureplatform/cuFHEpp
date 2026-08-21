#include <array>
#include <cstdlib>
#include <iostream>

#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trlwe.hpp>

namespace {

using cufhe::CuSafeCall__;
using Domain = TFHEpp::lvl0param;
using Target = TFHEpp::lvl1param;
using Bootstrap = TFHEpp::lvl01param;
using TLWE = TFHEpp::TLWE<Domain>;
using TRLWE = TFHEpp::TRLWE<Target>;
using TargetTLWE = TFHEpp::TLWE<Target>;

bool decryptsTo(const TRLWE& ciphertext, bool expected,
                const TFHEpp::SecretKey& secretKey)
{
    return TFHEpp::trlweSymDecrypt<Target>(
               ciphertext, secretKey.key.get<Target>())[0] == expected;
}

bool runBootstrapCase(const TLWE& input, bool expected,
                      const TFHEpp::EvalKey& evalKey,
                      const TFHEpp::SecretKey& secretKey,
                      Domain::T* deviceInput, Target::T* deviceOutput,
                      const char* label)
{
    TRLWE cpuOutput;
    TFHEpp::BlindRotate<Bootstrap>(
        cpuOutput, input, evalKey.getbkfft<Bootstrap>(),
        TFHEpp::μpolygen<Target, Target::μ>());
    if (!decryptsTo(cpuOutput, expected, secretKey)) {
        std::cerr << "CPU " << label << " failed for " << expected << '\n';
        return false;
    }

    CuSafeCall(cudaMemcpy(deviceInput, &input, sizeof(input),
                          cudaMemcpyHostToDevice));
    cufhe::BootstrapTLWE2TRLWE(deviceOutput, deviceInput, Target::μ,
                                cudaStreamDefault, 0);
    CuSafeCall(cudaDeviceSynchronize());

    TRLWE gpuOutput;
    CuSafeCall(cudaMemcpy(&gpuOutput, deviceOutput, sizeof(gpuOutput),
                          cudaMemcpyDeviceToHost));
    if (!decryptsTo(gpuOutput, expected, secretKey)) {
        std::cerr << "CUDA " << label << " failed for " << expected << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    int deviceCount = 0;
    CuSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA device is available\n";
        return EXIT_FAILURE;
    }
    CuSafeCall(cudaSetDevice(0));

    TFHEpp::SecretKey secretKey;
    TFHEpp::EvalKey evalKey(secretKey);
    evalKey.emplacebk<Bootstrap>(secretKey);
    evalKey.emplacebkfft<Bootstrap>(secretKey);
    // RAM sample extraction uses the complete key, not its prefix subset.
    evalKey.emplaceiksk<TFHEpp::lvl10param>(secretKey);
#ifdef USE_SUBSET_KEY
    evalKey.emplacesubiksk<TFHEpp::lvl10param>(secretKey);
#else
    evalKey.emplaceiksk<TFHEpp::lvl10param>(secretKey);
#endif

    cufhe::SetGPUNum(1);
    cufhe::Initialize(evalKey);

    Domain::T* deviceInput = nullptr;
    Target::T* deviceOutput = nullptr;
    CuSafeCall(cudaMalloc(&deviceInput, sizeof(TLWE)));
    CuSafeCall(cudaMalloc(&deviceOutput, sizeof(TRLWE)));

    bool passed = true;
    // Repetition covers fresh encryption noise as well as both plaintext signs.
    for (unsigned repetition = 0; repetition < 8; ++repetition) {
        for (const bool message : std::array<bool, 2>{false, true}) {
            TLWE input;
            TFHEpp::tlweSymEncrypt<Domain>(
                input, message ? Domain::μ : -Domain::μ,
                secretKey.key.get<Domain>());

            passed = runBootstrapCase(input, message, evalKey, secretKey,
                                      deviceInput, deviceOutput,
                                      "fresh-TLWE") && passed;

            TFHEpp::Polynomial<Target> polynomial{};
            polynomial[0] = message ? Target::μ : -Target::μ;
            TRLWE packed;
            TFHEpp::trlweSymEncrypt<Target>(packed, polynomial,
                                             secretKey.key.get<Target>());
            TargetTLWE extracted;
            TFHEpp::SampleExtractIndex<Target>(extracted, packed, 0);
            TLWE keySwitched;
            TFHEpp::IdentityKeySwitch<TFHEpp::lvl10param>(
                keySwitched, extracted,
                evalKey.getiksk<TFHEpp::lvl10param>());
            passed = runBootstrapCase(keySwitched, message, evalKey, secretKey,
                                      deviceInput, deviceOutput,
                                      "sample-extract-and-keyswitch") && passed;
        }
    }

    CuSafeCall(cudaFree(deviceOutput));
    CuSafeCall(cudaFree(deviceInput));
    cufhe::CleanUp();
    std::cout << "TFHEpp RAM bootstrap CPU/CUDA regression "
              << (passed ? "passed" : "failed") << '\n';
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
