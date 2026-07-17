#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>

#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>

#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/error_gpu.cuh>

namespace {

#define CUDA_CHECK(err) cufhe::CuSafeCall__(err, __FILE__, __LINE__)

template <class P>
void InitializeHandlers()
{
    if constexpr (P::n == TFHEpp::lvl1param::n)
        cufhe::InitializeNTThandlers(1);
    else
        cufhe::InitializeNTThandlers_lvl02(1);
}

template <class P>
void PolynomialAnnihilateKeyToCPUFFT(
    TFHEpp::AnnihilateKey<P>& out,
    const cufhe::AnnihilateKeyPolynomial<P>& in)
{
    for (uint32_t bit = 0; bit < P::nbit; bit++)
        for (uint32_t key_idx = 0; key_idx < P::k; key_idx++)
            TFHEpp::ApplyFFT2halftrgsw<P>(out[bit][key_idx],
                                          in[bit][key_idx]);
}

template <class P>
bool RunAnnihilateCase(const char* name)
{
    constexpr uint32_t num_tests = 4;
    std::cout << "=== " << name << " annihilate GPU test ===" << std::endl;

    std::default_random_engine engine(0xC0FFEEu + P::n);
    std::uniform_int_distribution<uint32_t> binary(0, 1);

    auto sk = std::make_unique<TFHEpp::SecretKey>();
    auto gpu_ahk = std::make_unique<cufhe::AnnihilateKeyPolynomial<P>>();
    cufhe::AnnihilateKeyPolynomialGen<P>(*gpu_ahk, *sk);
    auto cpu_ahk = std::make_unique<TFHEpp::AnnihilateKey<P>>();
    PolynomialAnnihilateKeyToCPUFFT<P>(*cpu_ahk, *gpu_ahk);

    cudaSetDevice(0);
    InitializeHandlers<P>();
    cufhe::AnnihilateKeyPolynomialToDevice<P>(*gpu_ahk, 1);

    typename P::T* d_in = nullptr;
    typename P::T* d_out = nullptr;
    constexpr size_t trlwe_bytes = sizeof(TFHEpp::TRLWE<P>);
    CUDA_CHECK(cudaMalloc(&d_in, trlwe_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, trlwe_bytes));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));

    bool ok = true;
    for (uint32_t test = 0; test < num_tests; test++) {
        std::array<uint8_t, P::n> plain{};
        std::array<typename P::T, P::n> encoded{};
        for (uint32_t i = 0; i < P::n; i++) {
            plain[i] = binary(engine) > 0;
            encoded[i] = plain[i] ? P::μ : -P::μ;
        }

        TFHEpp::TRLWE<P> input;
        TFHEpp::trlweSymEncrypt<P>(input, encoded, sk->key.get<P>());

        TFHEpp::TRLWE<P> cpu_out;
        TFHEpp::AnnihilateKeySwitching<P>(cpu_out, input, *cpu_ahk);

        CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), trlwe_bytes,
                                   cudaMemcpyHostToDevice, st));
        cufhe::AnnihilateKeySwitching<P>(d_out, d_in, st, 0);

        TFHEpp::TRLWE<P> gpu_out;
        CUDA_CHECK(cudaMemcpyAsync(gpu_out.data(), d_out, trlwe_bytes,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        const auto cpu_dec =
            TFHEpp::trlweSymDecrypt<P>(cpu_out, sk->key.get<P>());
        const auto gpu_dec =
            TFHEpp::trlweSymDecrypt<P>(gpu_out, sk->key.get<P>());

        const bool expected = plain[0] > 0;
        const bool pass = cpu_dec[0] == expected && gpu_dec[0] == expected;
        std::cout << "case " << test << ": expected=" << expected
                  << " cpu=" << cpu_dec[0] << " gpu=" << gpu_dec[0]
                  << (pass ? " pass" : " mismatch") << std::endl;
        ok = ok && pass;
    }

    CUDA_CHECK(cudaStreamDestroy(st));
    cudaFree(d_in);
    cudaFree(d_out);
    cufhe::DeleteAnnihilateKey<P>(1);

    return ok;
}

#undef CUDA_CHECK

}  // namespace

int main()
{
    bool ok = true;
    ok = RunAnnihilateCase<TFHEpp::AHlvl1param>("AHlvl1param") && ok;
    ok = RunAnnihilateCase<TFHEpp::AHlvl2param>("AHlvl2param") && ok;
    std::cout << (ok ? "ALL ANNIHILATE TESTS PASSED"
                     : "ANNIHILATE TESTS FAILED")
              << std::endl;
    return ok ? 0 : 1;
}
