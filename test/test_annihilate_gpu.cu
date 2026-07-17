#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/error_gpu.cuh>
#include <iostream>
#include <limits>
#include <memory>
#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>
#include <type_traits>

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
    TFHEpp::AnnihilateKey<P>& out, const cufhe::AnnihilateKeyPolynomial<P>& in)
{
    for (uint32_t bit = 0; bit < P::nbit; bit++)
        for (uint32_t key_idx = 0; key_idx < P::k; key_idx++)
            TFHEpp::ApplyFFT2halftrgsw<P>(out[bit][key_idx], in[bit][key_idx]);
}

template <class T>
uint64_t AbsTorusDiff(const T diff)
{
    using SignedT = std::make_signed_t<T>;
    const SignedT signed_diff = static_cast<SignedT>(diff);
    return signed_diff < 0 ? static_cast<uint64_t>(-signed_diff)
                           : static_cast<uint64_t>(signed_diff);
}

template <class P>
bool RunAnnihilateCase(const char* name)
{
    constexpr uint32_t num_tests = 4;
    std::cout << "=== " << name << " annihilate GPU test ===" << std::endl;

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
        auto input = std::make_unique<TFHEpp::TRLWE<P>>();
        const char* input_kind = nullptr;
        if (test == 0) {
            (*input)[P::k][0] = P::μ;
            input_kind = "trivial body";
        }
        else if (test <= P::k) {
            const uint32_t component = test - 1;
            for (uint32_t i = 0; i < P::n; i++)
                (*input)[component][i] =
                    static_cast<typename P::T>(i + 1)
                    << (std::numeric_limits<typename P::T>::digits / 2);
            input_kind = test == 1 ? "first mask" : "second mask";
        }
        else {
            std::array<typename P::T, P::n> encoded{};
            encoded[0] = P::μ;
            TFHEpp::trlweSymEncrypt<P>(*input, encoded, sk->key.get<P>());
            input_kind = "encrypted";
        }

        auto cpu_out = std::make_unique<TFHEpp::TRLWE<P>>();
        TFHEpp::AnnihilateKeySwitching<P>(*cpu_out, *input, *cpu_ahk);

        CUDA_CHECK(cudaMemcpyAsync(d_in, input->data(), trlwe_bytes,
                                   cudaMemcpyHostToDevice, st));
        cufhe::AnnihilateKeySwitching<P>(d_out, d_in, st, 0);

        auto gpu_out = std::make_unique<TFHEpp::TRLWE<P>>();
        CUDA_CHECK(cudaMemcpyAsync(gpu_out->data(), d_out, trlwe_bytes,
                                   cudaMemcpyDeviceToHost, st));
        CUDA_CHECK(cudaStreamSynchronize(st));

        const auto cpu_phase =
            TFHEpp::trlwePhase<P>(*cpu_out, sk->key.get<P>());
        const auto gpu_phase =
            TFHEpp::trlwePhase<P>(*gpu_out, sk->key.get<P>());
        uint64_t max_abs_diff = 0;
        for (uint32_t i = 0; i < P::n; i++)
            max_abs_diff = std::max(
                max_abs_diff,
                AbsTorusDiff<typename P::T>(gpu_phase[i] - cpu_phase[i]));
        constexpr uint64_t max_allowed =
            sizeof(typename P::T) == 4 ? (1ULL << 26) : (1ULL << 56);
        const bool pass = max_abs_diff <= max_allowed;
        const double max_log2 =
            max_abs_diff == 0 ? 0.0
                              : std::log2(static_cast<double>(max_abs_diff));
        std::cout << "case " << test << " (" << input_kind
                  << "): phase max abs diff=2^" << max_log2
                  << (pass ? " pass" : " mismatch") << std::endl;
        if (!pass) {
            for (uint32_t i = 0; i < 4; i++)
                std::cout << "  phase[" << i << "] cpu=0x" << std::hex
                          << cpu_phase[i] << " gpu=0x" << gpu_phase[i]
                          << std::dec << std::endl;
        }
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
#ifdef USE_DIFFERENT_AH_PARAM
    ok = RunAnnihilateCase<TFHEpp::cbAHlvl2param>("cbAHlvl2param") && ok;
#else
    ok = RunAnnihilateCase<TFHEpp::AHlvl2param>("AHlvl2param") && ok;
#endif
    std::cout << (ok ? "ALL ANNIHILATE TESTS PASSED"
                     : "ANNIHILATE TESTS FAILED")
              << std::endl;
    return ok ? 0 : 1;
}
