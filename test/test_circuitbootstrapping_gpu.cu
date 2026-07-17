#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>

#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/ntt_small_modulus.cuh>

namespace {

#define CUDA_CHECK(err) cufhe::CuSafeCall__(err, __FILE__, __LINE__)

template <class P>
constexpr uint32_t CBRows()
{
    static_assert(P::l == P::lₐ,
                  "This test expects the standard CB row layout");
    return (P::k + 1) * P::l;
}

template <class P>
constexpr size_t TRLWEElementCount()
{
    return static_cast<size_t>(P::k + 1) * P::n;
}

template <class P>
constexpr size_t CBElemCount()
{
    return static_cast<size_t>(CBRows<P>()) * TRLWEElementCount<P>();
}

template <class brP>
void InitializeBR()
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl1param::n) {
        cufhe::InitializeNTThandlers(1);
#ifdef USE_KEY_BUNDLE
        cufhe::InitializeXaiNTT(1);
        cufhe::InitializeOneTRGSWNTT(1);
#endif
    }
    else {
        cufhe::InitializeNTThandlers_lvl02(1);
#ifdef USE_KEY_BUNDLE
        cufhe::InitializeXaiNTT_lvl02(1);
        cufhe::InitializeOneTRGSWNTT_lvl02(1);
#endif
    }
}

template <class brP>
void DeleteBR()
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl1param::n) {
#ifdef USE_KEY_BUNDLE
        cufhe::DeleteXaiNTT();
        cufhe::DeleteOneTRGSWNTT();
#endif
        cufhe::DeleteBootstrappingKeyNTT(1);
    }
    else {
#ifdef USE_KEY_BUNDLE
        cufhe::DeleteXaiNTT_lvl02();
        cufhe::DeleteOneTRGSWNTT_lvl02();
#endif
        cufhe::DeleteBootstrappingKeyNTT_lvl02(1);
    }
}

template <class brP>
void BootstrappingKeyToCPUFFT(
    TFHEpp::BootstrappingKeyFFT<brP>& out,
    const TFHEpp::BootstrappingKey<brP>& in)
{
    for (uint32_t i = 0; i < in.size(); i++)
        for (uint32_t j = 0; j < in[i].size(); j++)
            TFHEpp::ApplyFFT2trgsw<typename brP::targetP>(out[i][j],
                                                          in[i][j]);
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
void PolynomialCBKeyToCPUFFT(TFHEpp::CBswitchingKey<P>& out,
                             const cufhe::CBswitchingKeyPolynomial<P>& in)
{
    for (uint32_t key_idx = 0; key_idx < P::k; key_idx++)
        TFHEpp::ApplyFFT2trgsw<P>(out[key_idx], in[key_idx]);
}

template <class P>
void FlatToTRGSW(TFHEpp::TRGSW<P>& out, const std::vector<typename P::T>& flat)
{
    size_t index = 0;
    for (uint32_t row = 0; row < CBRows<P>(); row++)
        for (uint32_t k = 0; k <= P::k; k++)
            for (uint32_t i = 0; i < P::n; i++) out[row][k][i] = flat[index++];
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
bool CompareCBPhase(const TFHEpp::TRGSW<P>& cpu, const TFHEpp::TRGSW<P>& gpu,
                    const TFHEpp::Key<P>& key)
{
    uint64_t max_abs_diff = 0;
    uint32_t over_threshold = 0;
    constexpr uint64_t max_allowed =
        sizeof(typename P::T) == 4 ? (1ULL << 26) : (1ULL << 56);

    for (uint32_t row = 0; row < CBRows<P>(); row++) {
        const auto cpu_phase = TFHEpp::trlwePhase<P>(cpu[row], key);
        const auto gpu_phase = TFHEpp::trlwePhase<P>(gpu[row], key);
        for (uint32_t i = 0; i < P::n; i++) {
            const uint64_t abs_diff =
                AbsTorusDiff<typename P::T>(gpu_phase[i] - cpu_phase[i]);
            max_abs_diff = std::max(max_abs_diff, abs_diff);
            if (abs_diff > max_allowed) over_threshold++;
        }
    }

    const double max_log2 =
        max_abs_diff == 0 ? 0.0 : std::log2(static_cast<double>(max_abs_diff));
    std::cout << "phase max abs diff=2^" << max_log2
              << " threshold_failures=" << over_threshold << std::endl;
    return over_threshold == 0;
}

template <class brP, class ahP>
void CPUAnnihilateCB(
    TFHEpp::TRGSW<typename brP::targetP>& out,
    const TFHEpp::TLWE<typename brP::domainP>& in,
    const TFHEpp::BootstrappingKeyFFT<brP>& bkfft,
    const TFHEpp::AnnihilateKey<ahP>& ahk,
    const TFHEpp::CBswitchingKey<ahP>& cbsk)
{
    using targetP = typename brP::targetP;
    std::array<TFHEpp::TLWE<targetP>, targetP::l> temp;
    TFHEpp::GateBootstrappingManyLUT<brP, targetP::l>(
        temp, in, bkfft, TFHEpp::CBtestvector<targetP, targetP>());

    for (uint32_t i = 0; i < targetP::l; i++) {
        temp[i][targetP::k * targetP::n] +=
            static_cast<typename targetP::T>(1)
            << (std::numeric_limits<typename targetP::T>::digits -
                (i + 1) * targetP::Bgbit - 1);

        TFHEpp::TRLWE<targetP> temptrlwe;
        TFHEpp::InvSampleExtractIndex<targetP>(temptrlwe, temp[i], 0);
        TFHEpp::AnnihilateKeySwitching<ahP>(
            out[i + targetP::k * targetP::l], temptrlwe, ahk);

        for (uint32_t k = 0; k < targetP::k; k++)
            TFHEpp::ExternalProduct<ahP>(
                out[i + k * targetP::l],
                out[i + targetP::k * targetP::l], cbsk[k]);
    }
}

template <class brP, class ahP>
bool RunDirectCBCase(const bool message)
{
    using domainP = typename brP::domainP;
    using targetP = typename brP::targetP;
    static_assert(targetP::n == ahP::n &&
                      sizeof(typename targetP::T) == sizeof(typename ahP::T),
                  "targetP and ahP must share the torus ring");

    std::cout << "=== direct AnnihilateCircuitBootstrapping GPU test, message="
              << message << " ===" << std::endl;

    TFHEpp::SecretKey sk;

    auto bk = std::make_unique<TFHEpp::BootstrappingKey<brP>>();
    TFHEpp::bkgen<brP>(*bk, sk);
    auto cpu_bkfft = std::make_unique<TFHEpp::BootstrappingKeyFFT<brP>>();
    BootstrappingKeyToCPUFFT<brP>(*cpu_bkfft, *bk);

    auto gpu_ahk = std::make_unique<cufhe::AnnihilateKeyPolynomial<ahP>>();
    cufhe::AnnihilateKeyPolynomialGen<ahP>(*gpu_ahk, sk);
    auto cpu_ahk = std::make_unique<TFHEpp::AnnihilateKey<ahP>>();
    PolynomialAnnihilateKeyToCPUFFT<ahP>(*cpu_ahk, *gpu_ahk);

    auto gpu_cbsk = std::make_unique<cufhe::CBswitchingKeyPolynomial<ahP>>();
    cufhe::CBswitchingKeyPolynomialGen<ahP>(*gpu_cbsk, sk);
    auto cpu_cbsk = std::make_unique<TFHEpp::CBswitchingKey<ahP>>();
    PolynomialCBKeyToCPUFFT<ahP>(*cpu_cbsk, *gpu_cbsk);

    TFHEpp::TLWE<domainP> input;
    const typename domainP::T plain =
        message ? domainP::μ : static_cast<typename domainP::T>(-domainP::μ);
    TFHEpp::tlweSymEncrypt<domainP>(input, plain, sk.key.get<domainP>());

    TFHEpp::TRGSW<targetP> cpu_out;
    const auto cpu_start = std::chrono::steady_clock::now();
    CPUAnnihilateCB<brP, ahP>(cpu_out, input, *cpu_bkfft, *cpu_ahk,
                              *cpu_cbsk);
    const auto cpu_end = std::chrono::steady_clock::now();
    const double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    cudaSetDevice(0);
    InitializeBR<brP>();
#ifdef USE_KEY_BUNDLE
    cufhe::BootstrappingKeyBundleToNTT<brP>(*bk, 1);
#else
    cufhe::BootstrappingKeyToNTT<brP>(*bk, 1);
#endif
    cufhe::AnnihilateKeyPolynomialToDevice<ahP>(*gpu_ahk, 1);
    cufhe::CBswitchingKeyPolynomialToDevice<ahP>(*gpu_cbsk, 1);

    typename domainP::T* d_in = nullptr;
    typename targetP::T* d_out = nullptr;
    constexpr size_t input_bytes = sizeof(TFHEpp::TLWE<domainP>);
    constexpr size_t output_bytes = CBElemCount<targetP>() *
                                    sizeof(typename targetP::T);
    CUDA_CHECK(cudaMalloc(&d_in, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, output_bytes));

    cudaStream_t st;
    CUDA_CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(d_in, input.data(), input_bytes,
                               cudaMemcpyHostToDevice, st));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, st));
    cufhe::AnnihilateCircuitBootstrapping<brP, ahP>(d_out, d_in, st, 0);
    CUDA_CHECK(cudaEventRecord(stop, st));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));

    std::vector<typename targetP::T> gpu_flat(CBElemCount<targetP>());
    CUDA_CHECK(cudaMemcpyAsync(gpu_flat.data(), d_out, output_bytes,
                               cudaMemcpyDeviceToHost, st));
    CUDA_CHECK(cudaStreamSynchronize(st));

    TFHEpp::TRGSW<targetP> gpu_out;
    FlatToTRGSW<targetP>(gpu_out, gpu_flat);

    const bool ok = CompareCBPhase<targetP>(cpu_out, gpu_out,
                                            sk.key.get<targetP>());
    std::cout << "CPU CB compute: " << cpu_ms << " ms" << std::endl;
    std::cout << "GPU CB compute: " << gpu_ms << " ms" << std::endl;
    if (gpu_ms > 0)
        std::cout << "speedup: " << (cpu_ms / static_cast<double>(gpu_ms))
                  << "x" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(st));
    cudaFree(d_in);
    cudaFree(d_out);
    cufhe::DeleteCBswitchingKey<ahP>(1);
    cufhe::DeleteAnnihilateKey<ahP>(1);
    DeleteBR<brP>();

    return ok;
}

#undef CUDA_CHECK

}  // namespace

int main()
{
    bool ok = true;
    ok = RunDirectCBCase<TFHEpp::lvl01param, TFHEpp::AHlvl1param>(false) && ok;
    ok = RunDirectCBCase<TFHEpp::lvl01param, TFHEpp::AHlvl1param>(true) && ok;
    std::cout << (ok ? "ALL CIRCUIT BOOTSTRAPPING TESTS PASSED"
                     : "CIRCUIT BOOTSTRAPPING TESTS FAILED")
              << std::endl;
    return ok ? 0 : 1;
}
