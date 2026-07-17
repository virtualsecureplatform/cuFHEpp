#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <include/annihilate_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <iostream>
#include <limits>
#include <memory>
#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>
#include <type_traits>
#include <vector>

namespace {

#define CUDA_CHECK(err) cufhe::CuSafeCall__(err, __FILE__, __LINE__)

template <class P>
constexpr uint32_t CBRows()
{
    return P::k * P::lₐ + P::l;
}

constexpr uint32_t BitsNeeded(uint32_t value)
{
    uint32_t bits = 0;
    while (value != 0) {
        bits++;
        value >>= 1;
    }
    return bits;
}

template <class P, uint32_t num_out>
constexpr typename P::T CBGadgetHalf(const uint32_t digit)
{
    if constexpr (num_out == P::l) {
        return static_cast<typename P::T>(1)
               << (std::numeric_limits<typename P::T>::digits -
                   (digit + 1) * P::Bgbit - 1);
    }
    else {
        if (digit < P::l)
            return static_cast<typename P::T>(1)
                   << (std::numeric_limits<typename P::T>::digits -
                       (digit + 1) * P::Bgbit - 1);
        if (digit < P::l + P::lₐ)
            return static_cast<typename P::T>(1)
                   << (std::numeric_limits<typename P::T>::digits -
                       (digit - P::l + 1) * P::Bgₐbit - 1);
        return 0;
    }
}

template <class P, uint32_t num_out>
constexpr TFHEpp::Polynomial<P> CBTestVector()
{
    TFHEpp::Polynomial<P> poly{};
    constexpr uint32_t bitwidth = BitsNeeded(num_out - 1);
    for (uint32_t i = 0; i < (P::n >> bitwidth); i++)
        for (uint32_t j = 0; j < (1U << bitwidth); j++)
            poly[(i << bitwidth) + j] = CBGadgetHalf<P, num_out>(j);
    return poly;
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
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        cufhe::InitializeXaiNTT(1);
#endif
#ifdef USE_KEY_BUNDLE
        cufhe::InitializeOneTRGSWNTT(1);
#endif
    }
    else {
        cufhe::InitializeNTThandlers_lvl02(1);
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        cufhe::InitializeXaiNTT_lvl02(1);
#endif
#ifdef USE_KEY_BUNDLE
        cufhe::InitializeOneTRGSWNTT_lvl02(1);
#endif
    }
}

template <class brP>
void DeleteBR()
{
    using targetP = typename brP::targetP;
    if constexpr (targetP::n == TFHEpp::lvl1param::n) {
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        cufhe::DeleteXaiNTT();
#endif
#ifdef USE_KEY_BUNDLE
        cufhe::DeleteOneTRGSWNTT();
#endif
        cufhe::DeleteBootstrappingKeyNTT(1);
    }
    else {
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
        cufhe::DeleteXaiNTT_lvl02();
#endif
#ifdef USE_KEY_BUNDLE
        cufhe::DeleteOneTRGSWNTT_lvl02();
#endif
        cufhe::DeleteBootstrappingKeyNTT_lvl02(1);
    }
}

template <class brP>
void BootstrappingKeyToCPUFFT(TFHEpp::BootstrappingKeyFFT<brP>& out,
                              const TFHEpp::BootstrappingKey<brP>& in)
{
    for (uint32_t i = 0; i < in.size(); i++)
        for (uint32_t j = 0; j < in[i].size(); j++)
            TFHEpp::ApplyFFT2trgsw<typename brP::targetP>(out[i][j], in[i][j]);
}

template <class P>
void PolynomialAnnihilateKeyToCPUFFT(
    TFHEpp::AnnihilateKey<P>& out, const cufhe::AnnihilateKeyPolynomial<P>& in)
{
    for (uint32_t bit = 0; bit < P::nbit; bit++)
        for (uint32_t key_idx = 0; key_idx < P::k; key_idx++)
            TFHEpp::ApplyFFT2halftrgsw<P>(out[bit][key_idx], in[bit][key_idx]);
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

template <class P>
bool ValidateCircuitBootstrap(const TFHEpp::TRGSW<P>& trgsw,
                              const TFHEpp::Key<P>& key, const bool selector)
{
    std::array<uint8_t, P::n> plain{};
    auto input = std::make_unique<TFHEpp::TRLWE<P>>();
    for (uint32_t i = 0; i < P::n; i++) {
        plain[i] = (i * 5 + 3) & 1U;
        (*input)[P::k][i] = plain[i] ? P::μ : -P::μ;
    }

    auto trgswfft = std::make_unique<TFHEpp::TRGSWFFT<P>>();
    TFHEpp::ApplyFFT2trgsw<P>(*trgswfft, trgsw);
    auto output = std::make_unique<TFHEpp::TRLWE<P>>();
    TFHEpp::ExternalProduct<P>(*output, *input, *trgswfft);
    const auto phase = TFHEpp::trlwePhase<P>(*output, key);

    uint32_t mismatches = 0;
    for (uint32_t i = 0; i < P::n; i++) {
        const typename P::T expected =
            selector ? (*input)[P::k][i] : typename P::T{0};
        if (AbsTorusDiff<typename P::T>(phase[i] - expected) > P::μ / 4)
            mismatches++;
    }
    std::cout << "external-product mismatches=" << mismatches << std::endl;
    return mismatches == 0;
}

template <class brP, class ahP>
void CPUAnnihilateCB(TFHEpp::TRGSW<typename brP::targetP>& out,
                     const TFHEpp::TLWE<typename brP::domainP>& in,
                     const TFHEpp::BootstrappingKeyFFT<brP>& bkfft,
                     const TFHEpp::AnnihilateKey<ahP>& ahk,
                     const TFHEpp::CBswitchingKey<ahP>& cbsk)
{
    using targetP = typename brP::targetP;
    constexpr uint32_t num_out = cufhe::CircuitBootstrapLUTCount<targetP>;
    std::array<TFHEpp::TLWE<targetP>, num_out> temp;
    TFHEpp::GateBootstrappingManyLUT<brP, num_out>(
        temp, in, bkfft, CBTestVector<targetP, num_out>());

    constexpr uint32_t main_row_offset = targetP::k * targetP::lₐ;
    if constexpr (!cufhe::CircuitBootstrapSharedGadget<targetP>) {
        for (uint32_t i = 0; i < targetP::lₐ; i++) {
            const uint32_t output = targetP::l + i;
            temp[output][targetP::k * targetP::n] +=
                CBGadgetHalf<targetP, num_out>(output);
            TFHEpp::TRLWE<targetP> source;
            TFHEpp::InvSampleExtractIndex<targetP>(source, temp[output], 0);
            TFHEpp::AnnihilateKeySwitching<ahP>(source, source, ahk);
            for (uint32_t k = 0; k < targetP::k; k++)
                TFHEpp::ExternalProduct<ahP>(out[i + k * targetP::lₐ], source,
                                             cbsk[k]);
        }
    }

    for (uint32_t i = 0; i < targetP::l; i++) {
        temp[i][targetP::k * targetP::n] += CBGadgetHalf<targetP, num_out>(i);

        TFHEpp::TRLWE<targetP> temptrlwe;
        TFHEpp::InvSampleExtractIndex<targetP>(temptrlwe, temp[i], 0);
        TFHEpp::AnnihilateKeySwitching<ahP>(out[i + main_row_offset], temptrlwe,
                                            ahk);

        if constexpr (cufhe::CircuitBootstrapSharedGadget<targetP>)
            for (uint32_t k = 0; k < targetP::k; k++)
                TFHEpp::ExternalProduct<ahP>(out[i + k * targetP::lₐ],
                                             out[i + main_row_offset], cbsk[k]);
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
    TFHEpp::tlweSymEncrypt<domainP>(input, plain, 0.0, sk.key.get<domainP>());

    auto cpu_out = std::make_unique<TFHEpp::TRGSW<targetP>>();
    const auto cpu_start = std::chrono::steady_clock::now();
    CPUAnnihilateCB<brP, ahP>(*cpu_out, input, *cpu_bkfft, *cpu_ahk, *cpu_cbsk);
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
    constexpr size_t output_bytes =
        CBElemCount<targetP>() * sizeof(typename targetP::T);
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

    auto gpu_out = std::make_unique<TFHEpp::TRGSW<targetP>>();
    FlatToTRGSW<targetP>(*gpu_out, gpu_flat);

    bool ok =
        CompareCBPhase<targetP>(*cpu_out, *gpu_out, sk.key.get<targetP>());
    if constexpr (sizeof(typename targetP::T) == 8) {
        const bool cpu_ok = ValidateCircuitBootstrap<targetP>(
            *cpu_out, sk.key.get<targetP>(), message);
        const bool gpu_ok = ValidateCircuitBootstrap<targetP>(
            *gpu_out, sk.key.get<targetP>(), message);
        ok = ok && cpu_ok && gpu_ok;
    }
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
#if defined(USE_DIFFERENT_BR_PARAM) && defined(USE_DIFFERENT_AH_PARAM)
    ok = RunDirectCBCase<TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>(false) &&
         ok;
    ok = RunDirectCBCase<TFHEpp::cblvl02param, TFHEpp::cbAHlvl2param>(true) &&
         ok;
#endif
    std::cout << (ok ? "ALL CIRCUIT BOOTSTRAPPING TESTS PASSED"
                     : "CIRCUIT BOOTSTRAPPING TESTS FAILED")
              << std::endl;
    return ok ? 0 : 1;
}
