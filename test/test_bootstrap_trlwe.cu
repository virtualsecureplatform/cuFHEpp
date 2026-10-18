#include <include/tfhepp_keygen.hpp>
#include <include/cufhe_gpu.cuh>
#ifndef CUFHE_TRLWE_BENCH_ONLY
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#endif
#include <tfhe/trlwe.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#ifndef CUFHE_TRLWE_BENCH_ONLY
namespace cufhe {
extern std::vector<NTTValue*> bk_ntts;
extern std::vector<CuNTTHandler<>*> ntt_handlers;
}
#endif
namespace {
using Small = TFHEpp::lvl0param;
using Ring = TFHEpp::lvl1param;
using BR = TFHEpp::lvl01param;
using IKS = TFHEpp::lvl10param;
constexpr int InputWords = Small::k * Small::n + 1;
constexpr int RingWords = (Ring::k + 1) * Ring::n;
#ifndef CUFHE_TRLWE_BENCH_ONLY
constexpr int Threads = cufhe::NUM_THREAD4HOMGATE<Ring>;
constexpr size_t DirectShared = cufhe::MEM4HOMGATE<Ring>;
constexpr size_t LegacyShared = ((Ring::k + 2 + Ring::k + 1) * Ring::n +
                                (Small::n + 1) / 2 + 1) * sizeof(cufhe::NTTValue);
#endif
void Check(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } }
#ifndef CUFHE_TRLWE_BENCH_ONLY
__global__ __launch_bounds__(Threads, cufhe::MIN_BLOCKS4HOMGATE<Ring>)
void DirectTRLWEReference(Ring::T* out, const Small::T* in, Ring::T mu,
    const cufhe::NTTValue* key, cufhe::CuNTTHandler<> ntt)
{
    cufhe::__BlindRotate__<BR>(out + blockIdx.x * RingWords,
        in + blockIdx.x * InputWords, mu, key, ntt);
}
// Preserve the legacy SEI entry point's arithmetic, including its original
// per-coefficient modulus switching. Only the final reference copy is cooperative.
__global__ __launch_bounds__(Threads, cufhe::MIN_BLOCKS4HOMGATE<Ring>)
void LegacySEIReference(Ring::T* out, const Ring::T* in, Ring::T mu,
    const cufhe::NTTValue* key, const Small::T* ksk, cufhe::CuNTTHandler<> ntt)
{
    extern __shared__ cufhe::NTTValue legacy_scratch[];
    auto* accumulator = reinterpret_cast<Ring::T*>(legacy_scratch + (Ring::k + 2) * Ring::n);
    auto* small = reinterpret_cast<Small::T*>(legacy_scratch + (Ring::k + 2 + Ring::k + 1) * Ring::n);
    cufhe::KeySwitch<IKS>(small, in + blockIdx.x * RingWords, ksk);
    __syncthreads();
    uint32_t bar = 2 * Ring::n - (small[Small::k * Small::n] >>
        (std::numeric_limits<Small::T>::digits - 1 - Ring::nbit));
    cufhe::RotatedTestVector<Ring>(accumulator, bar, mu);
    for (int i = 0; i < Small::n; ++i) {
        constexpr Small::T roundoffset = 1ULL <<
            (std::numeric_limits<Small::T>::digits - 2 - Ring::nbit);
        bar = cufhe::modSwitchFromTorus<BR>(small[i] + roundoffset);
        cufhe::Accumulate<BR>(accumulator, legacy_scratch, bar,
            key + (i << Ring::nbit) * (Ring::k + 1) * (Ring::k + 1) * Ring::l, ntt);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < RingWords; i += Threads)
        out[blockIdx.x * RingWords + i] = accumulator[i];
}


// Isolate one external product so every rotation exponent is checked against
// the original scalar implementation, independently of the full SEI chain.
template <bool Paired>
__global__ __launch_bounds__(Threads, cufhe::MIN_BLOCKS4HOMGATE<Ring>)
void LegacyExternalProductCheck(Ring::T* values, const cufhe::NTTValue* key,
    cufhe::CuNTTHandler<> ntt)
{
    extern __shared__ cufhe::NTTValue product_scratch[];
    auto* accumulator = reinterpret_cast<Ring::T*>(product_scratch + (Ring::k + 2) * Ring::n);
    auto* output = values + size_t(blockIdx.x) * RingWords;
    for (int i = threadIdx.x; i < RingWords; i += Threads) accumulator[i] = output[i];
    __syncthreads();
    const uint32_t exponent = blockIdx.x % (2 * Ring::n);
    const auto* row = key + size_t(blockIdx.x % 8) * (Ring::k + 1) *
        (Ring::k + 1) * Ring::l * Ring::n;
#if defined(USE_CUDA_PAIRED_NTT_WARP) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    if constexpr (Paired)
        cufhe::AccumulateLegacyPairedNTT<BR>(accumulator, product_scratch, exponent, row, ntt);
    else
#endif
        cufhe::Accumulate<BR>(accumulator, product_scratch, exponent, row, ntt);
    __syncthreads();
    for (int i = threadIdx.x; i < RingWords; i += Threads) output[i] = accumulator[i];
}

void CheckLegacyExternalProducts()
{
    constexpr int Cases = 4 * 2 * Ring::n;
    const size_t words = size_t(Cases) * RingWords;
    std::vector<Ring::T> input(words), expected(words), actual(words);
    std::mt19937_64 random(880131);
    constexpr std::array<Ring::T,8> boundaries{
        0, UINT32_MAX, 1, 1U << 31, (1U << 31) - 1, Ring::μ,
        (1U << (32 - Ring::l * Ring::Bgbit - 1)) - 1,
        1U << (32 - Ring::l * Ring::Bgbit - 1)};
    for (int sample = 0; sample < Cases; ++sample) {
        const int pattern = sample / (2 * Ring::n);
        for (int i = 0; i < RingWords; ++i)
            input[size_t(sample) * RingWords + i] = pattern == 0 ? 0 :
                pattern == 1 ? UINT32_MAX : pattern == 2 ? boundaries[i % boundaries.size()] : random();
    }
    Ring::T *reference_device, *actual_device;
    Check(cudaMalloc(&reference_device, words * sizeof(Ring::T)));
    Check(cudaMalloc(&actual_device, words * sizeof(Ring::T)));
    Check(cudaMemcpy(reference_device,input.data(),words*sizeof(Ring::T),cudaMemcpyHostToDevice));
    Check(cudaMemcpy(actual_device,input.data(),words*sizeof(Ring::T),cudaMemcpyHostToDevice));
    constexpr size_t Scratch = (2 * Ring::k + 3) * Ring::n * sizeof(cufhe::NTTValue);
    LegacyExternalProductCheck<false><<<Cases,Threads,Scratch>>>(reference_device,
        cufhe::bk_ntts[0],*cufhe::ntt_handlers[0]); Check(cudaGetLastError());
    LegacyExternalProductCheck<true><<<Cases,Threads,Scratch>>>(actual_device,
        cufhe::bk_ntts[0],*cufhe::ntt_handlers[0]); Check(cudaGetLastError());
    Check(cudaMemcpy(expected.data(),reference_device,words*sizeof(Ring::T),cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(),actual_device,words*sizeof(Ring::T),cudaMemcpyDeviceToHost));
    if (expected != actual) { std::fprintf(stderr,"legacy external-product coefficient mismatch\n"); std::exit(1); }
    Check(cudaFree(reference_device)); Check(cudaFree(actual_device));
    std::printf("legacy paired external product: PASS (%zu coefficients, all %d exponents, four input patterns)\n",words,2*Ring::n);
}

#endif
bool Truth(const std::vector<Ring::T>& values, int sample, const TFHEpp::SecretKey& key)
{
    std::array<Ring::T, Ring::k * Ring::n + 1> extracted;
    for (int i = 0; i < Ring::k * Ring::n; ++i) {
        const int component = i / Ring::n, coeff = i % Ring::n;
        extracted[i] = coeff == 0 ? values[sample * RingWords + i] :
            -values[sample * RingWords + (component + 1) * Ring::n - coeff];
    }
    extracted[Ring::k * Ring::n] = values[sample * RingWords + Ring::k * Ring::n];
    return TFHEpp::tlweSymDecrypt<Ring>(extracted, key.key.getSubset<Ring>());
}
}
int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "verify";
#ifdef CUFHE_TRLWE_BENCH_ONLY
    const bool verify = false, sei = mode == "sei";
#else
    const bool verify = mode == "verify", sei = mode == "sei";
#endif
    if (!verify && mode != "direct" && !sei) return 2;
    const int streams_count = argc > 2 ? std::atoi(argv[2]) : 1;
    const int per_stream = argc > 3 ? std::atoi(argv[3]) : 8;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 11;
    if (streams_count < 1 || per_stream < 1 || repeats < 1) return 2;
    const int count = verify ? 19 : streams_count * per_stream;
    Check(cudaSetDevice(0)); cufhe::SetGPUNum(1);
    TFHEpp::SecretKey secret; TFHEpp::EvalKey evaluation(secret);
    evaluation.emplacebk<BR>(secret);
#ifdef USE_SUBSET_KEY
    evaluation.emplacesubiksk<IKS>(secret);
#else
    cufhe::EmplaceGateKeySwitchingKey<IKS>(evaluation, secret);
#endif
    cufhe::Initialize(evaluation);
    std::vector<Small::T> small(count * InputWords);
    std::vector<Ring::T> large(count * RingWords);
    std::mt19937_64 random(860031);
    for (int sample = 0; sample < count; ++sample) {
        if (!verify || sample < 8) {
            std::array<Small::T, InputWords> a;
            TFHEpp::tlweSymEncrypt<Small>(a, (sample & 1) ? Small::μ : -Small::μ, secret.key.getSubset<Small>());
            std::copy(a.begin(), a.end(), small.begin() + sample * InputWords);
            TFHEpp::TRLWE<Ring> b; std::array<Ring::T, Ring::n> plaintext;
            for (int i = 0; i < Ring::n; ++i) plaintext[i] = ((sample + i) & 1) ? Ring::μ : -Ring::μ;
            TFHEpp::trlweSymEncrypt<Ring>(b, plaintext, secret.key.getSubset<Ring>());
            for (int j = 0; j <= Ring::k; ++j)
                std::copy(b[j].begin(), b[j].end(), large.begin() + sample * RingWords + j * Ring::n);
        } else {
            for (int i = 0; i < InputWords; ++i) small[sample * InputWords + i] =
                sample == 8 ? 0 : sample == 9 ? UINT16_MAX : sample == 10 ? 1U << 15 :
                sample == 11 ? 1U << (i % 16) : random();
            for (int i = 0; i < RingWords; ++i) large[sample * RingWords + i] =
                sample == 8 ? 0 : sample == 9 ? UINT32_MAX : sample == 10 ? 1U << 31 :
                sample == 11 ? 1U << (i % 32) : random();
        }
    }
    Small::T* small_device; Ring::T *large_device, *reference, *actual;
    Check(cudaMalloc(&small_device, small.size() * sizeof(Small::T)));
    Check(cudaMalloc(&large_device, large.size() * sizeof(Ring::T)));
    Check(cudaMalloc(&reference, large.size() * sizeof(Ring::T)));
    Check(cudaMalloc(&actual, large.size() * sizeof(Ring::T)));
    Check(cudaMemcpy(small_device, small.data(), small.size() * sizeof(Small::T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(large_device, large.data(), large.size() * sizeof(Ring::T), cudaMemcpyHostToDevice));
    std::vector<Ring::T> expected(large.size()), got(large.size());
    #ifndef CUFHE_TRLWE_BENCH_ONLY
    if (verify) {
        for (const auto mu : std::array<Ring::T,4>{0, Ring::μ/2, Ring::μ, Ring::μ*2}) {
            DirectTRLWEReference<<<count, Threads, DirectShared>>>(reference, small_device, mu,
                cufhe::bk_ntts[0], *cufhe::ntt_handlers[0]); Check(cudaGetLastError());
            for (int s = 0; s < count; ++s) cufhe::BootstrapTLWE2TRLWE(actual + s * RingWords,
                small_device + s * InputWords, mu, nullptr, 0);
            Check(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(Ring::T), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(got.data(), actual, got.size() * sizeof(Ring::T), cudaMemcpyDeviceToHost));
            if (expected != got) { std::fprintf(stderr,"direct TRLWE coefficient mismatch\n"); return 1; }
            if (mu == Ring::μ) for (int s=0;s<8;++s) if (Truth(got,s,secret)!=bool(s&1)) return 1;
            for (int round=0;round<2;++round) {
                LegacySEIReference<<<count, Threads, LegacyShared>>>(reference, round ? reference : large_device,
                    mu, cufhe::bk_ntts[0], cufhe::ksk_devs[0], *cufhe::ntt_handlers[0]); Check(cudaGetLastError());
                for (int s=0;s<count;++s) cufhe::SEIandBootstrap2TRLWE(actual+s*RingWords,
                    round ? actual+s*RingWords : large_device+s*RingWords, mu, nullptr, 0);
                Check(cudaMemcpy(expected.data(), reference, expected.size() * sizeof(Ring::T), cudaMemcpyDeviceToHost));
                Check(cudaMemcpy(got.data(), actual, got.size() * sizeof(Ring::T), cudaMemcpyDeviceToHost));
                if (expected != got) { std::fprintf(stderr,"SEI coefficient / in-place mismatch\n"); return 1; }
            }
        }
        std::puts("direct TLWE-to-TRLWE coefficient equivalence: PASS (155648 coefficients, 19 inputs, 4 runtime amplitudes)");
        std::puts("legacy SEI-to-TRLWE coefficient / in-place equivalence: PASS (311296 coefficients, 19 inputs, 4 runtime amplitudes, replay x2)");
        CheckLegacyExternalProducts();
    } else
    #endif
    {
        std::vector<cudaStream_t> streams(streams_count);
        for (auto& s:streams) Check(cudaStreamCreate(&s));
        auto submit=[&] { for (int s=0;s<count;++s) {
            if (sei) cufhe::SEIandBootstrap2TRLWE(actual+s*RingWords,large_device+s*RingWords,Ring::μ,streams[s%streams_count],0);
            else cufhe::BootstrapTLWE2TRLWE(actual+s*RingWords,small_device+s*InputWords,Ring::μ,streams[s%streams_count],0);
        }};
        std::vector<double> times;
        const auto warmup=std::chrono::steady_clock::now();
        for (int repeat=-1;repeat<repeats;) {
            const auto start=std::chrono::steady_clock::now(); submit(); Check(cudaDeviceSynchronize());
            const auto end=std::chrono::steady_clock::now();
            const double elapsed=std::chrono::duration<double,std::milli>(end-start).count()/count;
            if (repeat>=0) {times.push_back(elapsed);++repeat;}
            else if (end-warmup>=std::chrono::milliseconds(250)) repeat=0;
        }
        Check(cudaMemcpy(got.data(),actual,got.size()*sizeof(Ring::T),cudaMemcpyDeviceToHost));
        int errors=0; for(int s=0;s<count;++s) errors+=Truth(got,s,secret)!=bool(s&1);
        std::sort(times.begin(),times.end());
        std::printf("mode=%s streams=%d count=%d median_ms/call=%.8f errors=%d\n",mode.c_str(),streams_count,count,times[times.size()/2],errors);
        for(auto s:streams) Check(cudaStreamDestroy(s));
        if(errors) return 1;
    }
    Check(cudaFree(small_device));Check(cudaFree(large_device));Check(cudaFree(reference));Check(cudaFree(actual));
    cufhe::CleanUp();
}
