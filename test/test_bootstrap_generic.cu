#include <include/cufhe_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace cufhe {
extern std::vector<NTTValue*> bk_ntts;
extern std::vector<CuNTTHandler<>*> ntt_handlers;
}

namespace {
using P = TFHEpp::lvl0param;
using BR = TFHEpp::lvl01param;
using IKS = TFHEpp::lvl10param;
using Target = BR::targetP;
constexpr int Words = P::k * P::n + 1, Cases = 19;
constexpr int Threads = cufhe::NUM_THREAD4HOMGATE<Target>;
constexpr size_t Shared = cufhe::MEM4HOMGATE<Target>;
void Check(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}

// Arithmetic of the original generic API, with its required scratch allocated.
__global__ __launch_bounds__(Threads, cufhe::MIN_BLOCKS4HOMGATE<Target>)
void GenericBootstrapReference(P::T* out, const P::T* input, Target::T mu,
    const cufhe::NTTValueFor<Target::n, 32>* key, const P::T* ksk,
    cufhe::CuNTTHandler<Target::n, 32> ntt)
{
    __shared__ Target::T accumulator[(Target::k + 1) * Target::n];
    cufhe::__BlindRotate__<BR>(accumulator, input + blockIdx.x * Words, mu, key, ntt);
    cufhe::KeySwitch<IKS>(out + blockIdx.x * Words, accumulator, ksk);
}
}  // namespace

int main()
{
    Check(cudaSetDevice(0));
    cufhe::SetGPUNum(1);
    TFHEpp::SecretKey secret;
    TFHEpp::EvalKey evaluation(secret);
    evaluation.emplacebk<BR>(secret);
#ifdef USE_SUBSET_KEY
    evaluation.emplacesubiksk<IKS>(secret);
#else
    evaluation.emplaceiksk<IKS>(secret);
#endif
    cufhe::Initialize(evaluation);
    std::vector<P::T> input(Cases * Words);
    std::mt19937_64 random(850031);
    for (int sample = 0; sample < Cases; ++sample) {
        if (sample < 8) {
            std::array<P::T, Words> encrypted;
            TFHEpp::tlweSymEncrypt<P>(encrypted, (sample & 1) ? P::μ : -P::μ,
                                     secret.key.get<P>());
            std::copy(encrypted.begin(), encrypted.end(), input.begin() + sample * Words);
        } else {
            for (int i = 0; i < Words; ++i)
                input[sample * Words + i] = sample == 8 ? 0 : sample == 9 ? UINT16_MAX :
                    sample == 10 ? P::T{1} << 15 : sample == 11 ? P::T{1} << (i % 16) : random();
        }
    }
    P::T *device_input, *expected_device, *actual_device;
    Check(cudaMalloc(&device_input, input.size() * sizeof(P::T)));
    Check(cudaMalloc(&expected_device, input.size() * sizeof(P::T)));
    Check(cudaMalloc(&actual_device, input.size() * sizeof(P::T)));
    Check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(P::T), cudaMemcpyHostToDevice));
    const std::array<Target::T, 4> amplitudes = {0, Target::μ / 2, Target::μ, Target::μ * 2};
    std::vector<P::T> expected(input.size()), actual(input.size());
    for (const auto mu : amplitudes) {
        for (int round = 0; round < 2; ++round) {
            GenericBootstrapReference<<<Cases, Threads, Shared>>>(expected_device,
                round ? expected_device : device_input, mu, cufhe::bk_ntts[0],
                cufhe::ksk_devs[0], *cufhe::ntt_handlers[0]);
            Check(cudaGetLastError());
            for (int sample = 0; sample < Cases; ++sample) {
                auto* output = actual_device + sample * Words;
                cufhe::Bootstrap(output, round ? output : device_input + sample * Words,
                                 mu, nullptr, 0);
            }
            Check(cudaMemcpy(expected.data(), expected_device, expected.size() * sizeof(P::T), cudaMemcpyDeviceToHost));
            Check(cudaMemcpy(actual.data(), actual_device, actual.size() * sizeof(P::T), cudaMemcpyDeviceToHost));
            if (actual != expected) {
                std::fprintf(stderr, "generic Bootstrap coefficient mismatch (mu=%u, round=%d)\n", unsigned(mu), round);
                return 1;
            }
            if (mu == Target::μ && round == 0) {
                for (int sample = 0; sample < 8; ++sample) {
                    std::array<P::T, Words> ciphertext;
                    std::copy_n(actual.begin() + sample * Words, Words, ciphertext.begin());
                    if (TFHEpp::tlweSymDecrypt<P>(ciphertext, secret.key.get<P>()) != bool(sample & 1)) {
                        std::fprintf(stderr, "generic Bootstrap truth mismatch\n");
                        return 1;
                    }
                }
            }
        }
    }
    Check(cudaFree(device_input)); Check(cudaFree(expected_device)); Check(cudaFree(actual_device));
    cufhe::CleanUp();
    std::puts("generic Bootstrap coefficient / in-place equivalence: PASS (95912 coefficients, 19 inputs, 4 runtime amplitudes, replay x2)");
}
