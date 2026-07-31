#include <include/gpu_runtime.cuh>

#include <array>
#include <cstdint>
#include <iostream>
#include <random>

#include <include/ntt_small_modulus.cuh>

namespace {

template <uint32_t N>
uint64_t ReferenceMadd(const uint64_t a, const uint64_t b, const uint64_t c)
{
    return static_cast<uint64_t>(
        (static_cast<unsigned __int128>(a) * b + c) %
        cufhe::SmallNTTModulus<N>::P);
}

template <uint32_t N>
uint64_t ReferenceMadd3(const uint64_t a0, const uint64_t b0,
                        const uint64_t a1, const uint64_t b1,
                        const uint64_t a2, const uint64_t b2,
                        const uint64_t c)
{
    uint64_t result = ReferenceMadd<N>(a0, b0, c);
    result = ReferenceMadd<N>(a1, b1, result);
    return ReferenceMadd<N>(a2, b2, result);
}

template <uint32_t N>
bool TestHostMadd()
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N>::P;
    const std::array<uint64_t, 12> edge_values = {
        0,      1,      2,      3,      17,     65535,
        p >> 1, p - 18, p - 4,  p - 3,  p - 2,  p - 1};

    for (const uint64_t a : edge_values) {
        for (const uint64_t b : edge_values) {
            for (const uint64_t c : edge_values) {
                if (cufhe::small_mod_madd<N>(a, b, c) !=
                    ReferenceMadd<N>(a, b, c)) {
                    return false;
                }
            }
        }
    }

    std::mt19937_64 rng(0x4e54544d414444ULL + N);
    for (int i = 0; i < 250000; ++i) {
        const uint64_t a = rng() % p;
        const uint64_t b = rng() % p;
        const uint64_t c = rng() % p;
        if (cufhe::small_mod_madd<N>(a, b, c) !=
            ReferenceMadd<N>(a, b, c)) {
            return false;
        }
    }
    return true;
}

template <uint32_t N>
bool TestHostMadd3()
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N>::P;
    const std::array<uint64_t, 12> edge_values = {
        0,      1,      2,      3,      17,     65535,
        p >> 1, p - 18, p - 4,  p - 3,  p - 2,  p - 1};

    if (cufhe::small_mod_madd3<N>(p - 1, p - 1, p - 1, p - 1, p - 1,
                                  p - 1, p - 1) !=
        ReferenceMadd3<N>(p - 1, p - 1, p - 1, p - 1, p - 1, p - 1,
                          p - 1)) {
        return false;
    }

    for (size_t i = 0; i < edge_values.size(); ++i) {
        const uint64_t a0 = edge_values[i];
        const uint64_t b0 = edge_values[(i + 1) % edge_values.size()];
        const uint64_t a1 = edge_values[(i + 3) % edge_values.size()];
        const uint64_t b1 = edge_values[(i + 5) % edge_values.size()];
        const uint64_t a2 = edge_values[(i + 7) % edge_values.size()];
        const uint64_t b2 = edge_values[(i + 9) % edge_values.size()];
        const uint64_t c = edge_values[(i + 11) % edge_values.size()];
        if (cufhe::small_mod_madd3<N>(a0, b0, a1, b1, a2, b2, c) !=
            ReferenceMadd3<N>(a0, b0, a1, b1, a2, b2, c)) {
            return false;
        }
    }

    std::mt19937_64 rng(0x4e54544d414433ULL + N);
    for (int i = 0; i < 250000; ++i) {
        const uint64_t a0 = rng() % p;
        const uint64_t b0 = rng() % p;
        const uint64_t a1 = rng() % p;
        const uint64_t b1 = rng() % p;
        const uint64_t a2 = rng() % p;
        const uint64_t b2 = rng() % p;
        const uint64_t c = rng() % p;
        if (cufhe::small_mod_madd3<N>(a0, b0, a1, b1, a2, b2, c) !=
            ReferenceMadd3<N>(a0, b0, a1, b1, a2, b2, c)) {
            return false;
        }
    }
    return true;
}

template <uint32_t N>
__global__ void TestDeviceMaddKernel(unsigned int* const failures)
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N>::P;
    uint64_t state = 0x9e3779b97f4a7c15ULL ^
                     (static_cast<uint64_t>(blockIdx.x * blockDim.x +
                                            threadIdx.x) +
                      N);

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t a = (state * 0x2545f4914f6cdd1dULL) % p;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t b = (state * 0x2545f4914f6cdd1dULL) % p;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t c = (state * 0x2545f4914f6cdd1dULL) % p;

        const uint64_t expected = cufhe::small_mod_add<N>(
            cufhe::small_mod_mult<N>(a, b), c);
        if (cufhe::small_mod_madd<N>(a, b, c) != expected) {
            atomicAdd(failures, 1U);
        }
    }
}

template <uint32_t N>
bool TestDeviceMadd()
{
    unsigned int* failures;
    if (cudaMalloc(&failures, sizeof(*failures)) != cudaSuccess) return false;
    if (cudaMemset(failures, 0, sizeof(*failures)) != cudaSuccess) return false;

    TestDeviceMaddKernel<N><<<256, 256>>>(failures);
    if (cudaGetLastError() != cudaSuccess) return false;

    unsigned int host_failures = 0;
    if (cudaMemcpy(&host_failures, failures, sizeof(host_failures),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    cudaFree(failures);
    return host_failures == 0;
}

template <uint32_t N>
__global__ void TestDeviceMadd3Kernel(unsigned int* const failures)
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N>::P;
    uint64_t state = 0x243f6a8885a308d3ULL ^
                     (static_cast<uint64_t>(blockIdx.x * blockDim.x +
                                            threadIdx.x) +
                      N);

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        uint64_t values[7];
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            values[j] = (state * 0x2545f4914f6cdd1dULL) % p;
        }
        uint64_t expected =
            cufhe::small_mod_madd<N>(values[0], values[1], values[6]);
        expected = cufhe::small_mod_madd<N>(values[2], values[3], expected);
        expected = cufhe::small_mod_madd<N>(values[4], values[5], expected);
        if (cufhe::small_mod_madd3<N>(
                values[0], values[1], values[2], values[3], values[4],
                values[5], values[6]) != expected) {
            atomicAdd(failures, 1U);
        }
    }
}

template <uint32_t N>
bool TestDeviceMadd3()
{
    unsigned int* failures;
    if (cudaMalloc(&failures, sizeof(*failures)) != cudaSuccess) return false;
    if (cudaMemset(failures, 0, sizeof(*failures)) != cudaSuccess) return false;

    TestDeviceMadd3Kernel<N><<<256, 256>>>(failures);
    if (cudaGetLastError() != cudaSuccess) return false;

    unsigned int host_failures = 0;
    if (cudaMemcpy(&host_failures, failures, sizeof(host_failures),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    cudaFree(failures);
    return host_failures == 0;
}

}  // namespace

int main()
{
    const bool host31 = TestHostMadd<TFHEpp::lvl1param::n>();
    const bool host64 = TestHostMadd<TFHEpp::lvl2param::n>();
    const bool device31 = TestDeviceMadd<TFHEpp::lvl1param::n>();
    const bool device64 = TestDeviceMadd<TFHEpp::lvl2param::n>();
    const bool host31_madd3 = TestHostMadd3<TFHEpp::lvl1param::n>();
    const bool host64_madd3 = TestHostMadd3<TFHEpp::lvl2param::n>();
    const bool device31_madd3 = TestDeviceMadd3<TFHEpp::lvl1param::n>();
    const bool device64_madd3 = TestDeviceMadd3<TFHEpp::lvl2param::n>();

    std::cout << "31-bit host madd: " << (host31 ? "PASS" : "FAIL") << '\n';
    std::cout << "64-bit host madd: " << (host64 ? "PASS" : "FAIL") << '\n';
    std::cout << "31-bit device madd: " << (device31 ? "PASS" : "FAIL")
              << '\n';
    std::cout << "64-bit device madd: " << (device64 ? "PASS" : "FAIL")
              << '\n';
    std::cout << "31-bit host madd3: " << (host31_madd3 ? "PASS" : "FAIL")
              << '\n';
    std::cout << "64-bit host madd3: " << (host64_madd3 ? "PASS" : "FAIL")
              << '\n';
    std::cout << "31-bit device madd3: "
              << (device31_madd3 ? "PASS" : "FAIL") << '\n';
    std::cout << "64-bit device madd3: "
              << (device64_madd3 ? "PASS" : "FAIL") << '\n';
    return host31 && host64 && device31 && device64 && host31_madd3 &&
                   host64_madd3 && device31_madd3 && device64_madd3
               ? 0
               : 1;
}
