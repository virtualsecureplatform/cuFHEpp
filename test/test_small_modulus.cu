#include <include/gpu_runtime.cuh>

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <include/ntt_small_modulus.cuh>
#include <include/gatebootstrapping_gpu.cuh>

namespace {

#if defined(CUFHE_HAS_PAIR_NTT64)
__global__ void TestGoldilocksTorusKernel(const uint64_t* in, ulonglong3* out, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = make_ulonglong3(
        cufhe::ntt_mod_to_torus64<TFHEpp::lvl2param::n, 64, true>(in[i]),
        cufhe::ntt_mod_to_torus32<TFHEpp::lvl2param::n, 64>(in[i]),
        cufhe::ntt_mod_to_torus64<TFHEpp::lvl2param::n, 64>(in[i]));
}

bool TestGoldilocksTorus()
{
    constexpr uint64_t p = cufhe::small_ntt::P, half = p / 2, B = uint64_t{1} << 32;
    std::vector<uint64_t> input = {0, 1, 2, half - 1, half, half + 1,
                                   half + 2, p - 3, p - 2, p - 1};
    // Direct signed-conversion thresholds in both halves of the field.
    for (uint64_t hi : std::array<uint64_t, 8>{0, 1, 2147483646, 2147483647,
                                               2147483648, 2147483649, B - 2, B - 1}) {
        const uint64_t flipped = hi ^ (B >> 1);
        for (uint64_t lo : std::array<uint64_t, 11>{0, 1, 2, hi, B >> 1,
             (B >> 1) + 1, B - 1, flipped, (flipped + 1) & (B - 1),
             (flipped + 2) & (B - 1), (flipped + 3) & (B - 1)}) {
            if (hi * B + lo < p) input.push_back(hi * B + lo);
        }
    }
    const auto add_magnitude = [&](uint64_t v) {
        input.push_back(v); input.push_back(v == 0 ? 0 : p - v);
    };
    for (uint64_t v : std::array<uint64_t, 4>{0, 1, half - 1, half}) add_magnitude(v);
    for (uint64_t hi : std::array<uint64_t, 8>{0, 1, 2, 65535, 65536,
                                               2147483645, 2147483646, 2147483647}) {
        for (uint64_t lo : std::array<uint64_t, 7>{0, 1, B - 1, hi,
                 hi + (B >> 1) - 1, hi + (B >> 1), hi + (B >> 1) + 1}) {
            if (lo < B && hi * B + lo <= half) add_magnitude(hi * B + lo);
        }
    }
    std::mt19937_64 rng(0x544f5255533634ULL);
    for (int i = 0; i < 200000; ++i) add_magnitude(rng() % (half + 1));
    for (int i = 0; i < 100000; ++i) input.push_back(rng() % p);
    std::vector<ulonglong3> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const uint64_t v = input[i];
        const bool neg = v > half;
        const unsigned __int128 mag = neg ? p - v : v;
        const uint64_t rounded = ((mag << 64) + (mag << 32) - mag +
                                  (static_cast<unsigned __int128>(1) << 63)) >> 64;
        uint32_t rounded32 = static_cast<uint32_t>(rounded >> 32);
        rounded32 += static_cast<uint32_t>((rounded & UINT32_MAX) >= 0x80000000u);
        expected[i] = make_ulonglong3(neg ? uint64_t(-rounded) : rounded,
                                     neg ? uint32_t(-rounded32) : rounded32,
                                     neg ? uint64_t(-rounded) : rounded);
        if (cufhe::ntt_mod_to_torus64<TFHEpp::lvl2param::n, 64>(v) != expected[i].x ||
            cufhe::ntt_mod_to_torus32<TFHEpp::lvl2param::n, 64>(v) != expected[i].y) return false;
    }
    uint64_t* in = nullptr; ulonglong3* out = nullptr;
    bool ok = cudaMalloc(&in, input.size() * sizeof(uint64_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&out, actual.size() * sizeof(ulonglong3)) == cudaSuccess;
    ok = ok && cudaMemcpy(in, input.data(), input.size() * sizeof(uint64_t),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        TestGoldilocksTorusKernel<<<(input.size() + 255) / 256, 256>>>(in, out, input.size());
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(actual.data(), out, actual.size() * sizeof(ulonglong3),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(in); cudaFree(out);
    for (size_t i = 0; ok && i < input.size(); ++i)
        ok = actual[i].x == expected[i].x && actual[i].y == expected[i].y && actual[i].z == expected[i].z;
    return ok;
}

__global__ void TestGoldilocksAddKernel(const ulonglong2* in, ulonglong3* out, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        out[i] = make_ulonglong3(cufhe::small_mod64_add<0>(in[i].x, in[i].y),
                                cufhe::small_mod64_add<1>(in[i].x, in[i].y),
                                cufhe::small_mod64_add<2>(in[i].x, in[i].y));
}

bool TestGoldilocksAdd()
{
    constexpr uint64_t p = cufhe::small_ntt::P;
    const std::array<uint64_t, 16> edges = {0, 1, 2, 3, UINT32_MAX - 1, UINT32_MAX,
        uint64_t{1} << 32, (uint64_t{1} << 32) + 1, p / 2, p - 3, p - 2,
        p - 1, p, p + 1, UINT64_MAX - 1, UINT64_MAX};
    std::vector<ulonglong2> input;
    for (auto a : edges) for (auto b : edges) input.push_back(make_ulonglong2(a, b));
    std::mt19937_64 rng(0x4144444341525259ULL);
    for (int i = 0; i < 100000; ++i) input.push_back(make_ulonglong2(rng() % p, rng() % p));
    for (int i = 0; i < 100000; ++i) input.push_back(make_ulonglong2(rng(), rng()));
    for (int i = 0; i < 10000; ++i) {
        const uint64_t a = rng() % p;
        for (int delta = -2; delta <= 2; ++delta) {
            input.push_back(make_ulonglong2(a, p - a + uint64_t(delta)));
            input.push_back(make_ulonglong2(a, -a + uint64_t(delta)));
        }
    }
    std::vector<uint64_t> expected(input.size());
    std::vector<ulonglong3> actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const uint64_t a = input[i].x, b = input[i].y, sum = a + b;
        expected[i] = sum + static_cast<uint32_t>(-(sum < b || sum >= p));
        if (a < p && b < p && expected[i] !=
            (static_cast<unsigned __int128>(a) + b) % p) return false;
        if (cufhe::small_mod64_add<0>(a, b) != expected[i] ||
            cufhe::small_mod64_add<1>(a, b) != expected[i] ||
            cufhe::small_mod64_add<2>(a, b) != expected[i]) return false;
    }
    ulonglong2* in = nullptr;
    ulonglong3* out = nullptr;
    bool ok = cudaMalloc(&in, input.size() * sizeof(ulonglong2)) == cudaSuccess;
    ok = ok && cudaMalloc(&out, actual.size() * sizeof(ulonglong3)) == cudaSuccess;
    ok = ok && cudaMemcpy(in, input.data(), input.size() * sizeof(ulonglong2),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        TestGoldilocksAddKernel<<<(input.size() + 255) / 256, 256>>>(in, out, input.size());
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(actual.data(), out, actual.size() * sizeof(ulonglong3),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(in); cudaFree(out);
    for (size_t i = 0; ok && i < input.size(); ++i)
        ok = actual[i].x == expected[i] && actual[i].y == expected[i] && actual[i].z == expected[i];
    return ok;
}

__global__ void TestGoldilocksSubKernel(const ulonglong2* in, uint64_t* out, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = cufhe::small_mod64_sub_borrow(in[i].x, in[i].y);
}

bool TestGoldilocksSub()
{
    constexpr uint64_t p = cufhe::small_ntt::P;
    const std::array<uint64_t, 12> edges = {0, 1, UINT32_MAX - 1, UINT32_MAX,
        uint64_t{1} << 32, p / 2, p - 2, p - 1, p, p + 1, UINT64_MAX - 1, UINT64_MAX};
    std::vector<ulonglong2> input;
    for (auto a : edges) for (auto b : edges) input.push_back(make_ulonglong2(a, b));
    std::mt19937_64 rng(0x535542424f52524fULL);
    for (int i = 0; i < 100000; ++i) input.push_back(make_ulonglong2(rng() % p, rng() % p));
    for (int i = 0; i < 100000; ++i) input.push_back(make_ulonglong2(rng(), rng()));
    std::vector<uint64_t> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const uint64_t a = input[i].x, b = input[i].y, difference = a - b;
        expected[i] = difference - static_cast<uint32_t>(-(difference > a));
        if (a < p && b < p && expected[i] !=
            (static_cast<unsigned __int128>(a) + p - b) % p) return false;
        if (cufhe::small_mod64_sub_borrow(a, b) != expected[i]) return false;
    }
    ulonglong2* in = nullptr;
    uint64_t* out = nullptr;
    bool ok = cudaMalloc(&in, input.size() * sizeof(ulonglong2)) == cudaSuccess;
    ok = ok && cudaMalloc(&out, actual.size() * sizeof(uint64_t)) == cudaSuccess;
    ok = ok && cudaMemcpy(in, input.data(), input.size() * sizeof(ulonglong2),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        TestGoldilocksSubKernel<<<(input.size() + 255) / 256, 256>>>(in, out, input.size());
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(actual.data(), out, actual.size() * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(in); cudaFree(out);
    return ok && actual == expected;
}
#endif

__global__ void TestGoldilocksScaleKernel(const ulonglong2* input,
                                         ulonglong3* output, size_t count, uint64_t scaled_root)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    constexpr int power = TFHEpp::lvl2param::nbit;
    uint64_t u = input[i].x, v = input[i].y;
    const uint64_t divided = cufhe::small_mod64_div_pow2<power>(u);
    cufhe::SmallGentlemanSandeGoldilocksScaledUnit<power>(u, v, scaled_root);
    output[i] = make_ulonglong3(divided, u, v);
}

bool TestGoldilocksScale()
{
    constexpr uint64_t p = cufhe::small_ntt::P;
    constexpr int power = TFHEpp::lvl2param::nbit;
    constexpr uint64_t n = TFHEpp::lvl2param::n;
    cufhe::CuNTTHandler<n, 64> ntt;
    ntt.Create();
    ntt.SetDevicePointers(0);
    const uint64_t inverse = ntt.n_inverse_;
    uint64_t scaled_root = 0;
    bool ok = cudaMemcpy(&scaled_root, ntt.inverse_root_ + 1,
                          sizeof(scaled_root), cudaMemcpyDeviceToHost) == cudaSuccess;
    ok = ok && static_cast<unsigned __int128>(inverse) * n % p == 1 &&
         scaled_root == p - (uint64_t{1} << (48 - power));
    std::vector<ulonglong2> input;
    const std::array<uint64_t, 18> edges = {
        0, 1, 2, n - 1, n, n + 1, (uint64_t{1} << 32) - 1,
        uint64_t{1} << 32, (uint64_t{1} << 32) + 1,
        (uint64_t{1} << 48) - 1, uint64_t{1} << 48,
        (uint64_t{1} << 48) + 1, p / 2, p - n - 1, p - n,
        p - n + 1, p - 2, p - 1};
    for (uint64_t u : edges)
        for (uint64_t v : edges) input.push_back(make_ulonglong2(u, v));
    // Cover every low limb of the modular division near zero, mid-field,
    // and the upper boundary, along with independent butterfly partners.
    for (uint64_t low = 0; low < n; ++low)
        for (uint64_t base : {uint64_t{0}, p / 2 - n, p - n})
            input.push_back(make_ulonglong2(base + low, edges[low % edges.size()]));
    std::mt19937_64 random(0x474f4c445343414cULL);
    for (int i = 0; i < 100000; ++i)
        input.push_back(make_ulonglong2(random() % p, random() % p));
    std::vector<ulonglong3> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const uint64_t a = input[i].x, b = input[i].y;
        const uint64_t sum = (static_cast<unsigned __int128>(a) + b) % p;
        const uint64_t difference = (static_cast<unsigned __int128>(a) + p - b) % p;
        expected[i] = make_ulonglong3(
            static_cast<unsigned __int128>(a) * inverse % p,
            static_cast<unsigned __int128>(sum) * inverse % p,
            static_cast<unsigned __int128>(difference) * scaled_root % p);
        uint64_t u = a, v = b;
        cufhe::SmallGentlemanSandeGoldilocksScaledUnit<power>(u, v, scaled_root);
        if (cufhe::small_mod64_div_pow2<power>(a) != expected[i].x ||
            u != expected[i].y || v != expected[i].z) return false;
    }
    ulonglong2* device_input = nullptr;
    ulonglong3* device_output = nullptr;
    ok = ok && cudaMalloc(&device_input, input.size() * sizeof(ulonglong2)) == cudaSuccess;
    ok = ok && cudaMalloc(&device_output, actual.size() * sizeof(ulonglong3)) == cudaSuccess;
    ok = ok && cudaMemcpy(device_input, input.data(), input.size() * sizeof(ulonglong2),
                          cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        TestGoldilocksScaleKernel<<<(input.size() + 255) / 256, 256>>>(
            device_input, device_output, input.size(), scaled_root);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(ulonglong3),
                         cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device_input);
    cudaFree(device_output);
    if (!ok) return false;
    for (size_t i = 0; i < input.size(); ++i)
        if (actual[i].x != expected[i].x || actual[i].y != expected[i].y ||
            actual[i].z != expected[i].z) return false;
    return true;
}

template <int Sign0, int Sign1>
__global__ void TestCorrectionKernel(TFHEpp::lvl0param::T* out,
                                     const TFHEpp::lvl0param::T* in0,
                                     const TFHEpp::lvl0param::T* in1)
{
    constexpr uint32_t count = TFHEpp::lvl0param::k * TFHEpp::lvl0param::n;
    const auto correction = cufhe::BlockBinaryModSwitchCorrection<TFHEpp::lvl01param, Sign0, Sign1>(
        in0 + blockIdx.x * count, in1 + blockIdx.x * count);
    if (threadIdx.x == 0) out[blockIdx.x] = TFHEpp::lvl0param::T(correction);
}

template <int Sign0, int Sign1>
bool TestCorrection()
{
    using P = TFHEpp::lvl01param;
    using T = P::domainP::T;
    constexpr uint32_t count = P::domainP::k * P::domainP::n;
    constexpr uint32_t shift = std::numeric_limits<T>::digits - 1 - P::targetP::nbit;
    constexpr T rounding = T{1} << (shift - 1);
    constexpr uint32_t cases = 112;
    std::vector<T> a(count * cases), b(a.size()), expected(cases), actual(cases);
    std::mt19937 rng(0x434f5252);
    for (uint32_t block = 0; block < cases; ++block) {
        T correction = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const size_t index = block * count + i;
            a[index] = block < 105 ? T(index) : T(rng());
            b[index] = T(rng());
            if (block >= 108) {
                a[index] = block == 108 ? rounding : block == 109 ? rounding - 1 : block == 110 ? T(-1) : T(0);
                b[index] = 0;
            }
            const T phase = Sign0 * a[index] + Sign1 * b[index];
            const T rounded = phase + rounding;
            const uint32_t moded = rounded >> shift;
            correction += phase - (T(moded) << shift);
        }
        expected[block] = correction;
    }
    T *gpu_a = nullptr, *gpu_b = nullptr, *gpu_out = nullptr;
    if (cudaMalloc(&gpu_a, a.size() * sizeof(T)) != cudaSuccess ||
        cudaMalloc(&gpu_b, b.size() * sizeof(T)) != cudaSuccess ||
        cudaMalloc(&gpu_out, actual.size() * sizeof(T)) != cudaSuccess) return false;
    bool ok = cudaMemcpy(gpu_a, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(gpu_b, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
    TestCorrectionKernel<Sign0, Sign1><<<cases, 512>>>(gpu_out, gpu_a, gpu_b);
    ok = ok && cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(actual.data(), gpu_out, actual.size() * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_out);
    return ok && actual == expected;
}

__global__ void TestLazyRangeKernel(const uint4* input, uint2* output, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        uint32_t u = input[i].x, v = input[i].y;
        cufhe::SmallCooleyTukeyLazyShoupUnit(u, v, input[i].z, input[i].w);
        output[i] = make_uint2(u, v);
    }
}

bool TestLazyRange()
{
    constexpr uint32_t p = cufhe::small_ntt31::P;
    std::vector<uint4> input;
    auto append = [&](uint32_t u, uint32_t v, uint32_t root) {
        input.push_back(make_uint4(u, v, root, (uint64_t(root) << 32) / p));
    };
    for (uint32_t u : {0U, 1U, p - 1, p, p + 1, 2 * p - 2, 2 * p - 1})
        for (uint32_t v : {0U, 1U, p - 1, p, p + 1, 2 * p - 2, 2 * p - 1})
            for (uint32_t root : {0U, 1U, p / 2, p - 2, p - 1})
                append(u, v, root);
    std::mt19937_64 random(0x4c415a59);
    for (int i = 0; i < 100000; ++i)
        append(random() % (2ULL * p), random() % (2ULL * p), random() % p);
    uint4* device_input = nullptr;
    uint2* device_output = nullptr;
    if (cudaMalloc(&device_input, input.size() * sizeof(uint4)) != cudaSuccess)
        return false;
    if (cudaMalloc(&device_output, input.size() * sizeof(uint2)) != cudaSuccess) {
        cudaFree(device_input);
        return false;
    }
    bool ok = cudaMemcpy(device_input, input.data(), input.size() * sizeof(uint4),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    TestLazyRangeKernel<<<(input.size() + 255) / 256, 256>>>(
        device_input, device_output, input.size());
    ok = ok && cudaGetLastError() == cudaSuccess;
    std::vector<uint2> output(input.size());
    ok = ok && cudaMemcpy(output.data(), device_output,
                          output.size() * sizeof(uint2),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(device_input);
    cudaFree(device_output);
    if (!ok) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        const auto a = input[i];
        const auto b = output[i];
        const uint64_t product = uint64_t(a.y) * a.z % p;
        const uint32_t expected_u = (uint64_t(a.x) + product) % p;
        const uint32_t expected_v = (uint64_t(a.x) + p - product) % p;
        if (b.x >= 2ULL * p || b.y >= 2ULL * p ||
            b.x % p != expected_u || b.y % p != expected_v)
            return false;
    }
    return true;
}

__global__ void TestInverseButterflyKernel(const uint4* input, uint4* output,
                                          size_t count, uint32_t n_inverse)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const uint4 a = input[i];
        uint32_t u = a.x, v = a.y, scaled_u = a.x, scaled_v = a.y;
        cufhe::SmallGentlemanSandeShoupUnit(u, v, a.z, a.w);
        cufhe::SmallGentlemanSandeScaledShoupUnit(
            scaled_u, scaled_v, a.z, a.w, n_inverse);
        output[i] = make_uint4(u, v, scaled_u, scaled_v);
    }
}

bool TestInverseButterflies()
{
    constexpr uint32_t p = cufhe::small_ntt31::P;
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t n_inverse = p - (p - 1) / N;
    static_assert(uint64_t(n_inverse) * N % p == 1);
    std::vector<uint4> input;
    auto append = [&](uint32_t u, uint32_t v, uint32_t root) {
        input.push_back(make_uint4(u, v, root, (uint64_t(root) << 32) / p));
    };
    for (uint32_t u : {0U, 1U, p / 2, p - 2, p - 1})
        for (uint32_t v : {0U, 1U, p / 2, p - 2, p - 1})
            for (uint32_t root : {0U, 1U, p / 2, p - 2, p - 1})
                append(u, v, root);
    std::mt19937_64 random(0x494e5653);
    for (int i = 0; i < 100000; ++i)
        append(random() % p, random() % p, random() % p);
    uint4 *device_input = nullptr, *device_output = nullptr;
    if (cudaMalloc(&device_input, input.size() * sizeof(uint4)) != cudaSuccess)
        return false;
    if (cudaMalloc(&device_output, input.size() * sizeof(uint4)) != cudaSuccess) {
        cudaFree(device_input);
        return false;
    }
    bool ok = cudaMemcpy(device_input, input.data(), input.size() * sizeof(uint4),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    TestInverseButterflyKernel<<<(input.size() + 255) / 256, 256>>>(
        device_input, device_output, input.size(), n_inverse);
    ok = ok && cudaGetLastError() == cudaSuccess;
    std::vector<uint4> output(input.size());
    ok = ok && cudaMemcpy(output.data(), device_output,
                          output.size() * sizeof(uint4),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(device_input);
    cudaFree(device_output);
    if (!ok) return false;
    for (size_t i = 0; i < input.size(); ++i) {
        const auto a = input[i];
        const auto b = output[i];
        const uint32_t sum = (uint64_t(a.x) + a.y) % p;
        const uint32_t product = (uint64_t(a.x) + p - a.y) * a.z % p;
        const uint32_t scaled_sum = uint64_t(sum) * n_inverse % p;
        // In the scaled helper, the supplied root is already scaled.
        if (b.x != sum || b.y != product ||
            b.z != scaled_sum || b.w != product)
            return false;
    }
    return true;
}

__global__ void TestReduce31Kernel(const uint64_t* input, uint32_t* output, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = cufhe::small_mod31_reduce64(input[i]);
}

bool TestReduce31()
{
    constexpr uint64_t p = cufhe::small_ntt31::P;
    const std::array<uint32_t, 12> edges = {0, 1, 2, 10239, 20478, uint32_t(p - 1),
        uint32_t(p), 0x7fffffffU, 0x80000000U, 0xfffffffdU, 0xfffffffeU, 0xffffffffU};
    std::vector<uint64_t> input;
    for (auto hi : edges)
        for (auto lo : edges) input.push_back((uint64_t(hi) << 32) | lo);
    for (uint64_t k : std::array<uint64_t, 6>{0, 1, 2, 65535, 0xffffffffULL, UINT64_MAX / p}) {
        const uint64_t multiple = k * p;
        input.push_back(multiple);
        if (multiple > 0) input.push_back(multiple - 1);
        if (multiple < UINT64_MAX) input.push_back(multiple + 1);
    }
    std::mt19937_64 rng(0x5245445543453331ULL);
    for (int i = 0; i < 250000; ++i) input.push_back(rng());
    std::vector<uint32_t> expected(input.size()), output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        expected[i] = uint32_t(input[i] % p);
        if (cufhe::small_mod31_reduce64(input[i]) != expected[i]) return false;
    }
    uint64_t* gpu_input = nullptr;
    uint32_t* gpu_output = nullptr;
    if (cudaMalloc(&gpu_input, input.size() * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&gpu_output, output.size() * sizeof(uint32_t)) != cudaSuccess) return false;
    bool ok = cudaMemcpy(gpu_input, input.data(), input.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess;
    TestReduce31Kernel<<<(input.size() + 255) / 256, 256>>>(gpu_input, gpu_output, input.size());
    ok = ok && cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(output.data(), gpu_output, output.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    return ok && output == expected;
}

struct Wide64Case { uint64_t lo, hi; };

__global__ void TestReduce64Kernel(const Wide64Case* input, uint64_t* output,
                                  size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto a = input[i];
        output[i] = cufhe::small_mod64_reduce_wide<true>(
            (static_cast<unsigned __int128>(a.hi) << 64) | a.lo);
    }
}

bool TestReduce64()
{
    constexpr uint64_t p = cufhe::small_ntt::P, b = uint64_t{1} << 32;
    const std::array<uint64_t, 15> edges = {
        0, 1, 2, b - 2, b - 1, b, p - 2, p - 1, p, p + 1,
        UINT64_MAX, UINT64_MAX - 1, uint64_t{1} << 63, b + 1, ~b};
    std::vector<Wide64Case> input;
    for (uint64_t hi : edges)
        for (uint64_t lo : edges) input.push_back({lo, hi});
    // Borrow boundaries and noncanonical low limbs, across both high limbs.
    for (uint64_t h : {0ULL, 1ULL, 2ULL, 0xfffffffeULL, 0xffffffffULL})
        for (uint64_t l : {0ULL, 1ULL, 2ULL, 0xfffffffeULL, 0xffffffffULL})
            for (uint64_t lo : std::array<uint64_t, 7>{h - 1, h, h + 1,
                                                       p - 1, p, p + 1, UINT64_MAX})
                input.push_back({lo, (h << 32) | l});
    std::mt19937_64 rng(0x5245445543453634ULL);
    for (int i = 0; i < 1000000; ++i) input.push_back({rng(), rng()});
    std::vector<uint64_t> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const auto a = input[i];
        const auto value = (static_cast<unsigned __int128>(a.hi) << 64) | a.lo;
        expected[i] = value % p;
        if (cufhe::small_mod64_reduce_wide<true>(value) != expected[i]) return false;
    }
    Wide64Case* gpu_input = nullptr;
    uint64_t* gpu_output = nullptr;
    if (cudaMalloc(&gpu_input, input.size() * sizeof(Wide64Case)) != cudaSuccess)
        return false;
    if (cudaMalloc(&gpu_output, actual.size() * sizeof(uint64_t)) != cudaSuccess) {
        cudaFree(gpu_input);
        return false;
    }
    bool ok = cudaMemcpy(gpu_input, input.data(), input.size() * sizeof(Wide64Case),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    TestReduce64Kernel<<<(input.size() + 255) / 256, 256>>>(
        gpu_input, gpu_output, input.size());
    ok = ok && cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(actual.data(), gpu_output, actual.size() * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    return ok && actual == expected;
}

struct Madd2Case { uint64_t a0, b0, a1, b1, c; };

__global__ void TestMadd2Kernel(const Madd2Case* input, uint64_t* output, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto a = input[i];
        output[i] = cufhe::small_mod64_madd2<true>(a.a0, a.b0, a.a1, a.b1, a.c);
    }
}

bool TestMadd2()
{
    constexpr uint64_t p = cufhe::small_ntt::P;
    const std::array<uint64_t, 8> edges = {
        0, 1, 2, UINT32_MAX, uint64_t{1} << 63, p - 1, p, UINT64_MAX};
    std::vector<Madd2Case> input;
    for (uint64_t a0 : edges) for (uint64_t b0 : edges)
        for (uint64_t a1 : edges) for (uint64_t b1 : edges)
            for (uint64_t c : std::array<uint64_t, 4>{0, 1, p - 1, UINT64_MAX})
                input.push_back({a0, b0, a1, b1, c});
    std::mt19937_64 rng(0x4d41444432574944ULL);
    for (int i = 0; i < 250000; ++i)
        input.push_back({rng(), rng(), rng(), rng(), rng()});
    std::vector<uint64_t> expected(input.size()), actual(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const auto a = input[i];
        // Each reference sum fits 128 bits, even with full-range factors.
        const uint64_t first = (static_cast<unsigned __int128>(a.a0) * a.b0 + a.c) % p;
        expected[i] = (static_cast<unsigned __int128>(a.a1) * a.b1 + first) % p;
        if (cufhe::small_mod64_madd2<true>(a.a0, a.b0, a.a1, a.b1, a.c) != expected[i])
            return false;
    }
    Madd2Case* gpu_input = nullptr;
    uint64_t* gpu_output = nullptr;
    if (cudaMalloc(&gpu_input, input.size() * sizeof(Madd2Case)) != cudaSuccess)
        return false;
    if (cudaMalloc(&gpu_output, actual.size() * sizeof(uint64_t)) != cudaSuccess) {
        cudaFree(gpu_input);
        return false;
    }
    bool ok = cudaMemcpy(gpu_input, input.data(), input.size() * sizeof(Madd2Case),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    TestMadd2Kernel<<<(input.size() + 255) / 256, 256>>>(gpu_input, gpu_output, input.size());
    ok = ok && cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(actual.data(), gpu_output, actual.size() * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    return ok && expected == actual;
}

template <uint32_t N, uint32_t Q_BITS>
uint64_t ReferenceMadd(const uint64_t a, const uint64_t b, const uint64_t c)
{
    return static_cast<uint64_t>(
        (static_cast<unsigned __int128>(a) * b + c) %
        cufhe::SmallNTTModulus<N, Q_BITS>::P);
}

template <uint32_t N, uint32_t Q_BITS>
uint64_t ReferenceMadd3(const uint64_t a0, const uint64_t b0,
                        const uint64_t a1, const uint64_t b1,
                        const uint64_t a2, const uint64_t b2,
                        const uint64_t c)
{
    uint64_t result = ReferenceMadd<N, Q_BITS>(a0, b0, c);
    result = ReferenceMadd<N, Q_BITS>(a1, b1, result);
    return ReferenceMadd<N, Q_BITS>(a2, b2, result);
}

template <uint32_t N, uint32_t Q_BITS>
bool TestHostMadd()
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N, Q_BITS>::P;
    const std::array<uint64_t, 12> edge_values = {
        0,      1,      2,      3,      17,     65535,
        p >> 1, p - 18, p - 4,  p - 3,  p - 2,  p - 1};

    for (const uint64_t a : edge_values) {
        for (const uint64_t b : edge_values) {
            for (const uint64_t c : edge_values) {
                if (cufhe::small_mod_madd<N, Q_BITS>(a, b, c) !=
                    ReferenceMadd<N, Q_BITS>(a, b, c)) {
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
        if (cufhe::small_mod_madd<N, Q_BITS>(a, b, c) !=
            ReferenceMadd<N, Q_BITS>(a, b, c)) {
            return false;
        }
    }
    return true;
}

template <uint32_t N, uint32_t Q_BITS>
bool TestHostMadd3()
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N, Q_BITS>::P;
    const std::array<uint64_t, 12> edge_values = {
        0,      1,      2,      3,      17,     65535,
        p >> 1, p - 18, p - 4,  p - 3,  p - 2,  p - 1};

    if (cufhe::small_mod_madd3<N, Q_BITS>(p - 1, p - 1, p - 1, p - 1, p - 1,
                                  p - 1, p - 1) !=
        ReferenceMadd3<N, Q_BITS>(p - 1, p - 1, p - 1, p - 1, p - 1, p - 1,
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
        if (cufhe::small_mod_madd3<N, Q_BITS>(a0, b0, a1, b1, a2, b2, c) !=
            ReferenceMadd3<N, Q_BITS>(a0, b0, a1, b1, a2, b2, c)) {
            return false;
        }
        // Cover the two-product specialization used by block-binary keys.
        if (cufhe::small_mod_madd3<N, Q_BITS>(a0, b0, a1, b1, 0, 0, c) !=
            ReferenceMadd3<N, Q_BITS>(a0, b0, a1, b1, 0, 0, c)) {
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
        if (cufhe::small_mod_madd3<N, Q_BITS>(a0, b0, a1, b1, a2, b2, c) !=
            ReferenceMadd3<N, Q_BITS>(a0, b0, a1, b1, a2, b2, c)) {
            return false;
        }
        // Cover the two-product specialization used by block-binary keys.
        if (cufhe::small_mod_madd3<N, Q_BITS>(a0, b0, a1, b1, 0, 0, c) !=
            ReferenceMadd3<N, Q_BITS>(a0, b0, a1, b1, 0, 0, c)) {
            return false;
        }
    }
    return true;
}

uint64_t ReferenceMadd4(const uint64_t* v)
{
    unsigned __int128 sum = v[8];
    for (int j = 0; j < 8; j += 2) sum += static_cast<unsigned __int128>(v[j]) * v[j + 1];
    return uint64_t(sum % cufhe::small_ntt31::P);
}

__host__ __device__ uint64_t EvaluateMadd4(const uint64_t* v)
{
    return cufhe::small_mod31_madd4(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]);
}

__global__ void TestDeviceMadd4Kernel(const uint64_t* values, const uint64_t* expected,
                                     unsigned int* failures, size_t count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && EvaluateMadd4(values + i * 9) != expected[i]) atomicAdd(failures, 1U);
}

bool TestMadd4()
{
    constexpr uint64_t p = cufhe::small_ntt31::P;
    constexpr size_t count = 250016;
    std::vector<uint64_t> values(count * 9), expected(count);
    std::mt19937_64 rng(0x4e54544d414434ULL);
    for (size_t i = 0; i < count; ++i) {
        for (int j = 0; j < 9; ++j) {
            values[i * 9 + j] = i < 8 ? p - 1 - i : rng() % p;
            if (i >= 8 && i < 16) values[i * 9 + j] = ((i + j) & 1) ? p - 1 : 0;
        }
        expected[i] = ReferenceMadd4(values.data() + i * 9);
        if (EvaluateMadd4(values.data() + i * 9) != expected[i]) return false;
    }
    uint64_t *gpu_values = nullptr, *gpu_expected = nullptr;
    unsigned int* failures = nullptr;
    if (cudaMalloc(&gpu_values, values.size() * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&gpu_expected, expected.size() * sizeof(uint64_t)) != cudaSuccess ||
        cudaMalloc(&failures, sizeof(unsigned int)) != cudaSuccess) return false;
    bool ok = cudaMemcpy(gpu_values, values.data(), values.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(gpu_expected, expected.data(), expected.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemset(failures, 0, sizeof(unsigned int)) == cudaSuccess;
    TestDeviceMadd4Kernel<<<(count + 255) / 256, 256>>>(gpu_values, gpu_expected, failures, count);
    unsigned int host_failures = 0;
    ok = ok && cudaGetLastError() == cudaSuccess &&
         cudaMemcpy(&host_failures, failures, sizeof(unsigned int), cudaMemcpyDeviceToHost) == cudaSuccess;
    cudaFree(gpu_values);
    cudaFree(gpu_expected);
    cudaFree(failures);
    return ok && host_failures == 0;
}

template <uint32_t N, uint32_t Q_BITS>
__global__ void TestDeviceMaddKernel(unsigned int* const failures)
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N, Q_BITS>::P;
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

        const uint64_t expected = cufhe::small_mod_add<N, Q_BITS>(
            cufhe::small_mod_mult<N, Q_BITS>(a, b), c);
        if (cufhe::small_mod_madd<N, Q_BITS>(a, b, c) != expected) {
            atomicAdd(failures, 1U);
        }
    }
}

template <uint32_t N, uint32_t Q_BITS>
bool TestDeviceMadd()
{
    unsigned int* failures;
    if (cudaMalloc(&failures, sizeof(*failures)) != cudaSuccess) return false;
    if (cudaMemset(failures, 0, sizeof(*failures)) != cudaSuccess) return false;

    TestDeviceMaddKernel<N, Q_BITS><<<256, 256>>>(failures);
    if (cudaGetLastError() != cudaSuccess) return false;

    unsigned int host_failures = 0;
    if (cudaMemcpy(&host_failures, failures, sizeof(host_failures),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    cudaFree(failures);
    return host_failures == 0;
}

template <uint32_t N, uint32_t Q_BITS>
__global__ void TestDeviceMadd3Kernel(unsigned int* const failures)
{
    constexpr uint64_t p = cufhe::SmallNTTModulus<N, Q_BITS>::P;
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
        if (i & 1) {
            values[4] = 0;
            values[5] = 0;
            values[6] = 0;
        }
        uint64_t expected =
            cufhe::small_mod_madd<N, Q_BITS>(values[0], values[1], values[6]);
        expected = cufhe::small_mod_madd<N, Q_BITS>(values[2], values[3], expected);
        expected = cufhe::small_mod_madd<N, Q_BITS>(values[4], values[5], expected);
        if (cufhe::small_mod_madd3<N, Q_BITS>(
                values[0], values[1], values[2], values[3], values[4],
                values[5], values[6]) != expected) {
            atomicAdd(failures, 1U);
        }
    }
}

template <uint32_t N, uint32_t Q_BITS>
bool TestDeviceMadd3()
{
    unsigned int* failures;
    if (cudaMalloc(&failures, sizeof(*failures)) != cudaSuccess) return false;
    if (cudaMemset(failures, 0, sizeof(*failures)) != cudaSuccess) return false;

    TestDeviceMadd3Kernel<N, Q_BITS><<<256, 256>>>(failures);
    if (cudaGetLastError() != cudaSuccess) return false;

    unsigned int host_failures = 0;
    if (cudaMemcpy(&host_failures, failures, sizeof(host_failures),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    cudaFree(failures);
    return host_failures == 0;
}

template <uint32_t N, uint32_t Q_BITS, bool Square, bool Paired>
__global__ void TestTransformKernel(
    cufhe::NTTValueFor<N, Q_BITS>* out,
    const cufhe::NTTValueFor<N, Q_BITS>* in,
    cufhe::CuNTTHandler<N, Q_BITS> ntt)
{
    using Value = cufhe::NTTValueFor<N, Q_BITS>;
    constexpr int POLYS = Paired ? 2 : 1;
    __shared__ Value work[N * POLYS];
    const int tid = threadIdx.x;
    for (int p = 0; p < POLYS; ++p) {
        work[p * N + tid] = in[p * N + tid];
        work[p * N + tid + N / 2] = in[p * N + tid + N / 2];
    }
    __syncthreads();
    if constexpr (Paired) {
#if defined(CUFHE_HAS_PAIR_NTT64)
        if constexpr (Q_BITS == 64)
            cufhe::SmallForwardNTTPair64<cufhe::SmallLog2<N>()>(work, work + N, ntt.forward_root_, tid);
        else
#endif
            cufhe::SmallForwardNTTPair<cufhe::SmallLog2<N>(), Q_BITS>(work, work + N, ntt.forward_root_, tid);
    } else
        cufhe::SmallForwardNTT<cufhe::SmallLog2<N>(), Q_BITS, true>(work, ntt.forward_root_, tid);
    if constexpr (Square) {
        for (int p = 0; p < POLYS; ++p) {
            const int i = p * N + tid;
            work[i] = cufhe::small_mod_mult<N, Q_BITS>(work[i], work[i]);
            work[i + N / 2] = cufhe::small_mod_mult<N, Q_BITS>(
                work[i + N / 2], work[i + N / 2]);
        }
    }
    __syncthreads();
    if constexpr (Paired) {
#if defined(CUFHE_HAS_PAIR_NTT64)
        if constexpr (Q_BITS == 64)
            cufhe::SmallInverseNTTPair64<cufhe::SmallLog2<N>()>(work, work + N, ntt.inverse_root_, ntt.n_inverse_, tid);
        else
#endif
            cufhe::SmallInverseNTTPair<cufhe::SmallLog2<N>(), Q_BITS>(work, work + N, ntt.inverse_root_, ntt.n_inverse_, tid);
    } else
        cufhe::SmallInverseNTT<cufhe::SmallLog2<N>(), Q_BITS>(work, ntt.inverse_root_, ntt.n_inverse_, tid);
    for (int p = 0; p < POLYS; ++p) {
        out[p * N + tid] = work[p * N + tid];
        out[p * N + tid + N / 2] = work[p * N + tid + N / 2];
    }
}

// Compare against an independent signed-integer negacyclic convolution.
// Both widths deliberately use the configured N, including equal-size rings.
template <uint32_t N, uint32_t Q_BITS, bool Paired = false>
bool TestTransforms()
{
    using Value = cufhe::NTTValueFor<N, Q_BITS>;
    constexpr uint32_t COUNT = N * (Paired ? 2 : 1);
    cufhe::CuNTTHandler<N, Q_BITS> ntt;
    ntt.Create();
    ntt.SetDevicePointers(0);
    std::mt19937 rng(0x4e5454 + Q_BITS);
    std::vector<int32_t> coefficients(COUNT);
    std::vector<Value> input(COUNT), output(COUNT);
    for (uint32_t i = 0; i < COUNT; ++i) {
        coefficients[i] = static_cast<int32_t>(rng() % 15) - 7;
        input[i] = cufhe::signed_int_to_ntt_mod<N, Q_BITS>(coefficients[i]);
    }
    Value *in = nullptr, *out = nullptr;
    if (cudaMalloc(&in, COUNT * sizeof(Value)) != cudaSuccess) return false;
    if (cudaMalloc(&out, COUNT * sizeof(Value)) != cudaSuccess) {
        cudaFree(in);
        return false;
    }
    bool passed = cudaMemcpy(in, input.data(), COUNT * sizeof(Value),
                             cudaMemcpyHostToDevice) == cudaSuccess;
    if (passed) {
        TestTransformKernel<N, Q_BITS, false, Paired><<<1, N / 2>>>(out, in, ntt);
        passed = cudaGetLastError() == cudaSuccess &&
                 cudaMemcpy(output.data(), out, COUNT * sizeof(Value),
                            cudaMemcpyDeviceToHost) == cudaSuccess &&
                 output == input;
    }
    if (passed) {
        TestTransformKernel<N, Q_BITS, true, Paired><<<1, N / 2>>>(out, in, ntt);
        passed = cudaGetLastError() == cudaSuccess &&
                 cudaMemcpy(output.data(), out, COUNT * sizeof(Value),
                            cudaMemcpyDeviceToHost) == cudaSuccess;
        for (uint32_t i = 0; passed && i < COUNT; ++i) {
            int32_t expected = 0;
            const uint32_t base = (i / N) * N;
            const uint32_t index = i % N;
            for (uint32_t j = 0; j < N; ++j) {
                const int32_t product = coefficients[base + j] *
                    coefficients[base + ((index - j) & (N - 1))];
                expected += j <= index ? product : -product;
            }
            passed = output[i] ==
                     cufhe::signed_int_to_ntt_mod<N, Q_BITS>(expected);
        }
    }
    cudaFree(in);
    cudaFree(out);
    ntt.Destroy();
    return passed;
}

#if defined(CUFHE_HAS_PAIR_NTT64)
struct Madd4WideCase { uint64_t v[9]; };
__host__ __device__ uint64_t CallMadd4Wide(const Madd4WideCase& x) {
    return cufhe::small_mod64_madd4<2>(x.v[0],x.v[1],x.v[2],x.v[3],
                                  x.v[4],x.v[5],x.v[6],x.v[7],x.v[8]);
}
__global__ void TestMadd4WideKernel(const Madd4WideCase* in, uint64_t* out, size_t count) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = CallMadd4Wide(in[i]);
}
bool TestMadd4Wide() {
    constexpr uint64_t p = cufhe::small_ntt::P;
    const uint64_t edges[4] = {0,1,p-1,UINT64_MAX};
    std::vector<Madd4WideCase> input;
    for (uint32_t pattern=0;pattern<65536;++pattern)
        for (uint64_t c : std::array<uint64_t,3>{0,p-1,UINT64_MAX}) {
            Madd4WideCase x{};
            for (int j=0;j<8;++j) x.v[j]=edges[(pattern>>(2*j))&3];
            x.v[8]=c; input.push_back(x);
        }
    input.push_back({{uint64_t{1}<<63,2,UINT64_MAX,UINT64_MAX,0,0,0,0,UINT64_MAX}});
    input.push_back({{uint64_t{1}<<63,2,UINT64_MAX,UINT64_MAX,uint64_t{1}<<63,4,UINT64_MAX,UINT64_MAX,UINT64_MAX}});
    input.push_back({{p,p,p,p,p,p,p,p,0}});
    std::mt19937_64 rng(0x4d41444434313330ULL);
    for(int i=0;i<200000;++i) {
        Madd4WideCase x;for(auto& v:x.v)v=rng();input.push_back(x);
    }
    std::vector<uint64_t> expected(input.size()), actual(input.size());
    for(size_t i=0;i<input.size();++i) {
        const auto& x=input[i]; uint64_t r=x.v[8];
        for(int j=0;j<4;++j) r=(static_cast<unsigned __int128>(x.v[2*j])*x.v[2*j+1]+r)%p;
        expected[i]=r; if(CallMadd4Wide(x)!=r)return false;
    }
    Madd4WideCase* gpu_in=nullptr;uint64_t* gpu_out=nullptr;
    if(cudaMalloc(&gpu_in,input.size()*sizeof(Madd4WideCase))!=cudaSuccess)return false;
    if(cudaMalloc(&gpu_out,actual.size()*sizeof(uint64_t))!=cudaSuccess){cudaFree(gpu_in);return false;}
    bool ok=cudaMemcpy(gpu_in,input.data(),input.size()*sizeof(Madd4WideCase),cudaMemcpyHostToDevice)==cudaSuccess;
    TestMadd4WideKernel<<<(input.size()+255)/256,256>>>(gpu_in,gpu_out,input.size());
    ok=ok&&cudaGetLastError()==cudaSuccess&&cudaMemcpy(actual.data(),gpu_out,actual.size()*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess;
    cudaFree(gpu_in);cudaFree(gpu_out);return ok&&actual==expected;
}

template <class Param, bool Paired, bool Inverse, bool NativeSubtract = false, int NativeAdd = 0, bool UniformRoot = false>
__global__ void TestPair64CompareKernel(uint64_t* out,const uint64_t* in,
    cufhe::CuNTTHandler<Param::n,64> ntt) {
    constexpr int N=Param::n;
    __shared__ uint64_t sh[2*N];int tid=threadIdx.x;
    for(int k=0;k<2;++k){sh[k*N+tid]=in[k*N+tid];sh[k*N+tid+N/2]=in[k*N+tid+N/2];}
    __syncthreads();
    if constexpr(Paired) {
        if constexpr(Inverse)cufhe::SmallInverseNTTPair64<Param::nbit,NativeAdd,UniformRoot>(sh,sh+N,ntt.inverse_root_,ntt.n_inverse_,tid);
        else cufhe::SmallForwardNTTPair64<Param::nbit, NativeSubtract,NativeAdd,UniformRoot>(sh,sh+N,ntt.forward_root_,tid);
    } else {
        for(int k=0;k<2;++k) {
            if constexpr(Inverse)cufhe::SmallInverseNTT<Param::nbit,64,NativeAdd>(sh+k*N,ntt.inverse_root_,ntt.n_inverse_,tid);
            else cufhe::SmallForwardNTT<Param::nbit,64,NativeSubtract,NativeAdd>(sh+k*N,ntt.forward_root_,tid);
        }
    }
    for(int k=0;k<2;++k){out[k*N+tid]=sh[k*N+tid];out[k*N+tid+N/2]=sh[k*N+tid+N/2];}
}
template <class Param>
bool TestPair64Compare() {
    constexpr int N=Param::n;constexpr uint64_t p=cufhe::small_ntt::P;
    cufhe::CuNTTHandler<N,64> ntt;ntt.Create();ntt.SetDevicePointers(0);
    uint64_t *in=nullptr,*ref=nullptr,*got=nullptr;
    if(cudaMalloc(&in,2*N*sizeof(uint64_t))!=cudaSuccess||cudaMalloc(&ref,2*N*sizeof(uint64_t))!=cudaSuccess||cudaMalloc(&got,2*N*sizeof(uint64_t))!=cudaSuccess)return false;
    std::vector<uint64_t> input(2*N), expected(2*N), actual(2*N);std::mt19937_64 rng(320064);bool ok=true;
    for(int sample=0;sample<5&&ok;++sample) {
        for(int i=0;i<2*N;++i)input[i]=sample==0?0:sample==1?p-1:sample==2?(i%2?p-1:0):sample==3?rng()%p:(i%N==N-1?1:0);
        ok=cudaMemcpy(in,input.data(),2*N*sizeof(uint64_t),cudaMemcpyHostToDevice)==cudaSuccess;
        for(int inverse=0;inverse<2&&ok;++inverse) {
            if(inverse){TestPair64CompareKernel<Param,false,true><<<1,N/2>>>(ref,in,ntt);TestPair64CompareKernel<Param,true,true><<<1,N/2>>>(got,in,ntt);}
            else {TestPair64CompareKernel<Param,false,false><<<1,N/2>>>(ref,in,ntt);TestPair64CompareKernel<Param,true,false><<<1,N/2>>>(got,in,ntt);}
            ok=cudaGetLastError()==cudaSuccess&&cudaMemcpy(expected.data(),ref,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&cudaMemcpy(actual.data(),got,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected;
            if (ok) {
                if (inverse) TestPair64CompareKernel<Param,false,true,false,true><<<1,N/2>>>(got,in,ntt);
                else TestPair64CompareKernel<Param,false,false,false,true><<<1,N/2>>>(got,in,ntt);
                ok=cudaGetLastError()==cudaSuccess&&cudaMemcpy(actual.data(),got,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected;
            }
            if (ok) {
                if (inverse) TestPair64CompareKernel<Param,true,true,false,2><<<1,N/2>>>(got,in,ntt);
                else TestPair64CompareKernel<Param,true,false,false,2><<<1,N/2>>>(got,in,ntt);
                ok=cudaGetLastError()==cudaSuccess&&cudaMemcpy(actual.data(),got,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected;
            }
            if (ok) {
                if (inverse) TestPair64CompareKernel<Param,true,true,true,0,true><<<1,N/2>>>(got,in,ntt);
                else TestPair64CompareKernel<Param,true,false,true,0,true><<<1,N/2>>>(got,in,ntt);
                ok=cudaGetLastError()==cudaSuccess&&cudaMemcpy(actual.data(),got,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected;
            }
            if (!inverse && ok) {
                TestPair64CompareKernel<Param,true,false,true><<<1,N/2>>>(got,in,ntt);
                ok=cudaGetLastError()==cudaSuccess&&cudaMemcpy(actual.data(),got,2*N*sizeof(uint64_t),cudaMemcpyDeviceToHost)==cudaSuccess&&actual==expected;
            }
        }
    }
    cudaFree(in);cudaFree(ref);cudaFree(got);ntt.Destroy();return ok;
}
#endif


#if defined(CUFHE_HAS_PAIR_NTT64) && defined(USE_BLOCK_BINARY) && !defined(USE_FFT)
// Compare the complete cached external product with the independently selected
// uncached scalar schedule, including decomposition and final torus conversion.
template <bool Cached, bool Paired = false>
__global__ __launch_bounds__(TFHEpp::lvl2param::n / 2, 1)
void TestScalarMonomialCacheKernel(
    uint64_t* out, const uint64_t* input, const uint64_t* key,
    const uint32_t* exponents, cufhe::CuNTTHandler<TFHEpp::lvl2param::n, 64> ntt)
{
    using P = TFHEpp::lvl02param;
    using Target = P::targetP;
    constexpr int N = Target::n, Words = (Target::k + 1) * N;
    extern __shared__ uint64_t scratch[];
    auto* trlwe = scratch + (Target::k + 2 + (Paired ? 1 : 0)) * N;
    const int tid = threadIdx.x, sample = blockIdx.x;
    for (int i = tid; i < Words; i += N / 2)
        trlwe[i] = input[sample * Words + i];
    __syncthreads();
    const uint32_t bara[2] = {exponents[2 * sample], exponents[2 * sample + 1]};
    if constexpr (Paired)
        cufhe::AccumulateBlockBinaryPaired64<P, false, 2>(
            trlwe, scratch, 0, bara, key, ntt);
    else
        cufhe::AccumulateBlockBinary<P, false, false, Cached>(
            trlwe, scratch, 0, bara, key, ntt);
    for (int i = tid; i < Words; i += N / 2)
        out[sample * Words + i] = trlwe[i];
}

template <bool Paired = false>
bool TestScalarMonomialCache(bool& table_symmetry)
{
    using Target = TFHEpp::lvl2param;
    constexpr int N = Target::n, Cases = 16, Words = (Target::k + 1) * N;
    constexpr size_t KeyWords = 2 * cufhe::BootstrappingTRGSWRows<Target> *
        (Target::k + 1) * N;
    // This fixture explicitly instantiates the paired kernel even when the
    // production dispatch disables it. Match the fixture's workspace/TRLWE
    // offsets rather than the option-dependent production allocation.
    constexpr size_t Shared = (2 * Target::k + 3 + (Paired ? 1 : 0)) *
                              N * sizeof(typename Target::T);
    constexpr uint64_t p = cufhe::small_ntt::P;
    cufhe::CuNTTHandler<N, 64> ntt;
    ntt.Create(); ntt.SetDevicePointers(0);
    cufhe::InitializeXaiNTT_lvl02(1);
    // Verify the reconstruction identity against every word of the real table.
    std::vector<uint64_t> table(size_t(2) * N * N);
    table_symmetry = cudaMemcpy(table.data(), cufhe::xai_ntt_devs_lvl02[0],
        table.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess;
    for (int a = 0; a < 2 * N && table_symmetry; ++a) {
        for (int i = 0; i < N; i += 2) {
            const auto x = table[size_t(a) * N + i];
            const auto y = table[size_t(a) * N + i + 1];
            if (x > p - 2 || y > p - 2 ||
                ((a & 1) ? y != p - 2 - x : y != x)) {
                std::cerr << "monomial symmetry mismatch at exponent " << a
                          << " coefficient " << i << '\n';
                table_symmetry = false; break;
            }
        }
    }
    std::mt19937_64 rng(730064);
    const std::array<uint32_t, 8> powers = {0, 1, N / 2 - 1, N / 2,
                                           N - 1, N, N + 1, 2 * N - 1};
    const std::array<uint64_t, 8> edges = {0, 1, UINT64_MAX, uint64_t{1} << 63,
        (uint64_t{1} << 24) - 1, uint64_t{1} << 24,
        (uint64_t{1} << 25) - 1, uint64_t{1} << 25};
    std::vector<uint64_t> input(Cases * Words), key(KeyWords);
    std::vector<uint32_t> exponents(2 * Cases);
    for (int sample = 0; sample < Cases; ++sample) {
        exponents[2 * sample] = powers[sample % powers.size()];
        exponents[2 * sample + 1] = powers[(sample + 3) % powers.size()];
        for (int i = 0; i < Words; ++i)
            input[sample * Words + i] = sample < 4 ? edges[sample] :
                ((i & 7) == 0 ? edges[(i / 8) % edges.size()] : rng());
    }
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = i % 16 == 0 ? 0 : i % 16 == 1 ? p - 1 :
                 i % 16 == 2 ? p / 2 : rng() % p;
    uint64_t *dinput = nullptr, *dkey = nullptr, *dref = nullptr, *dgot = nullptr;
    uint32_t* dpowers = nullptr;
    bool ok = cudaMalloc(&dinput, input.size() * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&dkey, key.size() * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&dref, input.size() * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&dgot, input.size() * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc(&dpowers, exponents.size() * sizeof(uint32_t)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dinput, input.data(), input.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dkey, key.data(), key.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dpowers, exponents.data(), exponents.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaFuncSetAttribute(TestScalarMonomialCacheKernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess &&
        cudaFuncSetAttribute(TestScalarMonomialCacheKernel<true, Paired>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
    if (ok) {
        TestScalarMonomialCacheKernel<false><<<Cases, N / 2, Shared>>>(dref, dinput, dkey, dpowers, ntt);
        ok = cudaGetLastError() == cudaSuccess;
        TestScalarMonomialCacheKernel<true, Paired><<<Cases, N / 2, Shared>>>(dgot, dinput, dkey, dpowers, ntt);
        ok = cudaGetLastError() == cudaSuccess && ok;
        std::vector<uint64_t> expected(input.size()), actual(input.size());
        ok = cudaMemcpy(expected.data(), dref, expected.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(actual.data(), dgot, actual.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) == cudaSuccess && ok;
        if (ok && actual != expected) {
            for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
                std::cerr << "scalar monomial cache mismatch at coefficient " << i << '\n'; break;
            }
            ok = false;
        }
    }
    cudaFree(dinput); cudaFree(dkey); cudaFree(dref); cudaFree(dgot); cudaFree(dpowers);
    cufhe::DeleteXaiNTT_lvl02(); ntt.Destroy();
    return ok;
}
#endif

#if defined(USE_BLOCK_BINARY) && defined(USE_CUDA_PAIRED_NTT) && !defined(USE_FFT)
template <bool Cached, bool KeySwitchAfter31 = false>
__global__ void TestPairedMonomialCache31Kernel(
    uint32_t* out, const uint32_t* input, const uint32_t* key,
    const uint32_t* exponents, cufhe::CuNTTHandler<TFHEpp::lvl1param::n, 32> ntt)
{
    using P = TFHEpp::lvl01param;
    using Target = P::targetP;
    constexpr int N = Target::n, Words = (Target::k + 1) * N;
    extern __shared__ uint32_t scratch31[];
    auto* trlwe = scratch31 + 4 * N;
    const int tid = threadIdx.x, sample = blockIdx.x;
    for (int i = tid; i < Words; i += N / 2)
        trlwe[i] = input[sample * Words + i];
    __syncthreads();
    const uint32_t bara[2] = {exponents[2 * sample], exponents[2 * sample + 1]};
    cufhe::AccumulateBlockBinaryPairedNTT<P, Cached, KeySwitchAfter31, KeySwitchAfter31>(
        trlwe, scratch31, 0, bara, key, ntt);
    for (int i = tid; i < Words; i += N / 2)
        out[sample * Words + i] = trlwe[i];
}

bool TestPairedMonomialCache31()
{
    using Target = TFHEpp::lvl1param;
    constexpr int N = Target::n, Cases = 16, Words = (Target::k + 1) * N;
    constexpr size_t KeyWords = 2 * cufhe::BootstrappingTRGSWRows<Target> *
        (Target::k + 1) * N;
    // The fixture reserves four workspace polynomials even without staging.
    constexpr size_t Shared = (4 + Target::k + 1) * N * sizeof(typename Target::T);
    constexpr uint32_t p = cufhe::small_ntt31::P;
    cufhe::CuNTTHandler<N, 32> ntt;
    ntt.Create(); ntt.SetDevicePointers(0);
    cufhe::InitializeXaiNTT(1);
    std::mt19937_64 rng(810031);
    const std::array<uint32_t, 8> powers = {0, 1, N / 2 - 1, N / 2,
                                           N - 1, N, N + 1, 2 * N - 1};
    const std::array<uint32_t, 8> edges = {0, 1, UINT32_MAX, uint32_t{1} << 31,
        (uint32_t{1} << 13) - 1, uint32_t{1} << 13,
        (uint32_t{1} << 14) - 1, uint32_t{1} << 14};
    std::vector<uint32_t> input(Cases * Words), key(KeyWords);
    std::vector<uint32_t> exponents(2 * Cases);
    for (int sample = 0; sample < Cases; ++sample) {
        exponents[2 * sample] = powers[sample % powers.size()];
        exponents[2 * sample + 1] = powers[(sample + 3) % powers.size()];
        for (int i = 0; i < Words; ++i)
            input[sample * Words + i] = sample < 4 ? edges[sample] :
                ((i & 7) == 0 ? edges[(i / 8) % edges.size()] : rng());
    }
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = i % 16 == 0 ? 0 : i % 16 == 1 ? p - 1 :
                 i % 16 == 2 ? p / 2 : rng() % p;
    uint32_t *dinput = nullptr, *dkey = nullptr, *dref = nullptr, *dgot = nullptr;
    uint32_t* dpowers = nullptr;
    bool ok = cudaMalloc(&dinput, input.size() * sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&dkey, key.size() * sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&dref, input.size() * sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&dgot, input.size() * sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc(&dpowers, exponents.size() * sizeof(uint32_t)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(dinput, input.data(), input.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dkey, key.data(), key.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dpowers, exponents.data(), exponents.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaFuncSetAttribute(TestPairedMonomialCache31Kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess &&
        cudaFuncSetAttribute(TestPairedMonomialCache31Kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess &&
        cudaFuncSetAttribute(TestPairedMonomialCache31Kernel<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
    if (ok) {
        TestPairedMonomialCache31Kernel<false><<<Cases, N / 2, Shared>>>(dref, dinput, dkey, dpowers, ntt);
        ok = cudaGetLastError() == cudaSuccess;
        for (bool key_switch_after : {false, true}) {
            if (key_switch_after)
                TestPairedMonomialCache31Kernel<true, true><<<Cases, N / 2, Shared>>>(dgot, dinput, dkey, dpowers, ntt);
            else
                TestPairedMonomialCache31Kernel<true><<<Cases, N / 2, Shared>>>(dgot, dinput, dkey, dpowers, ntt);
            ok = cudaGetLastError() == cudaSuccess && ok;
            std::vector<uint32_t> expected(input.size()), actual(input.size());
            ok = cudaMemcpy(expected.data(), dref, expected.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess &&
                cudaMemcpy(actual.data(), dgot, actual.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) == cudaSuccess && ok;
            if (ok && actual != expected) {
                for (size_t i = 0; i < actual.size(); ++i) if (actual[i] != expected[i]) {
                    std::cerr << "31-bit paired monomial cache mismatch at coefficient " << i << '\n'; break;
                }
                ok = false;
            }
        }
    }
    cudaFree(dinput); cudaFree(dkey); cudaFree(dref); cudaFree(dgot); cudaFree(dpowers);
    cufhe::DeleteXaiNTT(); ntt.Destroy();
    return ok;
}
#endif

#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
template <int Sign0, int Sign1>
__global__ void TestOutputExponentsKernel(uint16_t* seen, uint16_t* out,
    uint16_t* a, uint16_t* b, int alias)
{
    using P = TFHEpp::lvl02param;
    constexpr int Words = P::domainP::n + 3;
    const int start = blockIdx.x * Words;
    auto* workspace = (alias == 1 ? a : alias == 2 ? b : out) + start;
    const auto captured = cufhe::CapturePublicOutputExponents<P, Sign0, Sign1, 512>(
        a + start, b + start);
    __syncthreads();
    cufhe::PublishPublicOutputExponents<P, 512>(workspace, captured);
    __syncthreads();
    for (int i = threadIdx.x; i < Words; i += blockDim.x)
        seen[start + i] = workspace[i];
}

bool TestOutputExponents()
{
    using P = TFHEpp::lvl02param;
    constexpr int Count = P::domainP::n, Words = Count + 3;
    constexpr int Blocks = (65536 + Count - 1) / Count, Total = Blocks * Words;
    constexpr int Shift = 15 - P::targetP::nbit;
    constexpr uint16_t Rounding = uint16_t{1} << (Shift - 1), Sentinel = 0x6dad;
    std::vector<uint16_t> a(Total), b(Total), initial(Total, Sentinel), got(Total);
    uint16_t *da = nullptr, *db = nullptr, *dout = nullptr, *dseen = nullptr;
    bool ok = cudaMalloc(&da, Total * sizeof(*da)) == cudaSuccess &&
        cudaMalloc(&db, Total * sizeof(*db)) == cudaSuccess &&
        cudaMalloc(&dout, Total * sizeof(*dout)) == cudaSuccess &&
        cudaMalloc(&dseen, Total * sizeof(*dseen)) == cudaSuccess;
    std::mt19937 random(1100630);
    for (int pattern = 0; pattern < 4 && ok; ++pattern) {
        for (int block = 0; block < Blocks; ++block)
            for (int i = 0; i < Words; ++i) {
                const int index = block * Words + i, phase = block * Count + i;
                a[index] = uint16_t(phase);
                b[index] = pattern == 0 ? 0 : pattern == 1 ? UINT16_MAX :
                    pattern == 2 ? uint16_t(~phase) : uint16_t(random());
            }
        const auto compare = [&]<int Sign0, int Sign1>() {
            for (int alias = 0; alias < 3 && ok; ++alias) {
                ok = cudaMemcpy(da, a.data(), Total * sizeof(*da), cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(db, b.data(), Total * sizeof(*db), cudaMemcpyHostToDevice) == cudaSuccess &&
                    cudaMemcpy(dout, initial.data(), Total * sizeof(*dout), cudaMemcpyHostToDevice) == cudaSuccess;
                if (!ok) break;
                TestOutputExponentsKernel<Sign0, Sign1><<<Blocks, 512>>>(dseen, dout, da, db, alias);
                ok = cudaGetLastError() == cudaSuccess &&
                    cudaMemcpy(got.data(), dseen, Total * sizeof(*dseen), cudaMemcpyDeviceToHost) == cudaSuccess;
                for (int index = 0; index < Total && ok; ++index) {
                    uint16_t expected = alias == 1 ? a[index] : alias == 2 ? b[index] : Sentinel;
                    if (index % Words < Count) {
                        const uint16_t phase = Sign0 * a[index] + Sign1 * b[index];
                        const uint16_t rounded = phase + Rounding;
                        expected = rounded >> Shift;
                    }
                    if (got[index] != expected) {
                        std::cerr << "Output exponent mismatch at pattern " << pattern
                                  << " alias " << alias << " index " << index << '\n';
                        ok = false;
                    }
                }
            }
        };
        compare.template operator()<-1, -1>(); compare.template operator()<1, 1>();
        compare.template operator()<2, 2>(); compare.template operator()<-2, -2>();
        compare.template operator()<-1, 1>(); compare.template operator()<1, -1>();
    }
    cudaFree(da); cudaFree(db); cudaFree(dout); cudaFree(dseen);
    return ok;
}
#endif

#if defined(CUFHE_HAS_PAIR_NTT64) && defined(USE_BLOCK_BINARY) && !defined(USE_FFT)
template <bool Cached>
__global__ __launch_bounds__(512, 1) void TestFullOutputCacheBRKernel(
    uint64_t* out, uint16_t* a, uint16_t* b, uint16_t* workspace,
    const uint64_t* key, cufhe::CuNTTHandler<TFHEpp::lvl2param::n, 64> ntt,
    int alias)
{
    using P = TFHEpp::lvl02param;
    using Target = P::targetP;
    constexpr int InputWords = P::domainP::n + 1;
    constexpr int OutputWords = (Target::k + 1) * Target::n;
    extern __shared__ __align__(16) unsigned char shared[];
    auto* trlwe = reinterpret_cast<uint64_t*>(shared + cufhe::MEM4HOMGATE<Target>);
    const int input_start = blockIdx.x * InputWords;
    auto* work = (alias == 1 || alias == 3 ? a :
                  alias == 2 || alias == 4 ? b : workspace) + input_start +
                 (alias >= 3 ? 1 : 0);
    cufhe::__BlindRotatePreAdd__<P, -1, -1, P::domainP::μ, true, true, Cached>(
        trlwe, a + input_start, b + input_start, key, ntt, work);
    for (int i = threadIdx.x; i < OutputWords; i += blockDim.x)
        out[blockIdx.x * OutputWords + i] = trlwe[i];
}

template <int Sign0, bool Optimized>
__global__ __launch_bounds__(512, 1) void TestMuxPreAddBRKernel(
    uint64_t* out, const uint16_t* a, const uint16_t* b,
    const uint64_t* key, cufhe::CuNTTHandler<TFHEpp::lvl2param::n, 64> ntt)
{
    using P = TFHEpp::lvl02param;
    using Target = P::targetP;
    extern __shared__ __align__(16) unsigned char shared[];
    auto* trlwe = reinterpret_cast<uint64_t*>(shared + cufhe::MEM4HOMGATE<Target>);
    const int input_start = blockIdx.x * (P::domainP::n + 1);
    cufhe::__BlindRotatePreAdd__<P, Sign0, 1, -P::domainP::μ,
        Optimized, false, false, Optimized>(
        trlwe, a + input_start, b + input_start, key, ntt);
    for (int i = threadIdx.x; i < (Target::k + 1) * Target::n; i += blockDim.x)
        out[blockIdx.x * (Target::k + 1) * Target::n + i] = trlwe[i];
}

bool TestFullOutputCacheBR()
{
    using P = TFHEpp::lvl02param;
    using Target = P::targetP;
    constexpr int Cases = 8, N = Target::n, InputWords = P::domainP::n + 1;
    constexpr int OutputWords = (Target::k + 1) * N;
    constexpr size_t KeyWords = size_t(P::domainP::n) *
        cufhe::BootstrappingTRGSWRows<Target> * (Target::k + 1) * N;
    constexpr size_t Shared = cufhe::MEM4HOMGATE_DYN<Target>;
    constexpr uint64_t prime = cufhe::small_ntt::P;
    cufhe::CuNTTHandler<N, 64> ntt;
    ntt.Create(); ntt.SetDevicePointers(0);
    cufhe::InitializeXaiNTT_lvl02(1);
    std::mt19937_64 random(1103000);
    const std::array<uint16_t, 8> edges = {0, 1, UINT16_MAX, 0x8000, 15, 16, 31, 32};
    std::vector<uint16_t> a(Cases * InputWords), b(a.size());
    for (int sample = 0; sample < Cases; ++sample)
        for (int i = 0; i < InputWords; ++i) {
            a[sample * InputWords + i] = sample < 4 ? edges[(sample + i) % edges.size()] : uint16_t(random());
            b[sample * InputWords + i] = sample < 4 ? edges[(sample * 2 + i / 3) % edges.size()] : uint16_t(random());
        }
    std::vector<uint64_t> key(KeyWords), expected(Cases * OutputWords), got(expected.size());
    for (size_t i = 0; i < key.size(); ++i)
        key[i] = i % 32 == 0 ? 0 : i % 32 == 1 ? prime - 1 : random() % prime;
    uint16_t *da = nullptr, *db = nullptr, *dw = nullptr;
    uint64_t *dk = nullptr, *dr = nullptr, *dg = nullptr;
    bool ok = cudaMalloc(&da, a.size() * sizeof(*da)) == cudaSuccess &&
        cudaMalloc(&db, b.size() * sizeof(*db)) == cudaSuccess &&
        cudaMalloc(&dw, a.size() * sizeof(*dw)) == cudaSuccess &&
        cudaMalloc(&dk, key.size() * sizeof(*dk)) == cudaSuccess &&
        cudaMalloc(&dr, expected.size() * sizeof(*dr)) == cudaSuccess &&
        cudaMalloc(&dg, got.size() * sizeof(*dg)) == cudaSuccess;
    if (ok) ok = cudaMemcpy(da, a.data(), a.size() * sizeof(*da), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(db, b.data(), b.size() * sizeof(*db), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(dk, key.data(), key.size() * sizeof(*dk), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaFuncSetAttribute(TestFullOutputCacheBRKernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess &&
        cudaFuncSetAttribute(TestFullOutputCacheBRKernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
    if (ok) {
        TestFullOutputCacheBRKernel<false><<<Cases, 512, Shared>>>(dr, da, db, dw, dk, ntt, 0);
        ok = cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(expected.data(), dr, expected.size() * sizeof(*dr), cudaMemcpyDeviceToHost) == cudaSuccess;
        for (int alias = 0; alias < 5 && ok; ++alias) {
            ok = cudaMemcpy(da, a.data(), a.size() * sizeof(*da), cudaMemcpyHostToDevice) == cudaSuccess &&
                cudaMemcpy(db, b.data(), b.size() * sizeof(*db), cudaMemcpyHostToDevice) == cudaSuccess;
            if (!ok) break;
            TestFullOutputCacheBRKernel<true><<<Cases, 512, Shared>>>(dg, da, db, dw, dk, ntt, alias);
            ok = cudaGetLastError() == cudaSuccess &&
                cudaMemcpy(got.data(), dg, got.size() * sizeof(*dg), cudaMemcpyDeviceToHost) == cudaSuccess &&
                got == expected;
            if (!ok) std::cerr << "Full output-cache BR mismatch, alias " << alias << '\n';
        }
    }
    const auto compare_mux = [&]<int Sign0>() {
        if (!ok) return;
        ok = cudaMemcpy(da, a.data(), a.size() * sizeof(*da), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(db, b.data(), b.size() * sizeof(*db), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaFuncSetAttribute(TestMuxPreAddBRKernel<Sign0, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess &&
            cudaFuncSetAttribute(TestMuxPreAddBRKernel<Sign0, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared) == cudaSuccess;
        if (!ok) return;
        TestMuxPreAddBRKernel<Sign0, false><<<Cases, 512, Shared>>>(dr, da, db, dk, ntt);
        TestMuxPreAddBRKernel<Sign0, true><<<Cases, 512, Shared>>>(dg, da, db, dk, ntt);
        ok = cudaGetLastError() == cudaSuccess &&
            cudaMemcpy(expected.data(), dr, expected.size() * sizeof(*dr), cudaMemcpyDeviceToHost) == cudaSuccess &&
            cudaMemcpy(got.data(), dg, got.size() * sizeof(*dg), cudaMemcpyDeviceToHost) == cudaSuccess &&
            got == expected;
        if (!ok) std::cerr << "MUX pre-add BR mismatch, selector sign " << Sign0 << '\n';
    };
    compare_mux.template operator()<1>();
    compare_mux.template operator()<-1>();
    cudaFree(da); cudaFree(db); cudaFree(dw); cudaFree(dk); cudaFree(dr); cudaFree(dg);
    cufhe::DeleteXaiNTT_lvl02(); ntt.Destroy();
    return ok;
}
#endif

}  // namespace

int main()
{
    bool torus64 = true;
#if defined(CUFHE_HAS_PAIR_NTT64)
    torus64 = TestGoldilocksTorus();
    std::cout << "64-bit host/device exact torus conversion (500200 cases): "
              << (torus64 ? "PASS" : "FAIL") << '\n';
#endif
    bool add64 = true;
#if defined(CUFHE_HAS_PAIR_NTT64)
    add64 = TestGoldilocksAdd();
    std::cout << "64-bit host/device modular addition (300256 cases): "
              << (add64 ? "PASS" : "FAIL") << '\n';
#endif
    bool sub64 = true;
#if defined(CUFHE_HAS_PAIR_NTT64)
    sub64 = TestGoldilocksSub();
    std::cout << "64-bit host/device modular subtraction (200144 cases): "
              << (sub64 ? "PASS" : "FAIL") << '\n';
#endif
    const bool inverse64_scale = TestGoldilocksScale();
    std::cout << "64-bit host/device fixed inverse scaling: "
              << (inverse64_scale ? "PASS" : "FAIL") << '\n';

    const bool correction = TestCorrection<1, 0>() && TestCorrection<1, 1>() && TestCorrection<-1, -1>() &&
                            TestCorrection<2, 2>() && TestCorrection<-2, -2>() &&
                            TestCorrection<1, -1>() && TestCorrection<-1, 1>();
    std::cout << "Block-binary rounding correction: " << (correction ? "PASS" : "FAIL") << '\n';
    const bool lazy31 = TestLazyRange();
    std::cout << "31-bit lazy butterfly range/residues (100245 cases): "
              << (lazy31 ? "PASS" : "FAIL") << '\n';
    const bool inverse31 = TestInverseButterflies();
    std::cout << "31-bit inverse butterflies (100125 cases): "
              << (inverse31 ? "PASS" : "FAIL") << '\n';
    const bool reduce31 = TestReduce31();
    std::cout << "31-bit full uint64 reduction: " << (reduce31 ? "PASS" : "FAIL") << '\n';
    const bool reduce64 = TestReduce64();
    std::cout << "64-bit host/device full uint128 reduction (1000400 cases): "
              << (reduce64 ? "PASS" : "FAIL") << '\n';
    const bool madd2 = TestMadd2();
    std::cout << "64-bit host/device madd2 (266384 cases): " << (madd2 ? "PASS" : "FAIL") << '\n';
    const bool madd4 = TestMadd4();
    std::cout << "31-bit host/device madd4: " << (madd4 ? "PASS" : "FAIL") << '\n';
    const bool host31 = TestHostMadd<TFHEpp::lvl1param::n, 32>();
    const bool host64 = TestHostMadd<TFHEpp::lvl2param::n, 64>();
    const bool device31 = TestDeviceMadd<TFHEpp::lvl1param::n, 32>();
    const bool device64 = TestDeviceMadd<TFHEpp::lvl2param::n, 64>();
    const bool host31_madd3 = TestHostMadd3<TFHEpp::lvl1param::n, 32>();
    const bool host64_madd3 = TestHostMadd3<TFHEpp::lvl2param::n, 64>();
    const bool device31_madd3 = TestDeviceMadd3<TFHEpp::lvl1param::n, 32>();
    const bool device64_madd3 = TestDeviceMadd3<TFHEpp::lvl2param::n, 64>();

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
    const bool transform31 = TestTransforms<TFHEpp::lvl1param::n, 32>();
    const bool transform64 = TestTransforms<TFHEpp::lvl2param::n, 64>();
    std::cout << "31-bit roundtrip / negacyclic square: "
              << (transform31 ? "PASS" : "FAIL") << '\n';
    std::cout << "64-bit roundtrip / negacyclic square: "
              << (transform64 ? "PASS" : "FAIL") << '\n';
    const bool paired31 = TestTransforms<TFHEpp::lvl1param::n, 32, true>();
    std::cout << "31-bit paired roundtrip / negacyclic square: "
              << (paired31 ? "PASS" : "FAIL") << '\n';
    bool paired64=true, compare64=true, madd4wide=true, scalar_cache=true, monomial_symmetry=true;
#if defined(CUFHE_HAS_PAIR_NTT64)
    const auto test_pair64 = [&](auto* param) {
        using Param = std::remove_pointer_t<decltype(param)>;
        if constexpr (Param::nbit >= 8 && (Param::nbit - 6) % 2 == 0) {
            paired64 = TestTransforms<Param::n, 64, true>();
            compare64 = TestPair64Compare<Param>();
        }
    };
    test_pair64(static_cast<TFHEpp::lvl2param*>(nullptr));
    madd4wide=TestMadd4Wide();
    std::cout<<"64-bit paired roundtrip / convolution: "<<(paired64?"PASS":"FAIL")<<'\n';
    std::cout<<"64-bit paired / scalar forward and inverse: "<<(compare64?"PASS":"FAIL")<<'\n';
    std::cout<<"64-bit host/device madd4 (396611 cases): "<<(madd4wide?"PASS":"FAIL")<<'\n';
#endif
#if defined(CUFHE_HAS_PAIR_NTT64) && defined(USE_BLOCK_BINARY) && !defined(USE_FFT)
    scalar_cache = TestScalarMonomialCache(monomial_symmetry);
    bool paired_symmetry = true;
    const bool paired_cache = TestScalarMonomialCache<true>(paired_symmetry);
    scalar_cache &= paired_cache && paired_symmetry;
    std::cout << "64-bit raw paired / uncached scalar external product (49152 coefficients): "
              << (paired_cache && paired_symmetry ? "PASS" : "FAIL") << '\n';
    std::cout << "64-bit complete monomial adjacent-pair symmetry (1048576 pairs): "
              << (monomial_symmetry ? "PASS" : "FAIL") << '\n';
    std::cout << "64-bit scalar cached / uncached external product (49152 coefficients): "
              << (scalar_cache ? "PASS" : "FAIL") << '\n';
#endif
    bool cache31 = true;
#if defined(USE_BLOCK_BINARY) && defined(USE_CUDA_PAIRED_NTT) && !defined(USE_FFT)
    cache31 = TestPairedMonomialCache31();
    std::cout << "31-bit register schedules / staged external product (65536 coefficients): "
              << (cache31 ? "PASS" : "FAIL") << '\n';
#endif
    bool output_exponents = true;
#if defined(USE_BLOCK_BINARY) && !defined(USE_FFT) && !defined(CUFHE_USE_HIP)
    output_exponents = TestOutputExponents();
    std::cout << "Output exponent cache (4762800 exponents / body-padding / both aliases): "
              << (output_exponents ? "PASS" : "FAIL") << '\n';
#endif
    bool full_output_br = true;
#if defined(CUFHE_HAS_PAIR_NTT64) && defined(USE_BLOCK_BINARY) && !defined(USE_FFT)
    full_output_br = TestFullOutputCacheBR();
    std::cout << "Full BR original / output-cache / MUX signs (172032 coefficients / exact-forward aliases): "
              << (full_output_br ? "PASS" : "FAIL") << '\n';
#endif
    return full_output_br && output_exponents && cache31 && scalar_cache && monomial_symmetry && add64 && torus64 && sub64 && inverse64_scale && paired64 && compare64 && madd4wide && madd2 && reduce64 && inverse31 && lazy31 && reduce31 && correction && madd4 && paired31 && transform31 && transform64 && host31 && host64 && device31 && device64 && host31_madd3 &&
                   host64_madd3 && device31_madd3 && device64_madd3
               ? 0
               : 1;
}
