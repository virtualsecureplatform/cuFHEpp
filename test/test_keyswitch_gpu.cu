#include <include/keyswitch_gpu.cuh>

#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#ifdef CUFHE_TEST_COMPACT_LVL20
using P = TFHEpp::lvl20param;
using Row = uint16_t;
#else
using P = TFHEpp::lvl10param;
using Row = uint32_t;
#endif
using Input = P::domainP::T;
using Output = P::targetP::T;
constexpr int input_size = P::domainP::k * P::domainP::n + 1;
constexpr int output_size = P::targetP::k * P::targetP::n + 1;
constexpr int switched_count = input_size - 1 -
    (cufhe::use_subset_key_switch<P> ? output_size - 1 : 0);
constexpr int key_size = switched_count * P::t *
    (1 << (P::basebit - 1)) * output_size;
constexpr int batch_size = 19;

void Check(cudaError_t error)
{
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
        std::exit(2);
    }
}

template <bool prepared, int a_sign, int b_sign, auto offset>
__global__ void TestKeySwitch(Output* out, const Input* a, const Input* b,
                              const Output* key)
{
    extern __shared__ Row rows[];
    const size_t sample = blockIdx.x;
    if constexpr (prepared) {
#ifdef CUFHE_TEST_COMPACT_LVL20
        cufhe::IdentityKeySwitchPreAddPrepared16<P, a_sign, b_sign, offset>(
#else
        cufhe::IdentityKeySwitchPreAddPrepared<P, a_sign, b_sign, offset>(
#endif
            out + sample * output_size, a + sample * input_size,
            b + sample * input_size, key, rows);
    }
    else
        cufhe::IdentityKeySwitchPreAdd<P, a_sign, b_sign, offset>(
            out + sample * output_size, a + sample * input_size,
            b + sample * input_size, key);
}

template <int a_sign, int b_sign, auto offset>
void Compare(Output* original, Output* prepared, const Input* a,
             const Input* b, const Output* key, int threads)
{
    TestKeySwitch<false, a_sign, b_sign, offset><<<batch_size, threads>>>(
        original, a, b, key);
    Check(cudaGetLastError());
    TestKeySwitch<true, a_sign, b_sign, offset>
        <<<batch_size, threads, switched_count * P::t * sizeof(Row)>>>(
            prepared, a, b, key);
    Check(cudaGetLastError());
    std::vector<Output> expected(batch_size * output_size);
    std::vector<Output> actual(expected.size());
    Check(cudaMemcpy(expected.data(), original, expected.size() * sizeof(Output),
                     cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(), prepared, actual.size() * sizeof(Output),
                     cudaMemcpyDeviceToHost));
    if (expected != actual) {
        std::fprintf(stderr, "prepared key-switch bit mismatch (%d threads)\n",
                     threads);
        std::exit(1);
    }
}

// The prepared path deliberately overwrites its own shared TRLWE input. The
// reference extracts to disjoint storage and runs the original key switch.
template <bool prepared, int Threads, bool ExtractFromTRLWE>
__global__ void TestExtractKeySwitch(Output* out, const Input* input,
                                     const Output* key)
{
    extern __shared__ uint64_t scratch[];
    Input* working = reinterpret_cast<Input*>(scratch);
    const Input* trlwe = input + blockIdx.x * input_size;
    if constexpr (prepared) {
        for (int i = threadIdx.x; i < input_size; i += Threads)
            working[i] = trlwe[i];
        __syncthreads();
#ifdef CUFHE_TEST_COMPACT_LVL20
        cufhe::KeySwitchPrepared16<P, Threads, ExtractFromTRLWE>(
#else
        cufhe::KeySwitchPrepared32<P, Threads, ExtractFromTRLWE>(
#endif
            out + blockIdx.x * output_size, working, key,
            reinterpret_cast<Row*>(working));
    } else {
        for (int i = threadIdx.x; i < input_size; i += Threads) {
            if (!ExtractFromTRLWE || i == input_size - 1) working[i] = trlwe[i];
            else {
                const int component = i / P::domainP::n;
                const int coeff = i % P::domainP::n;
                working[i] = coeff == 0 ? trlwe[i] :
                    -trlwe[(component + 1) * P::domainP::n - coeff];
            }
        }
        __syncthreads();
        cufhe::KeySwitchFromTLWE<P>(out + blockIdx.x * output_size, working, key);
    }
}

template <int Threads, bool ExtractFromTRLWE>
void CompareExtract(Output* original, Output* prepared, const Input* input,
                    const Output* key)
{
    constexpr size_t scratch = (switched_count * P::t + 1) * sizeof(Row);
    static_assert(scratch >= input_size * sizeof(Input));
    TestExtractKeySwitch<false, Threads, ExtractFromTRLWE><<<batch_size, Threads, scratch>>>(
        original, input, key);
    Check(cudaGetLastError());
    TestExtractKeySwitch<true, Threads, ExtractFromTRLWE><<<batch_size, Threads, scratch>>>(
        prepared, input, key);
    Check(cudaGetLastError());
    std::vector<Output> expected(batch_size * output_size), actual(expected.size());
    Check(cudaMemcpy(expected.data(), original, expected.size() * sizeof(Output),
                     cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(actual.data(), prepared, actual.size() * sizeof(Output),
                     cudaMemcpyDeviceToHost));
    if (expected != actual) {
        std::fprintf(stderr, "extraction / prepared key-switch mismatch (%d threads)\n", Threads);
        std::exit(1);
    }
}

int main()
{
    std::mt19937_64 random(1234);
    std::vector<Input> a(batch_size * input_size), b(a.size());
    std::vector<Output> key(key_size);
    for (auto& value : a) value = random();
    for (auto& value : b) value = random();
    for (auto& value : key) value = random();
    // Four boundary patterns, followed by independent uniform random samples.
    constexpr int bits = std::numeric_limits<Input>::digits;
    constexpr Input high_bit = Input{1} << (bits - 1);
    for (int i = 0; i < input_size; ++i) {
        a[i] = b[i] = 0;
        a[input_size + i] = b[input_size + i] = std::numeric_limits<Input>::max();
        a[2 * input_size + i] = high_bit;
        b[2 * input_size + i] = high_bit - 1;
        a[3 * input_size + i] = Input{1} << (i % bits);
        b[3 * input_size + i] = ~a[3 * input_size + i];
    }
    Input *device_a, *device_b;
    Output *device_key, *original, *prepared;
    Check(cudaMalloc(&device_a, a.size() * sizeof(Input)));
    Check(cudaMalloc(&device_b, b.size() * sizeof(Input)));
    Check(cudaMalloc(&device_key, key.size() * sizeof(Output)));
    Check(cudaMalloc(&original, batch_size * output_size * sizeof(Output)));
    Check(cudaMalloc(&prepared, batch_size * output_size * sizeof(Output)));
    Check(cudaMemcpy(device_a, a.data(), a.size() * sizeof(Input),
                     cudaMemcpyHostToDevice));
    Check(cudaMemcpy(device_b, b.data(), b.size() * sizeof(Input),
                     cudaMemcpyHostToDevice));
    Check(cudaMemcpy(device_key, key.data(), key.size() * sizeof(Output),
                     cudaMemcpyHostToDevice));
    for (int threads : {256, 512}) {
        Compare<-1, -1, P::domainP::μ>(original, prepared, device_a, device_b,
                                     device_key, threads);
        Compare<1, 1, -P::domainP::μ>(original, prepared, device_a, device_b,
                                     device_key, threads);
        Compare<-1, 1, 0>(original, prepared, device_a, device_b, device_key,
                          threads);
        Compare<2, 2, -2 * P::domainP::μ>(original, prepared, device_a, device_b,
                                         device_key, threads);
    }
    std::puts("prepared key-switch bit equivalence: PASS "
              "(4 sign/offset cases, 19 boundary/random samples, 256/512 threads)");
    CompareExtract<256, true>(original, prepared, device_a, device_key);
    CompareExtract<512, true>(original, prepared, device_a, device_key);
    CompareExtract<256, false>(original, prepared, device_a, device_key);
    CompareExtract<512, false>(original, prepared, device_a, device_key);
    std::puts("sample extraction / direct TLWE prepared key-switch bit equivalence: PASS "
              "(19 boundary/random samples, 256/512 threads, aliased scratch)");
    Check(cudaFree(device_a));
    Check(cudaFree(device_b));
    Check(cudaFree(device_key));
    Check(cudaFree(original));
    Check(cudaFree(prepared));
}
