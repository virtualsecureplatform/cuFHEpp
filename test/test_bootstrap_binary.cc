#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class P, class IKS, class BR>
bool TestBinary(const TFHEpp::SecretKey& sk)
{
    TFHEpp::EvalKey ek(sk);
    ek.template emplacebk<BR>(sk);
#ifdef USE_SUBSET_KEY
    if constexpr (sizeof(typename P::T) == 4) ek.template emplacesubiksk<IKS>(sk);
    else
#endif
        ek.template emplaceiksk<IKS>(sk);
    if constexpr (sizeof(typename P::T) == 4) cufhe::Initialize(ek);
    else cufhe::Initialize_lvl02(ek, sk);

    using T = typename P::T;
    using Launch = void (*)(T*, const T*, const T*, cudaStream_t, int);
    struct Gate { const char* name; int ca, cb, offset; unsigned truth; Launch launch; };
#define G(Name, ca, cb, offset, truth) \
    {#Name, ca, cb, offset, truth, &cufhe::Name##Bootstrap<IKS, BR, P::μ>}
    const Gate gates[] = {
        G(Nand, -1, -1, 1, 7), G(And, 1, 1, -1, 8),
        G(Nor, -1, -1, -1, 1), G(Or, 1, 1, 1, 14),
        G(Xor, 2, 2, 2, 6), G(Xnor, -2, -2, -2, 9),
        G(AndNY, -1, 1, -1, 4), G(AndYN, 1, -1, -1, 2),
        G(OrNY, -1, 1, 1, 13), G(OrYN, 1, -1, 1, 11)};
#undef G
    constexpr int count = 16, elements = P::k * P::n + 1;
    cudaDeviceProp prop;
    Check(cudaGetDeviceProperties(&prop, 0));
    // Keep reference chunks below the adaptive staging threshold. On the
    // RTX 4090 this selects the original scalar 64-bit transform schedule.
    const int chunk = std::max(1, std::min(count, prop.multiProcessorCount - 1));
    cufhe::Stream stream;
    stream.Create();
    std::vector<cufhe::Ctxt<P>> a(count), b(count), out(count);
    std::vector<TFHEpp::TLWE<P>> saved_a(count);
    for (int i = 0; i < count; ++i) {
        TFHEpp::tlweSymEncrypt<P>(a[i].tlwehost, (i & 1) ? P::μ : -P::μ, sk.key.template get<P>());
        TFHEpp::tlweSymEncrypt<P>(b[i].tlwehost, (i & 2) ? P::μ : -P::μ, sk.key.template get<P>());
        saved_a[i] = a[i].tlwehost;
        cufhe::CtxtCopyH2D(b[i], stream);
    }
    std::vector<T> linear(count * elements), reference(linear.size());
    T *dlinear, *dzero, *dref;
    Check(cudaMalloc(&dlinear, linear.size() * sizeof(T)));
    Check(cudaMalloc(&dzero, linear.size() * sizeof(T)));
    Check(cudaMalloc(&dref, linear.size() * sizeof(T)));
    Check(cudaMemsetAsync(dzero, 0, linear.size() * sizeof(T), stream.st()));
    bool all_passed = true;
    for (const auto& gate : gates) {
        // NAND(-ca*a-cb*b + (mu-offset) in its body, zero) has exactly
        // the gate's original pre-add ciphertext, including its body offset.
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < elements; ++j)
                linear[i * elements + j] = T(-gate.ca) * saved_a[i][j] +
                                          T(-gate.cb) * b[i].tlwehost[j];
            linear[i * elements + elements - 1] += P::μ - T(gate.offset) * P::μ;
        }
        Check(cudaMemcpyAsync(dlinear, linear.data(), linear.size() * sizeof(T),
                              cudaMemcpyHostToDevice, stream.st()));
        for (int first = 0; first < count; first += chunk) {
            const int size = std::min(chunk, count - first);
            cufhe::NandBootstrapBatch<IKS, BR, P::μ>(
                dref + first * elements, elements, dlinear + first * elements,
                elements, dzero + first * elements, elements, size, stream.st(), 0);
        }
        Check(cudaMemcpyAsync(reference.data(), dref, reference.size() * sizeof(T),
                              cudaMemcpyDeviceToHost, stream.st()));
        Check(cudaStreamSynchronize(stream.st()));
        bool passed = true;
        for (bool alias : {false, true}) {
            for (int i = 0; i < count; ++i) {
                a[i].tlwehost = saved_a[i];
                cufhe::CtxtCopyH2D(a[i], stream);
                auto& result = alias ? a[i] : out[i];
                gate.launch(result.tlwedevices[0], a[i].tlwedevices[0],
                            b[i].tlwedevices[0], stream.st(), 0);
                cufhe::CtxtCopyD2H(result, stream);
                Check(cudaStreamSynchronize(stream.st()));
#ifndef USE_FFT
                passed &= std::equal(result.tlwehost.begin(), result.tlwehost.end(),
                                     reference.begin() + i * elements);
#endif
                passed &= TFHEpp::tlweSymDecrypt<P>(result.tlwehost, sk.key.template get<P>()) ==
                          bool((gate.truth >> (i & 3)) & 1U);
            }
        }
        all_passed &= passed;
        std::cout << (sizeof(T) == 4 ? "lvl1" : "lvl02") << " raw " << gate.name
                  << " 16 masks / 4 truth cases / small-batch reference / output=input0: "
                  << (passed ? "PASS" : "FAIL") << std::endl;
    }
    Check(cudaFree(dlinear)); Check(cudaFree(dzero)); Check(cudaFree(dref));
    stream.Destroy();
    if constexpr (sizeof(T) == 4) cufhe::CleanUp();
    else cufhe::CleanUp_lvl02();
    return all_passed;
}
}

int main()
{
    try {
        Check(cudaSetDevice(0));
        cufhe::SetGPUNum(1);
        TFHEpp::SecretKey sk;
        const bool lvl1 = TestBinary<TFHEpp::lvl1param, TFHEpp::lvl10param, TFHEpp::lvl01param>(sk);
        const bool lvl02 = TestBinary<TFHEpp::lvl2param, TFHEpp::lvl20param, TFHEpp::lvl02param>(sk);
        return lvl1 && lvl02 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
