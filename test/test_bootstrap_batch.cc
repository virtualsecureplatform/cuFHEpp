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

template <class P, class brP, class iksP>
bool TestBatch(const TFHEpp::SecretKey& sk, uint32_t count = 17)
{
    using T = typename P::T;
    constexpr size_t elements = P::k * P::n + 1;
    constexpr size_t stride_a = elements + 3;
    constexpr size_t stride_b = elements + 7;
    constexpr size_t stride_out = elements + 11;
    cufhe::NandBootstrapBatch<iksP, brP, P::μ>(
        nullptr, 0, nullptr, 0, nullptr, 0, 0, nullptr, 0);
    bool rejected = false;
    try {
        cufhe::NandBootstrapBatch<iksP, brP, P::μ>(
            nullptr, 0, nullptr, 0, nullptr, 0, 1, nullptr, 0);
    }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) return false;

    TFHEpp::EvalKey ek(sk);
    ek.template emplacebk<brP>(sk);
#ifdef USE_SUBSET_KEY
    if constexpr (sizeof(T) == 4)
        ek.template emplacesubiksk<iksP>(sk);
    else
#endif
        ek.template emplaceiksk<iksP>(sk);
    if constexpr (sizeof(T) == 4) cufhe::Initialize(ek);
    else cufhe::Initialize_lvl02(ek, sk);

    std::vector<T> a(count * stride_a, 7), b(count * stride_b, 8);
    std::vector<T> output(count * stride_out, 9), reference(output);
    for (uint32_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> sample;
        TFHEpp::tlweSymEncrypt<P>(sample, (i & 1) ? P::μ : -P::μ, sk.key.get<P>());
        std::copy(sample.begin(), sample.end(), a.begin() + i * stride_a);
        TFHEpp::tlweSymEncrypt<P>(sample, (i & 2) ? P::μ : -P::μ, sk.key.get<P>());
        std::copy(sample.begin(), sample.end(), b.begin() + i * stride_b);
    }
    T *da = nullptr, *db = nullptr, *dout = nullptr, *dref = nullptr;
    Check(cudaMalloc(&da, a.size() * sizeof(T)));
    Check(cudaMalloc(&db, b.size() * sizeof(T)));
    Check(cudaMalloc(&dout, output.size() * sizeof(T)));
    Check(cudaMalloc(&dref, reference.size() * sizeof(T)));
    Check(cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dout, output.data(), output.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dref, reference.data(), reference.size() * sizeof(T), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cufhe::NandBootstrapBatch<iksP, brP, P::μ>(
        dout, stride_out, da, stride_a, db, stride_b, count, stream, 0);
    for (uint32_t i = 0; i < count; ++i)
        cufhe::NandBootstrap<iksP, brP, P::μ>(
            dref + i * stride_out, da + i * stride_a, db + i * stride_b, stream, 0);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(output.data(), dout, output.size() * sizeof(T), cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(reference.data(), dref, reference.size() * sizeof(T), cudaMemcpyDeviceToHost));
    bool passed = true;
#ifndef USE_FFT
    passed = output == reference;  // Includes untouched padding and exact NTT residues.
#endif
    for (uint32_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> sample;
        std::copy_n(output.begin() + i * stride_out, elements, sample.begin());
        passed &= TFHEpp::tlweSymDecrypt<P>(sample, sk.key.get<P>()) ==
                  !((i & 1) && (i & 2));
        for (size_t j = elements; j < stride_out; ++j)
            passed &= output[i * stride_out + j] == 9;
    }

    // Verify aliasing, then a chain that bootstraps its own previous outputs.
    cufhe::NandBootstrapBatch<iksP, brP, P::μ>(
        da, stride_a, da, stride_a, db, stride_b, count, stream, 0);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(a.data(), da, a.size() * sizeof(T), cudaMemcpyDeviceToHost));
#ifndef USE_FFT
    for (uint32_t i = 0; i < count; ++i)
        passed &= std::equal(a.begin() + i * stride_a,
                            a.begin() + i * stride_a + elements,
                            reference.begin() + i * stride_out);
#endif
    for (int depth = 0; depth < 8; ++depth) {
        cufhe::NandBootstrapBatch<iksP, brP, P::μ>(
            da, stride_a, da, stride_a, db, stride_b, count, stream, 0);
        // The small case also compares the scalar schedule in place at every
        // depth. In NTT builds this checks exact coefficients across schedules.
        if (count == 17)
            for (uint32_t i = 0; i < count; ++i)
                cufhe::NandBootstrap<iksP, brP, P::μ>(
                    dref + i * stride_out, dref + i * stride_out,
                    db + i * stride_b, stream, 0);
    }
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(a.data(), da, a.size() * sizeof(T), cudaMemcpyDeviceToHost));
    if (count == 17) {
        Check(cudaMemcpy(reference.data(), dref, reference.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
#ifndef USE_FFT
        for (uint32_t i = 0; i < count; ++i)
            passed &= std::equal(a.begin() + i * stride_a,
                                a.begin() + i * stride_a + elements,
                                reference.begin() + i * stride_out);
#endif
        for (uint32_t i = 0; i < count; ++i) {
            TFHEpp::TLWE<P> sample;
            std::copy_n(reference.begin() + i * stride_out, elements, sample.begin());
            passed &= TFHEpp::tlweSymDecrypt<P>(sample, sk.key.get<P>()) ==
                      !((i & 1) && (i & 2));
        }
    }
    for (uint32_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> sample;
        std::copy_n(a.begin() + i * stride_a, elements, sample.begin());
        passed &= TFHEpp::tlweSymDecrypt<P>(sample, sk.key.get<P>()) ==
                  !((i & 1) && (i & 2));
        for (size_t j = elements; j < stride_a; ++j)
            passed &= a[i * stride_a + j] == 7;
    }
    Check(cudaStreamDestroy(stream));
    Check(cudaFree(da)); Check(cudaFree(db));
    Check(cudaFree(dout)); Check(cudaFree(dref));
    if constexpr (sizeof(T) == 4) cufhe::CleanUp();
    else cufhe::CleanUp_lvl02();
    std::cout << (sizeof(T) == 4 ? "lvl1" : "lvl02")
              << " count=" << count << " batch strides / scalar equivalence / in-place chain: "
              << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}
}

int main()
{
    try {
        Check(cudaSetDevice(0));
        cufhe::SetGPUNum(1);
        TFHEpp::SecretKey sk;
        const bool lvl1 = TestBatch<TFHEpp::lvl1param, TFHEpp::lvl01param,
                                   TFHEpp::lvl10param>(sk);
        const bool lvl02 = TestBatch<TFHEpp::lvl2param, TFHEpp::lvl02param,
                                    TFHEpp::lvl20param>(sk);
        cudaDeviceProp properties;
        Check(cudaGetDeviceProperties(&properties, 0));
        const int multiprocessors = properties.multiProcessorCount;
        const bool lvl02_large = TestBatch<TFHEpp::lvl2param, TFHEpp::lvl02param,
            TFHEpp::lvl20param>(sk, static_cast<uint32_t>(multiprocessors) + 1);
        return lvl1 && lvl02 && lvl02_large ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
