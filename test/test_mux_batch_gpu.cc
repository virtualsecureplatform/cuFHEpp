#include <include/cufhe_gpu.cuh>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

int main()
{
    using P = TFHEpp::lvl0param;
    using BrP = TFHEpp::lvl01param;
    using IksP = TFHEpp::lvl10param;
    using T = P::T;
    constexpr size_t stride = P::n + 1;
    // Seventeen exercises two complete truth tables and a partial tail.
    constexpr size_t count = 17;

    TFHEpp::SecretKey secret;
    TFHEpp::EvalKey eval(secret);
    eval.emplacebk<BrP>(secret);
#ifdef USE_SUBSET_KEY
    eval.emplacesubiksk<IksP>(secret);
#else
    eval.emplaceiksk<IksP>(secret);
#endif
    cufhe::Initialize(eval);
    cudaSetDevice(0);
#ifdef USE_BLOCK_BINARY
    const auto lvl0key = secret.key.getSubset<P>();
#else
    const auto lvl0key = secret.key.get<P>();
#endif

    std::vector<T> selector(count * stride);
    std::vector<T> whenTrue(count * stride);
    std::vector<T> whenFalse(count * stride);
    std::vector<T> output(count * stride);
    for (size_t i = 0; i < count; ++i) {
        const bool s = (i & 4) != 0;
        const bool t = (i & 2) != 0;
        const bool f = (i & 1) != 0;
        TFHEpp::TLWE<P> ciphertext;
        const auto encrypt = [&](bool value, T* destination) {
            TFHEpp::tlweSymEncrypt<P>(ciphertext,
                                      value ? P::μ : -P::μ,
                                      lvl0key);
            std::copy(ciphertext.begin(), ciphertext.end(), destination);
        };
        encrypt(s, selector.data() + i * stride);
        encrypt(t, whenTrue.data() + i * stride);
        encrypt(f, whenFalse.data() + i * stride);
    }

    T *selectorDevice = nullptr, *trueDevice = nullptr, *falseDevice = nullptr,
      *outputDevice = nullptr;
    const size_t bytes = count * stride * sizeof(T);
    cudaMalloc(reinterpret_cast<void**>(&selectorDevice), bytes);
    cudaMalloc(reinterpret_cast<void**>(&trueDevice), bytes);
    cudaMalloc(reinterpret_cast<void**>(&falseDevice), bytes);
    cudaMalloc(reinterpret_cast<void**>(&outputDevice), bytes);
    cudaMemcpy(selectorDevice, selector.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(trueDevice, whenTrue.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(falseDevice, whenFalse.data(), bytes, cudaMemcpyHostToDevice);

    cufhe::MuxBootstrapBatch<BrP, TFHEpp::lvl1param::μ, IksP>(
        outputDevice, selectorDevice, trueDevice, falseDevice, count, nullptr,
        0);
    cudaMemcpy(output.data(), outputDevice, bytes, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> ciphertext;
        std::copy_n(output.data() + i * stride, stride, ciphertext.begin());
        const bool actual = TFHEpp::tlweSymDecrypt<P>(ciphertext, lvl0key);
        const bool expected = (i & 4) != 0 ? (i & 2) != 0 : (i & 1) != 0;
        correct &= actual == expected;
    }

    cudaFree(outputDevice);
    cudaFree(falseDevice);
    cudaFree(trueDevice);
    cudaFree(selectorDevice);
    cufhe::CleanUp();
    std::cout << (correct ? "PASS" : "FAIL") << std::endl;
    return correct ? 0 : 1;
}
