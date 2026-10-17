#include <algorithm>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void Check(cudaError_t e)
{
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

template <class BR, class IKS>
bool Test(const TFHEpp::SecretKey& sk, uint32_t count)
{
    using P = typename BR::domainP;
    using T = typename P::T;
    constexpr size_t n = P::k * P::n + 1;
    constexpr size_t sa = n + 3, sb = n + 7, so = n + 11;
    constexpr auto batch = &cufhe::NandBootstrapBatch<BR, BR::targetP::μ, IKS>;
    constexpr auto scalar = &cufhe::NandBootstrap<BR, BR::targetP::μ, IKS>;
    batch(nullptr, 0, nullptr, 0, nullptr, 0, 0, nullptr, 0);
    for (int bad = 0; bad < 3; ++bad) {
        bool rejected = false;
        try {
            batch(nullptr, bad == 0 ? n - 1 : so, nullptr,
                  bad == 1 ? n - 1 : sa, nullptr, bad == 2 ? n - 1 : sb, 1,
                  nullptr, 0);
        }
        catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) return false;
    }
    TFHEpp::EvalKey ek(sk);
    ek.template emplacebk<BR>(sk);
#ifdef USE_SUBSET_KEY
    if constexpr (sizeof(typename BR::targetP::T) == 4)
        ek.template emplacesubiksk<IKS>(sk);
    else
#endif
        ek.template emplaceiksk<IKS>(sk);
    if constexpr (sizeof(typename BR::targetP::T) == 4)
        cufhe::Initialize(ek);
    else
        cufhe::Initialize_lvl02(ek, sk);

    std::vector<T> a(count * sa, 7), b(count * sb, 8), got(count * so, 9),
        ref(got);
    for (uint32_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> sample;
        TFHEpp::tlweSymEncrypt<P>(sample, i & 1 ? P::μ : -P::μ,
                                  sk.key.get<P>());
        std::copy(sample.begin(), sample.end(), a.begin() + i * sa);
        TFHEpp::tlweSymEncrypt<P>(sample, i & 2 ? P::μ : -P::μ,
                                  sk.key.get<P>());
        std::copy(sample.begin(), sample.end(), b.begin() + i * sb);
    }
    T *da = nullptr, *db = nullptr, *dout = nullptr, *dref = nullptr;
    Check(cudaMalloc(&da, a.size() * sizeof(T)));
    Check(cudaMalloc(&db, b.size() * sizeof(T)));
    Check(cudaMalloc(&dout, got.size() * sizeof(T)));
    Check(cudaMalloc(&dref, ref.size() * sizeof(T)));
    Check(
        cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(
        cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dout, got.data(), got.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
    Check(cudaMemcpy(dref, ref.data(), ref.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
    cudaStream_t stream;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    bool passed = true;
#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
    // First use of this batch kernel is captured, with no warm-up launch.
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    batch(dout, so, da, sa, db, sb, count, stream, 0);
    Check(cudaStreamEndCapture(stream, &graph));
    size_t nodes = 0;
    Check(cudaGraphGetNodes(graph, nullptr, &nodes));
    passed &= nodes == 1;
    if (nodes == 1) {
        cudaGraphNode_t node;
        Check(cudaGraphGetNodes(graph, &node, &nodes));
        cudaGraphNodeType type;
        Check(cudaGraphNodeGetType(node, &type));
        passed &= type == cudaGraphNodeTypeKernel;
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            Check(cudaGraphKernelNodeGetParams(node, &params));
            passed &= params.gridDim.x == count &&
                      params.blockDim.x == BR::targetP::n / 2;
        }
    }
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    Check(cudaGraphLaunch(exec, stream));
#else
    batch(dout, so, da, sa, db, sb, count, stream, 0);
#endif
    for (uint32_t i = 0; i < count; ++i)
        scalar(dref + i * so, da + i * sa, db + i * sb, stream, 0);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(got.data(), dout, got.size() * sizeof(T),
                     cudaMemcpyDeviceToHost));
    Check(cudaMemcpy(ref.data(), dref, ref.size() * sizeof(T),
                     cudaMemcpyDeviceToHost));
#ifndef USE_FFT
    passed &= got == ref;
#endif
#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
    Check(cudaGraphLaunch(exec, stream));
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(got.data(), dout, got.size() * sizeof(T),
                     cudaMemcpyDeviceToHost));
#ifndef USE_FFT
    passed &= got == ref;
#endif
    Check(cudaGraphExecDestroy(exec));
    Check(cudaGraphDestroy(graph));
#endif
    batch(dout, so, da, sa, db, sb, count, stream, 0);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(got.data(), dout, got.size() * sizeof(T),
                     cudaMemcpyDeviceToHost));
#ifndef USE_FFT
    passed &= got == ref;
#endif
    for (uint32_t i = 0; i < count; ++i) {
        TFHEpp::TLWE<P> sample;
        std::copy_n(got.begin() + i * so, n, sample.begin());
        passed &= TFHEpp::tlweSymDecrypt<P>(sample, sk.key.get<P>()) ==
                  !((i & 1) && (i & 2));
        for (size_t j = n; j < so; ++j) passed &= got[i * so + j] == 9;
    }
    const auto single_call_ref = ref;
    // Match scalar coefficients through both input aliases and eight in-place
    // rounds.
    for (int alias = 0; alias < 2; ++alias) {
        Check(cudaMemcpy(da, a.data(), a.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
        Check(cudaMemcpy(db, b.data(), b.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
        T* target = alias == 0 ? da : db;
        const size_t stride = alias == 0 ? sa : sb;
        std::vector<T> expected = alias == 0 ? a : b, actual(expected);
        Check(cudaMemcpy(dref, ref.data(), ref.size() * sizeof(T),
                         cudaMemcpyHostToDevice));
        for (uint32_t i = 0; i < count; ++i)
            Check(cudaMemcpyAsync(dref + i * so, expected.data() + i * stride,
                                  n * sizeof(T), cudaMemcpyHostToDevice,
                                  stream));
#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
        Check(cudaStreamSynchronize(stream));
        cudaGraph_t alias_graph;
        cudaGraphExec_t alias_exec;
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        batch(target, stride, da, sa, db, sb, count, stream, 0);
        Check(cudaStreamEndCapture(stream, &alias_graph));
        size_t alias_nodes = 0;
        Check(cudaGraphGetNodes(alias_graph, nullptr, &alias_nodes));
        passed &= alias_nodes == 1;
        Check(cudaGraphInstantiate(&alias_exec, alias_graph, nullptr, nullptr,
                                   0));
#endif
        for (int round = 0; round < 8; ++round) {
#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
            Check(cudaGraphLaunch(alias_exec, stream));
#else
            batch(target, stride, da, sa, db, sb, count, stream, 0);
#endif
            for (uint32_t i = 0; i < count; ++i)
                scalar(dref + i * so, alias == 0 ? dref + i * so : da + i * sa,
                       alias == 1 ? dref + i * so : db + i * sb, stream, 0);
        }
        Check(cudaStreamSynchronize(stream));
#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
        Check(cudaGraphExecDestroy(alias_exec));
        Check(cudaGraphDestroy(alias_graph));
#endif
        Check(cudaMemcpy(actual.data(), target, actual.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(ref.data(), dref, ref.size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < count; ++i) {
#ifndef USE_FFT
            passed &= std::equal(actual.begin() + i * stride,
                                 actual.begin() + i * stride + n,
                                 ref.begin() + i * so);
#endif
            for (size_t j = n; j < stride; ++j)
                passed &= actual[i * stride + j] == (alias == 0 ? 7 : 8);
            TFHEpp::TLWE<P> sample;
            std::copy_n(actual.begin() + i * stride, n, sample.begin());
            // Eight toggles with a true fixed operand restore the starting bit;
            // a false fixed operand makes NAND true on every round.
            const bool expected_bit = alias == 0
                                          ? ((i & 2) ? bool(i & 1) : true)
                                          : ((i & 1) ? bool(i & 2) : true);
            passed &= TFHEpp::tlweSymDecrypt<P>(sample, sk.key.get<P>()) ==
                      expected_bit;
        }
    }
    // Forward overlaps remain within each sample's padded allocation.
    // The cached implementation must capture the original body before using
    // a shifted output prefix for metadata; retain the original single call.
    for (int alias = 0; alias < 2; ++alias) {
        Check(cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice));
        T* target = alias == 0 ? da : db;
        const size_t stride = alias == 0 ? sa : sb;
        const auto& original = alias == 0 ? a : b;
        for (uint32_t i = 0; i < count; ++i)
            scalar(target + i * stride + 1, da + i * sa, db + i * sb, stream, 0);
        Check(cudaStreamSynchronize(stream));
        std::vector<T> actual(original.size());
        Check(cudaMemcpy(actual.data(), target, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < count; ++i) {
            passed &= std::equal(actual.begin() + i * stride + 1,
                                 actual.begin() + i * stride + 1 + n,
                                 single_call_ref.begin() + i * so);
            passed &= actual[i * stride] == original[i * stride];
            for (size_t j = n + 1; j < stride; ++j)
                passed &= actual[i * stride + j] == original[i * stride + j];
        }
    }

    Check(cudaStreamDestroy(stream));
    Check(cudaFree(da));
    Check(cudaFree(db));
    Check(cudaFree(dout));
    Check(cudaFree(dref));
    if constexpr (sizeof(typename BR::targetP::T) == 4)
        cufhe::CleanUp();
    else
        cufhe::CleanUp_lvl02();
    std::cout << "public lvl"
              << (sizeof(typename BR::targetP::T) == 4 ? "1" : "02")
              << " count=" << count
              << " exact batch/scalar, graph replay, both aliases: "
              << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}
}  // namespace
int main(int argc, char** argv)
{
    try {
        Check(cudaSetDevice(0));
        cufhe::SetGPUNum(1);
        TFHEpp::SecretKey sk;
        if (argc > 1) {
            const std::string level(argv[1]);
            const auto parsed = argc > 2 ? std::stoull(argv[2]) : 17ULL;
            if (parsed == 0 || parsed > UINT32_MAX)
                throw std::invalid_argument("invalid count");
            const auto count = static_cast<uint32_t>(parsed);
            if (level == "lvl01")
                return Test<TFHEpp::lvl01param, TFHEpp::lvl10param>(sk, count)
                           ? 0
                           : 1;
            if (level == "lvl02")
                return Test<TFHEpp::lvl02param, TFHEpp::lvl20param>(sk, count)
                           ? 0
                           : 1;
            throw std::invalid_argument("expected lvl01 or lvl02");
        }
        cudaDeviceProp prop;
        Check(cudaGetDeviceProperties(&prop, 0));
        bool passed = Test<TFHEpp::lvl01param, TFHEpp::lvl10param>(sk, 17);
        passed = Test<TFHEpp::lvl02param, TFHEpp::lvl20param>(sk, 17) && passed;
        passed = Test<TFHEpp::lvl01param, TFHEpp::lvl10param>(
                     sk, prop.multiProcessorCount + 1) &&
                 passed;
        passed = Test<TFHEpp::lvl02param, TFHEpp::lvl20param>(
                     sk, prop.multiProcessorCount + 1) &&
                 passed;
        return passed ? 0 : 1;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
