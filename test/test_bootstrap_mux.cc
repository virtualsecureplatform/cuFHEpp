#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
#define CUFHE_MUX_TEST_HAS_GRAPH 1
#endif

namespace {
void Check(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class P, class IKS, class BR>
bool TestMux(const TFHEpp::SecretKey& sk, bool graph_mode)
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

    constexpr int count = 16;
    constexpr int elements = P::k * P::n + 1;
    cufhe::Stream stream;
    stream.Create();
    std::vector<cufhe::Ctxt<P>> a(count), b(count), selector(count), first(count), second(count), out(count);
    std::vector<TFHEpp::TLWE<P>> saved_a(count), saved_b(count), saved_selector(count);
    for (int i = 0; i < count; ++i) {
        TFHEpp::tlweSymEncrypt<P>(a[i].tlwehost, (i & 1) ? P::μ : -P::μ, sk.key.template get<P>());
        TFHEpp::tlweSymEncrypt<P>(b[i].tlwehost, (i & 2) ? P::μ : -P::μ, sk.key.template get<P>());
        TFHEpp::tlweSymEncrypt<P>(selector[i].tlwehost, (i & 4) ? P::μ : -P::μ, sk.key.template get<P>());
        saved_a[i] = a[i].tlwehost;
        saved_b[i] = b[i].tlwehost;
        saved_selector[i] = selector[i].tlwehost;
        cufhe::CtxtCopyH2D(a[i], stream);
        cufhe::CtxtCopyH2D(b[i], stream);
        cufhe::CtxtCopyH2D(selector[i], stream);
        cufhe::AndBootstrap<IKS, BR, P::μ>(first[i].tlwedevices[0], selector[i].tlwedevices[0],
                                         b[i].tlwedevices[0], stream.st(), 0);
        cufhe::AndNYBootstrap<IKS, BR, P::μ>(second[i].tlwedevices[0], selector[i].tlwedevices[0],
                                           a[i].tlwedevices[0], stream.st(), 0);
        cufhe::CtxtCopyD2H(first[i], stream);
        cufhe::CtxtCopyD2H(second[i], stream);
    }
    Check(cudaStreamSynchronize(stream.st()));
    bool passed = true;
    for (bool negate : {false, true}) {
        for (int alias : {0, 1, 2, 3}) {
            for (int i = 0; i < count; ++i) {
                auto& output = alias == 1 ? b[i] : alias == 2 ? selector[i] : alias == 3 ? a[i] : out[i];
                auto launch = [&]() {
                    if (negate)
                        cufhe::NMuxBootstrap<IKS, BR, P::μ>(output.tlwedevices[0], selector[i].tlwedevices[0],
                            b[i].tlwedevices[0], a[i].tlwedevices[0], stream.st(), 0);
                    else
                        cufhe::MuxBootstrap<IKS, BR, P::μ>(output.tlwedevices[0], selector[i].tlwedevices[0],
                            b[i].tlwedevices[0], a[i].tlwedevices[0], stream.st(), 0);
                };
#ifdef CUFHE_MUX_TEST_HAS_GRAPH
                cudaGraph_t graph = nullptr;
                cudaGraphExec_t graph_exec = nullptr;
                if (graph_mode) {
                    // No ordinary MUX warmup: configure the graph kernels on
                    // their first use while capture is already active.
                    Check(cudaStreamBeginCapture(stream.st(), cudaStreamCaptureModeGlobal));
                    launch();
                    Check(cudaStreamEndCapture(stream.st(), &graph));
                    size_t nodes = 0;
                    Check(cudaGraphGetNodes(graph, nullptr, &nodes));
#ifndef USE_FFT
                    const size_t expected_nodes =
                        cufhe::USE_PAIRED_NTT64<P> && BR::domainP::ell == 2 && alias < 2 ? 2 : 1;
#else
                    constexpr size_t expected_nodes = 1;
#endif
                    passed &= nodes == expected_nodes;
                    Check(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
                }
#endif
                for (int replay = 0; replay < (graph_mode ? 2 : 1); ++replay) {
                    // Restore every input before an aliased replay.
                    a[i].tlwehost = saved_a[i];
                    b[i].tlwehost = saved_b[i];
                    selector[i].tlwehost = saved_selector[i];
                    cufhe::CtxtCopyH2D(a[i], stream);
                    cufhe::CtxtCopyH2D(selector[i], stream);
                    cufhe::CtxtCopyH2D(b[i], stream);
#ifdef CUFHE_MUX_TEST_HAS_GRAPH
                    if (graph_mode) Check(cudaGraphLaunch(graph_exec, stream.st()));
                    else
#endif
                        launch();
                    cufhe::CtxtCopyD2H(output, stream);
                    Check(cudaStreamSynchronize(stream.st()));
                    TFHEpp::TLWE<P> expected;
                    for (int j = 0; j < elements; ++j) {
                        expected[j] = first[i].tlwehost[j] + second[i].tlwehost[j];
                        if (negate) expected[j] = -expected[j];
                    }
                    if (negate) expected[elements - 1] -= P::μ;
                    else expected[elements - 1] += P::μ;
#ifndef USE_FFT
                    passed &= output.tlwehost == expected;
#endif
                    const bool selected = (i & 4) ? bool(i & 2) : bool(i & 1);
                    passed &= TFHEpp::tlweSymDecrypt<P>(output.tlwehost, sk.key.template get<P>()) ==
                              (negate ? !selected : selected);
                }
#ifdef CUFHE_MUX_TEST_HAS_GRAPH
                if (graph_mode) {
                    Check(cudaGraphExecDestroy(graph_exec));
                    Check(cudaGraphDestroy(graph));
                }
#endif
            }
        }
    }
    stream.Destroy();
    if constexpr (sizeof(typename P::T) == 4) cufhe::CleanUp();
    else cufhe::CleanUp_lvl02();
    std::cout << (sizeof(typename P::T) == 4 ? "lvl1" : "lvl02")
              << " raw MUX/NMux: 16 independent masks, 8 truth cases, "
#ifndef USE_FFT
              << "two-AND coefficient reference, "
#endif
              << (graph_mode ? "graph replay x2 / all input aliases: " : "all input aliases: ")
              << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}
}

int main(int argc, char** argv)
{
    const bool graph_mode = argc > 1 && std::string_view(argv[1]) == "graph";
    if (argc > 1 && !graph_mode) return 2;
#ifndef CUFHE_MUX_TEST_HAS_GRAPH
    if (graph_mode) return 2;
#endif
    try {
        Check(cudaSetDevice(0));
        cufhe::SetGPUNum(1);
        TFHEpp::SecretKey sk;
        const bool lvl1 = TestMux<TFHEpp::lvl1param, TFHEpp::lvl10param, TFHEpp::lvl01param>(sk, graph_mode);
        const bool lvl02 = TestMux<TFHEpp::lvl2param, TFHEpp::lvl20param, TFHEpp::lvl02param>(sk, graph_mode);
        return lvl1 && lvl02 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
