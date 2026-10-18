#include <include/tfhepp_keygen.hpp>
#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#if !defined(CUFHE_USE_HIP) && CUDART_VERSION >= 11040
#define CUFHE_BENCH_HAS_CUDA_GRAPH 1
#endif

// Real encrypted inputs and keys: unlike zero-key microbenchmarks, the
// block-binary monomial accesses and key-switch digits vary across samples.
// Arguments: streams, gates per stream, repetitions, [batch|packed|gate name], [graph].
// MUX/NMux use three independent ciphertexts and cover all eight truth cases.
int main(int argc, char** argv)
{
#ifdef CUFHE_BENCH_LVL02
    using IKS = TFHEpp::lvl20param;
    using BR = TFHEpp::lvl02param;
#else
    using IKS = TFHEpp::lvl10param;
    using BR = TFHEpp::lvl01param;
#endif
#ifdef CUFHE_BENCH_BR_IKS
    using P = BR::domainP;
#ifdef CUFHE_BENCH_LVL02
    const auto public_nand = &cufhe::Nand_lvl02;
    const auto public_mux = &cufhe::Mux_lvl02;
    const auto public_nmux = &cufhe::NMux_lvl02;
#else
    const auto public_nand = &cufhe::Nand<P>;
    const auto public_mux = &cufhe::Mux<P>;
    const auto public_nmux = &cufhe::NMux<P>;
#endif
#else
    using P = IKS::domainP;
#endif
    const int streams_count = argc > 1 ? std::atoi(argv[1]) : 128;
    const int per_stream = argc > 2 ? std::atoi(argv[2]) : 4;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 3;
    const std::string_view mode = argc > 4 ? argv[4] : "nand";
    const bool graph_mode = argc > 5 && std::string_view(argv[5]) == "graph";
    if (argc > 5 && !graph_mode) return 2;
#ifndef CUFHE_BENCH_HAS_CUDA_GRAPH
    if (graph_mode) return 2;
#endif
    const bool batch = mode == "batch";
    const bool packed_scalar = mode == "packed";
    const bool packed_buffers = batch || packed_scalar;
    const bool bootstrap = mode == "bootstrap";
#if !defined(CUFHE_BENCH_BR_IKS) || defined(CUFHE_BENCH_LVL02)
    if (bootstrap) return 2;
#endif
    const bool nmux = mode == "nmux";
    const bool mux = nmux || mode == "mux";
    using BinaryGate = void (*)(P::T*, const P::T*, const P::T*, cudaStream_t, int);
    struct NamedGate { std::string_view name; unsigned truth; BinaryGate launch; };
#ifdef CUFHE_BENCH_BR_IKS
#define BINARY(Name) &cufhe::Name##Bootstrap<BR, BR::targetP::μ, IKS>
#else
#define BINARY(Name) &cufhe::Name##Bootstrap<IKS, BR, P::μ>
#endif
    const NamedGate gates[] = {
        {"nand", 7, BINARY(Nand)}, {"and", 8, BINARY(And)},
        {"nor", 1, BINARY(Nor)}, {"or", 14, BINARY(Or)},
        {"xor", 6, BINARY(Xor)}, {"xnor", 9, BINARY(Xnor)},
        {"andny", 4, BINARY(AndNY)}, {"andyn", 2, BINARY(AndYN)},
        {"orny", 13, BINARY(OrNY)}, {"oryn", 11, BINARY(OrYN)}};
#undef BINARY
    const NamedGate* binary = nullptr;
    for (const auto& gate : gates)
        if (gate.name == (packed_buffers ? "nand" : mode)) binary = &gate;
    if (!binary && !mux && !bootstrap) {
        std::cerr << "Unknown gate: " << mode << '\n';
        return 2;
    }
    if (streams_count < 1 || per_stream < 1 || repeats < 1) return 2;
    auto check = [](cudaError_t error) {
        if (error != cudaSuccess) {
            std::cerr << cudaGetErrorString(error) << '\n';
            std::exit(1);
        }
    };
    check(cudaSetDevice(0));
    cudaDeviceProp prop;
    check(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU=" << prop.name << " n=" << TFHEpp::lvl0param::n
              << " N=" << BR::targetP::n << " k=" << BR::targetP::k
              << " l=" << BR::targetP::l
              << " torus_bits=" << std::numeric_limits<P::T>::digits
#ifdef CUFHE_BENCH_BR_IKS
              << " order=br-iks"
#else
              << " order=iks-br"
#endif
              << " gate=" << (packed_buffers ? "nand" : mode)
#ifdef USE_FFT
              << " backend=fft"
#else
              << " backend=ntt"
#endif
#ifdef USE_BLOCK_BINARY
              << " block_binary=1"
#else
              << " block_binary=0"
#endif
              << std::endl;
    cufhe::SetGPUNum(1);
    TFHEpp::SecretKey sk;
    TFHEpp::EvalKey ek(sk);
    ek.emplacebk<BR>(sk);
#ifdef CUFHE_BENCH_LVL02
    cufhe::EmplaceGateKeySwitchingKey<IKS>(ek, sk);
    cufhe::Initialize_lvl02(ek, sk);
#else
#ifdef USE_SUBSET_KEY
    ek.emplacesubiksk<IKS>(sk);
#else
    cufhe::EmplaceGateKeySwitchingKey<IKS>(ek, sk);
#endif
    cufhe::Initialize(ek);
#endif
    const int count = streams_count * per_stream;
    std::vector<cufhe::Stream> streams(streams_count);
    for (auto& stream : streams) stream.Create();
    std::vector<cufhe::Ctxt<P>> a(count), b(count), out(count);
    std::vector<cufhe::Ctxt<P>> c(mux ? count : 0);
    for (int i = 0; i < count; ++i) {
        TFHEpp::tlweSymEncrypt<P>(a[i].tlwehost, (i & 1) ? P::μ : -P::μ,
                                  sk.key.getSubset<P>());
        TFHEpp::tlweSymEncrypt<P>(b[i].tlwehost, (i & 2) ? P::μ : -P::μ,
                                  sk.key.getSubset<P>());
        cufhe::CtxtCopyH2D(a[i], streams[i % streams_count]);
        cufhe::CtxtCopyH2D(b[i], streams[i % streams_count]);
        if (mux) {
            TFHEpp::tlweSymEncrypt<P>(c[i].tlwehost, (i & 4) ? P::μ : -P::μ,
                                      sk.key.getSubset<P>());
            cufhe::CtxtCopyH2D(c[i], streams[i % streams_count]);
        }
    }
    check(cudaDeviceSynchronize());
    constexpr size_t stride = P::k * P::n + 1;
    P::T *batch_a = nullptr, *batch_b = nullptr, *batch_out = nullptr;
    if (packed_buffers) {
        check(cudaMalloc(&batch_a, count * stride * sizeof(P::T)));
        check(cudaMalloc(&batch_b, count * stride * sizeof(P::T)));
        check(cudaMalloc(&batch_out, count * stride * sizeof(P::T)));
        std::vector<P::T> staging(count * stride);
        for (int i = 0; i < count; ++i)
            std::copy(a[i].tlwehost.begin(), a[i].tlwehost.end(),
                      staging.begin() + i * stride);
        check(cudaMemcpy(batch_a, staging.data(), staging.size() * sizeof(P::T),
                         cudaMemcpyHostToDevice));
        for (int i = 0; i < count; ++i)
            std::copy(b[i].tlwehost.begin(), b[i].tlwehost.end(),
                      staging.begin() + i * stride);
        check(cudaMemcpy(batch_b, staging.data(), staging.size() * sizeof(P::T),
                         cudaMemcpyHostToDevice));
        // Complete pageable DMA before submission on nonblocking streams.
        check(cudaStreamSynchronize(nullptr));
    }
    auto submit = [&]() {
        if (batch)
#ifdef CUFHE_BENCH_BR_IKS
            cufhe::NandBootstrapBatch<BR, BR::targetP::μ, IKS>(
#else
            cufhe::NandBootstrapBatch<IKS, BR, P::μ>(
#endif
                batch_out, stride, batch_a, stride, batch_b, stride, count,
                streams[0].st(), 0);
        else if (packed_scalar)
            for (int i = 0; i < count; ++i)
                binary->launch(batch_out + i * stride, batch_a + i * stride,
                               batch_b + i * stride, streams[i % streams_count].st(), 0);
        else
            for (int i = 0; i < count; ++i) {
#ifdef CUFHE_BENCH_BR_IKS
#ifndef CUFHE_BENCH_LVL02
                if (bootstrap)
                    cufhe::Bootstrap(out[i].tlwedevices[0], a[i].tlwedevices[0],
                        BR::targetP::μ, streams[i % streams_count].st(), 0);
                else
#endif
                if (nmux)
                    public_nmux(out[i], c[i], b[i], a[i], streams[i % streams_count]);
                else if (mux)
                    public_mux(out[i], c[i], b[i], a[i], streams[i % streams_count]);
                else if (binary->name == "nand")
                    public_nand(out[i], a[i], b[i], streams[i % streams_count]);
                else
                    binary->launch(out[i].tlwedevices[0], a[i].tlwedevices[0],
                                   b[i].tlwedevices[0], streams[i % streams_count].st(), 0);
#else
                auto& stream = streams[i % streams_count];
                if (nmux)
                    cufhe::NMuxBootstrap<IKS, BR, P::μ>(out[i].tlwedevices[0],
                        c[i].tlwedevices[0], b[i].tlwedevices[0], a[i].tlwedevices[0], stream.st(), 0);
                else if (mux)
                    cufhe::MuxBootstrap<IKS, BR, P::μ>(out[i].tlwedevices[0],
                        c[i].tlwedevices[0], b[i].tlwedevices[0], a[i].tlwedevices[0], stream.st(), 0);
                else {
#ifndef CUFHE_BENCH_LVL02
                    if (binary->name == "nand")
                        cufhe::gNand(out[i], a[i], b[i], stream);
                    else
#endif
                        binary->launch(out[i].tlwedevices[0], a[i].tlwedevices[0],
                                       b[i].tlwedevices[0], stream.st(), 0);
                }
#endif
            }
        check(cudaGetLastError());
    };
#ifdef CUFHE_BENCH_HAS_CUDA_GRAPH
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t fork_event = nullptr;
    std::vector<cudaEvent_t> join_events(graph_mode ? streams_count - 1 : 0);
    if (graph_mode) {
        // Load kernels and configure their shared limits before capture.
        submit();
        check(cudaDeviceSynchronize());
        check(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming));
        for (auto& event : join_events)
            check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        check(cudaStreamBeginCapture(streams[0].st(), cudaStreamCaptureModeGlobal));
        check(cudaEventRecord(fork_event, streams[0].st()));
        for (int i = 1; i < streams_count; ++i)
            check(cudaStreamWaitEvent(streams[i].st(), fork_event, 0));
        submit();
        for (int i = 1; i < streams_count; ++i) {
            check(cudaEventRecord(join_events[i - 1], streams[i].st()));
            check(cudaStreamWaitEvent(streams[0].st(), join_events[i - 1], 0));
        }
        check(cudaStreamEndCapture(streams[0].st(), &graph));
        check(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
        size_t nodes = 0;
        check(cudaGraphGetNodes(graph, nullptr, &nodes));
        std::cout << "graph_nodes=" << nodes << std::endl;
    }
#endif
    std::vector<double> times;
    // A single small launch may leave the GPU at its idle clock. Warm up for
    // at least 250 ms before collecting latency or throughput samples.
    const auto warmup_start = std::chrono::steady_clock::now();
    for (int repeat = -1; repeat < repeats;) {
        const auto start = std::chrono::steady_clock::now();
#ifdef CUFHE_BENCH_HAS_CUDA_GRAPH
        if (graph_mode) check(cudaGraphLaunch(graph_exec, streams[0].st()));
        else
#endif
            submit();
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start).count();
        if (repeat >= 0) {
            times.push_back(ms / count);
            std::cout << "repeat=" << repeat << " ms/gate=" << ms / count
                      << std::endl;
            ++repeat;
        }
        else if (std::chrono::steady_clock::now() - warmup_start >=
                 std::chrono::milliseconds(250)) {
            repeat = 0;
        }
    }
#ifdef CUFHE_BENCH_HAS_CUDA_GRAPH
    if (graph_mode) {
        check(cudaGraphExecDestroy(graph_exec));
        check(cudaGraphDestroy(graph));
        check(cudaEventDestroy(fork_event));
        for (auto event : join_events) check(cudaEventDestroy(event));
    }
#endif
    if (packed_buffers) {
        std::vector<P::T> staging(count * stride);
        check(cudaMemcpy(staging.data(), batch_out, staging.size() * sizeof(P::T),
                         cudaMemcpyDeviceToHost));
        for (int i = 0; i < count; ++i)
            std::copy_n(staging.begin() + i * stride, stride,
                        out[i].tlwehost.begin());
        check(cudaFree(batch_a));
        check(cudaFree(batch_b));
        check(cudaFree(batch_out));
    }
    else {
        for (int i = 0; i < count; ++i)
            cufhe::CtxtCopyD2H(out[i], streams[i % streams_count]);
        check(cudaDeviceSynchronize());
    }
    int errors = 0;
    long double squared_noise = 0;
    long double max_noise = 0;
    for (int i = 0; i < count; ++i) {
        const bool selected = (i & 4) ? bool(i & 2) : bool(i & 1);
        const bool expected = bootstrap ? bool(i & 1) :
            mux ? (nmux ? !selected : selected) :
            bool((binary->truth >> (i & 3)) & 1U);
        errors += TFHEpp::tlweSymDecrypt<P>(out[i].tlwehost, sk.key.getSubset<P>()) !=
                  expected;
        const auto phase = TFHEpp::tlweSymPhase<P>(out[i].tlwehost,
                                                   sk.key.getSubset<P>());
        const auto residual = static_cast<std::make_signed_t<P::T>>(
            phase - (expected ? P::μ : -P::μ));
        const long double noise = std::ldexp(static_cast<long double>(residual),
                                            -std::numeric_limits<P::T>::digits);
        squared_noise += noise * noise;
        max_noise = std::max(max_noise, std::abs(noise));
    }
    std::sort(times.begin(), times.end());
    std::cout << "dispatch=" << (graph_mode ? "graph" : batch ? "batch" : "streams")
              << " streams=" << (batch ? 1 : streams_count) << " gates=" << count
              << " median_ms/gate=" << times[times.size() / 2]
              << " errors=" << errors << std::endl;
    std::cout << "observed_noise_rms=" << std::sqrt(squared_noise / count)
              << " observed_max_abs_noise=" << max_noise << std::endl;
    for (auto& stream : streams) stream.Destroy();
#ifdef CUFHE_BENCH_LVL02
    cufhe::CleanUp_lvl02();
#else
    cufhe::CleanUp();
#endif
    return errors ? 1 : 0;
}
