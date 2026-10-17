#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/key.hpp>
#include <include/cufhe_gpu.cuh>
#include <include/bootstrap_gpu.cuh>
#include <include/annihilate_gpu.cuh>
#include <include/circuitbootstrapping_gpu.cuh>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <numeric>
#include <string_view>
#include <vector>

namespace {
using BR = TFHEpp::lvl01param;
using AH = TFHEpp::AHlvl1param;
using Input = BR::domainP;
using Ring = BR::targetP;
constexpr size_t InputWords = Input::k * Input::n + 1;
constexpr size_t RingWords = (Ring::k + 1) * Ring::n;
constexpr size_t OutputWords = (Ring::k * Ring::lₐ + Ring::l) * RingWords;
constexpr size_t TempWords = cufhe::CircuitBootstrapLUTCount<Ring> * RingWords;
void Check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); }
}
}
// Host-only driver: its device link matches ordinary library applications.
// Arguments: streams, calls per stream, repetitions, [batch]. All key generation,
// transfers, workspace allocation and coefficient validation are outside timing.
int main(int argc, char** argv)
{
    const int stream_count = argc > 1 ? std::atoi(argv[1]) : 1;
    const int per_stream = argc > 2 ? std::atoi(argv[2]) : 4;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 11;
    const bool batch = argc > 4 && std::string_view(argv[4]) == "batch";
    if (stream_count < 1 || per_stream < 1 || repeats < 1 || (argc > 4 && !batch)) return 2;
    const size_t count = size_t(stream_count) * per_stream;
    Check(cudaSetDevice(0)); cufhe::SetGPUNum(1);
    TFHEpp::SecretKey secret;
    auto bk = std::make_unique<TFHEpp::BootstrappingKey<BR>>();
    TFHEpp::bkgen<BR>(*bk,secret);
    auto ahk = std::make_unique<cufhe::AnnihilateKeyPolynomial<AH>>();
    cufhe::AnnihilateKeyPolynomialGen<AH>(*ahk,secret);
    auto cbsk = std::make_unique<cufhe::CBswitchingKeyPolynomial<AH>>();
    cufhe::CBswitchingKeyPolynomialGen<AH>(*cbsk,secret);
    cufhe::InitializeNTThandlers(1);
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
    cufhe::InitializeXaiNTT(1);
#endif
#ifdef USE_KEY_BUNDLE
    cufhe::InitializeOneTRGSWNTT(1);
    cufhe::BootstrappingKeyBundleToNTT<BR>(*bk,1);
#else
    cufhe::BootstrappingKeyToNTT<BR>(*bk,1);
#endif
    cufhe::AnnihilateKeyPolynomialToDevice<AH>(*ahk,1);
    cufhe::CBswitchingKeyPolynomialToDevice<AH>(*cbsk,1);
    std::vector<Input::T> input(count * InputWords);
    for (size_t i=0;i<count;++i) {
        TFHEpp::TLWE<Input> sample;
        TFHEpp::tlweSymEncrypt<Input>(sample,(i&1)?Input::μ:-Input::μ,secret.key.get<Input>());
        std::copy(sample.begin(),sample.end(),input.begin()+i*InputWords);
    }
    Input::T* device_input; Ring::T *output,*acc,*temp;
    Check(cudaMalloc(&device_input,input.size()*sizeof(Input::T)));
    Check(cudaMalloc(&output,count*OutputWords*sizeof(Ring::T)));
    Check(cudaMalloc(&acc,count*RingWords*sizeof(Ring::T)));
    Check(cudaMalloc(&temp,count*TempWords*sizeof(Ring::T)));
    Check(cudaMemcpy(device_input,input.data(),input.size()*sizeof(Input::T),cudaMemcpyHostToDevice));
    std::vector<cudaStream_t> streams(stream_count);
    for (auto& st:streams) Check(cudaStreamCreateWithFlags(&st,cudaStreamNonBlocking));
    auto submit=[&](bool batched) {
        if (batched) cufhe::AnnihilateCircuitBootstrappingBatchWithWorkspace<BR,AH>(
            output,OutputWords,device_input,InputWords,acc,temp,count,streams[0],0);
        else for (size_t i=0;i<count;++i)
            cufhe::AnnihilateCircuitBootstrappingWithWorkspace<BR,AH>(output+i*OutputWords,
                device_input+i*InputWords,acc+i*RingWords,temp+i*TempWords,streams[i%stream_count],0);
    };
    std::vector<double> times;
    const auto warmup=std::chrono::steady_clock::now();
    for (int repeat=-1;repeat<repeats;) {
        const auto start=std::chrono::steady_clock::now();submit(batch);Check(cudaDeviceSynchronize());
        const auto end=std::chrono::steady_clock::now();
        const double elapsed=std::chrono::duration<double,std::milli>(end-start).count()/count;
        if (repeat>=0) {times.push_back(elapsed);++repeat;}
        else if (end-warmup>=std::chrono::milliseconds(250)) repeat=0;
    }
    std::vector<Ring::T> actual(count*OutputWords),reference(actual.size());
    Check(cudaMemcpy(actual.data(),output,actual.size()*sizeof(Ring::T),cudaMemcpyDeviceToHost));
    submit(!batch);Check(cudaDeviceSynchronize());
    Check(cudaMemcpy(reference.data(),output,reference.size()*sizeof(Ring::T),cudaMemcpyDeviceToHost));
    const size_t errors=std::inner_product(actual.begin(),actual.end(),reference.begin(),size_t(0),
        std::plus<size_t>{},[](auto a,auto b){return size_t(a!=b);});
    std::sort(times.begin(),times.end());
    std::printf("circuit mode=%s streams=%d count=%zu median_ms/call=%.8f coefficients=%zu errors=%zu\n",
        batch?"batch":"streams",stream_count,count,times[times.size()/2],actual.size(),errors);
    for(auto st:streams)Check(cudaStreamDestroy(st));
    Check(cudaFree(temp));Check(cudaFree(acc));Check(cudaFree(output));Check(cudaFree(device_input));
    cufhe::DeleteCBswitchingKey<AH>(1);cufhe::DeleteAnnihilateKey<AH>(1);
#if defined(USE_KEY_BUNDLE) || defined(USE_BLOCK_BINARY)
    cufhe::DeleteXaiNTT();
#endif
#ifdef USE_KEY_BUNDLE
    cufhe::DeleteOneTRGSWNTT();
#endif
    cufhe::DeleteBootstrappingKeyNTT(1);
    return errors?1:0;
}
