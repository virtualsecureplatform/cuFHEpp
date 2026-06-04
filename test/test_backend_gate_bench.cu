#include <cuda_runtime.h>

#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/ntt_small_modulus.cuh>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace cufhe {
extern std::vector<NTTValue*> bk_ntts;
}

namespace {

const char* BackendName()
{
#ifdef USE_FFT
#ifdef USE_GPU_FFT
    return "gpu_fft";
#else
    return "tfhe_rs_fft";
#endif
#else
    return "small_ntt";
#endif
}

void CudaCheck(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) CudaCheck((expr), __FILE__, __LINE__)

void AllocateFakeEvaluationKeys(const int gpu_num)
{
    using brP = TFHEpp::lvl01param;
    using iksP = TFHEpp::lvl10param;
    using targetP = typename brP::targetP;

    cufhe::InitializeNTThandlers(gpu_num);

    constexpr uint32_t num_pairs =
        brP::domainP::k * brP::domainP::n / brP::Addends;
    constexpr uint32_t bk_elements_per_pair = (1 << brP::Addends) - 1;
    constexpr uint32_t trgsw_polys =
        (targetP::k + 1) * targetP::l * (targetP::k + 1);
#ifdef USE_KEY_BUNDLE
#ifdef USE_FFT
    constexpr uint32_t bk_coeffs = targetP::n >> 1;
#else
    constexpr uint32_t bk_coeffs = targetP::n;
#endif
    constexpr size_t bk_elements = static_cast<size_t>(num_pairs) *
                                   bk_elements_per_pair * trgsw_polys *
                                   bk_coeffs;
#else
#ifdef USE_FFT
    constexpr uint32_t bk_coeffs = targetP::n >> 1;
#else
    constexpr uint32_t bk_coeffs = targetP::n;
#endif
    constexpr size_t bk_elements = static_cast<size_t>(brP::domainP::n) *
                                   (targetP::k + 1) * targetP::l *
                                   (targetP::k + 1) * bk_coeffs;
#endif

    cufhe::bk_ntts.resize(gpu_num);
    cufhe::ksk_devs.resize(gpu_num);
    for (int gpu = 0; gpu < gpu_num; gpu++) {
        cudaSetDevice(gpu);
        CUDA_CHECK(cudaMalloc(&cufhe::bk_ntts[gpu],
                              sizeof(cufhe::NTTValue) * bk_elements));
        CUDA_CHECK(cudaMemset(cufhe::bk_ntts[gpu], 0,
                              sizeof(cufhe::NTTValue) * bk_elements));

        CUDA_CHECK(cudaMalloc(&cufhe::ksk_devs[gpu],
                              sizeof(TFHEpp::KeySwitchingKey<iksP>)));
        CUDA_CHECK(cudaMemset(cufhe::ksk_devs[gpu], 0,
                              sizeof(TFHEpp::KeySwitchingKey<iksP>)));
    }

#ifdef USE_KEY_BUNDLE
    cufhe::InitializeXaiNTT(gpu_num);
    cufhe::InitializeOneTRGSWNTT(gpu_num);
#endif
}

void FreeFakeEvaluationKeys(const int gpu_num)
{
#ifdef USE_KEY_BUNDLE
    cufhe::DeleteXaiNTT();
    cufhe::DeleteOneTRGSWNTT();
#endif
    cufhe::DeleteBootstrappingKeyNTT(gpu_num);
    cufhe::DeleteKeySwitchingKey(gpu_num);
    cufhe::bk_ntts.clear();
    cufhe::ksk_devs.clear();
}

}  // namespace

int main()
{
    using Param = TFHEpp::lvl1param;

    cudaSetDevice(0);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    const int gpu_num = 1;
    const uint32_t num_streams = prop.multiProcessorCount;
    const uint32_t num_tests = num_streams * 32;
    const uint32_t gates_per_stream = num_tests / num_streams;

    cufhe::SetGPUNum(gpu_num);
    AllocateFakeEvaluationKeys(gpu_num);

    std::vector<cufhe::Stream> streams(num_streams);
    for (auto& stream : streams) stream.Create();

    std::vector<cufhe::Ctxt<Param>> out(num_tests);
    std::vector<cufhe::Ctxt<Param>> in0(num_tests);
    std::vector<cufhe::Ctxt<Param>> in1(num_tests);

    for (uint32_t i = 0; i < num_tests; i++) {
        CUDA_CHECK(
            cudaMemset(out[i].tlwedevices[0], 0, sizeof(out[i].tlwehost)));
        CUDA_CHECK(
            cudaMemset(in0[i].tlwedevices[0], 0, sizeof(in0[i].tlwehost)));
        CUDA_CHECK(
            cudaMemset(in1[i].tlwedevices[0], 0, sizeof(in1[i].tlwehost)));
    }
    cufhe::Synchronize();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, 0));

    for (uint32_t i = 0; i < num_tests; i++) {
        cufhe::gNand<Param>(out[i], in0[i], in1[i],
                            streams[i % num_streams]);
    }
    cufhe::Synchronize();

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::cout << "Backend: " << BackendName() << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Gate: gNAND" << std::endl;
    std::cout << "Number of streams:\t" << num_streams << std::endl;
    std::cout << "Number of tests:\t" << num_tests << std::endl;
    std::cout << "Number of tests per stream:\t" << gates_per_stream
              << std::endl;
    std::cout << "Total: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Throughput: " << elapsed_ms / num_tests << " ms/gate"
              << std::endl;
    std::cout << "Latency: " << elapsed_ms / gates_per_stream << " ms/gate"
              << std::endl;

    for (auto& stream : streams) stream.Destroy();
    FreeFakeEvaluationKeys(gpu_num);
    return 0;
}

#undef CUDA_CHECK
