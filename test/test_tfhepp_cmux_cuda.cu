#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/error_gpu.cuh>
#include <tfhe/detwfa.hpp>
#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>

namespace {

using cufhe::CuSafeCall__;

using P = TFHEpp::lvl1param;
using T = P::T;
using TRLWE = TFHEpp::TRLWE<P>;
using TRGSWFFT = TFHEpp::TRGSWFFT<P>;
using Polynomial = TFHEpp::Polynomial<P>;

constexpr size_t trlweBytes = sizeof(TRLWE);
constexpr size_t trgswFftBytes = sizeof(TRGSWFFT);

struct DeviceBuffer {
    void* data = nullptr;

    explicit DeviceBuffer(size_t bytes)
    {
        CuSafeCall(cudaMalloc(&data, bytes));
    }

    ~DeviceBuffer()
    {
        if (data != nullptr)
            CuSafeCall(cudaFree(data));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

TRLWE encryptBits(const std::array<bool, P::n>& bits,
                  const TFHEpp::SecretKey& secretKey)
{
    std::array<T, P::n> message{};
    for (size_t i = 0; i < message.size(); ++i)
        message[i] = bits[i] ? P::μ : -P::μ;

    TRLWE result;
    TFHEpp::trlweSymEncrypt<P>(result, message, secretKey.key.get<P>());
    return result;
}

TRGSWFFT encryptSelector(bool value, const TFHEpp::SecretKey& secretKey)
{
    Polynomial message{};
    message[0] = value;
    TRGSWFFT result;
    TFHEpp::trgswSymEncrypt<P>(result, message, secretKey.key.get<P>());
    return result;
}

std::array<bool, P::n> bitPattern(unsigned salt)
{
    std::array<bool, P::n> result{};
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = ((i * 17 + salt * 29 + (i >> 2)) & 1U) != 0;
    return result;
}

bool checkDecryption(const TRLWE& ciphertext,
                     const std::array<bool, P::n>& expected,
                     const TFHEpp::SecretKey& secretKey,
                     const std::string& label, unsigned gpu)
{
    const auto decrypted =
        TFHEpp::trlweSymDecrypt<P>(ciphertext, secretKey.key.get<P>());
    for (size_t i = 0; i < decrypted.size(); ++i) {
        if (decrypted[i] != expected[i]) {
            std::cerr << label << " failed on GPU " << gpu << " at coefficient "
                      << i << ": expected " << expected[i] << ", got "
                      << decrypted[i] << '\n';
            return false;
        }
    }
    return true;
}

enum class InPlaceMode { None, True, False };

bool runSingleCMUX(bool selectorBit, InPlaceMode inPlace,
                   const TFHEpp::SecretKey& secretKey, unsigned gpu)
{
    const unsigned mode = static_cast<unsigned>(inPlace);
    const auto trueBits = bitPattern(11 + selectorBit + 2 * mode);
    const auto falseBits = bitPattern(31 + selectorBit + 2 * mode);
    const auto expected = selectorBit ? trueBits : falseBits;
    const TRLWE trueInput = encryptBits(trueBits, secretKey);
    const TRLWE falseInput = encryptBits(falseBits, secretKey);
    const TRGSWFFT selector = encryptSelector(selectorBit, secretKey);

    TRLWE cpuOutput;
    if (inPlace == InPlaceMode::True) {
        cpuOutput = trueInput;
        TFHEpp::CMUXFFT<P>(cpuOutput, selector, cpuOutput, falseInput);
    }
    else if (inPlace == InPlaceMode::False) {
        cpuOutput = falseInput;
        // TFHEpp's final add-back reads c0 after res has been overwritten,
        // so the CPU reference needs the same preserved false branch as the
        // Tangor in-place codelet.
        const TRLWE preservedFalse = falseInput;
        TFHEpp::CMUXFFT<P>(cpuOutput, selector, trueInput, preservedFalse);
    }
    else {
        TFHEpp::CMUXFFT<P>(cpuOutput, selector, trueInput, falseInput);
    }
    if (!checkDecryption(cpuOutput, expected, secretKey, "CPU CMUX", gpu))
        return false;

    DeviceBuffer deviceSelector(trgswFftBytes);
    DeviceBuffer deviceTrue(trlweBytes);
    DeviceBuffer deviceFalse(trlweBytes);
    DeviceBuffer deviceOutput(trlweBytes);
    CuSafeCall(cudaMemcpy(deviceSelector.data, &selector, trgswFftBytes,
                          cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceTrue.data, &trueInput, trlweBytes,
                          cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceFalse.data, &falseInput, trlweBytes,
                          cudaMemcpyHostToDevice));

    T* const output = static_cast<T*>(
        inPlace == InPlaceMode::True  ? deviceTrue.data
        : inPlace == InPlaceMode::False ? deviceFalse.data
                                        : deviceOutput.data);
    cufhe::CMUXTFHEppFFTkernel(
        output, static_cast<const double*>(deviceSelector.data),
        static_cast<T*>(deviceTrue.data), static_cast<T*>(deviceFalse.data),
        cudaStreamDefault, gpu);
    CuSafeCall(cudaDeviceSynchronize());

    TRLWE gpuOutput;
    CuSafeCall(cudaMemcpy(&gpuOutput, output, trlweBytes, cudaMemcpyDeviceToHost));
    return checkDecryption(gpuOutput, expected, secretKey, "CUDA CMUX", gpu);
}

template <unsigned depth>
bool runInPlaceRamChain(const TFHEpp::SecretKey& secretKey, unsigned gpu)
{
    TRLWE cpuAccumulator = encryptBits(bitPattern(101), secretKey);
    std::array<bool, P::n> expected = bitPattern(101);

    std::array<TRGSWFFT, depth> selectors;
    std::array<TRLWE, depth> alternatives;
    std::array<std::unique_ptr<DeviceBuffer>, depth> deviceSelectors;
    std::array<std::unique_ptr<DeviceBuffer>, depth> deviceAlternatives;

    DeviceBuffer deviceAccumulator(trlweBytes);
    CuSafeCall(cudaMemcpy(deviceAccumulator.data, &cpuAccumulator, trlweBytes,
                          cudaMemcpyHostToDevice));

    for (unsigned level = 0; level < depth; ++level) {
        const bool select = ((level * 5 + 3) & 1U) != 0;
        const auto alternativeBits = bitPattern(200 + level);
        selectors[level] = encryptSelector(select, secretKey);
        alternatives[level] = encryptBits(alternativeBits, secretKey);
        TFHEpp::CMUXFFT<P>(cpuAccumulator, selectors[level], cpuAccumulator,
                           alternatives[level]);
        if (!select)
            expected = alternativeBits;

        deviceSelectors[level] = std::make_unique<DeviceBuffer>(trgswFftBytes);
        deviceAlternatives[level] = std::make_unique<DeviceBuffer>(trlweBytes);
        CuSafeCall(cudaMemcpy(deviceSelectors[level]->data, &selectors[level],
                              trgswFftBytes, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(deviceAlternatives[level]->data, &alternatives[level],
                              trlweBytes, cudaMemcpyHostToDevice));
        cufhe::CMUXTFHEppFFTkernel(
            static_cast<T*>(deviceAccumulator.data),
            static_cast<const double*>(deviceSelectors[level]->data),
            static_cast<T*>(deviceAccumulator.data),
            static_cast<T*>(deviceAlternatives[level]->data), cudaStreamDefault,
            gpu);
    }
    CuSafeCall(cudaDeviceSynchronize());

    if (!checkDecryption(cpuAccumulator, expected, secretKey, "CPU RAM CMUX", gpu))
        return false;
    TRLWE gpuAccumulator;
    CuSafeCall(cudaMemcpy(&gpuAccumulator, deviceAccumulator.data, trlweBytes,
                          cudaMemcpyDeviceToHost));
    return checkDecryption(gpuAccumulator, expected, secretKey,
                           "CUDA RAM CMUX", gpu);
}

bool runIndependentBatch(const TFHEpp::SecretKey& secretKey, unsigned gpu)
{
    constexpr unsigned count = 8;
    std::array<TRLWE, count> trueInputs;
    std::array<TRLWE, count> falseInputs;
    std::array<TRGSWFFT, count> selectors;
    std::array<std::array<bool, P::n>, count> expected;
    std::array<std::unique_ptr<DeviceBuffer>, count> deviceSelectors;
    std::array<std::unique_ptr<DeviceBuffer>, count> deviceTrue;
    std::array<std::unique_ptr<DeviceBuffer>, count> deviceFalse;
    std::array<std::unique_ptr<DeviceBuffer>, count> deviceOutput;
    std::array<T*, count> outputs{};
    std::array<const double*, count> selectorPointers{};
    std::array<T*, count> truePointers{};
    std::array<T*, count> falsePointers{};

    for (unsigned i = 0; i < count; ++i) {
        const bool select = (i & 1U) != 0;
        const auto trueBits = bitPattern(300 + i);
        const auto falseBits = bitPattern(400 + i);
        expected[i] = select ? trueBits : falseBits;
        trueInputs[i] = encryptBits(trueBits, secretKey);
        falseInputs[i] = encryptBits(falseBits, secretKey);
        selectors[i] = encryptSelector(select, secretKey);
        deviceSelectors[i] = std::make_unique<DeviceBuffer>(trgswFftBytes);
        deviceTrue[i] = std::make_unique<DeviceBuffer>(trlweBytes);
        deviceFalse[i] = std::make_unique<DeviceBuffer>(trlweBytes);
        deviceOutput[i] = std::make_unique<DeviceBuffer>(trlweBytes);
        CuSafeCall(cudaMemcpy(deviceSelectors[i]->data, &selectors[i],
                              trgswFftBytes, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(deviceTrue[i]->data, &trueInputs[i], trlweBytes,
                              cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(deviceFalse[i]->data, &falseInputs[i], trlweBytes,
                              cudaMemcpyHostToDevice));
        outputs[i] = static_cast<T*>(deviceOutput[i]->data);
        selectorPointers[i] = static_cast<const double*>(deviceSelectors[i]->data);
        truePointers[i] = static_cast<T*>(deviceTrue[i]->data);
        falsePointers[i] = static_cast<T*>(deviceFalse[i]->data);
    }

    DeviceBuffer deviceOutputs(sizeof(outputs));
    DeviceBuffer deviceSelectorPointers(sizeof(selectorPointers));
    DeviceBuffer deviceTruePointers(sizeof(truePointers));
    DeviceBuffer deviceFalsePointers(sizeof(falsePointers));
    CuSafeCall(cudaMemcpy(deviceOutputs.data, outputs.data(), sizeof(outputs),
                          cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceSelectorPointers.data, selectorPointers.data(),
                          sizeof(selectorPointers), cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceTruePointers.data, truePointers.data(),
                          sizeof(truePointers), cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceFalsePointers.data, falsePointers.data(),
                          sizeof(falsePointers), cudaMemcpyHostToDevice));
    cufhe::CMUXTFHEppFFTBatchKernel(
        static_cast<T* const*>(deviceOutputs.data),
        static_cast<const double* const*>(deviceSelectorPointers.data),
        static_cast<T* const*>(deviceTruePointers.data),
        static_cast<T* const*>(deviceFalsePointers.data), count,
        cudaStreamDefault, gpu);
    CuSafeCall(cudaDeviceSynchronize());

    for (unsigned i = 0; i < count; ++i) {
        TRLWE output;
        CuSafeCall(cudaMemcpy(&output, outputs[i], trlweBytes,
                              cudaMemcpyDeviceToHost));
        if (!checkDecryption(output, expected[i], secretKey, "CUDA CMUX batch",
                             gpu))
            return false;
    }
    return true;
}

// RAM/ROM addresses are not directly encrypted TRGSWs: they are produced by
// CircuitBootstrappingWithInv.  Exercise that exact split-Fourier selector
// representation here, so a CPU/CUDA CMUX match covers the integration path
// rather than only the simpler trgswSymEncrypt case above.
bool runCircuitBootstrapSelectorCMUX(const TFHEpp::SecretKey& secretKey,
                                     unsigned gpu)
{
    using InputP = TFHEpp::lvl0param;
    using BootstrapP = TFHEpp::lvl02param;
    using PrivateP = TFHEpp::lvl21param;

    TFHEpp::EvalKey evalKey;
    evalKey.emplacebkfft<BootstrapP>(secretKey);
    evalKey.emplaceprivksk4cb<PrivateP>(secretKey);

    TFHEpp::TLWE<InputP> encryptedBit;
    TFHEpp::tlweSymEncrypt<InputP>(encryptedBit, InputP::μ,
                                   secretKey.key.get<InputP>());
    TRGSWFFT selector, invertedSelector;
    TFHEpp::CircuitBootstrappingWithInv<BootstrapP, PrivateP>(
        selector, invertedSelector, encryptedBit, evalKey);

    const auto trueBits = bitPattern(701);
    const auto falseBits = bitPattern(907);
    const TRLWE trueInput = encryptBits(trueBits, secretKey);
    const TRLWE falseInput = encryptBits(falseBits, secretKey);
    TRLWE cpuOutput;
    TFHEpp::CMUXFFT<P>(cpuOutput, selector, trueInput, falseInput);
    const auto expected =
        TFHEpp::trlweSymDecrypt<P>(cpuOutput, secretKey.key.get<P>());

    DeviceBuffer deviceSelector(trgswFftBytes);
    DeviceBuffer deviceTrue(trlweBytes);
    DeviceBuffer deviceFalse(trlweBytes);
    DeviceBuffer deviceOutput(trlweBytes);
    CuSafeCall(cudaMemcpy(deviceSelector.data, &selector, trgswFftBytes,
                          cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceTrue.data, &trueInput, trlweBytes,
                          cudaMemcpyHostToDevice));
    CuSafeCall(cudaMemcpy(deviceFalse.data, &falseInput, trlweBytes,
                          cudaMemcpyHostToDevice));
    cufhe::CMUXTFHEppFFTkernel(
        static_cast<T*>(deviceOutput.data),
        static_cast<const double*>(deviceSelector.data),
        static_cast<T*>(deviceTrue.data), static_cast<T*>(deviceFalse.data),
        cudaStreamDefault, gpu);
    CuSafeCall(cudaDeviceSynchronize());
    TRLWE gpuOutput;
    CuSafeCall(cudaMemcpy(&gpuOutput, deviceOutput.data, trlweBytes,
                          cudaMemcpyDeviceToHost));
    return checkDecryption(gpuOutput, expected, secretKey,
                           "CUDA CMUX circuit-bootstrap selector", gpu);
}

bool runOnDevice(unsigned gpu, const TFHEpp::SecretKey& secretKey)
{
    CuSafeCall(cudaSetDevice(gpu));
    return runSingleCMUX(false, InPlaceMode::None, secretKey, gpu) &&
           runSingleCMUX(true, InPlaceMode::None, secretKey, gpu) &&
           runSingleCMUX(false, InPlaceMode::True, secretKey, gpu) &&
           runSingleCMUX(true, InPlaceMode::True, secretKey, gpu) &&
           runSingleCMUX(false, InPlaceMode::False, secretKey, gpu) &&
           runSingleCMUX(true, InPlaceMode::False, secretKey, gpu) &&
           runInPlaceRamChain<8>(secretKey, gpu) &&
           runInPlaceRamChain<32>(secretKey, gpu) &&
           runIndependentBatch(secretKey, gpu) &&
           runCircuitBootstrapSelectorCMUX(secretKey, gpu);
}

}  // namespace

int main()
{
    int deviceCount = 0;
    CuSafeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "No CUDA device is available\n";
        return EXIT_FAILURE;
    }

    const unsigned testedDevices = static_cast<unsigned>(std::min(deviceCount, 2));
    cufhe::SetGPUNum(testedDevices);
    cufhe::Initialize();

    TFHEpp::SecretKey secretKey;
    bool passed = true;
    for (unsigned gpu = 0; gpu < testedDevices; ++gpu)
        passed = runOnDevice(gpu, secretKey) && passed;

    std::cout << "TFHEpp FFT CMUX CPU/CUDA regression "
              << (passed ? "passed" : "failed") << " on " << testedDevices
              << " GPU(s)\n";
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
