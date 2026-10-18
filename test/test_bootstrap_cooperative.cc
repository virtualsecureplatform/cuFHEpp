#include <include/tfhepp_keygen.hpp>
#include <include/bootstrap_gpu.cuh>
#include <include/cufhe_gpu.cuh>

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef CUFHE_TEST_PUBLIC_COOPERATIVE
#define COOP_NAND_TEMPLATE BR, TFHEpp::lvl2param::μ, IKS
#else
#define COOP_NAND_TEMPLATE IKS, BR, P::μ
#endif

namespace {
#ifdef CUFHE_TEST_PUBLIC_COOPERATIVE
using P = TFHEpp::lvl0param;
#else
using P = TFHEpp::lvl2param;
#endif
using IKS = TFHEpp::lvl20param;
using BR = TFHEpp::lvl02param;
using T = P::T;
constexpr size_t elements = P::k * P::n + 1;
constexpr size_t stride = elements + 10;
constexpr size_t capacity = 129;
constexpr size_t words = capacity * stride + 2;
constexpr T guard = static_cast<T>(0x129a63e84b7d05c1ULL);

void Check(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void Batch(T* out, const T* a, const T* b, unsigned count, cudaStream_t stream)
{
    cufhe::NandBootstrapBatch<COOP_NAND_TEMPLATE>(
        out, stride, a, stride, b, stride, count, stream, 0);
}
}

// Exercise both sides of the small-batch dispatch (raw SM80/SM89, public SM80),
// including stream-ordered
// scratch lifetime, graph allocation/free nodes, aliases and untouched padding.
int main()
{
    try {
        Check(cudaSetDevice(0));
        cufhe::SetGPUNum(1);
        // A caller-owned current pool must remain untouched by the private cache.
        cudaMemPool_t original_pool = nullptr, caller_pool = nullptr;
        int pool_support = 0;
        Check(cudaDeviceGetAttribute(&pool_support, cudaDevAttrMemoryPoolsSupported, 0));
        constexpr uint64_t caller_threshold = 1234567;
        if (pool_support) {
            Check(cudaDeviceGetMemPool(&original_pool, 0));
            cudaMemPoolProps properties{};
            properties.allocType = cudaMemAllocationTypePinned;
            properties.location.type = cudaMemLocationTypeDevice;
            properties.location.id = 0;
            Check(cudaMemPoolCreate(&caller_pool, &properties));
            uint64_t threshold = caller_threshold;
            Check(cudaMemPoolSetAttribute(caller_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
            int disabled = 0;
            Check(cudaMemPoolSetAttribute(caller_pool, cudaMemPoolReuseAllowOpportunistic, &disabled));
            Check(cudaMemPoolSetAttribute(caller_pool, cudaMemPoolReuseAllowInternalDependencies, &disabled));
            Check(cudaDeviceSetMemPool(0, caller_pool));
        }
        auto check_caller_pool = [&]() {
            if (!pool_support) return;
            cudaMemPool_t current = nullptr;
            uint64_t threshold = 0, high = 0;
            int opportunistic = -1, dependencies = -1;
            Check(cudaDeviceGetMemPool(&current, 0));
            Check(cudaMemPoolGetAttribute(caller_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
            Check(cudaMemPoolGetAttribute(caller_pool, cudaMemPoolAttrReservedMemHigh, &high));
            Check(cudaMemPoolGetAttribute(caller_pool, cudaMemPoolReuseAllowOpportunistic, &opportunistic));
            Check(cudaMemPoolGetAttribute(caller_pool, cudaMemPoolReuseAllowInternalDependencies, &dependencies));
            if (current != caller_pool || threshold != caller_threshold || high != 0 ||
                opportunistic != 0 || dependencies != 0)
                throw std::runtime_error("Caller pool identity/settings/allocation isolation mismatch");
        };
        TFHEpp::SecretKey sk;
        TFHEpp::EvalKey ek(sk);
        ek.emplacebk<BR>(sk);
        cufhe::EmplaceGateKeySwitchingKey<IKS>(ek, sk);
        cufhe::Initialize_lvl02(ek, sk);

        std::vector<T> a(words, guard), b(words, guard), blank(words, guard);
        std::vector<T> reference(words, guard), actual(words);
        for (size_t i = 0; i < capacity; ++i) {
            TFHEpp::TLWE<P> sample;
            TFHEpp::tlweSymEncrypt<P>(
                sample, (i & 1) ? P::μ : -P::μ, sk.key.getSubset<P>());
            std::copy(sample.begin(), sample.end(), a.begin() + 1 + i * stride);
            TFHEpp::tlweSymEncrypt<P>(
                sample, (i & 2) ? P::μ : -P::μ, sk.key.getSubset<P>());
            std::copy(sample.begin(), sample.end(), b.begin() + 1 + i * stride);
        }
        T *da, *db, *dout, *dref;
        for (auto pointer : {&da, &db, &dout, &dref})
            Check(cudaMalloc(pointer, words * sizeof(T)));
        cudaStream_t stream, other;
        Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        Check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking));
        auto upload = [&](T* device, const std::vector<T>& host) {
            Check(cudaMemcpy(device, host.data(), words * sizeof(T), cudaMemcpyHostToDevice));
            // Pageable H2D may return after staging, before DMA completion.
            // Nonblocking test streams do not inherit default-stream ordering.
            Check(cudaStreamSynchronize(nullptr));
        };
        auto equal = [&](T* device, const std::vector<T>& expected) {
            Check(cudaMemcpy(actual.data(), device, words * sizeof(T), cudaMemcpyDeviceToHost));
            if (actual != expected)
                throw std::runtime_error("Full-word/padding reference mismatch");
        };
        upload(da, a);
        upload(db, b);
        upload(dref, blank);
#ifdef USE_CUDA_COOPERATIVE_SCALAR
        // A capacity-sized launch exceeds every cooperative dispatch window,
        // preserving an independent one-CTA reference for scalar dispatch.
        Batch(dref + 1, da + 1, db + 1, capacity, stream);
        upload(dout, blank);
        for (size_t i = 0; i < capacity; ++i)
            cufhe::NandBootstrap<COOP_NAND_TEMPLATE>(
                dout + 1 + i * stride, da + 1 + i * stride,
                db + 1 + i * stride, stream, 0);
        Check(cudaStreamSynchronize(stream));
        std::vector<T> scalar_expected(words);
        Check(cudaMemcpy(scalar_expected.data(), dref, words * sizeof(T), cudaMemcpyDeviceToHost));
        equal(dout, scalar_expected);
        std::cout << "cooperative scalar / ordinary batch reference: PASS" << std::endl;
#else
        for (size_t i = 0; i < capacity; ++i)
            cufhe::NandBootstrap<COOP_NAND_TEMPLATE>(
                dref + 1 + i * stride, da + 1 + i * stride,
                db + 1 + i * stride, stream, 0);
#endif
        Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(reference.data(), dref, words * sizeof(T), cudaMemcpyDeviceToHost));
        size_t compared = 0;
        for (unsigned count : {1, 2, 4, 8, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 32, 35, 36, 37, 42, 43, 44, 48, 53, 54, 55, 63, 64, 65, 71, 72, 73, 96, 107, 108, 109, 127, 128, 129}) {
            for (unsigned placement = 0; placement < 5; ++placement) {
                upload(da, a);
                upload(db, b);
                upload(dout, blank);
                const bool side_b = placement == 2 || placement == 4;
                const bool separate = placement == 0;
                const size_t shift = placement >= 3 ? 1 : 0;
                T* destination = separate ? dout + 1 :
                    (side_b ? db : da) + 1 + shift;
                auto expected = separate ? blank : side_b ? b : a;
                for (unsigned i = 0; i < count; ++i)
                    std::copy_n(reference.begin() + 1 + i * stride, elements,
                                expected.begin() + 1 + shift + i * stride);
                Batch(destination, da + 1, db + 1, count, stream);
                Check(cudaStreamSynchronize(stream));
                equal(separate ? dout : side_b ? db : da, expected);
                if (separate) {
                    equal(da, a);
                    equal(db, b);
                }
                else equal(side_b ? da : db, side_b ? a : b);
                compared += size_t(count) * elements;
            }
            std::cout << "count=" << count
                      << " separate/exact aliases/forward aliases/padding: PASS"
                      << std::endl;
        }
        for (unsigned count : {1, 8, 12, 13, 14, 15, 20, 21, 22, 25, 26, 32, 35, 36, 37, 42, 43, 44, 48, 53, 54, 55, 63, 64, 65, 71, 72, 73, 107, 108, 109, 127, 128}) {
            upload(da, a);
            upload(db, b);
            upload(dref, a);
            // Warm the API before capture, then restore the original input.
            Batch(da + 1, da + 1, db + 1, count, stream);
            Check(cudaStreamSynchronize(stream));
            upload(da, a);
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
            Batch(da + 1, da + 1, db + 1, count, stream);
            Check(cudaStreamEndCapture(stream, &graph));
            Check(cudaGraphInstantiateWithFlags(&executable, graph, 0));
            for (int depth = 0; depth < 4; ++depth) {
                Check(cudaGraphLaunch(executable, stream));
                for (unsigned i = 0; i < count; ++i)
                    cufhe::NandBootstrap<COOP_NAND_TEMPLATE>(
                        dref + 1 + i * stride, dref + 1 + i * stride,
                        db + 1 + i * stride, other, 0);
                Check(cudaDeviceSynchronize());
                std::vector<T> expected(words);
                Check(cudaMemcpy(expected.data(), dref, words * sizeof(T), cudaMemcpyDeviceToHost));
                equal(da, expected);
                compared += size_t(count) * elements;
            }
            Check(cudaGraphExecDestroy(executable));
            Check(cudaGraphDestroy(graph));
            std::cout << "count=" << count
                      << " captured in-place chain / four replays: PASS" << std::endl;
        }
        // Concurrent calls must have independent scratch, even when their
        // combined grids exceed the cooperative residency limit of one call.
        upload(da, a);
        upload(db, b);
        upload(dout, blank);
        Batch(dout + 1, da + 1, db + 1, 8, stream);
        Batch(dout + 1 + 8 * stride, da + 1 + 8 * stride,
              db + 1 + 8 * stride, 8, other);
        Check(cudaDeviceSynchronize());
        auto expected = blank;
        for (unsigned i = 0; i < 16; ++i)
            std::copy_n(reference.begin() + 1 + i * stride, elements,
                        expected.begin() + 1 + i * stride);
        equal(dout, expected);
        compared += 16 * elements;
        std::cout << "concurrent independent small batches: PASS" << std::endl;
        // Full-word fixtures exercise cooperative paired schedules at arithmetic
        // boundaries independently of encryption validity. Compare every word
        // with the original scalar schedule, including all surrounding padding.
        for (int pattern = 0; pattern < 4; ++pattern) {
            auto fixture_a = blank, fixture_b = blank;
            for (size_t i = 0; i < 43; ++i)
                for (size_t j = 0; j < elements; ++j) {
                    T value = pattern == 0 ? T{0} : pattern == 1 ? ~T{0} :
                              pattern == 2 ? (T{1} << (std::numeric_limits<T>::digits - 1)) :
                              (T{1} << ((i + j) % std::numeric_limits<T>::digits));
                    fixture_a[1 + i * stride + j] = value;
                    fixture_b[1 + i * stride + j] = pattern == 2 ? value - 1 :
                                                   pattern == 3 ? ~value : value;
                }
            upload(da, fixture_a);
            upload(db, fixture_b);
            upload(dout, blank);
            upload(dref, blank);
            Batch(dout + 1, da + 1, db + 1, 43, stream);
            for (size_t i = 0; i < 43; ++i)
                cufhe::NandBootstrap<COOP_NAND_TEMPLATE>(
                    dref + 1 + i * stride, da + 1 + i * stride,
                    db + 1 + i * stride, other, 0);
            Check(cudaDeviceSynchronize());
            std::vector<T> fixture_expected(words);
            Check(cudaMemcpy(fixture_expected.data(), dref, words * sizeof(T), cudaMemcpyDeviceToHost));
            equal(dout, fixture_expected);
            equal(da, fixture_a);
            equal(db, fixture_b);
            compared += 43 * elements;
            for (unsigned count : {25, 32}) {
                upload(dout, blank);
                Batch(dout + 1, da + 1, db + 1, count, stream);
                Check(cudaStreamSynchronize(stream));
                auto medium_expected = blank;
                for (size_t i = 0; i < count; ++i)
                    std::copy_n(fixture_expected.begin() + 1 + i * stride, elements,
                                medium_expected.begin() + 1 + i * stride);
                equal(dout, medium_expected);
                equal(da, fixture_a);
                equal(db, fixture_b);
                compared += count * elements;
            }
        }
        std::cout << "paired full-word boundary fixtures / padding: PASS" << std::endl;
        cufhe::CleanUp_lvl02();
        check_caller_pool();
        for (int iteration = 0; iteration < 3; ++iteration) {
            cufhe::Initialize_lvl02(ek, sk);
            for (unsigned count : {25, 42, 64}) {
                upload(da, a);
                upload(db, b);
                upload(dout, blank);
                upload(dref, blank);
                Batch(dout + 1, da + 1, db + 1, count, stream);
                Batch(dref + 1, da + 1 + count * stride, db + 1 + count * stride, count, other);
                Check(cudaDeviceSynchronize());
                auto first = blank, second = blank;
                for (unsigned i = 0; i < count; ++i) {
                    std::copy_n(reference.begin() + 1 + i * stride, elements,
                                first.begin() + 1 + i * stride);
                    std::copy_n(reference.begin() + 1 + (i + count) * stride, elements,
                                second.begin() + 1 + i * stride);
                }
                equal(dout, first);
                equal(dref, second);
                compared += 2 * count * elements;
            }
            cufhe::CleanUp_lvl02();
            check_caller_pool();
        }
        if (pool_support) {
            Check(cudaDeviceSetMemPool(0, original_pool));
            Check(cudaMemPoolDestroy(caller_pool));
        }
        std::cout << "caller-pool isolation / three cleanup-reinitialize cycles: PASS" << std::endl;
        for (auto device : {da, db, dout, dref}) Check(cudaFree(device));
        Check(cudaStreamDestroy(stream));
        Check(cudaStreamDestroy(other));
        int visible_devices = 0;
        Check(cudaGetDeviceCount(&visible_devices));
        if (visible_devices >= 2) {
            cufhe::SetGPUNum(2);
            cufhe::Initialize_lvl02(ek, sk);
            for (int gpu = 0; gpu < 2; ++gpu) {
                Check(cudaSetDevice(gpu));
                T *ma, *mb, *mo;
                for (auto pointer : {&ma, &mb, &mo})
                    Check(cudaMalloc(pointer, words * sizeof(T)));
                upload(ma, a);
                upload(mb, b);
                upload(mo, blank);
                cudaStream_t device_stream;
                Check(cudaStreamCreateWithFlags(&device_stream, cudaStreamNonBlocking));
                for (unsigned count : {25, 42, 64}) {
                    upload(mo, blank);
                    cufhe::NandBootstrapBatch<COOP_NAND_TEMPLATE>(
                        mo + 1, stride, ma + 1, stride, mb + 1, stride,
                        count, device_stream, gpu);
                    Check(cudaStreamSynchronize(device_stream));
                    auto device_expected = blank;
                    for (unsigned i = 0; i < count; ++i)
                        std::copy_n(reference.begin() + 1 + i * stride, elements,
                                    device_expected.begin() + 1 + i * stride);
                    equal(mo, device_expected);
                    compared += count * elements;
                }
                for (auto device : {ma, mb, mo}) Check(cudaFree(device));
                Check(cudaStreamDestroy(device_stream));
            }
            cufhe::CleanUp_lvl02();
            cufhe::SetGPUNum(1);
            Check(cudaSetDevice(0));
            std::cout << "two-device initialization / private scratch / cleanup: PASS" << std::endl;
        }
        std::cout << "reference_coefficients=" << compared << " PASS" << std::endl;
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
