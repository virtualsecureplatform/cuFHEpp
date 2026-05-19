#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <params.hpp>

#include "ntt_small_modulus.cuh"

namespace TFHEpp {
struct SecretKey;
}

namespace cufhe {

template <class P>
using CBswitchingKeyPolynomial = std::array<TFHEpp::TRGSW<P>, P::k>;

template <class P>
void CBswitchingKeyPolynomialGen(CBswitchingKeyPolynomial<P>& cbsk,
                                 const TFHEpp::Key<P>& key);

template <class P>
void CBswitchingKeyPolynomialGen(CBswitchingKeyPolynomial<P>& cbsk,
                                 const TFHEpp::SecretKey& sk);

template <class P>
void CBswitchingKeyPolynomialToDevice(
    const CBswitchingKeyPolynomial<P>& cbsk, const int gpuNum);

template <class P>
void DeleteCBswitchingKey(const int gpuNum);

template <class brP, class ahP>
void AnnihilateCircuitBootstrapping(typename brP::targetP::T* const out,
                                    const typename brP::domainP::T* const in,
                                    const cudaStream_t st, const int gpuNum);

template <class brP, class ahP>
void AnnihilateCircuitBootstrappingWithWorkspace(
    typename brP::targetP::T* const out,
    const typename brP::domainP::T* const in,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temptrlwe, const cudaStream_t st,
    const int gpuNum);

template <class brP, class ahP>
void AnnihilateCircuitBootstrappingBatchWithWorkspace(
    typename brP::targetP::T* const out, const size_t out_stride,
    const typename brP::domainP::T* const in, const size_t in_stride,
    typename brP::targetP::T* const acc,
    typename brP::targetP::T* const temptrlwe,
    const size_t batch_count, const cudaStream_t st, const int gpuNum);

template <class iksP, class brP, class ahP>
void AnnihilateCircuitBootstrapping(typename brP::targetP::T* const out,
                                    const typename iksP::domainP::T* const in,
                                    const cudaStream_t st, const int gpuNum);

}  // namespace cufhe
