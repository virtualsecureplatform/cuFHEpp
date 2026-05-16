#pragma once

#include <cuda_runtime.h>

#include <array>
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

template <class iksP, class brP, class ahP>
void AnnihilateCircuitBootstrapping(typename brP::targetP::T* const out,
                                    const typename iksP::domainP::T* const in,
                                    const cudaStream_t st, const int gpuNum);

}  // namespace cufhe
