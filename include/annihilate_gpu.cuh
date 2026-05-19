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
using EvalAutoKeyPolynomial = std::array<TFHEpp::HalfTRGSW<P>, P::k>;

template <class P>
using AnnihilateKeyPolynomial =
    std::array<EvalAutoKeyPolynomial<P>, P::nbit>;

template <class P>
void AnnihilateKeyPolynomialGen(AnnihilateKeyPolynomial<P>& ahk,
                                const TFHEpp::Key<P>& key);

template <class P>
void AnnihilateKeyPolynomialGen(AnnihilateKeyPolynomial<P>& ahk,
                                const TFHEpp::SecretKey& sk);

template <class P>
void AnnihilateKeyPolynomialToDevice(
    const AnnihilateKeyPolynomial<P>& ahk, const int gpuNum);

template <class P>
void DeleteAnnihilateKey(const int gpuNum);

template <class P>
void AnnihilateKeySwitching(typename P::T* const out,
                            const typename P::T* const in,
                            const cudaStream_t st, const int gpuNum);

template <class P>
void AnnihilateKeySwitchingWithWorkspace(
    typename P::T* const out, const typename P::T* const in,
    typename P::T* const evaledauto, const cudaStream_t st, const int gpuNum);

template <class P>
void AnnihilateKeySwitchingBatchWithWorkspace(
    typename P::T* const out, const size_t out_stride,
    const typename P::T* const in, const size_t in_stride,
    typename P::T* const evaledauto, const size_t evaledauto_stride,
    const size_t batch_count, const cudaStream_t st, const int gpuNum);

}  // namespace cufhe
