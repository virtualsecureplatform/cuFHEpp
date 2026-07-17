#pragma once

#include <array>

#include <tfhe/circuitbootstrapping.hpp>
#include <tfhe/cloudkey.hpp>
#include <tfhe/cmuxmem.hpp>
#include <tfhe/gate.hpp>
#include <tfhe/gatebootstrapping.hpp>
#include <tfhe/key.hpp>
#include <tfhe/keyswitch.hpp>
#include <tfhe/tlwe.hpp>
#include <tfhe/trgsw.hpp>
#include <tfhe/trlwe.hpp>
#include <tfhe/aes.hpp>

#include "cufhe_gpu.cuh"

namespace cufhe {

template <class brP, class ahP>
void InitializeAES(const TFHEpp::EvalKey& ek, const TFHEpp::SecretKey& sk);

template <class brP, class ahP>
void CleanUpAES();

template <class iksP, class brP, class ahP>
void AESDec(
    std::array<TFHEpp::TLWE<typename brP::targetP>, 128>& plain,
    const std::array<TFHEpp::TLWE<typename iksP::domainP>, 128>& cipher,
    const std::array<std::array<TFHEpp::TLWE<typename brP::targetP>, 128>,
                     TFHEpp::Nr + 1>& expandedkey,
    const TFHEpp::EvalKey& ek, Stream st);

}  // namespace cufhe
