#pragma once

#include <array>

#include <circuitbootstrapping.hpp>
#include <cloudkey.hpp>
#include <cmuxmem.hpp>
#include <gate.hpp>
#include <gatebootstrapping.hpp>
#include <key.hpp>
#include <keyswitch.hpp>
#include <tlwe.hpp>
#include <trgsw.hpp>
#include <trlwe.hpp>
#include <aes.hpp>

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
