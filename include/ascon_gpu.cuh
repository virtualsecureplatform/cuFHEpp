#pragma once

#include <span>

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
#include <ascon.hpp>

#include "cufhe_gpu.cuh"

namespace cufhe {

template <class brP, class ahP>
void InitializeASCON(const TFHEpp::EvalKey& ek, const TFHEpp::SecretKey& sk);

template <class brP, class ahP>
void CleanUpASCON();

template <class iksP, class brP, class ahP>
void ASCONXOFInitialize(TFHEpp::ASCONState<typename brP::targetP>& state,
                        const TFHEpp::EvalKey& ek, Stream st);

template <class iksP, class brP, class ahP>
void ASCONXOFAbsorb(TFHEpp::ASCONState<typename brP::targetP>& state,
                    std::span<const TFHEpp::TLWE<typename brP::targetP>> input,
                    const TFHEpp::EvalKey& ek, Stream st);

template <class iksP, class brP, class ahP>
void ASCONXOFSqueeze(TFHEpp::ASCONState<typename brP::targetP>& state,
                     std::span<TFHEpp::TLWE<typename brP::targetP>> output,
                     const TFHEpp::EvalKey& ek, Stream st);

template <class iksP, class brP, class ahP>
void ASCONXOF(std::span<TFHEpp::TLWE<typename brP::targetP>> output,
              std::span<const TFHEpp::TLWE<typename brP::targetP>> input,
              const TFHEpp::EvalKey& ek, Stream st);

}  // namespace cufhe
