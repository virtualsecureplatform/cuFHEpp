#pragma once

#include <span>

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
#include <tfhe/ascon.hpp>

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
