#pragma once

#include <memory>
#include <type_traits>
#include <tfhe/cloudkey.hpp>

namespace cufhe {

// TFHEpp's ordinary lvl01/lvl02 bootstrapping keys use the subset chain,
// while EvalKey::emplaceiksk now uses the independent chain. Full gate key
// switches must use the same secrets as the ordinary bootstrapping key.
// Select the subset switch for lvl10 when the GPU build enables it.
template <class P>
void EmplaceGateKeySwitchingKey(TFHEpp::EvalKey& evaluation,
                                const TFHEpp::SecretKey& secret)
{
#ifdef USE_SUBSET_KEY
    if constexpr (std::is_same_v<P, TFHEpp::lvl10param>) {
        evaluation.template emplacesubiksk<P>(secret);
        return;
    }
#endif
    auto& key = evaluation.template get<TFHEpp::KeySwitchingKey<P>>();
    if (key) return;
    key = std::make_unique_for_overwrite<TFHEpp::KeySwitchingKey<P>>();
    TFHEpp::ikskgen<P>(*key,
        secret.key.template getSubset<typename P::domainP>(),
        secret.key.template getSubset<typename P::targetP>());
}

}  // namespace cufhe
