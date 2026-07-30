#pragma once

#include <cmath>

// libstdc++ does not make std::pow constexpr until C++26. HIP otherwise emits
// dynamic device initializers for TFHEpp's inline noise constants, which are
// incompatible with relocatable device code in current ROCm toolchains. Keep
// the normal floating-point overload while making integer powers constant
// expressions during HIP compilation.
namespace std {
constexpr double cufhe_hip_pow(double base, int exponent)
{
    double result = 1.0;
    if (exponent < 0) {
        base = 1.0 / base;
        exponent = -exponent;
    }
    while (exponent != 0) {
        if ((exponent & 1) != 0) result *= base;
        base *= base;
        exponent >>= 1;
    }
    return result;
}

inline double cufhe_hip_pow(double base, double exponent)
{
    return __builtin_pow(base, exponent);
}
}  // namespace std

#define pow cufhe_hip_pow
