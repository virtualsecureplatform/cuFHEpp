#pragma once
#include <array>
#include <cmath>
#include <cstdint>

namespace TFHEpp {
using namespace std;

constexpr uint32_t DEF_n = 500;
constexpr uint32_t DEF_Nbit = 10;
constexpr uint32_t DEF_N = 1 << DEF_Nbit;
constexpr uint32_t DEF_l = 2;
constexpr uint32_t DEF_Bgbit = 10;
constexpr uint32_t DEF_Bg = 1 << DEF_Bgbit;
constexpr uint32_t DEF_t = 8;
constexpr uint32_t DEF_basebit = 2;

constexpr uint32_t DEF_nbarbit = 11;
constexpr uint32_t DEF_nbar = 1 << DEF_nbarbit;
constexpr uint32_t DEF_lbar = 4;
constexpr uint32_t DEF_Bgbitbar = 9;
constexpr uint32_t DEF_Bgbar = 1 << DEF_Bgbitbar;
constexpr uint32_t DEF_tbar = 10;
constexpr uint32_t DEF_basebitlvl21 = 3;

using Keylvl1 = array<uint32_t, DEF_N>;
using Keylvl2 = array<uint64_t, DEF_nbar>;

using TLWElvl0 = array<uint32_t, DEF_n + 1>;
using TLWElvl1 = array<uint32_t, DEF_N + 1>;
using TLWElvl2 = array<uint64_t, DEF_nbar + 1>;

using Polynomiallvl1 = array<uint32_t, DEF_N>;
using Polynomiallvl2 = array<uint64_t, DEF_nbar>;
using PolynomialInFDlvl1 = array<double, DEF_N>;
using PolynomialInFDlvl2 = array<double, DEF_nbar>;

using TRLWElvl1 = array<Polynomiallvl1, 2>;
using TRLWElvl2 = array<Polynomiallvl2, 2>;
using TRLWEInFDlvl1 = array<PolynomialInFDlvl1, 2>;
using TRLWEInFDlvl2 = array<PolynomialInFDlvl2, 2>;
using DecomposedTRLWElvl1 = array<Polynomiallvl1, 2 * DEF_l>;
using DecomposedTRLWElvl2 = array<Polynomiallvl2, 2 * DEF_lbar>;
using DecomposedTRLWEInFDlvl1 = array<PolynomialInFDlvl1, 2 * DEF_l>;
using DecomposedTRLWEInFDlvl2 = array<PolynomialInFDlvl2, 2 * DEF_lbar>;

using TRGSWlvl1 = array<TRLWElvl1, 2 * DEF_l>;
using TRGSWlvl2 = array<TRLWElvl2, 2 * DEF_lbar>;
using TRGSWFFTlvl1 = array<TRLWEInFDlvl1, 2 * DEF_l>;
using TRGSWFFTlvl2 = array<TRLWEInFDlvl2, 2 * DEF_lbar>;

using BootStrappingKeyFFTlvl01 = array<TRGSWFFTlvl1, DEF_n>;
using BootStrappingKeyFFTlvl02 = array<TRGSWFFTlvl2, DEF_n>;

using KeySwitchingKey =
    array<array<array<TLWElvl0, (1 << DEF_basebit) - 1>, DEF_t>, DEF_N>;
using PrivKeySwitchKey =
    array<array<array<array<TRLWElvl1, (1 << DEF_basebitlvl21) - 1>, DEF_tbar>,
                DEF_nbar + 1>,
          2>;

}  // namespace TFHEpp

namespace SPCULIOS{
    using cuKeylvl1 = uint32_t[TFHEpp::DEF_N];
    using cuKeylvl2 = uint64_t[TFHEpp::DEF_nbar];

    using cuTLWElvl0 = uint32_t[TFHEpp::DEF_n + 1];
    using cuTLWElvl1 = uint32_t[TFHEpp::DEF_N + 1];
    using cuTLWElvl2 = uint64_t[TFHEpp::DEF_nbar + 1];

    using cuPolynomiallvl1 = uint32_t[TFHEpp::DEF_N];
    using cuPolynomiallvl2 = uint64_t[TFHEpp::DEF_nbar];
    using cuPolynomialInFDlvl1 = double[TFHEpp::DEF_N];
    using cuPolynomialInFDlvl2 = double[TFHEpp::DEF_nbar];

    using cuTRLWElvl1 = cuPolynomiallvl1[2];
    using cuTRLWElvl2 = cuPolynomiallvl2[2];
    using cuTRLWEInFDlvl1 = cuPolynomialInFDlvl1[2];
    using cuTRLWEInFDlvl2 = cuPolynomialInFDlvl2[2];
    using cuDecomposedTRLWElvl1 = cuPolynomiallvl1[2 * TFHEpp::DEF_l];
    using cuDecomposedTRLWElvl2 = cuPolynomiallvl2[2 * TFHEpp::DEF_lbar];
    using cuDecomposedTRLWEInFDlvl1 = cuPolynomialInFDlvl1[2 * TFHEpp::DEF_l];
    using cuDecomposedTRLWEInFDlvl2 = cuPolynomialInFDlvl2[2 * TFHEpp::DEF_lbar];

    using cuTRGSWlvl1 = cuTRLWElvl1[2 * TFHEpp::DEF_l];
    using cuTRGSWlvl2 = cuTRLWElvl2[2 * TFHEpp::DEF_lbar];
    using cuTRGSWFFTlvl1 = cuTRLWEInFDlvl1[2 * TFHEpp::DEF_l];
    using cuTRGSWFFTlvl2 = cuTRLWEInFDlvl2[2 * TFHEpp::DEF_lbar];

    using cuBootStrappingKeyFFTlvl01 = cuTRGSWFFTlvl1[TFHEpp::DEF_n];
    using cuBootStrappingKeyFFTlvl02 = cuTRGSWFFTlvl2[TFHEpp::DEF_n];

    using cuKeySwitchingKey =
        cuTLWElvl0[TFHEpp::DEF_N][TFHEpp::DEF_t][(1 << TFHEpp::DEF_basebit) - 1];
    using cuPrivKeySwitchKey =
        cuTRLWElvl1[TFHEpp::DEF_nbar][TFHEpp::DEF_tbar][(1 << TFHEpp::DEF_basebitlvl21) - 1];
}