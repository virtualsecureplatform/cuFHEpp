#pragma once
#include <include/ntt_small_modulus.cuh>

namespace cufhe {

// Four products plus c fit 130 bits. The high carry contributes -carry*2^32
// modulo the existing Goldilocks prime; it is folded with the high-high limb.
template <int NativeAdd = 0>
__host__ __device__ __forceinline__ uint64_t small_mod64_madd4(
    uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1,
    uint64_t a2, uint64_t b2, uint64_t a3, uint64_t b3, uint64_t c)
{
    const unsigned __int128 first = static_cast<unsigned __int128>(a0) * b0 + c;
    const unsigned __int128 p1 = static_cast<unsigned __int128>(a1) * b1;
    const unsigned __int128 p2 = static_cast<unsigned __int128>(a2) * b2;
    const unsigned __int128 p3 = static_cast<unsigned __int128>(a3) * b3;
    uint64_t lo, hi, carry;
#if defined(__CUDA_ARCH__) && !defined(CUFHE_USE_HIP)
    uint32_t top;
    asm("{\n\t"
        ".reg .u64 la, ha, lb, hb;\n\t.reg .u32 ta, tb, t;\n\t"
        "add.cc.u64 la, %3, %5;\n\taddc.cc.u64 ha, %4, %6;\n\taddc.u32 ta, 0, 0;\n\t"
        "add.cc.u64 lb, %7, %9;\n\taddc.cc.u64 hb, %8, %10;\n\taddc.u32 tb, 0, 0;\n\t"
        "add.cc.u64 la, la, lb;\n\taddc.cc.u64 ha, ha, hb;\n\taddc.u32 t, ta, tb;\n\t"
        "mov.u64 %0, la;\n\tmov.u64 %1, ha;\n\tmov.u32 %2, t;\n\t}"
        : "=l"(lo), "=l"(hi), "=r"(top)
        : "l"(static_cast<uint64_t>(first)), "l"(static_cast<uint64_t>(first >> 64)),
          "l"(static_cast<uint64_t>(p1)), "l"(static_cast<uint64_t>(p1 >> 64)),
          "l"(static_cast<uint64_t>(p2)), "l"(static_cast<uint64_t>(p2 >> 64)),
          "l"(static_cast<uint64_t>(p3)), "l"(static_cast<uint64_t>(p3 >> 64)));
    carry = top;
#else
    auto sum = first;
    carry = 0;
    auto previous = sum; sum += p1; carry += sum < previous;
    previous = sum; sum += p2; carry += sum < previous;
    previous = sum; sum += p3; carry += sum < previous;
    lo = static_cast<uint64_t>(sum); hi = static_cast<uint64_t>(sum >> 64);
#endif
    const uint64_t subtraction = (hi >> 32) | (carry << 32);
    const uint64_t difference = lo - subtraction;
    const uint64_t corrected = difference - (lo < subtraction ? uint64_t{UINT32_MAX} : 0);
    const uint64_t hi_lo = static_cast<uint32_t>(hi);
    return small_mod64_add<NativeAdd>(corrected, (hi_lo << 32) - hi_lo);
}

#if defined(CUFHE_GPU_DEVICE_COMPILER) && !defined(CUFHE_USE_HIP)
#define CUFHE_HAS_PAIR_NTT64 1

template <int Power, bool Inverse, bool NativeSubtract = false, int NativeAdd = 0,
          bool UniformRoot = false>
__device__ __forceinline__ void PairButterfly64(
    uint64_t& u0, uint64_t& v0, uint64_t& u1, uint64_t& v1,
    const uint64_t* roots, int index)
{
    const uint64_t root = UniformRoot
        ? (Inverse ? d_const_inverse_root_64[index] : d_const_forward_root_64[index])
        : __ldg(roots + index);
    if constexpr (Inverse) {
        SmallGentlemanSandeUnit<Power, 64, NativeAdd>(u0, v0, root);
        SmallGentlemanSandeUnit<Power, 64, NativeAdd>(u1, v1, root);
    } else {
        SmallCooleyTukeyUnit<Power, 64, NativeSubtract, NativeAdd>(u0, v0, root);
        SmallCooleyTukeyUnit<Power, 64, NativeSubtract, NativeAdd>(u1, v1, root);
    }
}

template <int Power, bool NativeSubtract = false, int NativeAdd = 0, bool UniformRoot = false>
__device__ __forceinline__ void SmallForwardNTTPair64(
    uint64_t* sh0, uint64_t* sh1, const uint64_t* roots, int tid)
{
    static_assert(Power == TFHEpp::lvl2param::nbit && Power >= 8 && (Power - 6) % 2 == 0);
    constexpr int Quarter = 1 << (Power - 2);
    uint64_t* work = tid < Quarter ? sh0 : sh1;
    const int ptid = tid & (Quarter - 1);
    int t = 1 << (Power - 1), shift = Power - 1, m = 1;
#pragma unroll
    for (int stage = 0; stage < (Power - 6) / 2; ++stage) {
        const int half_t = t >> 1;
        const int group = ptid >> (shift - 1);
        const int address = (group << (shift + 1)) + (ptid & (half_t - 1));
        const int r0 = m + group, rlo = 2 * m + 2 * group, rhi = rlo + 1;
        uint64_t a = work[address], b = work[address + half_t];
        uint64_t c = work[address + t], d = work[address + t + half_t];
        if (stage == 0) {
            SmallCooleyTukeyGoldilocksRoot1Unit<NativeAdd>(a, c);
            SmallCooleyTukeyGoldilocksRoot1Unit<NativeAdd>(b, d);
            SmallCooleyTukeyUnit<Power, 64, false, NativeAdd>(a, b, d_const_forward_root_64[rlo]);
            SmallCooleyTukeyUnit<Power, 64, false, NativeAdd>(c, d, d_const_forward_root_64[rhi]);
        } else {
            SmallCooleyTukeyRadix4Unit<Power, 64, false, NativeAdd>(a, b, c, d,
                d_const_forward_root_64[r0], d_const_forward_root_64[rlo],
                d_const_forward_root_64[rhi]);
        }
        work[address] = a; work[address + half_t] = b;
        work[address + t] = c; work[address + t + half_t] = d;
        t >>= 2; shift -= 2; m <<= 2;
        __syncthreads();
    }
    const int address = ((tid >> 5) << 5) + tid;
    uint64_t u0 = sh0[address], v0 = sh0[address + 32];
    uint64_t u1 = sh1[address], v1 = sh1[address + 32];
    PairButterfly64<Power, false, NativeSubtract, NativeAdd, UniformRoot>(u0, v0, u1, v1, roots, m + (tid >> 5));
    const bool upper = (tid & 16) != 0;
    auto pu0 = SmallNTTShuffleXor(u0, 16), pv0 = SmallNTTShuffleXor(v0, 16);
    auto pu1 = SmallNTTShuffleXor(u1, 16), pv1 = SmallNTTShuffleXor(v1, 16);
    u0 = upper ? pv0 : u0; v0 = upper ? v0 : pu0;
    u1 = upper ? pv1 : u1; v1 = upper ? v1 : pu1;
    shift = 4; m <<= 1;
    PairButterfly64<Power, false, NativeSubtract, NativeAdd>(u0, v0, u1, v1, roots, m + (tid >> shift));
#pragma unroll
    for (int mask = 8; mask >= 1; mask >>= 1) {
        const bool high = (tid & mask) != 0;
        const auto p0 = SmallNTTShuffleXor(high ? u0 : v0, mask);
        const auto p1 = SmallNTTShuffleXor(high ? u1 : v1, mask);
        if (high) { u0 = p0; u1 = p1; } else { v0 = p0; v1 = p1; }
        --shift; m <<= 1;
        PairButterfly64<Power, false, NativeSubtract, NativeAdd>(u0, v0, u1, v1, roots, m + (tid >> shift));
    }
    __syncwarp();
    sh0[2 * tid] = u0; sh0[2 * tid + 1] = v0;
    sh1[2 * tid] = u1; sh1[2 * tid + 1] = v1;
    __syncthreads();
}

template <int Power, int NativeAdd = 0, bool UniformRoot = false>
__device__ __forceinline__ void SmallInverseNTTPair64(
    uint64_t* sh0, uint64_t* sh1, const uint64_t* roots,
    uint64_t n_inverse, int tid)
{
    static_assert(Power == TFHEpp::lvl2param::nbit && Power >= 8 && (Power - 6) % 2 == 0);
    uint64_t u0 = sh0[2 * tid], v0 = sh0[2 * tid + 1];
    uint64_t u1 = sh1[2 * tid], v1 = sh1[2 * tid + 1];
    int shift = 0, m = 1 << (Power - 1);
    PairButterfly64<Power, true, false, NativeAdd>(u0, v0, u1, v1, roots, m + (tid >> shift));
#pragma unroll
    for (int mask = 1; mask <= 8; mask <<= 1) {
        const bool high = (tid & mask) != 0;
        const auto p0 = SmallNTTShuffleXor(high ? u0 : v0, mask);
        const auto p1 = SmallNTTShuffleXor(high ? u1 : v1, mask);
        if (high) { u0 = p0; u1 = p1; } else { v0 = p0; v1 = p1; }
        ++shift; m >>= 1;
        PairButterfly64<Power, true, false, NativeAdd>(u0, v0, u1, v1, roots, m + (tid >> shift));
    }
    const bool upper = (tid & 16) != 0;
    const auto pu0 = SmallNTTShuffleXor(u0, 16), pv0 = SmallNTTShuffleXor(v0, 16);
    const auto pu1 = SmallNTTShuffleXor(u1, 16), pv1 = SmallNTTShuffleXor(v1, 16);
    u0 = upper ? pv0 : u0; v0 = upper ? v0 : pu0;
    u1 = upper ? pv1 : u1; v1 = upper ? v1 : pu1;
    ++shift; m >>= 1;
    PairButterfly64<Power, true, false, NativeAdd, UniformRoot>(u0, v0, u1, v1, roots, m + (tid >> shift));
    const int address = ((tid >> 5) << 5) + tid;
    __syncwarp();
    sh0[address] = u0; sh0[address + 32] = v0;
    sh1[address] = u1; sh1[address + 32] = v1;
    __syncthreads();
    constexpr int Quarter = 1 << (Power - 2);
    uint64_t* work = tid < Quarter ? sh0 : sh1;
    const int ptid = tid & (Quarter - 1);
    int t = 64; shift = 6; m >>= 1;
#pragma unroll
    for (int stage = 0; stage < (Power - 6) / 2; ++stage) {
        const int group = ptid >> shift, offset = ptid & (t - 1);
        const int addr = (group << (shift + 2)) + offset;
        const int rlo = m + 2 * group, rhi = rlo + 1, r1 = (m >> 1) + group;
        uint64_t a = work[addr], b = work[addr + t];
        uint64_t c = work[addr + 2 * t], d = work[addr + 3 * t];
        if (stage == (Power - 6) / 2 - 1) {
            SmallGentlemanSandeUnit<Power, 64, NativeAdd>(a, b, d_const_inverse_root_64[rlo]);
            SmallGentlemanSandeUnit<Power, 64, NativeAdd>(c, d, d_const_inverse_root_64[rhi]);
            SmallGentlemanSandeScaledUnit<Power, 64, NativeAdd>(a, c, d_const_inverse_root_64[r1], n_inverse);
            SmallGentlemanSandeScaledUnit<Power, 64, NativeAdd>(b, d, d_const_inverse_root_64[r1], n_inverse);
        } else {
            SmallGentlemanSandeRadix4Unit<Power, 64, NativeAdd>(a, b, c, d,
                d_const_inverse_root_64[rlo], d_const_inverse_root_64[rhi],
                d_const_inverse_root_64[r1]);
        }
        work[addr] = a; work[addr + t] = b;
        work[addr + 2 * t] = c; work[addr + 3 * t] = d;
        t <<= 2; shift += 2; m >>= 2;
        __syncthreads();
    }
}

// The final scalar inverse stage has N/4 workers. Retain its low/even-quarter
// outputs, publish the other half, then return two coefficients to every worker.
template <int Power, int NativeAdd = 0>
__device__ __forceinline__ void SmallInverseNTTReturnPair64(
    uint64_t* sh, const uint64_t* roots, uint64_t n_inverse, int tid,
    uint64_t* output_pair)
{
    static_assert(Power == TFHEpp::lvl2param::nbit && Power >= 8 && (Power - 6) % 2 == 0);
    constexpr int N = 1 << Power, Quarter = N / 4;
    uint64_t u = sh[2 * tid], v = sh[2 * tid + 1];
    int shift = 0, m = N / 2;
    SmallGentlemanSandeUnit<Power, 64, NativeAdd>(u, v, __ldg(roots + m + tid));
    ++shift; m >>= 1;
#pragma unroll
    for (int mask = 1; mask <= 8; mask <<= 1) {
        const bool high = (tid & mask) != 0;
        const uint64_t peer = SmallNTTShuffleXor(high ? u : v, mask);
        if (high) u = peer; else v = peer;
        SmallGentlemanSandeUnit<Power, 64, NativeAdd>(
            u, v, __ldg(roots + m + (tid >> shift)));
        ++shift; m >>= 1;
    }
    const bool upper = (tid & 16) != 0;
    const uint64_t peer_u = SmallNTTShuffleXor(u, 16);
    const uint64_t peer_v = SmallNTTShuffleXor(v, 16);
    u = upper ? peer_v : u; v = upper ? v : peer_u;
    SmallGentlemanSandeUnit<Power, 64, NativeAdd>(
        u, v, __ldg(roots + m + (tid >> shift)));
    const int boundary = ((tid >> 5) << 5) + tid;
    __syncwarp();
    sh[boundary] = u; sh[boundary + 32] = v;
    __syncthreads();
    int t = 64; shift = 6; m >>= 1;
#pragma unroll
    for (int stage = 0; stage < (Power - 6) / 2; ++stage) {
        if (tid < Quarter) {
            const int group = tid >> shift, offset = tid & (t - 1);
            const int address = (group << (shift + 2)) + offset;
            const int rlo = m + 2 * group, rhi = rlo + 1, r1 = (m >> 1) + group;
            uint64_t a = sh[address], b = sh[address + t];
            uint64_t c = sh[address + 2 * t], d = sh[address + 3 * t];
            if (stage == (Power - 6) / 2 - 1) {
                SmallGentlemanSandeUnit<Power, 64, NativeAdd>(a, b, d_const_inverse_root_64[rlo]);
                SmallGentlemanSandeUnit<Power, 64, NativeAdd>(c, d, d_const_inverse_root_64[rhi]);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
                SmallGentlemanSandeGoldilocksScaledUnit<Power, NativeAdd>(a, c, d_const_inverse_root_64[r1]);
                SmallGentlemanSandeGoldilocksScaledUnit<Power, NativeAdd>(b, d, d_const_inverse_root_64[r1]);
#else
                SmallGentlemanSandeScaledUnit<Power, 64, NativeAdd>(a, c, d_const_inverse_root_64[r1], n_inverse);
                SmallGentlemanSandeScaledUnit<Power, 64, NativeAdd>(b, d, d_const_inverse_root_64[r1], n_inverse);
#endif
                output_pair[0] = a; output_pair[1] = c;
            } else {
                SmallGentlemanSandeRadix4Unit<Power, 64, NativeAdd>(a, b, c, d,
                    d_const_inverse_root_64[rlo], d_const_inverse_root_64[rhi],
                    d_const_inverse_root_64[r1]);
                sh[address] = a; sh[address + 2 * t] = c;
            }
            sh[address + t] = b; sh[address + 3 * t] = d;
        }
        t <<= 2; shift += 2; m >>= 2;
        __syncthreads();
    }
    if (tid >= Quarter) {
        output_pair[0] = sh[tid]; output_pair[1] = sh[tid + N / 2];
    }
}
#endif
}  // namespace cufhe
