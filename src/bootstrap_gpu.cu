/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include <include/bootstrap_gpu.cuh>
#include <include/gatebootstrapping_gpu.cuh>
#include <include/keyswitch_gpu.cuh>
#include <include/cufhe_gpu.cuh>
#include <include/details/error_gpu.cuh>
#include <include/details/utils_gpu.cuh>
#include <include/ntt_gpu/ntt.cuh>
#include <limits>
#include <vector>
#include <algorithm>

namespace cufhe {

using namespace std;
using namespace TFHEpp;

vector<NTTValue*> bk_ntts;
vector<CuNTTHandler<>*> ntt_handlers;

template<class P = TFHEpp::lvl1param>
__global__ void __TRGSW2NTT__(NTTValue* const bk_ntt, const typename P::T* const bk,
                              CuNTTHandler<> ntt)
{
    __shared__ NTTValue sh_temp[P::n];
    const int index = blockIdx.z * ((P::k+1) * P::l * (P::k+1) * P::n) +
                      blockIdx.y * (P::k+1) * P::n + blockIdx.x * P::n;
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    // Load and convert each element: Torus -> NTT mod
    for (int i = tid; i < P::n; i += bdim) {
        sh_temp[i] = torus32_to_ntt_mod(bk[index + i]);
    }
    __syncthreads();
    // Forward NTT
    if constexpr (P::n == 1024) {
        SmallForwardNTT32_1024(sh_temp, ntt.forward_root_, tid);
    }
    // Store result
    for (int i = tid; i < P::n; i += bdim) {
        bk_ntt[index + i] = sh_temp[i];
    }
}

void TRGSW2NTT(cuFHETRGSWNTTlvl1& trgswntt,
               const TFHEpp::TRGSW<TFHEpp::lvl1param>& trgsw, Stream& st)
{
    cudaSetDevice(st.device_id());
    TFHEpp::lvl1param::T* d_trgsw;
    cudaMalloc((void**)&d_trgsw, sizeof(trgsw));
    cudaMemcpyAsync(d_trgsw, trgsw.data(), sizeof(trgsw),
                    cudaMemcpyHostToDevice, st.st());

    constexpr uint32_t num_threads = lvl1param::n >> 1;  // N/2
    dim3 grid(lvl1param::k+1, (lvl1param::k+1) * lvl1param::l, 1);
    dim3 block(num_threads);
    __TRGSW2NTT__<<<grid, block, 0, st.st()>>>(
        trgswntt.trgswdevices[st.device_id()], d_trgsw,
        *ntt_handlers[st.device_id()]);
    CuCheckError();
    // Copy NTT results back to host (NTTValue sized)
    cudaMemcpyAsync(
        trgswntt.trgswhost.data(), trgswntt.trgswdevices[st.device_id()],
        cuFHETRGSWNTTlvl1::kNumElements * sizeof(NTTValue),
        cudaMemcpyDeviceToHost, st.st());
    cudaFree(d_trgsw);
}

void InitializeNTThandlers(const int gpuNum)
{
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        ntt_handlers.push_back(new CuNTTHandler<>());
        ntt_handlers[i]->Create();
        ntt_handlers[i]->CreateConstant();
        ntt_handlers[i]->SetDevicePointers(i);  // Set device pointers for this GPU
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

template<class P>
void BootstrappingKeyToNTT(const BootstrappingKey<P>& bk,
                           const int gpuNum)
{
    bk_ntts.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);

        cudaMalloc((void**)&bk_ntts[i], sizeof(NTTValue) * P::domainP::n *
                                            (P::targetP::k+1) * P::targetP::l *
                                            (P::targetP::k+1) * P::targetP::n);

        typename P::targetP::T* d_bk;
        cudaMalloc((void**)&d_bk, sizeof(bk));
        cudaMemcpy(d_bk, bk.data(), sizeof(bk), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        CuCheckError();

        dim3 grid(P::targetP::k+1, (P::targetP::k+1) * P::targetP::l, P::domainP::n);
        dim3 block(P::targetP::n >> NTT_THREAD_UNITBIT);
        __TRGSW2NTT__<<<grid, block>>>(bk_ntts[i], d_bk, *ntt_handlers[i]);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_bk);
    }
}
#define INST(P)                                                \
template void BootstrappingKeyToNTT<P>(const BootstrappingKey<P>& bk, \
                           const int gpuNum)
INST(TFHEpp::lvl01param);
#undef INST

void DeleteBootstrappingKeyNTT(const int gpuNum)
{
    for (int i = 0; i < bk_ntts.size(); i++) {
        cudaSetDevice(i);
        cudaFree(bk_ntts[i]);

        ntt_handlers[i]->Destroy();
        delete ntt_handlers[i];
    }
    ntt_handlers.clear();
}

// ============================================================================
// CMUX operation - small modulus sequential NTT approach
// Same pattern as Accumulate in gatebootstrapping_gpu.cuh
// ============================================================================

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CMUXNTT__(
    TFHEpp::lvl1param::T* out, const NTTValue* const tgsw_ntt,
    const TFHEpp::lvl1param::T* const trlwe1,
    const TFHEpp::lvl1param::T* const trlwe0, const CuNTTHandler<> ntt)
{
    const uint32_t tid = ThisThreadRankInBlock();

    constexpr uint32_t N = lvl1param::n;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512

    extern __shared__ NTTValue sh_cmux[];
    NTTValue* const sh_work = &sh_cmux[0];         // Working buffer for NTT (N elements)
    NTTValue* const sh_accum = &sh_cmux[N];        // Accumulated results ((k+1)*N elements)

    // Initialize accumulated results to zero
    if (tid < NUM_THREADS) {
        for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
            sh_accum[k_idx * N + tid] = 0;
            sh_accum[k_idx * N + tid + NUM_THREADS] = 0;
        }
    }
    __syncthreads();

    // Decomposition constants
    constexpr uint32_t decomp_mask = (1 << lvl1param::Bgbit) - 1;
    constexpr int32_t decomp_half = 1 << (lvl1param::Bgbit - 1);
    constexpr uint32_t decomp_offset = offsetgen<lvl1param>();
    constexpr typename lvl1param::T roundoffset =
        1ULL << (std::numeric_limits<typename lvl1param::T>::digits -
                 lvl1param::l * lvl1param::Bgbit - 1);

    // Process each TRLWE component and decomposition level sequentially
    for (int j = 0; j <= lvl1param::k; j++) {
        for (int digit = 0; digit < lvl1param::l; digit++) {
            // Step 1: Decompose (trlwe1 - trlwe0) for component j, digit
            if (tid < NUM_THREADS) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    typename lvl1param::T temp = trlwe1[j * N + i] -
                                                  trlwe0[j * N + i] +
                                                  decomp_offset + roundoffset;
                    int32_t digit_val = static_cast<int32_t>(
                        ((temp >>
                          (std::numeric_limits<typename lvl1param::T>::digits -
                           (digit + 1) * lvl1param::Bgbit)) &
                         decomp_mask) -
                        decomp_half);
                    sh_work[i] = intToNTT(digit_val);
                }
            }
            __syncthreads();

            // Step 2: Forward NTT on decomposed polynomial
            if (tid < NUM_THREADS) {
                SmallForwardNTT32_1024(sh_work, ntt.forward_root_, tid);
            } else {
                for (int s = 0; s < 5; s++) __syncthreads();
            }

            // Step 3: Multiply with TRGSW_NTT and accumulate
            int digit_linear = j * lvl1param::l + digit;
            if (tid < NUM_THREADS) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int i = tid + e * NUM_THREADS;
                    NTTValue ntt_val = sh_work[i];
                    #pragma unroll
                    for (int out_k = 0; out_k <= lvl1param::k; out_k++) {
                        NTTValue bk_val = tgsw_ntt[(((lvl1param::k + 1) * digit_linear + out_k) << lvl1param::nbit) + i];
                        sh_accum[out_k * N + i] = nttAdd(sh_accum[out_k * N + i], nttMult(ntt_val, bk_val));
                    }
                }
            }
            __syncthreads();
        }
    }

    // Step 4: Inverse NTT on accumulated results, modswitch, add to trlwe0, write out
    for (int k_idx = 0; k_idx <= lvl1param::k; k_idx++) {
        // Copy accumulated data to work buffer
        if (tid < NUM_THREADS) {
            sh_work[tid] = sh_accum[k_idx * N + tid];
            sh_work[tid + NUM_THREADS] = sh_accum[k_idx * N + tid + NUM_THREADS];
        }
        __syncthreads();

        // Inverse NTT
        if (tid < NUM_THREADS) {
            SmallInverseNTT32_1024(sh_work, ntt.inverse_root_, ntt.n_inverse_, tid);
        } else {
            for (int s = 0; s < 6; s++) __syncthreads();
        }

        // Convert with modulus switching and add to trlwe0
        constexpr uint32_t half_mod = small_ntt::P / 2;
        if (tid < NUM_THREADS) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int i = tid + e * NUM_THREADS;
                uint32_t val = sh_work[i];
                int32_t signed_val = (val > half_mod) ? static_cast<int32_t>(val - small_ntt::P) : static_cast<int32_t>(val);
                out[k_idx * N + i] = trlwe0[k_idx * N + i] + ntt_mod_to_torus32(signed_val);
            }
        }
        __syncthreads();
    }
}

template <class brP, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __Bootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in,
    const typename brP::targetP::T mu, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T tlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    __BlindRotate__<brP>(tlwe,in,mu,bk,ntt);
    KeySwitch<iksP>(out, tlwe, ksk);
    __threadfence();
}

// template <class iksP, class bkP>
// __global__ __launch_bounds__(NUM_THREAD4HOMGATE) void __Bootstrap__(
//     typename iksP::domainP::T* const out, const typename iksP::domainP::T* const in,
//     const typename bkP::targetP::T mu, const NTTValue* const bk,
//     const typename iksP::targetP::T* const ksk, const CuNTTHandler<> ntt)
// {
//     __shared__ typename bkP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 

//     KeySwitch<iksP>(tlwe, in, ksk);
//     __threadfence();
//     __BlindRotate__<bkP>(out,tlwe,mu,bk,ntt);
//     __threadfence();
// }

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename P::targetP>) void __BlindRotateGlobal__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
    const TFHEpp::lvl1param::T mu, const NTTValue* const bk, const CuNTTHandler<> ntt)
{
    __BlindRotate__<P>(out, in, mu, bk, ntt);
}

__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __SEIandBootstrap2TRLWE__(
    TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
    const TFHEpp::lvl1param::T mu, const NTTValue* const bk, const TFHEpp::lvl0param::T* const ksk,
    const CuNTTHandler<> ntt)
{
    extern __shared__ NTTValue sh[];
    // Shared memory layout (all uint32_t):
    // [0, (k+2)*N): sh_acc_ntt for Accumulate (working buf + accum)
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE with (k+1)*N elements)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE)
    NTTValue* sh_acc_ntt = &sh[0];
    TFHEpp::lvl1param::T* tlwe =
        (TFHEpp::lvl1param::T*)&sh[(lvl1param::k + 2) * lvl1param::n];

    lvl0param::T* tlwelvl0 =
        (lvl0param::T*)&sh[(lvl1param::k + 2 + lvl1param::k + 1) * lvl1param::n];

    KeySwitch<lvl10param>(tlwelvl0, in, ksk);
    __syncthreads();

    // test vector
    // acc.a = 0; acc.b = vec(mu) * x ^ (in.b()/2048)
    register uint32_t bar = 2 * TFHEpp::lvl1param::n - (tlwelvl0[TFHEpp::lvl0param::k * TFHEpp::lvl0param::n] >> (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 1 - TFHEpp::lvl1param::nbit));
    RotatedTestVector<lvl1param>(tlwe, bar, mu);

    // accumulate
    for (int i = 0; i < lvl0param::n; i++) {
        constexpr typename TFHEpp::lvl0param::T roundoffset =
                1ULL << (std::numeric_limits<typename TFHEpp::lvl0param::T>::digits - 2 -
                        TFHEpp::lvl1param::nbit);
        bar = modSwitchFromTorus<lvl01param>(tlwelvl0[i]+roundoffset);
        Accumulate<lvl01param>(tlwe, sh_acc_ntt, bar,
                   bk + (i << lvl1param::nbit) * (lvl1param::k+1) * (lvl1param::k+1) * lvl1param::l, ntt);
    }
    __syncthreads();
    for (int i = 0; i < (lvl1param::k+1) * lvl1param::n; i++) {
        out[i] = tlwe[i];
    }
    __threadfence();
}

template<class P, uint index>
__device__ inline void __SampleExtractIndex__(typename P::T* const res, const typename P::T* const in){
    const uint32_t tid = ThisThreadRankInBlock();
    const uint32_t bdim = ThisBlockSize();
    constexpr uint nmask = P::n-1; 
    for (uint i = tid; i <= P::k*P::n; i += bdim) {
        if (i == P::k*P::n){
            res[P::k*P::n] = in[P::k*P::n+index];
        }else {
            const uint k = i >> P::nbit; 
            const uint n = i & nmask;
            if (n  <= index) res[i] = in[k*P::n + index - n];
            else res[i] = -in[k*P::n + P::n + index-n];
        }
    }
}

template <class iksP, class brP, std::make_signed_t<typename brP::targetP::T> μ, int casign, int cbsign, std::make_signed_t<typename iksP::domainP::T> offset>
__device__ inline void __HomGate__(typename brP::targetP::T* const out,
                                   const typename iksP::domainP::T* const in0,
                                   const typename iksP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwe[iksP::targetP::k*iksP::targetP::n+1]; 

    IdentityKeySwitchPreAdd<iksP, casign, cbsign, offset>(tlwe, in0, in1, ksk);
    __syncthreads();

    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n]; 

    __BlindRotate__<brP>(trlwe, tlwe, μ, bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out,trlwe);
    __threadfence();
}

template <class brP, std::make_signed_t<typename brP::targetP::T> μ, class iksP, int casign, int cbsign, std::make_signed_t<typename brP::domainP::T> offset>
__device__ inline void __HomGate__(typename iksP::targetP::T* const out,
                                   const typename brP::domainP::T* const in0,
                                   const typename brP::domainP::T* const in1, const NTTValue* const bk,
                                   const typename iksP::targetP::T* const ksk,
                                   const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, casign,cbsign,offset>(trlwe,in0,in1,bk,ntt);

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// br iks ver.
template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NandBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XnorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -2, -2, -2 * brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XorBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 2, 2, 2 * brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, -brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrNYBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, -1, 1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ, class iksP = TFHEpp::lvl10param>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrYNBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
    const typename brP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<brP, μ, iksP, 1, -1, brP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,ntt);
    __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Add all (k+1)*n elements of the TRLWE ciphertexts
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] += trlwe0[i];
    }
    // Add μ to b[0] (the constant term of polynomial b)
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] += μ;
    }

    __syncthreads();

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template<class brP, typename brP::targetP::T μ, class iksP>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrap__(
    typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
    const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename brP::targetP::T trlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __shared__ typename brP::targetP::T trlwe0[(brP::targetP::k+1)*brP::targetP::n];

    __BlindRotatePreAdd__<brP, 1, 1, -brP::domainP::μ>(trlwe1,inc,in1,bk,ntt);
    __BlindRotatePreAdd__<brP, -1, 1, -brP::domainP::μ>(trlwe0,inc,in0,bk,ntt);

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Negate and add all (k+1)*n elements of the TRLWE ciphertexts
    constexpr uint32_t trlwe_size = (brP::targetP::k+1)*brP::targetP::n;
#pragma unroll
    for (int i = tid; i < trlwe_size; i += bdim) {
        trlwe1[i] = -trlwe1[i] - trlwe0[i];
    }
    // Subtract μ from b[0] (the constant term of polynomial b)
    if (tid == 0) {
        trlwe1[brP::targetP::k*brP::targetP::n] -= μ;
    }

    __syncthreads();

    // Extract TLWE from TRLWE at index 0 before key switching
    // This is required for k > 1 (concrete.hpp uses k=2)
    __shared__ typename brP::targetP::T tlwe[brP::targetP::k*brP::targetP::n+1];
    __SampleExtractIndex__<typename brP::targetP, 0>(tlwe, trlwe1);
    __syncthreads();

    KeySwitchFromTLWE<iksP>(out, tlwe, ksk);
    __threadfence();
}

// iks br ver.
template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NandBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, -1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XnorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -2, -2, -2 * iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, 1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __XorBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 2, 2, 2 * iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndNYBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __AndYNBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, -iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrNYBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, -1, 1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class iksP = TFHEpp::lvl10param, class brP = TFHEpp::lvl01param, typename brP::targetP::T μ = TFHEpp::lvl1param::μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __OrYNBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
    const typename iksP::domainP::T* const in1, NTTValue* bk, const typename iksP::targetP::T* const ksk,
    const CuNTTHandler<> ntt)
{
    __HomGate__<iksP, brP, μ, 1, -1, iksP::domainP::μ>(out, in0, in1, bk, ksk, ntt);
}

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __CopyBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k*P::n; i += bdim) 
        out[i] = in[i];
    __syncthreads();
    __threadfence();
}

template<class P>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<TFHEpp::lvl1param>) void __NotBootstrap__(
    typename P::T* const out, const typename P::T* const in)
{
    const uint tid = ThisThreadRankInBlock();
    const uint bdim = ThisBlockSize();
    for (int i = tid; i <= P::k*P::n; i += bdim) 
        out[i] = -in[i];
    __syncthreads();
    __threadfence();
}

// Mux(inc,in1,in0) = inc?in1:in0 = inc&in1 + (!inc)&in0
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __MuxBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1]; 

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n]; 
    __BlindRotate__<brP>(tlwe1,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n]; 
    __BlindRotate__<brP>(tlwe0,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);
    
    __syncthreads();

    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Add TLWE ciphertexts: k*n elements for 'a' parts plus 1 for 'b'
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] += tlwe1[i];
    }
    // Add μ to b (the last element)
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] += μ;
    }
    __threadfence();
}

// NMux(inc,in1,in0) = !(inc?in1:in0) = !(inc&in1 + (!inc)&in0)
template<class iksP, class brP, typename brP::targetP::T μ>
__global__ __launch_bounds__(NUM_THREAD4HOMGATE<typename brP::targetP>) void __NMuxBootstrap__(
    typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
    const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, const NTTValue* const bk,
    const typename iksP::targetP::T* const ksk,  const CuNTTHandler<> ntt)
{
    __shared__ typename iksP::targetP::T tlwelvl0[iksP::targetP::k*iksP::targetP::n+1];

    IdentityKeySwitchPreAdd<iksP, 1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in1, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe1[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotate__<brP>(tlwe1,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(out, tlwe1);

    IdentityKeySwitchPreAdd<iksP, -1, 1, -iksP::domainP::μ>(tlwelvl0, inc, in0, ksk);
    __syncthreads();
    __shared__ typename brP::targetP::T tlwe0[(brP::targetP::k+1)*brP::targetP::n];
    __BlindRotate__<brP>(tlwe0,tlwelvl0,μ,bk,ntt);
    __SampleExtractIndex__<typename brP::targetP,0>(tlwe1, tlwe0);

    __syncthreads();


    volatile const uint32_t tid = ThisThreadRankInBlock();
    volatile const uint32_t bdim = ThisBlockSize();
    // Negate and add TLWE ciphertexts: k*n elements for 'a' parts plus 1 for 'b'
    constexpr uint32_t tlwe_size = brP::targetP::k*brP::targetP::n + 1;
#pragma unroll
    for (int i = tid; i < tlwe_size; i += bdim) {
        out[i] = -out[i] - tlwe1[i];
    }
    // Subtract μ from b (the last element)
    if (tid == 0) {
        out[brP::targetP::k*brP::targetP::n] -= μ;
    }
    __threadfence();
}

void Bootstrap(TFHEpp::lvl0param::T* const out, const TFHEpp::lvl0param::T* const in,
               const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    __Bootstrap__<lvl01param,lvl10param><<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, 0, st>>>(
        out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void CMUXNTTkernel(TFHEpp::lvl1param::T* const res, const NTTValue* const cs,
                   TFHEpp::lvl1param::T* const c1,
                   TFHEpp::lvl1param::T* const c0, cudaStream_t st,
                   const int gpuNum)
{
    constexpr size_t shmem_size = (TFHEpp::lvl1param::k + 2) * TFHEpp::lvl1param::n * sizeof(NTTValue);
    cudaFuncSetAttribute(__CMUXNTT__,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_size);
    __CMUXNTT__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, shmem_size, st>>>(
        res, cs, c1, c0, *ntt_handlers[gpuNum]);
    CuCheckError();
}

void BootstrapTLWE2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl0param::T* const in,
                         const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__BlindRotateGlobal__<TFHEpp::lvl01param>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<TFHEpp::lvl1param>);
    __BlindRotateGlobal__<TFHEpp::lvl01param><<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>, MEM4HOMGATE<TFHEpp::lvl1param>, st>>>(
        out, in, mu, bk_ntts[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

void SEIandBootstrap2TRLWE(TFHEpp::lvl1param::T* const out, const TFHEpp::lvl1param::T* const in,
                           const lvl1param::T mu, const cudaStream_t st, const int gpuNum)
{
    // Shared memory layout (all uint32_t / NTTValue):
    // [0, (k+2)*N): Accumulate working space
    // [(k+2)*N, (2k+3)*N): tlwe (TRLWE)
    // [(2k+3)*N, ...): tlwelvl0 (lvl0 TLWE, ceil((n0+1)/2+1) uint32_ts)
    constexpr size_t shmem_elems = (lvl1param::k + 2 + lvl1param::k + 1) * lvl1param::n +
                                   (lvl0param::n + 1) / 2 + 1;
    constexpr size_t shmem_size = shmem_elems * sizeof(NTTValue);
    cudaFuncSetAttribute(
        __SEIandBootstrap2TRLWE__, cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size);
    __SEIandBootstrap2TRLWE__<<<1, NUM_THREAD4HOMGATE<TFHEpp::lvl1param>,
                              shmem_size, st>>>
        (out, in, mu, bk_ntts[gpuNum], ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}

template<class brP, typename brP::targetP::T μ, class iksP>
void NandBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NandBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrap__<brP, μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NandBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void NandBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NandBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NandBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NandBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrYNBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrYNBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrYNBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrYNBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrYNBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void OrNYBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void OrNYBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void OrNYBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__OrNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __OrNYBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void OrNYBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndYNBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndYNBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndYNBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndYNBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndYNBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndYNBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndYNBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void AndNYBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndNYBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void AndNYBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void AndNYBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__AndNYBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __AndNYBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void AndNYBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void NorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void NorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __NorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void XorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__XorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void XorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void XorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__XorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void XorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void XnorBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0,
                   const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__XnorBootstrap__<brP, brP::targetP::μ, iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrap__<brP, brP::targetP::μ, iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void XnorBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const in0, \
                                        const typename brP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void XnorBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0,
                   const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__XnorBootstrap__<iksP, brP, brP::targetP::μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MEM4HOMGATE<typename brP::targetP>);
    __XnorBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>, MEM4HOMGATE<typename brP::targetP>, st>>>(
        out, in0, in1, bk_ntts[gpuNum], ksk_devs[gpuNum],
        *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void XnorBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const in0, \
                                         const typename iksP::domainP::T* const in1, const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class P>
void CopyBootstrap(typename P::T* const out, const typename P::T* const in,
                   const cudaStream_t st, const int gpuNum)
{
    __CopyBootstrap__<P><<<1, std::min(P::n + 1,NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0, st>>>(out, in);
    CuCheckError();
}
#define INST(P) \
template void CopyBootstrap<P>(typename P::T* const out, const typename P::T* const in, \
                   const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template<class P>
void NotBootstrap(typename P::T* const out, const typename P::T* const in,
                  const cudaStream_t st, const int gpuNum)
{
    __NotBootstrap__<P><<<1, std::min(P::n + 1,NUM_THREAD4HOMGATE<TFHEpp::lvl1param>), 0, st>>>(out, in);
    CuCheckError();
}
#define INST(P) \
template void NotBootstrap<P>(typename P::T* const out, const typename P::T* const in, \
                   const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl0param);
INST(TFHEpp::lvl1param);
#undef INST

template<class brP, typename brP::targetP::T μ, class iksP>
void MuxBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
                  const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__MuxBootstrap__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrap__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void MuxBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc, \
                                       const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, \
                                       const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST
template<class iksP, class brP, typename brP::targetP::T μ>
void MuxBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__MuxBootstrap__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __MuxBootstrap__<iksP,brP,μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void MuxBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc, \
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, \
                  const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST

template<class iksP, class brP, typename brP::targetP::T μ>
void NMuxBootstrap(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc,
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0, 
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NMuxBootstrap__<iksP,brP,μ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrap__<iksP, brP, μ><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(iksP, brP, μ)                                                \
template void NMuxBootstrap<iksP, brP,μ>(typename brP::targetP::T* const out, const typename iksP::domainP::T* const inc, \
                  const typename iksP::domainP::T* const in1, const typename iksP::domainP::T* const in0,  \
                  const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl10param, TFHEpp::lvl01param, TFHEpp::lvl1param::μ);
#undef INST
template<class brP, typename brP::targetP::T μ, class iksP>
void NMuxBootstrap(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc,
                  const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0,
                  const cudaStream_t st, const int gpuNum)
{
    cudaFuncSetAttribute(__NMuxBootstrap__<brP,μ,iksP>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue));
    __NMuxBootstrap__<brP,μ,iksP><<<1, NUM_THREAD4HOMGATE<typename brP::targetP>,
                       ((brP::targetP::k+1) * brP::targetP::l + 3) * brP::targetP::n * sizeof(NTTValue),
                       st>>>(out, inc, in1, in0, bk_ntts[gpuNum],
                             ksk_devs[gpuNum], *ntt_handlers[gpuNum]);
    CuCheckError();
}
#define INST(brP, μ, iksP)                                                \
template void NMuxBootstrap<brP,μ,iksP>(typename iksP::targetP::T* const out, const typename brP::domainP::T* const inc, \
                                       const typename brP::domainP::T* const in1, const typename brP::domainP::T* const in0, \
                                       const cudaStream_t st, const int gpuNum)
INST(TFHEpp::lvl01param, TFHEpp::lvl1param::μ, TFHEpp::lvl10param);
#undef INST

}  // namespace cufhe
