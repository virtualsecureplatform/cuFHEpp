/**
 * Goldilocks-prime NTT implementation for cuFHE
 * Host-side initialization functions
 *
 * Modulus: P = 2^64 - 2^32 + 1
 * P - 1 is divisible by 2^32, so lvl1 and lvl2 have primitive 2N-th roots.
 */

#include <cmath>
#include <complex>
#include <include/error_gpu.cuh>
#include <include/ntt_small_modulus.cuh>
#include <stdexcept>
#include <vector>

namespace cufhe {

// Host-side storage for NTT parameters per GPU
std::vector<SmallNTTParams> g_small_ntt_params;
std::vector<SmallNTTParams> g_small_ntt_params_lvl02;

namespace {

// Bit reversal helper
int bitreverse(int index, int n_power)
{
    int res = 0;
    for (int i = 0; i < n_power; i++) {
        res <<= 1;
        res = (index & 1) | res;
        index >>= 1;
    }
    return res;
}

SmallNTTValue mod_exp(SmallNTTValue base, uint64_t exp)
{
    SmallNTTValue result = 1;
    base = small_mod_normalize(base);
    while (exp > 0) {
        if (exp & 1) {
            result = small_mod_mult(result, base);
        }
        exp >>= 1;
        base = small_mod_mult(base, base);
    }
    return result;
}

SmallNTTValue mod_inv(SmallNTTValue a)
{
    return mod_exp(a, small_ntt::P - 2);
}

/**
 * Find a primitive 2N-th root of unity for negacyclic NTT
 */
SmallNTTValue find_primitive_root(int log_n)
{
    if (log_n < 1 || log_n + 1 > 32) {
        throw std::runtime_error("Unsupported Goldilocks NTT length");
    }

    const uint64_t exponent = 1ULL << (32 - (log_n + 1));
    SmallNTTValue psi = mod_exp(small_ntt::ROOT_2_32, exponent);

    SmallNTTValue psi_N = mod_exp(psi, 1ULL << log_n);
    if (psi_N != small_ntt::P_MINUS_ONE) {
        throw std::runtime_error("Computed root does not satisfy psi^N = -1");
    }

    return psi;
}

struct SmallRootTableState {
    std::vector<SmallNTTValue> forward_table;
    std::vector<SmallNTTValue> inverse_table;
    SmallNTTValue n_inverse = 0;
    int log_n = 0;
};

SmallRootTableState g_small_tables_1024;
SmallRootTableState g_small_tables_2048;

template <uint32_t length>
SmallRootTableState& RootTablesForLength()
{
    if constexpr (length == TFHEpp::lvl2param::n) {
        return g_small_tables_2048;
    }
    else {
        static_assert(length == TFHEpp::lvl1param::n,
                      "Unsupported Goldilocks NTT length");
        return g_small_tables_1024;
    }
}

template <uint32_t length>
std::vector<SmallNTTParams>& ParamsForLength()
{
    if constexpr (length == TFHEpp::lvl2param::n) {
        return g_small_ntt_params_lvl02;
    }
    else {
        static_assert(length == TFHEpp::lvl1param::n,
                      "Unsupported Goldilocks NTT length");
        return g_small_ntt_params;
    }
}

/**
 * Generate NTT root tables for the Goldilocks modulus.
 */
void GenerateSmallRootTables(int log_n, SmallRootTableState& state)
{
    if (state.log_n == log_n && !state.forward_table.empty()) {
        return;  // Already generated
    }

    const int n = 1 << log_n;

    SmallNTTValue psi = find_primitive_root(log_n);
    SmallNTTValue psi_inv = mod_inv(psi);

    state.n_inverse = mod_inv(static_cast<SmallNTTValue>(n));

    // Generate forward root table: psi^0, psi^1, ..., psi^(n-1) in bit-reversed
    // order For Cooley-Tukey, we need powers of psi^2 (omega = psi^2 is the
    // N-th root of unity)
    state.forward_table.resize(n);
    state.forward_table[0] = 1;
    for (int i = 1; i < n; i++) {
        state.forward_table[i] = small_mod_mult(state.forward_table[i - 1], psi);
    }

    // Generate inverse root table: psi_inv^0, psi_inv^1, ...
    state.inverse_table.resize(n);
    state.inverse_table[0] = 1;
    for (int i = 1; i < n; i++) {
        state.inverse_table[i] =
            small_mod_mult(state.inverse_table[i - 1], psi_inv);
    }

    // Convert to bit-reversed order for NTT algorithm
    std::vector<SmallNTTValue> br_forward(n);
    std::vector<SmallNTTValue> br_inverse(n);
    for (int i = 0; i < n; i++) {
        int br_idx = bitreverse(i, log_n);
        br_forward[i] = state.forward_table[br_idx];
        br_inverse[i] = state.inverse_table[br_idx];
    }
    state.forward_table = std::move(br_forward);
    state.inverse_table = std::move(br_inverse);

    state.log_n = log_n;
}

#if defined(USE_FFT) && defined(USE_GPU_FFT)
// Generate the 4 FFNT tables for a ring of size N.
// Replaces gpufft::FFNT<Float64>: same math, no external dependency.
// - forward_root[i] = exp(2πi·bitrev(i,logH-1)/H),  H = N/2
// - inverse_root[i] = conj(forward_root[i])
// - twist[i]        = exp(2πi·i/(2N))
// - untwist[i]      = conj(twist[i])
static void GenerateFFNTTables(int N,
    std::vector<double2>& forward_root,
    std::vector<double2>& inverse_root,
    std::vector<double2>& twist,
    std::vector<double2>& untwist)
{
    const int half_n = N >> 1;
    const int logH   = (int)std::log2((double)half_n);  // log2(N/2)

    // Root table for N/2-point FFT: omega^k, omega = e^(2πi/(N/2))
    const double root_angle = 2.0 * M_PI / half_n;
    std::vector<std::complex<double>> roots_new(half_n);
    for (int i = 0; i < half_n; i++)
        roots_new[i] = std::exp(std::complex<double>(0.0, i * root_angle));

    // Twist table: psi^k, psi = e^(2πi/(2N))
    const double twist_angle = 2.0 * M_PI / (2 * N);
    std::vector<std::complex<double>> roots_twist(half_n);
    for (int i = 0; i < half_n; i++)
        roots_twist[i] = std::exp(std::complex<double>(0.0, i * twist_angle));

    forward_root.resize(half_n);
    inverse_root.resize(half_n);
    twist.resize(half_n);
    untwist.resize(half_n);

    // n_inverse is folded into the untwist table so that the IFFT can skip
    // the separate n_inverse multiply pass (saves one __syncthreads).
    const double n_inv = 1.0 / half_n;

    for (int i = 0; i < half_n; i++) {
        int br = bitreverse(i, logH - 1);
        forward_root[i] = { roots_new[br].real(),  roots_new[br].imag() };
        inverse_root[i] = { roots_new[br].real(), -roots_new[br].imag() };
        twist[i]   = {  roots_twist[i].real(),  roots_twist[i].imag() };
        untwist[i] = {  roots_twist[i].real() * n_inv,
                        -roots_twist[i].imag() * n_inv };
    }
}
#endif  // USE_FFT && USE_GPU_FFT

}  // anonymous namespace

//=============================================================================
// Static host functions for initialization
//=============================================================================

template <uint32_t length>
void CuSmallNTTHandler<length>::Create()
{
    constexpr int log_n = kLogLength;
    GenerateSmallRootTables(log_n, RootTablesForLength<length>());
}

template <uint32_t length>
void CuSmallNTTHandler<length>::Destroy()
{
    auto& params_vec = ParamsForLength<length>();
    auto& tables = RootTablesForLength<length>();

    for (auto& params : params_vec) {
        if (params.forward_root) {
            cudaFree(params.forward_root);
            params.forward_root = nullptr;
        }
        if (params.inverse_root) {
            cudaFree(params.inverse_root);
            params.inverse_root = nullptr;
        }
        params.initialized = false;
    }
    params_vec.clear();
    tables.forward_table.clear();
    tables.inverse_table.clear();
    tables.n_inverse = 0;
    tables.log_n = 0;
}

template <uint32_t length>
void CuSmallNTTHandler<length>::SetDevicePointers(int device_id)
{
    auto& params_vec = ParamsForLength<length>();
    auto& tables = RootTablesForLength<length>();

    // Resize if needed
    if (params_vec.size() <= static_cast<size_t>(device_id)) {
        params_vec.resize(device_id + 1);
    }

    SmallNTTParams& params = params_vec[device_id];

    // Initialize if not already done
    if (!params.initialized) {
        // Allocate device memory for root tables
        CuSafeCall(
            cudaMalloc(&params.forward_root, sizeof(SmallNTTValue) * kLength));
        CuSafeCall(
            cudaMalloc(&params.inverse_root, sizeof(SmallNTTValue) * kLength));

        // Copy root tables to device
        CuSafeCall(cudaMemcpy(params.forward_root, tables.forward_table.data(),
                              sizeof(SmallNTTValue) * kLength,
                              cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.inverse_root, tables.inverse_table.data(),
                              sizeof(SmallNTTValue) * kLength,
                              cudaMemcpyHostToDevice));

        params.n_inverse = tables.n_inverse;
        params.initialized = true;
    }

    // Set device pointers
    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    n_inverse_ = params.n_inverse;
}

// Explicit template instantiation
template class CuSmallNTTHandler<TFHEpp::lvl1param::n>;
template class CuSmallNTTHandler<TFHEpp::lvl2param::n>;

//=============================================================================
// GPU-FFT Handler implementation
//=============================================================================
#if defined(USE_FFT) && defined(USE_GPU_FFT)

namespace {

// Host-side storage for GPU-FFT tables
struct GPUFFTParams {
    double2* forward_root;
    double2* inverse_root;
    double2* twist;
    double2* untwist;
    bool initialized;
};

std::vector<GPUFFTParams> g_gpufft_params;

// Host-side tables (generated once, copied to each GPU)
std::vector<double2> g_gpufft_forward_root;
std::vector<double2> g_gpufft_inverse_root;
std::vector<double2> g_gpufft_twist;
std::vector<double2> g_gpufft_untwist;

}  // anonymous namespace

template <>
void CuGPUFFTHandler<TFHEpp::lvl1param::n>::Create()
{
    if (!g_gpufft_forward_root.empty()) return;  // Already generated

    constexpr uint32_t N = TFHEpp::lvl1param::n;  // 1024
    GenerateFFNTTables(N,
        g_gpufft_forward_root, g_gpufft_inverse_root,
        g_gpufft_twist, g_gpufft_untwist);
}

template <>
void CuGPUFFTHandler<TFHEpp::lvl1param::n>::Destroy()
{
    for (auto& params : g_gpufft_params) {
        if (params.forward_root) {
            cudaFree(params.forward_root);
            params.forward_root = nullptr;
        }
        if (params.inverse_root) {
            cudaFree(params.inverse_root);
            params.inverse_root = nullptr;
        }
        if (params.twist) {
            cudaFree(params.twist);
            params.twist = nullptr;
        }
        if (params.untwist) {
            cudaFree(params.untwist);
            params.untwist = nullptr;
        }
        params.initialized = false;
    }
    g_gpufft_params.clear();
    g_gpufft_forward_root.clear();
    g_gpufft_inverse_root.clear();
    g_gpufft_twist.clear();
    g_gpufft_untwist.clear();
}

template <>
void CuGPUFFTHandler<TFHEpp::lvl1param::n>::SetDevicePointers(int device_id)
{
    if (g_gpufft_params.size() <= static_cast<size_t>(device_id)) {
        g_gpufft_params.resize(device_id + 1);
    }

    GPUFFTParams& params = g_gpufft_params[device_id];

    if (!params.initialized) {
        constexpr size_t table_bytes = kHalfLength * sizeof(double2);

        CuSafeCall(cudaMalloc(&params.forward_root, table_bytes));
        CuSafeCall(cudaMalloc(&params.inverse_root, table_bytes));
        CuSafeCall(cudaMalloc(&params.twist, table_bytes));
        CuSafeCall(cudaMalloc(&params.untwist, table_bytes));

        CuSafeCall(cudaMemcpy(params.forward_root, g_gpufft_forward_root.data(),
                              table_bytes, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.inverse_root, g_gpufft_inverse_root.data(),
                              table_bytes, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.twist, g_gpufft_twist.data(), table_bytes,
                              cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.untwist, g_gpufft_untwist.data(),
                              table_bytes, cudaMemcpyHostToDevice));

        params.initialized = true;
    }

    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    twist_ = params.twist;
    untwist_ = params.untwist;
}

template class CuGPUFFTHandler<TFHEpp::lvl1param::n>;

// =========================================================================
// CuGPUFFTHandler<2048> for lvl2param (N=2048, HALF_N=1024)
// =========================================================================

namespace {

std::vector<GPUFFTParams> g_gpufft2048_params;
std::vector<double2> g_gpufft2048_forward_root;
std::vector<double2> g_gpufft2048_inverse_root;
std::vector<double2> g_gpufft2048_twist;
std::vector<double2> g_gpufft2048_untwist;

}  // anonymous namespace

template <>
void CuGPUFFTHandler<TFHEpp::lvl2param::n>::Create()
{
    if (!g_gpufft2048_forward_root.empty()) return;

    constexpr uint32_t N = TFHEpp::lvl2param::n;  // 2048
    GenerateFFNTTables(N,
        g_gpufft2048_forward_root, g_gpufft2048_inverse_root,
        g_gpufft2048_twist, g_gpufft2048_untwist);
}

template <>
void CuGPUFFTHandler<TFHEpp::lvl2param::n>::Destroy()
{
    for (auto& params : g_gpufft2048_params) {
        if (params.forward_root) {
            cudaFree(params.forward_root);
            params.forward_root = nullptr;
        }
        if (params.inverse_root) {
            cudaFree(params.inverse_root);
            params.inverse_root = nullptr;
        }
        if (params.twist) {
            cudaFree(params.twist);
            params.twist = nullptr;
        }
        if (params.untwist) {
            cudaFree(params.untwist);
            params.untwist = nullptr;
        }
        params.initialized = false;
    }
    g_gpufft2048_params.clear();
    g_gpufft2048_forward_root.clear();
    g_gpufft2048_inverse_root.clear();
    g_gpufft2048_twist.clear();
    g_gpufft2048_untwist.clear();
}

template <>
void CuGPUFFTHandler<TFHEpp::lvl2param::n>::SetDevicePointers(int device_id)
{
    if (g_gpufft2048_params.size() <= static_cast<size_t>(device_id)) {
        g_gpufft2048_params.resize(device_id + 1);
    }

    GPUFFTParams& params = g_gpufft2048_params[device_id];

    if (!params.initialized) {
        constexpr size_t table_bytes = kHalfLength * sizeof(double2);

        CuSafeCall(cudaMalloc(&params.forward_root, table_bytes));
        CuSafeCall(cudaMalloc(&params.inverse_root, table_bytes));
        CuSafeCall(cudaMalloc(&params.twist, table_bytes));
        CuSafeCall(cudaMalloc(&params.untwist, table_bytes));

        CuSafeCall(cudaMemcpy(params.forward_root,
                              g_gpufft2048_forward_root.data(), table_bytes,
                              cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.inverse_root,
                              g_gpufft2048_inverse_root.data(), table_bytes,
                              cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.twist, g_gpufft2048_twist.data(),
                              table_bytes, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.untwist, g_gpufft2048_untwist.data(),
                              table_bytes, cudaMemcpyHostToDevice));

        params.initialized = true;
    }

    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    twist_ = params.twist;
    untwist_ = params.untwist;
}

template class CuGPUFFTHandler<TFHEpp::lvl2param::n>;

#endif  // USE_FFT && USE_GPU_FFT

#ifdef USE_KEY_BUNDLE
//=============================================================================
// XaiNTT table: precomputed NTT(X^a) for a = 0..2N-1
// Used for on-the-fly keybundle computation in key-bundle bootstrapping
//=============================================================================

std::vector<NTTValue*> xai_ntt_devs;
std::vector<NTTValue*> one_trgsw_ntt_devs;

#ifdef USE_FFT
#ifdef USE_GPU_FFT
//=============================================================================
// GPU-FFT Key-bundle initialization
//=============================================================================

// GPU kernel: compute GPU-FFT of (X^a - 1) mod (X^N+1) for a = 0..2N-1
// Uses fold + twist + GPUFFTForward512
__global__ void __ComputeXaiFFT__(NTTValue* const xai_fft,
                                  const double2* const twist_table,
                                  const double2* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t a = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    uint32_t a_mod = a & (N - 1);
    bool negate = (a >= N);

    // Build (X^a - 1) mod (X^N + 1) as integer polynomial, fold + twist
    if (tid < HALF_N) {
        double re = 0.0, im = 0.0;

        if (tid == 0) re = -1.0;
        if (!negate) {
            if (tid == a_mod) re += 1.0;
            if (tid + HALF_N == a_mod) im += 1.0;
        }
        else {
            if (tid == a_mod) re -= 1.0;
            if (tid + HALF_N == a_mod) im -= 1.0;
        }

        // Fold + twist
        double2 folded = {re, im};
        double2 tw = __ldg(&twist_table[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward512(sh_fft, forward_root, tid);
    }
    else {
        for (int s = 0; s < 5; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        xai_fft[a * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeXaiNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t table_entries = 2 * N;
    constexpr size_t table_size = table_entries * HALF_N * sizeof(NTTValue);

    xai_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&xai_ntt_devs[i], table_size));

        dim3 grid(table_entries);
        dim3 block(HALF_N);
        __ComputeXaiFFT__<<<grid, block>>>(xai_ntt_devs[i],
                                           g_gpufft_params[i].twist,
                                           g_gpufft_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

// GPU kernel: FFT Torus32 polynomials using GPU-FFT (for OneTRGSW identity)
// Normalizes by 1/2^32, fold + twist + GPUFFTForward512
__global__ void __FFTPolynomials__(NTTValue* const out,
                                   const uint32_t* const in,
                                   const double2* const twist_table,
                                   const double2* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 256

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    constexpr double norm = 1.0 / 4294967296.0;  // 1/2^32
    if (tid < HALF_N) {
        double re =
            static_cast<double>(static_cast<int32_t>(in[poly_idx * N + tid])) *
            norm;
        double im = static_cast<double>(
                        static_cast<int32_t>(in[poly_idx * N + tid + HALF_N])) *
                    norm;
        // Fold + twist
        double2 folded = {re, im};
        double2 tw = __ldg(&twist_table[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward512(sh_fft, forward_root, tid);
    }
    else {
        for (int s = 0; s < 5; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        out[poly_idx * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeOneTRGSWNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t k = TFHEpp::lvl1param::k;
    constexpr uint32_t l = TFHEpp::lvl1param::l;
    constexpr uint32_t Bgbit = TFHEpp::lvl1param::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * HALF_N * sizeof(NTTValue);

    std::vector<uint32_t> h(l);
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<uint32_t>(1)
               << (std::numeric_limits<uint32_t>::digits - (i + 1) * Bgbit);
    }

    std::vector<uint32_t> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            host_polys[poly_idx * N + 0] = h[digit];
        }
    }

    one_trgsw_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&one_trgsw_ntt_devs[i], total_size));

        uint32_t* d_polys;
        CuSafeCall(cudaMalloc(&d_polys, num_polys * N * sizeof(uint32_t)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(HALF_N);
        __FFTPolynomials__<<<grid, block>>>(one_trgsw_ntt_devs[i], d_polys,
                                            g_gpufft_params[i].twist,
                                            g_gpufft_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

#else  // !USE_GPU_FFT (tfhe-rs FFT)
//=============================================================================
// tfhe-rs FFT Key-bundle initialization (negacyclic FFT over double2)
//=============================================================================

// GPU kernel to compute FFT of (X^a - 1) mod (X^N+1) for a = 0..2N-1
// Integer coefficients, no normalization — xai multiplies normalized BSK values
__global__ void __ComputeXaiFFT__(NTTValue* const xai_fft)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t a = blockIdx.x;  // 0..2N-1
    const uint32_t tid = threadIdx.x;

    uint32_t a_mod = a & (N - 1);
    bool negate = (a >= N);

    // Build (X^a - 1) mod (X^N + 1) as integer polynomial packed into complex
    // Packing: Complex[i] = {Poly[i], Poly[i + N/2]}
    if (tid < HALF_N) {
        double re = 0.0, im = 0.0;

        // Constant term (-1) at coefficient 0
        if (tid == 0) re = -1.0;

        // X^a term: +1 at a_mod if a < N, -1 at a_mod if a >= N
        if (!negate) {
            if (tid == a_mod) re += 1.0;
            if (tid + HALF_N == a_mod) im += 1.0;
        }
        else {
            if (tid == a_mod) re -= 1.0;
            if (tid + HALF_N == a_mod) im -= 1.0;
        }

        sh_fft[tid] = {re, im};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    }
    else {
        for (int s = 0; s < 11; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        xai_fft[a * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeXaiNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t table_entries = 2 * N;
    constexpr size_t table_size = table_entries * HALF_N * sizeof(NTTValue);

    xai_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&xai_ntt_devs[i], table_size));

        dim3 grid(table_entries);
        dim3 block(HALF_N);
        __ComputeXaiFFT__<<<grid, block>>>(xai_ntt_devs[i]);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

// GPU kernel to FFT Torus32 polynomials (for OneTRGSW identity)
// Normalizes by 1/2^32 (matching BSK normalization), packs into N/2 complex
__global__ void __FFTPolynomials__(NTTValue* const out,
                                   const uint32_t* const in)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    constexpr double norm = 1.0 / 4294967296.0;  // 1/2^32
    if (tid < HALF_N) {
        sh_fft[tid] = {
            static_cast<double>(static_cast<int32_t>(in[poly_idx * N + tid])) *
                norm,
            static_cast<double>(
                static_cast<int32_t>(in[poly_idx * N + tid + HALF_N])) *
                norm};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    }
    else {
        for (int s = 0; s < 11; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        out[poly_idx * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeOneTRGSWNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t k = TFHEpp::lvl1param::k;
    constexpr uint32_t l = TFHEpp::lvl1param::l;
    constexpr uint32_t Bgbit = TFHEpp::lvl1param::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * HALF_N * sizeof(NTTValue);

    std::vector<uint32_t> h(l);
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<uint32_t>(1)
               << (std::numeric_limits<uint32_t>::digits - (i + 1) * Bgbit);
    }

    // Build identity TRGSW as Torus32 polynomials on host
    std::vector<uint32_t> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            host_polys[poly_idx * N + 0] = h[digit];  // Torus32 value
        }
    }

    one_trgsw_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&one_trgsw_ntt_devs[i], total_size));

        uint32_t* d_polys;
        CuSafeCall(cudaMalloc(&d_polys, num_polys * N * sizeof(uint32_t)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(HALF_N);
        __FFTPolynomials__<<<grid, block>>>(one_trgsw_ntt_devs[i], d_polys);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT
//=============================================================================
// NTT Key-bundle initialization (small modulus NTT)
//=============================================================================

// GPU kernel to NTT multiple polynomials
template <uint32_t N>
__global__ void __NTTPolynomials__(NTTValue* const out,
                                   const NTTValue* const in,
                                   const NTTValue* const forward_root)
{
    constexpr uint32_t LOG_N = SmallLog2<N>();
    __shared__ NTTValue sh_temp[N];

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t NUM_THREADS = N >> 1;

    // Load polynomial
    if (tid < NUM_THREADS) {
        sh_temp[tid] = in[poly_idx * N + tid];
        sh_temp[tid + NUM_THREADS] = in[poly_idx * N + tid + NUM_THREADS];
    }
    __syncthreads();

    // Forward NTT
    if (tid < NUM_THREADS) {
        SmallForwardNTT<LOG_N>(sh_temp, forward_root, tid);
    }
    else {
        for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
            __syncthreads();
    }

    // Store
    if (tid < NUM_THREADS) {
        out[poly_idx * N + tid] = sh_temp[tid];
        out[poly_idx * N + tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
    }
}

template <uint32_t N>
__global__ void __ComputeXaiNTT__(NTTValue* const xai_ntt,
                                  const NTTValue* const forward_root)
{
    constexpr uint32_t LOG_N = SmallLog2<N>();
    __shared__ NTTValue poly[N];

    const uint32_t a = blockIdx.x;  // 0..2N-1
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512

    // Initialize polynomial: (X^a - 1) mod (X^N + 1) in Z_P
    // Following TFHEpp's XaittGen: xai[0] = -1; if (a < N) xai[a] += 1; else
    // xai[a-N] -= 1;
    const uint32_t a_mod = a & (N - 1);
    const bool negate = (a >= N);

    // Set poly to zero
    if (tid < NUM_THREADS) {
        poly[tid] = 0;
        poly[tid + NUM_THREADS] = 0;
    }
    __syncthreads();

    // Set poly = (X^a - 1) mod (X^N + 1)
    if (tid == 0) {
        // Start with -1 at constant term
        poly[0] = small_ntt::P_MINUS_ONE;  // -1 mod P
        if (!negate) {
            // a < N: add 1 to coefficient a_mod
            poly[a_mod] = small_mod_add(poly[a_mod], 1);
        }
        else {
            // a >= N: subtract 1 from coefficient a_mod
            poly[a_mod] = small_mod_add(poly[a_mod], small_ntt::P_MINUS_ONE);
        }
    }
    __syncthreads();

    // Forward NTT
    if (tid < NUM_THREADS) {
        SmallForwardNTT<LOG_N>(poly, forward_root, tid);
    }
    else {
        for (int s = 0; s < SmallForwardNTTSyncCount<N>(); s++)
            __syncthreads();
    }

    // Store result to global memory
    if (tid < NUM_THREADS) {
        xai_ntt[a * N + tid] = poly[tid];
        xai_ntt[a * N + tid + NUM_THREADS] = poly[tid + NUM_THREADS];
    }
}

template <class P>
void InitializeXaiNTTForLength(std::vector<NTTValue*>& storage,
                               std::vector<SmallNTTParams>& params,
                               const int gpuNum)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t table_entries = 2 * N;
    constexpr size_t table_size = table_entries * N * sizeof(NTTValue);

    storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&storage[i], table_size));

        dim3 grid(table_entries);
        dim3 block(N >> 1);
        __ComputeXaiNTT__<N><<<grid, block>>>(storage[i],
                                              params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

void InitializeXaiNTT(const int gpuNum)
{
    InitializeXaiNTTForLength<TFHEpp::lvl1param>(
        xai_ntt_devs, g_small_ntt_params, gpuNum);
}

template <class P>
void InitializeOneTRGSWNTTForLength(std::vector<NTTValue*>& storage,
                                    std::vector<SmallNTTParams>& params,
                                    const int gpuNum)
{
    constexpr uint32_t N = P::n;
    constexpr uint32_t k = P::k;
    constexpr uint32_t l = P::l;
    constexpr uint32_t Bgbit = P::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * N * sizeof(NTTValue);

    std::vector<typename P::T> h(l);
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<typename P::T>(1)
               << (std::numeric_limits<typename P::T>::digits -
                   (i + 1) * Bgbit);
    }

    std::vector<NTTValue> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            host_polys[poly_idx * N] = torus_to_ntt_mod(h[digit]);
        }
    }

    storage.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&storage[i], total_size));

        NTTValue* d_polys;
        CuSafeCall(cudaMalloc(&d_polys, num_polys * N * sizeof(NTTValue)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(NTTValue),
                              cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(N >> 1);
        __NTTPolynomials__<N><<<grid, block>>>(storage[i], d_polys,
                                               params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

void InitializeOneTRGSWNTT(const int gpuNum)
{
    InitializeOneTRGSWNTTForLength<TFHEpp::lvl1param>(
        one_trgsw_ntt_devs, g_small_ntt_params, gpuNum);
}

#endif  // USE_FFT

void DeleteXaiNTT()
{
    for (size_t i = 0; i < xai_ntt_devs.size(); i++) {
        cudaSetDevice(i);
        cudaFree(xai_ntt_devs[i]);
    }
    xai_ntt_devs.clear();
}

void DeleteOneTRGSWNTT()
{
    for (size_t i = 0; i < one_trgsw_ntt_devs.size(); i++) {
        cudaSetDevice(i);
        cudaFree(one_trgsw_ntt_devs[i]);
    }
    one_trgsw_ntt_devs.clear();
}

//=============================================================================
// lvl02 (N=2048) key-bundle tables: xai and one_trgsw for GPUFFTForward1024
//=============================================================================

std::vector<NTTValue*> xai_ntt_devs_lvl02;
std::vector<NTTValue*> one_trgsw_ntt_devs_lvl02;

#ifdef USE_FFT
#ifdef USE_GPU_FFT

// GPU kernel: FFT of (X^a - 1) mod (X^N+1) for a=0..2N-1 with N=2048
__global__ void __ComputeXaiFFT_lvl2__(NTTValue* const xai_fft,
                                       const double2* const twist_table,
                                       const double2* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 512

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t a = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    uint32_t a_mod = a & (N - 1);
    bool negate = (a >= N);

    if (tid < HALF_N) {
        double re = 0.0, im = 0.0;

        if (tid == 0) re = -1.0;
        if (!negate) {
            if (tid == a_mod) re += 1.0;
            if (tid + HALF_N == a_mod) im += 1.0;
        }
        else {
            if (tid == a_mod) re -= 1.0;
            if (tid + HALF_N == a_mod) im -= 1.0;
        }

        double2 folded = {re, im};
        double2 tw = __ldg(&twist_table[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward1024(sh_fft, forward_root, tid);
    }
    else {
        for (int s = 0; s < 6; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        xai_fft[a * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeXaiNTT_lvl02(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t table_entries = 2 * N;
    constexpr size_t table_size = table_entries * HALF_N * sizeof(NTTValue);

    xai_ntt_devs_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&xai_ntt_devs_lvl02[i], table_size));

        dim3 grid(table_entries);
        dim3 block(HALF_N);
        __ComputeXaiFFT_lvl2__<<<grid, block>>>(xai_ntt_devs_lvl02[i],
                                                g_gpufft2048_params[i].twist,
                                                g_gpufft2048_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

// GPU kernel: FFT of lvl2param identity TRGSW polynomials (uint64_t Torus)
// Normalizes by 1/2^64 = 1/(2^32 * 2^32), uses fold+twist+GPUFFTForward1024
__global__ void __FFTPolynomials_lvl2__(NTTValue* const out,
                                        const uint64_t* const in,
                                        const double2* const twist_table,
                                        const double2* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N >> 1;  // 512

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    // Normalize: 1/2^64 = 1/(2^32) * 1/(2^32)
    constexpr double norm = 1.0 / 4294967296.0 / 4294967296.0;
    if (tid < HALF_N) {
        double re =
            static_cast<double>(static_cast<int64_t>(in[poly_idx * N + tid])) *
            norm;
        double im = static_cast<double>(
                        static_cast<int64_t>(in[poly_idx * N + tid + HALF_N])) *
                    norm;
        double2 folded = {re, im};
        double2 tw = __ldg(&twist_table[tid]);
        sh_fft[tid] = folded * tw;
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        GPUFFTForward1024(sh_fft, forward_root, tid);
    }
    else {
        for (int s = 0; s < 6; s++) __syncthreads();
    }

    if (tid < HALF_N) {
        out[poly_idx * HALF_N + tid] = sh_fft[tid];
    }
}

void InitializeOneTRGSWNTT_lvl02(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t k = TFHEpp::lvl2param::k;
    constexpr uint32_t l = TFHEpp::lvl2param::l;
    constexpr uint32_t Bgbit = TFHEpp::lvl2param::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * HALF_N * sizeof(NTTValue);

    // h[i] = 2^(64 - (i+1)*Bgbit) as uint64_t
    uint64_t h[l];
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<uint64_t>(1)
               << (std::numeric_limits<uint64_t>::digits - (i + 1) * Bgbit);
    }

    std::vector<uint64_t> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            host_polys[poly_idx * N + 0] = h[digit];
        }
    }

    one_trgsw_ntt_devs_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&one_trgsw_ntt_devs_lvl02[i], total_size));

        uint64_t* d_polys;
        CuSafeCall(
            cudaMalloc(&d_polys, num_polys * N * sizeof(uint64_t)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(HALF_N);
        __FFTPolynomials_lvl2__<<<grid, block>>>(one_trgsw_ntt_devs_lvl02[i],
                                                 d_polys,
                                                 g_gpufft2048_params[i].twist,
                                                 g_gpufft2048_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

#else  // !USE_GPU_FFT

// GPU kernel: tfhe-rs FFT of (X^a - 1) mod (X^N+1) for N=2048
__global__ void __ComputeXaiFFT_lvl2__(NTTValue* const xai_fft)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t a = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    uint32_t a_mod = a & (N - 1);
    bool negate = (a >= N);

    if (tid < HALF_N) {
        double re = 0.0, im = 0.0;

        if (tid == 0) re = -1.0;
        if (!negate) {
            if (tid == a_mod) re += 1.0;
            if (tid + HALF_N == a_mod) im += 1.0;
        }
        else {
            if (tid == a_mod) re -= 1.0;
            if (tid + HALF_N == a_mod) im -= 1.0;
        }

        sh_fft[tid] = {re, im};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    }
    else {
        for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
            __syncthreads();
    }

    if (tid < HALF_N) xai_fft[a * HALF_N + tid] = sh_fft[tid];
}

void InitializeXaiNTT_lvl02(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t table_entries = 2 * N;
    constexpr size_t table_size = table_entries * HALF_N * sizeof(NTTValue);

    xai_ntt_devs_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&xai_ntt_devs_lvl02[i], table_size));

        dim3 grid(table_entries);
        dim3 block(HALF_N);
        __ComputeXaiFFT_lvl2__<<<grid, block>>>(xai_ntt_devs_lvl02[i]);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

// GPU kernel: tfhe-rs FFT of lvl2 identity TRGSW Torus64 polynomials.
__global__ void __FFTPolynomials_lvl2__(NTTValue* const out,
                                        const uint64_t* const in)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t FFT_THREADS = HALF_N / (Degree<N>::opt / 2);

    __shared__ double2 sh_fft[HALF_N];

    const uint32_t poly_idx = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    constexpr double norm = 1.0 / 4294967296.0 / 4294967296.0;
    if (tid < HALF_N) {
        sh_fft[tid] = {
            static_cast<double>(static_cast<int64_t>(in[poly_idx * N + tid])) *
                norm,
            static_cast<double>(
                static_cast<int64_t>(in[poly_idx * N + tid + HALF_N])) *
                norm};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    }
    else {
        for (int s = 0; s < TfheRsFFTSharedSyncCount<N>(); s++)
            __syncthreads();
    }

    if (tid < HALF_N) out[poly_idx * HALF_N + tid] = sh_fft[tid];
}

void InitializeOneTRGSWNTT_lvl02(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl2param::n;
    constexpr uint32_t HALF_N = N >> 1;
    constexpr uint32_t k = TFHEpp::lvl2param::k;
    constexpr uint32_t l = TFHEpp::lvl2param::l;
    constexpr uint32_t Bgbit = TFHEpp::lvl2param::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * HALF_N * sizeof(NTTValue);

    uint64_t h[l];
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<uint64_t>(1)
               << (std::numeric_limits<uint64_t>::digits - (i + 1) * Bgbit);
    }

    std::vector<uint64_t> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            host_polys[poly_idx * N + 0] = h[digit];
        }
    }

    one_trgsw_ntt_devs_lvl02.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&one_trgsw_ntt_devs_lvl02[i], total_size));

        uint64_t* d_polys;
        CuSafeCall(
            cudaMalloc(&d_polys, num_polys * N * sizeof(uint64_t)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(HALF_N);
        __FFTPolynomials_lvl2__<<<grid, block>>>(one_trgsw_ntt_devs_lvl02[i],
                                                 d_polys);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

#endif  // USE_GPU_FFT

#else  // !USE_FFT

void InitializeXaiNTT_lvl02(const int gpuNum)
{
    InitializeXaiNTTForLength<TFHEpp::lvl2param>(
        xai_ntt_devs_lvl02, g_small_ntt_params_lvl02, gpuNum);
}

void InitializeOneTRGSWNTT_lvl02(const int gpuNum)
{
    InitializeOneTRGSWNTTForLength<TFHEpp::lvl2param>(
        one_trgsw_ntt_devs_lvl02, g_small_ntt_params_lvl02, gpuNum);
}

#endif  // USE_FFT

void DeleteXaiNTT_lvl02()
{
    for (size_t i = 0; i < xai_ntt_devs_lvl02.size(); i++) {
        cudaSetDevice(i);
        cudaFree(xai_ntt_devs_lvl02[i]);
    }
    xai_ntt_devs_lvl02.clear();
}

void DeleteOneTRGSWNTT_lvl02()
{
    for (size_t i = 0; i < one_trgsw_ntt_devs_lvl02.size(); i++) {
        cudaSetDevice(i);
        cudaFree(one_trgsw_ntt_devs_lvl02[i]);
    }
    one_trgsw_ntt_devs_lvl02.clear();
}

#endif  // USE_KEY_BUNDLE

}  // namespace cufhe
