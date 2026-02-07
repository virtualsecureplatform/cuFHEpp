/**
 * Small Modulus NTT implementation for cuFHE
 * Host-side initialization functions
 *
 * Modulus: P = 1048571 * 2^11 + 1 = 2147473409 (~31.0 bits)
 * P - 1 = 2^11 * 1048571 (prime factors: 2, 1048571)
 */

#include <include/ntt_small_modulus.cuh>
#include <include/error_gpu.cuh>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace cufhe {

// Define constant memory for NTT root tables
__constant__ uint32_t d_const_forward_root[1024];
__constant__ uint32_t d_const_inverse_root[1024];

// Host-side storage for small NTT parameters per GPU
std::vector<SmallNTTParams> g_small_ntt_params;

namespace {

// Bit reversal helper
int bitreverse(int index, int n_power) {
    int res = 0;
    for (int i = 0; i < n_power; i++) {
        res <<= 1;
        res = (index & 1) | res;
        index >>= 1;
    }
    return res;
}

// Modular exponentiation for 32-bit values
uint32_t mod_exp(uint64_t base, uint64_t exp, uint32_t mod) {
    uint64_t result = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp & 1) {
            result = (result * base) % mod;
        }
        exp >>= 1;
        base = (base * base) % mod;
    }
    return static_cast<uint32_t>(result);
}

// Modular inverse using extended Euclidean algorithm
uint32_t mod_inv(uint32_t a, uint32_t mod) {
    // Using Fermat's little theorem: a^(-1) = a^(p-2) mod p (for prime p)
    return mod_exp(a, mod - 2, mod);
}

// Modular multiplication for 32-bit values
uint32_t mod_mult(uint32_t a, uint32_t b, uint32_t mod) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) % mod);
}

/**
 * Find a primitive 2N-th root of unity for negacyclic NTT
 *
 * For negacyclic convolution, we need psi where psi^N = -1 (mod P)
 * which means psi^(2N) = 1 (mod P) but psi^k != 1 for k < 2N
 *
 * For P = 2147473409:
 * P - 1 = 2147473408 = 2^11 * 1048571
 *
 * Since 2^11 divides (P-1), we have primitive 2^11-th roots of unity
 * For N = 1024 = 2^10, we need 2N = 2048 = 2^11, which divides 2^11
 *
 * The primitive (P-1)-th root is found by:
 * g = generator of Z_P^*
 * Then psi_k = g^((P-1)/k) is a primitive k-th root of unity
 */
uint32_t find_primitive_root(int log_n) {
    constexpr uint32_t P = small_ntt::P;  // 2147473409
    constexpr uint32_t P_minus_1 = P - 1; // 2147473408 = 2^11 * 1048571

    // For N = 2^log_n, we need primitive 2^(log_n+1)-th root of unity
    uint32_t two_n = 1U << (log_n + 1);

    // First, find a generator g of Z_P^*
    // P - 1 = 2^11 * 1048571, so we need g where g^((P-1)/q) != 1 for each prime factor q
    uint32_t g = 3;  // Common generator candidate

    // Verify g is a generator by checking g^((P-1)/q) != 1 for prime divisors q of P-1
    // Prime divisors of P-1: 2 and 1048571
    while (true) {
        bool is_generator = true;

        // Check g^((P-1)/2) != 1
        if (mod_exp(g, P_minus_1 / 2, P) == 1) {
            is_generator = false;
        }
        // Check g^((P-1)/1048571) != 1
        if (mod_exp(g, P_minus_1 / 1048571, P) == 1) {
            is_generator = false;
        }

        if (is_generator) break;
        g++;
        if (g > 1000) {
            throw std::runtime_error("Could not find generator for small modulus");
        }
    }

    // Now compute primitive 2N-th root of unity
    // psi = g^((P-1)/(2N))
    uint32_t exponent = P_minus_1 / two_n;
    uint32_t psi = mod_exp(g, exponent, P);

    // Verify: psi^N should be P-1 (which is -1 mod P)
    uint32_t psi_N = mod_exp(psi, 1U << log_n, P);
    if (psi_N != P - 1) {
        throw std::runtime_error("Computed root does not satisfy psi^N = -1");
    }

    return psi;
}

// Storage for root tables (host-side)
std::vector<uint32_t> g_small_forward_table;
std::vector<uint32_t> g_small_inverse_table;
uint32_t g_small_n_inverse;
int g_small_log_n = 0;

/**
 * Generate NTT root tables for small modulus
 */
void GenerateSmallRootTables(int log_n) {
    if (g_small_log_n == log_n && !g_small_forward_table.empty()) {
        return;  // Already generated
    }

    const int n = 1 << log_n;
    constexpr uint32_t P = small_ntt::P;

    // Find primitive 2N-th root of unity (psi)
    uint32_t psi = find_primitive_root(log_n);
    uint32_t psi_inv = mod_inv(psi, P);

    // Compute n^(-1) mod P
    g_small_n_inverse = mod_inv(static_cast<uint32_t>(n), P);

    // Generate forward root table: psi^0, psi^1, ..., psi^(n-1) in bit-reversed order
    // For Cooley-Tukey, we need powers of psi^2 (omega = psi^2 is the N-th root of unity)
    g_small_forward_table.resize(n);
    g_small_forward_table[0] = 1;
    for (int i = 1; i < n; i++) {
        g_small_forward_table[i] = mod_mult(g_small_forward_table[i - 1], psi, P);
    }

    // Generate inverse root table: psi_inv^0, psi_inv^1, ...
    g_small_inverse_table.resize(n);
    g_small_inverse_table[0] = 1;
    for (int i = 1; i < n; i++) {
        g_small_inverse_table[i] = mod_mult(g_small_inverse_table[i - 1], psi_inv, P);
    }

    // Convert to bit-reversed order for NTT algorithm
    std::vector<uint32_t> br_forward(n);
    std::vector<uint32_t> br_inverse(n);
    for (int i = 0; i < n; i++) {
        int br_idx = bitreverse(i, log_n);
        br_forward[i] = g_small_forward_table[br_idx];
        br_inverse[i] = g_small_inverse_table[br_idx];
    }
    g_small_forward_table = std::move(br_forward);
    g_small_inverse_table = std::move(br_inverse);

    g_small_log_n = log_n;
}

}  // anonymous namespace

//=============================================================================
// Static host functions for initialization
//=============================================================================

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::Create() {
    constexpr int log_n = kLogLength;
    GenerateSmallRootTables(log_n);
}

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::Destroy() {
    for (auto& params : g_small_ntt_params) {
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
    g_small_ntt_params.clear();
    g_small_forward_table.clear();
    g_small_inverse_table.clear();
    g_small_log_n = 0;
}

template <>
void CuSmallNTTHandler<TFHEpp::lvl1param::n>::SetDevicePointers(int device_id) {
    // Resize if needed
    if (g_small_ntt_params.size() <= static_cast<size_t>(device_id)) {
        g_small_ntt_params.resize(device_id + 1);
    }

    SmallNTTParams& params = g_small_ntt_params[device_id];

    // Initialize if not already done
    if (!params.initialized) {
        // Allocate device memory for root tables
        CuSafeCall(cudaMalloc(&params.forward_root, sizeof(uint32_t) * kLength));
        CuSafeCall(cudaMalloc(&params.inverse_root, sizeof(uint32_t) * kLength));

        // Copy root tables to device
        CuSafeCall(cudaMemcpy(params.forward_root, g_small_forward_table.data(),
                              sizeof(uint32_t) * kLength, cudaMemcpyHostToDevice));
        CuSafeCall(cudaMemcpy(params.inverse_root, g_small_inverse_table.data(),
                              sizeof(uint32_t) * kLength, cudaMemcpyHostToDevice));

        // Also copy root tables to constant memory for this device
        CuSafeCall(cudaMemcpyToSymbol(d_const_forward_root, g_small_forward_table.data(),
                                      sizeof(uint32_t) * kLength));
        CuSafeCall(cudaMemcpyToSymbol(d_const_inverse_root, g_small_inverse_table.data(),
                                      sizeof(uint32_t) * kLength));

        params.n_inverse = g_small_n_inverse;
        params.initialized = true;
    }

    // Set device pointers
    forward_root_ = params.forward_root;
    inverse_root_ = params.inverse_root;
    n_inverse_ = params.n_inverse;
}

// Explicit template instantiation
template class CuSmallNTTHandler<TFHEpp::lvl1param::n>;

#ifdef USE_KEY_BUNDLE
//=============================================================================
// XaiNTT table: precomputed NTT(X^a) for a = 0..2N-1
// Used for on-the-fly keybundle computation in key-bundle bootstrapping
//=============================================================================

std::vector<NTTValue*> xai_ntt_devs;
std::vector<NTTValue*> one_trgsw_ntt_devs;

#ifdef USE_FFT
//=============================================================================
// FFT Key-bundle initialization (negacyclic FFT over double2)
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
        } else {
            if (tid == a_mod) re -= 1.0;
            if (tid + HALF_N == a_mod) im -= 1.0;
        }

        sh_fft[tid] = {re, im};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    } else {
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
__global__ void __FFTPolynomials__(
    NTTValue* const out,
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
        sh_fft[tid] = {static_cast<double>(static_cast<int32_t>(in[poly_idx * N + tid])) * norm,
                       static_cast<double>(static_cast<int32_t>(in[poly_idx * N + tid + HALF_N])) * norm};
    }
    __syncthreads();

    if (tid < FFT_THREADS) {
        NSMFFT_direct<HalfDegree<Degree<N>>>(sh_fft);
    } else {
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
        h[i] = static_cast<uint32_t>(1) << (std::numeric_limits<uint32_t>::digits - (i + 1) * Bgbit);
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
                              num_polys * N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        dim3 grid(num_polys);
        dim3 block(HALF_N);
        __FFTPolynomials__<<<grid, block>>>(one_trgsw_ntt_devs[i], d_polys);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
}

#else  // !USE_FFT
//=============================================================================
// NTT Key-bundle initialization (small modulus NTT)
//=============================================================================

// Host-side torus32 to NTT mod conversion (same formula as device version)
static uint32_t torus32_to_ntt_mod_host(uint32_t torus_val) {
    uint64_t prod = static_cast<uint64_t>(torus_val) * small_ntt::P;
    uint32_t hi = static_cast<uint32_t>(prod >> 32);
    uint32_t lo = static_cast<uint32_t>(prod);
    hi += (lo >= 0x80000000u);
    return hi;
}

// GPU kernel to NTT multiple polynomials
__global__ void __NTTPolynomials__(
    NTTValue* const out,
    const uint32_t* const in,
    const uint32_t* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    __shared__ uint32_t sh_temp[N];

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
        SmallForwardNTT32_1024(sh_temp, forward_root, tid);
    } else {
        for (int s = 0; s < 5; s++) __syncthreads();
    }

    // Store
    if (tid < NUM_THREADS) {
        out[poly_idx * N + tid] = sh_temp[tid];
        out[poly_idx * N + tid + NUM_THREADS] = sh_temp[tid + NUM_THREADS];
    }
}

__global__ void __ComputeXaiNTT__(
    NTTValue* const xai_ntt,
    const uint32_t* const forward_root)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;  // 1024
    __shared__ uint32_t poly[N];

    const uint32_t a = blockIdx.x;  // 0..2N-1
    const uint32_t tid = threadIdx.x;
    constexpr uint32_t NUM_THREADS = N >> 1;  // 512

    // Initialize polynomial: (X^a - 1) mod (X^N + 1) in Z_P
    // Following TFHEpp's XaittGen: xai[0] = -1; if (a < N) xai[a] += 1; else xai[a-N] -= 1;
    uint32_t a_mod = a & (N - 1);
    bool negate = (a >= N);

    // Set poly to zero
    if (tid < NUM_THREADS) {
        poly[tid] = 0;
        poly[tid + NUM_THREADS] = 0;
    }
    __syncthreads();

    // Set poly = (X^a - 1) mod (X^N + 1)
    if (tid == 0) {
        // Start with -1 at constant term
        poly[0] = small_ntt::P - 1;  // -1 mod P
        if (!negate) {
            // a < N: add 1 to coefficient a_mod
            poly[a_mod] = small_mod_add(poly[a_mod], 1);
        } else {
            // a >= N: subtract 1 from coefficient a_mod
            poly[a_mod] = small_mod_add(poly[a_mod], small_ntt::P - 1);
        }
    }
    __syncthreads();

    // Forward NTT
    if (tid < NUM_THREADS) {
        SmallForwardNTT32_1024(poly, forward_root, tid);
    } else {
        for (int s = 0; s < 5; s++) __syncthreads();
    }

    // Store result to global memory
    if (tid < NUM_THREADS) {
        xai_ntt[a * N + tid] = poly[tid];
        xai_ntt[a * N + tid + NUM_THREADS] = poly[tid + NUM_THREADS];
    }
}

void InitializeXaiNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t table_entries = 2 * N;  // a = 0..2N-1
    constexpr size_t table_size = table_entries * N * sizeof(NTTValue);

    xai_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&xai_ntt_devs[i], table_size));

        dim3 grid(table_entries);  // 2N blocks
        dim3 block(N >> 1);       // N/2 threads
        __ComputeXaiNTT__<<<grid, block>>>(
            xai_ntt_devs[i],
            g_small_ntt_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();
    }
}

void InitializeOneTRGSWNTT(const int gpuNum)
{
    constexpr uint32_t N = TFHEpp::lvl1param::n;
    constexpr uint32_t k = TFHEpp::lvl1param::k;
    constexpr uint32_t l = TFHEpp::lvl1param::l;
    constexpr uint32_t Bgbit = TFHEpp::lvl1param::Bgbit;
    constexpr uint32_t num_polys = (k + 1) * l * (k + 1);
    constexpr size_t total_size = num_polys * N * sizeof(NTTValue);

    // Compute h values on host: h[i] = 1 << (32 - (i+1)*Bgbit)
    // These are the diagonal elements of the identity TRGSW
    std::vector<uint32_t> h(l);
    for (uint32_t i = 0; i < l; i++) {
        h[i] = static_cast<uint32_t>(1) << (std::numeric_limits<uint32_t>::digits - (i + 1) * Bgbit);
    }

    // Build the TRGSW identity as polynomials (pre-NTT) on host
    // TRGSW layout: (k+1)*l rows, each row has (k+1) polynomials of N elements
    // Identity TRGSW: row (j*l + digit), polynomial j gets constant h[digit]
    // All other polynomials are zero
    std::vector<uint32_t> host_polys(num_polys * N, 0);
    for (uint32_t j = 0; j <= k; j++) {
        for (uint32_t digit = 0; digit < l; digit++) {
            uint32_t row = j * l + digit;
            uint32_t poly_idx = row * (k + 1) + j;
            // Constant polynomial: coefficient 0 is h[digit] (in torus)
            host_polys[poly_idx * N + 0] = torus32_to_ntt_mod_host(h[digit]);
        }
    }

    one_trgsw_ntt_devs.resize(gpuNum);
    for (int i = 0; i < gpuNum; i++) {
        cudaSetDevice(i);
        CuSafeCall(cudaMalloc(&one_trgsw_ntt_devs[i], total_size));

        // Copy host polynomials to device
        uint32_t* d_polys;
        CuSafeCall(cudaMalloc(&d_polys, num_polys * N * sizeof(uint32_t)));
        CuSafeCall(cudaMemcpy(d_polys, host_polys.data(),
                              num_polys * N * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // NTT each polynomial
        dim3 grid(num_polys);
        dim3 block(N >> 1);
        __NTTPolynomials__<<<grid, block>>>(one_trgsw_ntt_devs[i], d_polys,
                                            g_small_ntt_params[i].forward_root);
        cudaDeviceSynchronize();
        CuCheckError();

        cudaFree(d_polys);
    }
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
#endif  // USE_KEY_BUNDLE

}  // namespace cufhe
