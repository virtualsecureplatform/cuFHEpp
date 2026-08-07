# cuFHEpp
CUDA- and ROCm-accelerated Fully Homomorphic Encryption over the Torus Library.
This includes some bug fixes and performance improvements. 

# Citation 
We provide the BibTeX for citing this library, but since this is a forked version, we recommend that you also cite the original. 

@misc{cufhepp,
  title        = "cuFHEpp: CUDA implementation of TFHE",
  author       = "Matsuoka, Kotaro",
  year         =  2026,
  howpublished = "\url{https://github.com/virtualsecureplatform/cuFHEpp}"
}


## What is cuFHEpp?
The cuFHEpp library is an open-source library for Fully Homomorphic Encryption (FHE) on NVIDIA CUDA and AMD ROCm GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in GPU C++. Compared to the [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp), which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHEpp library yields almost the same performance per SM. Since GPUs have many parallel compute units, cuFHEpp delivers better performance when there are enough parallelizable tasks.

By default, cuFHEpp uses a negacyclic FFT over double-precision complex numbers (FFNT algorithm from [OS23]). The half-size FFT trick packs N real coefficients into N/2 complex values, eliminating modular reduction overhead and leveraging native FMA instructions. Root and twist tables are generated internally using standard C++ `<complex>` math, and the FFT itself runs as custom shared-memory Cooley-Tukey/Gentleman-Sande butterfly kernels optimized for N=512, N=1024, and N=2048. An alternative FFT backend adapted from the [tfhe-rs](https://github.com/zama-ai/tfhe-rs) CUDA backend is available via `-DUSE_GPU_FFT=OFF`. A custom small-modulus NTT path is also available via `-DUSE_FFT=OFF`.

Key bundle bootstrapping (`-DUSE_KEY_BUNDLE=ON`, default) processes 2 LWE bits per blind rotation step, reducing the number of iterations by half at the cost of a slightly more complex per-step computation. This yields a ~10-17% throughput improvement over the standard 1-bit blind rotation.

Block-binary keys are available with `-DUSE_BLOCK_BINARY=ON`. This selects TFHEpp's block-binary parameters, enables subset key switching, disables incompatible key bundling, and uses a fused block external product for Boolean gates at levels 0 and 1. Both the custom GPU FFT backend (the default) and the NTT backend (`USE_FFT=OFF`) support block-binary bootstrapping. Circuit bootstrapping, AES, and ASCON targets are not built with block-binary parameters because those CUDA paths currently require GLWE dimension 1.

## Performance

Benchmarked on **NVIDIA A100-PCIE-40GB** (108 SMs) and **Intel Xeon Silver 4216 @ 2.10 GHz**.
Each benchmark ran exclusively on the GPU — no concurrent workloads.

- **Latency**: sequential time per gate on a single stream / thread
- **Throughput**: total time ÷ total gates with all 108 SM streams active (cuFHEpp only; 3456 concurrent gates, 32 per SM)

### NAND gate comparison

| Library | Backend | N | Parameters | Latency | Throughput |
|---|---|---|---|---|---|
| [tfhe-rs](https://github.com/zama-ai/tfhe-rs) | CPU, `TFHE_LIB_PARAMETERS` | 1024 | n=630, k=1, l=3, Bg=128 | ~18 ms | — |
| [tfhe-rs](https://github.com/zama-ai/tfhe-rs) | GPU, `PARAM_GPU_MULTI_BIT_GROUP_4`¹ | 2048 | n=920, k=1, l=1, Bgbit=22, group=4 | **4.3 ms** | — |
| cuFHEpp | GPU, lvl1, FFT | 1024 | n=636, k=1, l=2, Bg=256 | 15.2 ms | 0.14 ms/gate |
| cuFHEpp | GPU, lvl1, FFT + KeyBundle | 1024 | n=636, k=1, l=2, Bg=256, KB=2 | 12.6 ms | **0.12 ms/gate** |
| cuFHEpp | GPU, lvl2, FFT | 2048 | n=636, k=1, l=4, Bg=1024 | 37–38 ms | 0.35 ms/gate |
| cuFHEpp | GPU, lvl2, FFT + KeyBundle | 2048 | n=636, k=1, l=4, Bg=1024, KB=2 | 31–33 ms | 0.29 ms/gate |

¹ `PARAM_GPU_MULTI_BIT_GROUP_4_MESSAGE_2_CARRY_2_KS_PBS_TUNIFORM_2M128`.
NAND is measured as NOT(AND); NOT is a trivial polynomial negation (~0.16 ms) with no bootstrapping.

**Note on methodology**: tfhe-rs GPU latency is single-stream (one gate at a time). cuFHEpp throughput exploits all 108 SMs simultaneously; its latency is measured on one stream out of 108.

### ROCm validation — AMD Radeon AI PRO R9700

The ROCm backend was validated on an R9700 (`gfx1201`) with ROCm 7.14, KeyBundle, 32 streams, and 1024 encrypted gates per test. All Boolean gate correctness tests passed for both lvl1 and lvl02 with the custom GPU-FFT and small-modulus NTT backends.

| Parameters | Backend | Binary gates | MUX / NMUX | Faster backend |
|---|---|---:|---:|---|
| lvl1 (N=1024) | GPU-FFT | ~3.53 ms/gate | ~6.98 ms/gate | |
| lvl1 (N=1024) | NTT | **~3.26 ms/gate** | **~6.97 ms/gate** | **NTT (~1.08x for binary gates)** |
| lvl02 (N=2048) | GPU-FFT | ~6.4 ms/gate | ~12.1 ms/gate | |
| lvl02 (N=2048) | NTT | **~5.6 ms/gate** | **~10.4 ms/gate** | **NTT (~1.15x)** |

These figures are throughput measurements from the encrypted correctness suite, not single-stream latency. NTT is slightly faster for lvl1 binary gates and approximately tied for lvl1 MUX/NMUX, while it remains faster for lvl02 on this GPU. In the separate saturated NAND benchmark (`test_backend_gate_bench`), the low-LDS lvl1 backends use two independent streams per compute unit. Median throughput was ~2.70 ms/gate for NTT and ~3.05 ms/gate for GPU-FFT, leaving NTT about 1.13x faster.

### All gates — cuFHEpp GPU, lvl1 (N=1024), FFT + KeyBundle

| Gate | Latency | Throughput |
|---|---|---|
| Binary (NAND/AND/OR/XOR/…) | ~13 ms | ~0.12 ms/gate |
| MUX / NMUX | ~25 ms | ~0.23 ms/gate |
| NOT / COPY | ~1.2 ms | ~0.01 ms/gate |

### All gates — cuFHEpp GPU, lvl1 (N=1024), FFT

| Gate | Latency | Throughput |
|---|---|---|
| Binary (NAND/AND/OR/XOR/…) | ~15 ms | ~0.14 ms/gate |
| MUX / NMUX | ~29–30 ms | ~0.27 ms/gate |
| NOT / COPY | ~1.2 ms | ~0.01 ms/gate |

### All gates — cuFHEpp GPU, lvl2 (N=2048), FFT + KeyBundle

| Gate | Latency | Throughput |
|---|---|---|
| Binary (NAND/AND/OR/XOR/…) | ~32 ms | ~0.29 ms/gate |
| MUX / NMUX | ~59–60 ms | ~0.55 ms/gate |
| NOT / COPY | ~1.0 ms | ~0.01 ms/gate |

### All gates — cuFHEpp GPU, lvl2 (N=2048), FFT

| Gate | Latency | Throughput |
|---|---|---|
| Binary (NAND/AND/OR/XOR/…) | ~37 ms | ~0.34 ms/gate |
| MUX / NMUX | ~68–69 ms | ~0.63 ms/gate |
| NOT / COPY | ~1.1 ms | ~0.01 ms/gate |

### System Requirements
**The library has been tested on Ubuntu Desktop 24.04 with NVIDIA A100, NVIDIA GeForce RTX 4070, and AMD Radeon AI PRO R9700 (`gfx1201`).**

- NVIDIA builds require an NVIDIA driver and CUDA Toolkit.
- AMD builds require a ROCm development installation containing the HIP Clang compiler, HIP headers, and `amdhip64` runtime library. The R9700 port was validated with ROCm 7.14.
- The AMD kernels currently target 32-lane wavefront GPUs. The default HIP architecture is therefore `gfx1201` for the R9700.

The R9700's 64 KiB per-workgroup LDS supports lvl1 directly. The optimized lvl1 NTT keeps its final five butterfly stages and pointwise accumulators in registers, using wave32 shuffles for cross-thread partners. The custom lvl1 GPU-FFT also keeps pointwise accumulators in registers, reducing dynamic gate scratch from 32 KiB to 16 KiB and removing the LDS barrier to a second resident block. For lvl02, the HIP kernels use a 49,160-byte low-LDS layout: transform accumulators stay in registers, the transform area is reused for extracted-TLWE scratch, and MUX/NMUX reuse a single TRLWE buffer. This path supports the default custom GPU-FFT backend and the NTT backend. The optional tfhe-rs-style FFT (`-DUSE_GPU_FFT=OFF`) remains limited to lvl1 on this GPU.

### Installation (Linux)
Do the standard CMake compilation process.
```
cd cufhepp
cmake -B build -DENABLE_TEST=ON
cd build
make
```

The default CUDA architecture list is `80;89`, covering A100 and RTX 4070.
For an RTX 4070-only build, pass `-DCMAKE_CUDA_ARCHITECTURES=89`.

The ROCm-tuned transform paths are also available on CUDA builds (always
active on HIP builds); defaults reflect RTX 4070 benchmarks:

- `USE_CUDA_PAIRED_GPU_FFT` (default ON) — run two GPU-FFT transforms per
  block in the KeyBundle path instead of idling half of the gate block.
  ~20% faster gNAND throughput on lvl1 and lvl02.
- `USE_CUDA_REGISTER_NTT_ACCUM` (default ON) — keep pointwise
  multiply-accumulate results in registers (NTT backend). Small lvl1 win.
- `USE_CUDA_LOW_LDS_BOOTSTRAP` (default ON) — use the low-shared-memory
  bootstrap layout that reuses the transform area as extracted-TLWE scratch
  and a single TRLWE buffer for MUX/NMUX. Performance-neutral, but required
  for lvl02 MUX/NMUX on GPUs with a ~99 KiB dynamic shared memory cap such as
  Ada (RTX 4070); the traditional layout requests 112 KiB and fails to launch.
- `-DUSE_CUDA_WARP_TRANSFORM=ON` (default OFF) — run the final NTT butterfly
  stages in registers with warp shuffles instead of shared-memory round trips.
  A win on RDNA4 but a regression on NVIDIA, where the 64-register ceiling for
  1024-thread blocks forces spills and lvl1 loses a resident block.

Three further lvl1 NTT optimizations (default ON, CUDA only) together cut
lvl1 gate time by ~29% on the RTX 4070 (0.0735 → 0.0522 ms/gate gNAND):

- `USE_CUDA_PAIRED_NTT` — transform two decomposition digits per forward pass
  and both output components per inverse pass, so the radix-4 stages occupy
  the whole block instead of idling half of it. −16% alone.
- `USE_CUDA_SHOUP_NTT` — Shoup precomputed-quotient multiplication for
  twiddle factors (stored alongside the root tables), replacing the two-fold
  pseudo-Mersenne reduction. −14% alone.
- `USE_CUDA_MIN_BLOCKS` — request three resident lvl1 gate blocks per SM via
  launch bounds on Ada (`sm_89`) and newer architectures. Pays another ~7% on
  the RTX 4070 when at least three streams per SM feed the GPU. Ampere
  (`sm_80`) keeps unconstrained register allocation because the 42-register
  cap reduces A100 throughput.

`./bench_rocm_ports.sh build && ./bench_rocm_ports.sh run` builds and
benchmarks these combinations for `CMAKE_CUDA_ARCHITECTURES=89`.

For ROCm on the Radeon AI PRO R9700:
```
cmake -S . -B build-rocm \
    -DCUFHE_GPU_BACKEND=HIP \
    -DCMAKE_HIP_ARCHITECTURES=gfx1201 \
    -DENABLE_TEST=ON
cmake --build build-rocm
./build-rocm/test/test_fft_roundtrip
./build-rocm/test/test_gate_gpu
./build-rocm/test/test_gate_gpu_lvl02
```

For the optimized NTT backend on R9700, configure with `-DUSE_FFT=OFF` and run
`test_gate_gpu` for lvl1 or `test_gate_gpu_lvl02` for lvl02.

If CMake cannot locate the ROCm compiler automatically, also pass
`-DCMAKE_HIP_COMPILER="$(hipconfig -l)/clang++"`. `CUFHE_GPU_BACKEND=ROCM`
is accepted as an alias for `HIP`.

For a block-binary build:
```
cmake -B build-block -DENABLE_TEST=ON -DUSE_BLOCK_BINARY=ON
cmake --build build-block
./build-block/test/test_block_binary_gpu
```

For block-binary keys with the NTT backend:
```
cmake -B build-block-ntt -DENABLE_TEST=ON -DUSE_BLOCK_BINARY=ON -DUSE_FFT=OFF
cmake --build build-block-ntt
./build-block-ntt/test/test_block_binary_gpu
```

### User Manual
See files in `test/` as examples. The library uses [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp) types for key generation, encryption, and decryption. cuFHEpp handles the GPU-accelerated gate evaluation.

```c++
#include <include/cufhe_gpu.cuh>
using namespace cufhe;

using P = TFHEpp::lvl1param;       // Parameter set for ciphertexts
using brP = TFHEpp::lvl01param;    // Blind rotation parameters
using iksP = TFHEpp::lvl10param;   // Key switching parameters

// --- Key generation (TFHEpp) ---
TFHEpp::SecretKey sk;
TFHEpp::EvalKey ek(sk);
ek.emplacebk<brP>(sk);    // Bootstrapping key
ek.emplaceiksk<iksP>(sk); // Key switching key

// --- Encryption (TFHEpp) ---
Ctxt<P> ct0, ct1, ct_out;
TFHEpp::tlweSymEncrypt<P>(ct0.tlwehost, P::μ, sk.key.get<P>());   // Encrypt 1
TFHEpp::tlweSymEncrypt<P>(ct1.tlwehost, -P::μ, sk.key.get<P>());  // Encrypt 0

// --- GPU initialization ---
Initialize(ek);  // Upload keys to GPU

Stream st;
st.Create();

// --- Gate evaluation on GPU ---
Nand<P>(ct_out, ct0, ct1, st);  // Homomorphic NAND gate

Synchronize();  // Wait for all GPU operations to complete

// --- Decryption (TFHEpp) ---
uint8_t result = TFHEpp::tlweSymDecrypt<P>(ct_out.tlwehost, sk.key.get<P>());

// --- Cleanup ---
st.Destroy();
CleanUp();
```

The lvl02 Boolean API keeps lvl0 ciphertext inputs and outputs while using
`lvl02param` for blind rotation. Generate `lvl02param`/`lvl20param` evaluation
keys, call `Initialize_lvl02(ek, sk)`, use gates such as `Nand_lvl02` and
`Mux_lvl02`, then call `CleanUp_lvl02()`. See
`test/test_gate_gpu_lvl02.cc` for a complete example.

With `USE_BLOCK_BINARY=ON`, generate the subset key-switching key instead:
```c++
ek.emplacebk<brP>(sk);
ek.emplacesubiksk<iksP>(sk);
```

#### Multi-GPU
```c++
SetGPUNum(2);  // Use 2 GPUs (call before Initialize, default is 1)
Initialize(ek);

Stream st_gpu0(0);  // Stream on GPU 0
Stream st_gpu1(1);  // Stream on GPU 1
st_gpu0.Create();
st_gpu1.Create();

Nand<P>(ct0, ct0, ct1, st_gpu0);  // Run on GPU 0
Nand<P>(ct2, ct2, ct3, st_gpu1);  // Run on GPU 1

Synchronize();
```

#### Available Gates
Binary: `And, AndNY, AndYN, Or, OrNY, OrYN, Nand, Nor, Xor, Xnor`
Ternary: `Mux, NMux`
Unary: `Not, Copy`

## Reference
[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.

[OS23]: Özcan, A. Ş., & Savaş, E. (2023). Two Algorithms for Fast GPU Implementation of NTT. Cryptology ePrint Archive, Paper 2023/1410. https://eprint.iacr.org/2023/1410
