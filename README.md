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

TFHEpp is pinned to `4309482` (upstream master as of 2026-10-18). CUDA/A100
compatibility checks and logs are recorded in
[measurements/tfhepp-4309482/validation.json](measurements/tfhepp-4309482/validation.json).

### A100 block-binary bootstrap profiles

The performance results in this section were collected with TFHEpp `3c7dca7`
(lvl1 decomposition length 3). The current TFHEpp pin `4309482` increases
block-binary lvl1 decomposition to 4; those historical timings should be
remeasured when comparing the updated parameters.

Use the `a100-ntt` preset for lvl1 throughput and small raw lvl02 NAND batches.
It enables the paired register NTT tail on Ampere, exact fused products,
monomial staging, warp rounding correction, and prepared key-switch row
descriptors reused across the block. Raw lvl02 batch NAND also
uses the cooperative row schedule on SM80: nine CTAs per sample for counts
1–12, five for 13–21, three for 22–36, and two for 37–54 on a 108-SM A100.
Full-grid residency and cooperative-launch support are checked before dispatch.
Larger batches use the ordinary fused block path.

```sh
cmake --preset a100-ntt
cmake --build build-a100-ntt-optimized --target test_bootstrap_bench \
  test_bootstrap_bench_lvl02 test_bootstrap_cooperative test_small_modulus \
  test_bootstrap_batch -j8
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
  ./build-a100-ntt-optimized/test/test_bootstrap_bench 1 432 15 batch
```

Select an idle GPU before running. The measurements below used an exclusive
A100-PCIE-40GB, CUDA 13.3, Release, SM80 device LTO, unchanged block-binary
parameters, and fresh encrypted inputs and keys. Times exclude key setup and
transfers, include host submission/synchronization, and report the median of
9 baseline or 15–27 optimized repetitions after warmup. Every run checked output
decryption. These are raw IKS→BR NAND measurements; see the API-order caveats
below. Logs and exact build options are in `measurements/a100/`.
Prepared-row scheduling passed exact comparisons with the original lvl10 and
lvl20 key switches, including boundary words and aliased extraction scratch.
Focused memory and synchronization sanitizer checks also passed.

| Raw NAND workload | Default NTT | A100 NTT preset | Reduction |
|---|---:|---:|---:|
| lvl1, one sample | 5.577 ms | 3.779 ms | 32.2% |
| lvl1, batch 432, per gate | 50.05 µs | 30.62 µs | 38.8% |
| lvl02, one sample | 29.696 ms | 5.153 ms | 82.6% |
| lvl02, batch 25, total | 25.694 ms | 11.296 ms | 56.0% |
| lvl02, batch 432, per gate | 228.55 µs | 153.73 µs | 32.7% |

The optional `a100-ntt-throughput` preset applies an A100-specific two-block
register budget to lvl1 NTT gates. It measured **29.32 µs/gate** at batch 432
(41.4% below the default NTT), with a 64-register budget per thread.
Its one-sample latency is higher, 3.91 ms, so prefer `a100-ntt` for latency.
At larger batches, it measured 28.12 µs/gate at 648 and 27.42 µs/gate at
3456. A three-block budget reached higher occupancy but was slower for both
raw and public lvl1, so the throughput preset retains two blocks.
This separate budget leaves Ada scheduling and 64-bit gate launch bounds intact.

For scalar lvl02 NAND latency, `cmake --preset a100-ntt-latency` enables
`USE_CUDA_COOPERATIVE_SCALAR=ON` and builds in `build-a100-ntt-latency`.
This option routes public and raw scalar NAND through the cooperative kernels
on A100. The ordinary presets leave it OFF to favor many concurrent calls.

| Scalar lvl02 NAND, one stream | Ordinary preset | Latency preset |
|---|---:|---:|
| Public BR→IKS | 18.91 ms | 5.14 ms |
| Raw IKS→BR | 16.64 ms | 5.80 ms |

The latency preset also improved eight-stream throughput. At 128 streams it
was slower: public 422.97 versus 293.83 µs/gate, and raw 473.21 versus
259.10 µs/gate. Use the default presets or the packed batch API for high
concurrency. Scalar dispatch retains the batch path's residency checks,
private scratch allocation, aliases and graph-capture behavior.

For **large raw lvl02 batches**, the `a100-fft` preset is faster: batch 432
measured **88.96 µs/gate** with the custom GPU FFT. Build it with
`cmake --preset a100-fft`, then use `build-a100-fft-optimized` for the build and
benchmark paths. The FFT profile measured 9.65 ms for one lvl02 sample;
the NTT cooperative profile is preferable for the measured small batches.
These backend choices keep the same cryptographic parameter types and do not
introduce a common-mask assumption or change torus precision.

The `a100-fft-throughput` preset requests two resident block-binary lvl02 FFT
blocks on A100. With the latest pre-add schedule, it measured
**97.88 µs/gate** for raw lvl02 at batch 432 and 12.66 ms for one sample,
so use the one-block preset for this raw workload. The default FFT budget remains
one block. Other GPU architectures keep their existing FFT launch bounds.

Raw FFT now also prepares pre-add key-switch rows before blind rotation.
The lvl1 table temporarily uses the full dynamic shared allocation; a CTA
barrier retires it before rotation initializes that storage. Lvl02 uses the
existing two-output native-word key-switch schedule on SM80 as well as Ada.
Exact key-switch, batch/alias, captured memory and synchronization checks passed.
Lvl1 measured
5.59 ms for one sample and 50.92 µs/gate at batch 432.

The public **BR→IKS** NAND batch API keeps lvl0 ciphertext inputs and outputs.
Its post-rotation key switch now reuses prepared row descriptors and native-word
paired-output sums on A100 with
the NTT presets and with block-binary GPU-FFT. These separate measurements use
`test_bootstrap_bench_lvl01_br_iks` or `test_bootstrap_bench_lvl02_br_iks`, with
the same parameters and backend before and after the change:

| Public NAND workload | Before | Optimized | Reduction |
|---|---:|---:|---:|
| NTT lvl1, one sample | 3.986 ms | 3.234 ms | 18.9% |
| NTT lvl1, batch 432, per gate | 34.01 µs | 30.76 µs | 9.6% |
| NTT lvl02, one sample through batch API | 28.410 ms | 5.112 ms | 82.0% |
| NTT lvl02, batch 432, per gate | 219.60 µs | 172.41 µs | 21.5% |
| FFT lvl02, one sample | 19.630 ms | 9.699 ms | 50.6% |
| FFT lvl02, batch 432, per gate | 139.42 µs | 88.92 µs | 36.2% |

The NTT throughput preset measured **29.14 µs/gate** for public lvl1 at batch
432. FFT was faster at batch 432; cooperative NTT is preferable for the
measured small public lvl02 batches. NTT public batches
passed exact scalar comparisons. Both backends passed padded strides, aliases,
encrypted truth cases and captured replay at counts 17 and 109 for both levels.

The A100 NTT preset also runs public lvl02 NAND batches cooperatively at counts
1–72, using the raw path's residency checks and private stream-ordered scratch.
The public variant combines lvl0 inputs before rotation and key-switches the
final extracted state back to lvl0. Batch 25 measured **8.79 ms total**, versus
18.63 ms with the same native-word key switch and one CTA per sample.
At counts 37–43 and 55–72, two resident CTAs use five or three row groups
per sample, with 64 registers per thread and 40 KiB dynamic shared memory.
Batch 42 measured 273.11 µs/gate versus 306.02 with the previous schedule;
batch 64 measured 249.84 µs/gate versus 291.02. Other counts retain the
previous schedules; the denser variant was slower outside these windows.
The 5.11 ms one-sample result above uses the batch API. With the default
presets, the standalone scalar public API measured 18.91 ms; the latency
preset enables cooperative scalar dispatch. Exact comparisons cover dispatch boundaries,
both aliases, forward overlap, graph replay, concurrent calls and cleanup.

For public lvl02, `a100-fft-throughput` measured **74.09 µs/gate** at batch
432 with the native-word key-switch schedule. Its one-sample latency is 11.25 ms,
so retain `a100-fft` for latency. Profiling confirmed 64 registers per thread
and 50% theoretical occupancy instead of 110 registers and 25%. A three-block
budget was slower even on a balanced batch; the throughput preset selects two.
All ten binary gates and MUX/NMux also passed fresh encrypted truth-case checks
with the public lvl02 FFT throughput profile.

Custom GPU-FFT transforms now execute block barriers uniformly across the CTA,
with butterfly work restricted to the active threads. The previous split
barrier pattern failed synchronization checking during monomial-table setup.
Tests verify bit-identical forward/inverse results with active-only and
oversized blocks at N=512, 1024, and 2048. Roundtrip/convolution, memory, and
synchronization checks passed, including the distinct N=2048 lvl2 configuration.

### RTX 4090 block-binary bootstrap optimizations

The CUDA paths now distinguish torus word width explicitly, including when
lvl1 and lvl2 share N=1024. NTT arithmetic continues to use the existing
31-bit and Goldilocks primes. Paired transforms, exact fused modular products,
register accumulators, prepared key-switch row descriptors, and workspace reuse
reduce redundant arithmetic and shared-memory traffic. FFT block products also
reuse their inverse workspace and keep partial sums in registers.

Cryptographic parameters, secret and noise distributions, torus precision,
decomposition and rounding, and security/noise-estimation inputs are unchanged.
The backend defaults remain unchanged. The options below control implementation
and scheduling for the existing block-binary parameter set.

Configure the measured RTX 4090 NTT profile as follows
(recorded with CUDA 13.3, Release, SM89, and device LTO):

```sh
cmake -S . -B build-4090-ntt -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_TEST=ON \
  -DUSE_BLOCK_BINARY=ON -DUSE_FFT=OFF \
  -DUSE_CUDA_WARP_TRANSFORM=OFF -DUSE_CUDA_WARP_TRANSFORM64=ON \
  -DUSE_CUDA_PAIRED_NTT=ON -DUSE_CUDA_PAIRED_NTT_WARP=ON \
  -DUSE_CUDA_PAIRED_NTT_MADD4=ON -DUSE_CUDA_MIN_BLOCKS=ON \
  -DCUFHE_CUDA_NTT_MIN_BLOCKS=2 \
  -DUSE_CUDA_WARP_MODSWITCH_CORRECTION=ON \
  -DUSE_CUDA_ASYNC_MONOMIALS=ON -DUSE_CUDA_PACKED_PAIRED_ROOTS=ON \
  -DUSE_CUDA_UNREDUCED_INVERSE_DIFF=ON -DUSE_CUDA_WORD32_REDUCTION=ON
cmake --build build-4090-ntt --target test_bootstrap_bench_lvl02 \
  test_bootstrap_cooperative test_bootstrap_batch test_bootstrap_public_batch \
  test_small_modulus test_keyswitch_gpu test_keyswitch_gpu_lvl20 -j8
```

Select an idle GPU using `CUDA_VISIBLE_DEVICES` before running tests or timing.
For example, with an idle GPU already selected:

```sh
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_bootstrap_bench_lvl02 1 25 45 batch
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_bootstrap_cooperative
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_bootstrap_batch
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_bootstrap_public_batch
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_small_modulus
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_keyswitch_gpu
OMP_NUM_THREADS=8 ./build-4090-ntt/test/test_keyswitch_gpu_lvl20
```

`test_bootstrap_bench` measures raw lvl1 IKS→BR NAND; the `_lvl02` target
measures raw lvl2-input/lvl2-output IKS→BR NAND. The `_br_iks` targets measure
the public BR→IKS ordering, whose lvl02 API has lvl0 inputs and outputs.
Arguments are streams, samples per stream, repetitions, gate/mode, and optional
`graph`. Each run creates fresh encrypted inputs and keys, excludes setup and
transfers from timing, and checks the final outputs by decryption. `batch`
uses one batch launch per stream; `packed` uses scalar launches on equivalent
buffers. Keep these API orders and dispatch modes separate when comparing results.

`NandBootstrapBatch` accepts GPU-resident independent ciphertexts with strides
in torus coefficients. Transfers belong to the caller. Each stride must fit a
complete TLWE, and distinct outputs must not overlap. An output may alias its
corresponding input, including a partial overlap; cross-sample aliases are not
supported. A zero count is a no-op. Both IKS→BR and BR→IKS overloads are available
in `include/bootstrap_gpu.cuh`.

For raw lvl02 NAND on an RTX 4090 (128 SMs), cooperative scheduling uses nine
CTAs per sample for counts 1–14, five for 15–25, three for 26–42, and two for
43–64. Larger batches use one CTA per ciphertext. Selection checks cooperative
launch support and full-grid residency; other configurations use the ordinary
batch path. Scratch allocation and release are stream ordered, support graph
capture, and use private per-device pools without changing the caller's pool.

Recent isolated comparisons on an idle RTX 4090 measured the following total
batch times. Each row compares the retained schedule with its immediate
predecessor, using the same cryptographic parameters and API order:

| Raw lvl02 NAND samples | Previous schedule | Retained schedule | Reduction |
|---|---:|---:|---:|
| 15 | 4.12227 ms | 3.70995 ms | 10.0% |
| 20 | 4.34568 ms | 3.89722 ms | 10.3% |
| 25 | 4.53915 ms | 4.08580 ms | 10.0% |
| 26 | 5.86661 ms | 4.92848 ms | 16.0% |
| 32 | 6.00675 ms | 5.13699 ms | 14.5% |
| 42 | 6.33419 ms | 5.64136 ms | 10.9% |
| 43 | 8.38401 ms | 7.36246 ms | 12.2% |
| 48 | 8.39170 ms | 7.52880 ms | 10.3% |
| 64 | 8.41530 ms | 8.17254 ms | 2.9% |

The final cooperative implementation passed 19,563,852 exact coefficient
comparisons on one device and 20,100,690 on two devices. Coverage includes
independent scalar references, dispatch boundaries, odd strides, padding,
in-place aliases, captured replay, concurrent streams, pool isolation, and
cleanup/reinitialization. Focused Compute Sanitizer memcheck, racecheck, and
synccheck passed; FFT and no-warp compile controls also passed. These checks
validate the tested implementation, not a new security or failure bound.

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

// --- Matching gate key generation ---
TFHEpp::SecretKey sk;
TFHEpp::EvalKey ek(sk);
ek.emplacebk<brP>(sk);    // Bootstrapping key
cufhe::EmplaceGateKeySwitchingKey<iksP>(ek, sk); // Key switching key

// --- Encryption (TFHEpp) ---
Ctxt<P> ct0, ct1, ct_out;
TFHEpp::tlweSymEncrypt<P>(ct0.tlwehost, P::μ, sk.key.getSubset<P>());   // Encrypt 1
TFHEpp::tlweSymEncrypt<P>(ct1.tlwehost, -P::μ, sk.key.getSubset<P>());  // Encrypt 0

// --- GPU initialization ---
Initialize(ek);  // Upload keys to GPU

Stream st;
st.Create();

// --- Gate evaluation on GPU ---
Nand<P>(ct_out, ct0, ct1, st);  // Homomorphic NAND gate

Synchronize();  // Wait for all GPU operations to complete

// --- Decryption (TFHEpp) ---
uint8_t result = TFHEpp::tlweSymDecrypt<P>(ct_out.tlwehost, sk.key.getSubset<P>());

// --- Cleanup ---
st.Destroy();
CleanUp();
```

TFHEpp now provides separate subset and independent secret-key chains.
Ordinary `lvl01param`/`lvl02param` gate bootstrapping keys use the subset
chain. `cufhe::EmplaceGateKeySwitchingKey<P>(ek, sk)` generates the matching
switch, selecting subset lvl10 switching when enabled. TFHEpp's direct
`ek.emplaceiksk<P>(sk)` generates independent-chain switches. Circuit and
annihilation bootstrapping use `sk.key.getIndependent<P>()` with their
matching evaluation keys.

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
