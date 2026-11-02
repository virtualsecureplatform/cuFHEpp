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

TFHEpp is pinned to `1dd9a1a` (upstream master verified on 2026-10-31).
R9700 validation and measurements are recorded in
[measurements/r9700/validation.json](measurements/r9700/validation.json).
The earlier CUDA/A100 dependency checks used `4309482`; those historical logs
remain in [measurements/tfhepp-4309482/validation.json](measurements/tfhepp-4309482/validation.json).

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
`graph`. Each run creates fresh encrypted inputs and keys and checks the final
outputs by decryption. `batch` uses one batch launch; `packed` uses scalar
launches on equivalent GPU-resident buffers. These modes exclude setup and
transfers from timing. Named public `nand`/`mux` modes exercise Ctxt wrappers
that include transfers; keep them separate from GPU-resident measurements.
Keep API orders and dispatch modes separate when comparing results.

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

### R9700 block-binary profiles

The R9700 profiles use ROCm 7.15, wave32, TFHEpp `1dd9a1a`, and the upstream
block-binary parameters: lvl1 N=1024, k=1, l=4, and lvl02 N=1024, k=2, l=3.
The dependency update preserves these parameters. Kernel tuning does not change
torus precision, key distributions, decomposition, rounding, or the NTT primes.

```sh
./bench_r9700.sh build r9700-ntt
HIP_VISIBLE_DEVICES=0 ./bench_r9700.sh check r9700-ntt
HIP_VISIBLE_DEVICES=0 ./bench_r9700.sh run r9700-ntt
```

The script selects an R9700 preset, builds its tests, and writes results under
`measurements/r9700/<preset>/`. `JOBS`, `OMP_NUM_THREADS`, `HIP_VISIBLE_DEVICES`,
`ROCM_PATH`, and `REPEATS` are configurable; timings default to 15 repetitions
after warmup. Run GPU jobs exclusively. Direct CMake use is also supported:
`cmake --preset r9700-ntt`, followed by `cmake --build build-r9700-ntt`.

These public NAND measurements use the batch API on an idle R9700. Values
are medians of two fresh-key runs, each with 15 repetitions after warmup.
One-sample rows use `r9700-ntt`; batch-512 rows use `r9700-ntt-throughput`.
Baseline and tuned builds use the same TFHEpp revision and parameters.

| Public BR→IKS NAND | Baseline NTT | R9700 NTT preset | Reduction |
|---|---:|---:|---:|
| lvl1, one sample | 4.55 ms | 2.85 ms | 37.3% |
| lvl1, batch 512, per gate | 66.89 µs | 56.32 µs | 15.8% |
| lvl02, one sample | 17.73 ms | 10.66 ms | 39.9% |
| lvl02, batch 512, per gate | 284.66 µs | 244.80 µs | 14.0% |

Raw IKS→BR results, smaller batches, scalar concurrency, and all repetitions
are retained in [comparison.json](measurements/r9700/comparison.json) and
[summary.json](measurements/r9700/summary.json). The dependency-only comparison
uses unchanged cuFHEpp source with `4309482` and `1dd9a1a`; its timings are
separate from these kernel comparisons.

The NTT presets enable paired lvl1 transforms with register tails, exact fused
products, prepared key-switch rows, wave rounding correction, and low-LDS
workspaces. The lvl02 paired path keeps three output accumulators in registers
and reuses two transform buffers for inverse transforms. Public MUX/NMux reuse one
TRLWE buffer, preserving the first extracted result in registers between
rotations. Raw MUX/NMux consume both key switches before publishing an aliased
partial result, then reuse the same TRLWE buffer. Prepared key rows may overwrite
extraction scratch only after every thread has captured its input.

`r9700-ntt` enables paired lvl02 products for latency and medium batches.
`r9700-ntt-throughput` keeps the scalar lvl02 transform schedule for large
batches. Both dispatch small lvl02 NAND batches cooperatively. HIP reports 32
multiprocessors on this 64-CU R9700: counts 1–6 use five row groups,
7–10 use three, and 11–16 use two. Direct 2/3/5/9-group comparisons selected
the five-group schedule even for one sample. Larger batches use one workgroup
per sample.
Dispatch additionally checks the ISA, wave size, cooperative support, allocator
support, and actual full-grid residency with static and dynamic LDS included.
Scratch comes from private per-device pools and supports capture, concurrent
calls, and cleanup without changing the caller's pool.

`r9700-ntt-latency` additionally enables cooperative standalone lvl02 NAND.
This favors isolated calls; many streams can serialize and become substantially
slower. In a separate 15-repetition GPU-resident scalar run, public lvl02
measured 15.02 ms with `r9700-ntt` versus 10.55 ms with the latency preset.
At 128 streams, the same profiles measured 3.46 versus 13.37 ms/gate.
Use the other presets or the packed batch API for throughput.

All new `USE_HIP_*` options default OFF and are independent of CUDA scheduling
controls. `USE_HIP_PAIRED_NTT64` requires `USE_HIP_LOW_LDS`.
`USE_HIP_MONOMIAL_STAGING` requires paired lvl02 transforms; it stages monomials
with ordinary HIP loads and a barrier, rather than CUDA asynchronous-copy
instructions. The 31-bit staging area is used only with fused products.
Staging and native-word key-switch sums remain available for
experiments but are excluded from presets unless the measurements justify them.
`CUFHE_HIP_MIN_WAVES_PER_EU` controls HIP gate launch bounds independently of
CUDA residency settings. A four-wave candidate did not show a repeatable gain,
so the presets retain one wave. `CUFHE_HIP_COOPERATIVE_GROUPS` selects automatic
scheduling (0) or a fixed 2/3/5/9-group schedule for measurement; every selection
still obeys the runtime residency checks.

The FFT profile retains the existing register accumulators and inverse-workspace
reuse. Prepared-row and wave-correction candidates passed correctness checks but
did not improve the initial FFT sweep, so `r9700-fft` leaves those additions off.
The baseline presets explicitly disable new features for comparable timing.
The original NTT baseline's lvl02 MUX exceeds the R9700 LDS limit; the optimized
low-LDS profiles fix that launch limitation.

Measurements keep raw IKS→BR and public BR→IKS ordering separate. The public
lvl02 API still accepts and returns lvl0 ciphertexts. Reported batch and packed
scalar timings use fresh encrypted inputs and keys, exclude setup/transfers,
include submission and synchronization, and check decryption. Named-wrapper
experiments that include transfers are kept separate. Exact arithmetic, batch padding,
input aliases, graph replay, cooperative boundaries, and pool isolation are
covered separately. Logs, build options, dependency comparisons, and timing
summaries are in [measurements/r9700](measurements/r9700/).

### Historical ROCm validation — AMD Radeon AI PRO R9700

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
- AMD builds require a ROCm development installation containing the HIP Clang compiler, HIP headers, and `amdhip64` runtime library. The current block-binary profiles were validated with ROCm 7.15; older KeyBundle results used ROCm 7.14.
- The AMD kernels currently target 32-lane wavefront GPUs. The default HIP architecture is therefore `gfx1201` for the R9700.

The R9700 exposes 64 KiB of LDS per workgroup. The block-binary NTT presets use 16 KiB of dynamic LDS for paired lvl1 gates and 40,968 bytes for lvl02 gates, with static storage counted separately by the occupancy checks. Register accumulators and workspace reuse keep lvl02 MUX/NMux within this limit. The custom GPU-FFT already uses register accumulators and inverse-workspace reuse on HIP. The optional tfhe-rs-style FFT (`-DUSE_GPU_FFT=OFF`) remains limited to lvl1 on this GPU. Historical KeyBundle configurations use different scratch sizes from these current block-binary profiles.

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

For current block-binary NTT tuning on R9700, use `cmake --preset r9700-ntt`
and the `bench_r9700.sh` commands above. This also enables the low-LDS layout
needed by lvl02 MUX/NMux.

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

### Remaining CUDA ideas on R9700 (experimental)

The second R9700 pass exposes the remaining schedules independently. These
options default OFF; the existing production presets retain their measured
settings until the new comparisons meet the promotion criteria. The TFHEpp
revision, parameters, torus rounding, and public gate interfaces are unchanged.

| HIP option (`USE_HIP_` prefix) | Candidate behavior |
|---|---|
| `MULTIOUT_KEYSWITCH` | Traverse prepared descriptors once for two outputs, or five adjacent outputs for lvl02 MUX/NMux |
| `WAVE_KEYSWITCH` | Broadcast prepared descriptors within complete wave32 groups; usable independently of multiple-output scheduling |
| `OUTPUT_EXPONENT_CACHE` | Cache public lvl02 binary-gate exponents in the output workspace after capturing overlapping inputs |
| `SCALAR_MONOMIAL_CACHE64` | Retain scalar Goldilocks monomials and input coefficients, reconstruct adjacent monomials exactly |
| `REGISTER_NTT_IO64` | Consume forward pairs and inverse return pairs directly in registers, including the low-LDS accumulator layout |
| `REGISTER_MONOMIALS31` | Reuse monomials and input coefficients, select consumer-specific products and inverse additions without requiring staging |
| `NATIVE_NTT64` | Use explicit 32-bit carry/borrow chains for Goldilocks arithmetic and four-product accumulation |
| `FAST_TORUS64` | Use the exact word-based Goldilocks-to-torus conversion |
| `SCALED_INVERSE64` | Normalize the final Goldilocks inverse butterfly through exact division by a power of two |
| `PACKED_PAIRED_ROOTS` | Load a 31-bit root and its Shoup quotient together from the existing packed table |
| `UNREDUCED_INVERSE_DIFF` | Pass the unreduced inverse difference directly to Shoup multiplication |
| `MONOMIAL_PREFETCH` | Issue aligned vector monomial loads before decomposition and publish to LDS after the first forward transform |

Multiple-output and wave key switching require prepared key switching.
Register-31 caching requires paired/fused-31 support. Register-64 I/O requires
scalar caching and low LDS. Prefetch requires monomial staging (and its existing
paired-64 prerequisite). Packed roots and unreduced differences require Shoup
NTT. The scalar-64 candidates are measured with paired-64 dispatch disabled so
that the scalar branch actually executes.

The experimental `r9700-remaining-all` preset exercises the scalar-64 and
paired-31 paths together. `r9700-remaining-prefetch` exercises staged paired-64
and paired-31 products. Neither is a performance recommendation. Focused tests
record the policies actually executed; disabled cache paths report `SKIP`.

The per-feature runner builds isolated configurations and uses an exclusive GPU
lock. For example:

```sh
./bench_r9700.sh remaining build --profiles r9700-ntt-throughput --variants baseline scalar-cache register-io --focused
./bench_r9700.sh remaining check --profiles r9700-ntt-throughput --variants baseline scalar-cache register-io --focused
./bench_r9700.sh remaining screen --profiles r9700-ntt-throughput --variants baseline scalar-cache register-io
./bench_r9700.sh remaining report --profiles r9700-ntt-throughput --variants baseline scalar-cache register-io
```

Use `run` for the full batch/concurrency grid and omit `--focused` from build/check
for exhaustive batch/cooperative validation of final configurations. Screening
and full measurements default to three independent fresh-key runs with 15
repetitions after warmup. Each candidate is alternated with its matched control.
Register I/O compares against scalar caching; prefetch compares against staging;
FFT key-switch schedules compare against prepared-key switching. Logs, commands,
effective caches, source hashes, and summaries are stored in
`measurements/r9700/remaining-cuda/<source fingerprint>/`.

The benchmark also accepts `packed-mux` and `packed-nmux` for resident-buffer
measurements without timed transfers. `CUFHE_BENCH_RESOURCES=1` records kernel
registers, private bytes, static/dynamic LDS, geometry, and the runtime occupancy
limit from a captured submission; the runner enables this automatically. No NVIDIA runtime
validation is implied by CUDA compile checks.

A preset promotion requires a repeatable improvement of at least 3% in the
intended workload in all three fresh-key runs, with no reproducible regression
above 2% elsewhere enabled by the same profile. Uncertain points receive 30
repetitions and otherwise remain experimental. The report identifies screening
candidates; full validation and profile-specific review are required before
changing a preset.

Direct asynchronous global-to-LDS copying is unavailable through the installed
gfx1201 builtins. The capability probe and native arithmetic instruction captures
are retained with the measurements. Software prefetch supplies the corresponding
load-overlap experiment without claiming CUDA `cp.async` equivalence.

The register-return regression fixture deliberately delays peer-half LDS readers
before workspace reuse. A control with the reuse barrier removed reproduces
coefficient mismatches; the corrected schedule passes the same stress case.

The runner also accepts `waves2`, `waves3`, `waves4`, `all-waves2`, `all-waves3`,
`all-waves4`, and `groups2`, `groups3`, `groups5`, `groups9` as variants for the
existing occupancy controls. `all-waves*` compares against the combined `all`
configuration. Effective caches and captured kernel attributes record the
actual resource budgets; register counts alone do not determine residency.

Feature qualification is complete. Candidates selected from the individual
screens passed exhaustive correctness checks, then completed the 116-point
original-backend grid and required longer reruns. Paired multiple-output
switching, its wave combination, throughput 31-bit register caching, and the FFT wave
combination all failed the promotion gate through median regressions or
unresolved variation. Earlier combined latency/throughput and paired native
arithmetic candidates also failed their gates. The new feature switches remain
experimental, and no new production preset has been promoted.

The final feature comparisons against the original backend are below. Each
row covers 116 workload points and three independent fresh-key runs, including
the required 30-repetition reruns. Wins require at least 3% in every run;
median losses above 2% and unresolved losses prevent promotion.

| Profile / candidate | Winning points | Median regressions | Unresolved regressions |
|---|---:|---:|---:|
| Paired NTT / multiple outputs | 45 | 2 | 7 |
| Paired NTT / multiple outputs + wave broadcast | 45 | 1 | 8 |
| Throughput NTT / register-31 caching | 17 | 8 | 18 |
| FFT / multiple outputs + wave broadcast | 37 | 5 | 24 |

The [per-feature outcomes](measurements/r9700/remaining-cuda/f3f746602ae5f7c2/completed-feature-outcomes.json)
reconcile all 26 feature experiments with their completed qualification results
and retain references to the original screening evidence.

Launch-setting screens are complete for all four profiles. Waves 2, 3, and 4
produced no repeatable screen gains in any profile; forced groups 2, 3, 5, and 9
also produced none in the three NTT profiles. Final configuration qualification
is complete, and the existing production settings are retained. Captured launches limit
forced groups of 2, 3, 5, and 9 to batches of 16, 10, 6, and 7 respectively in
all three NTT profiles. The public API's initial count guard of 21 does not
guarantee that a cooperative launch fits. Completed NTT group screens cover
every batch count from 1 through 22, including the first nine-group fallback at
eight, and found no repeatable gains. Final NTT qualification uses the same batch
range. Every retained configuration passed exhaustive gate, batch, and graph
checks. The final comparisons against the original backend are:

| Retained profile | Workload points | Points rerun with 30 repetitions | Median regressions | Unresolved points |
|---|---:|---:|---:|---:|
| Paired NTT | 156 | 20 | 5 | 12 |
| Throughput NTT | 156 | 33 | 7 | 17 |
| Latency NTT | 156 | 15 | 3 | 6 |
| FFT | 116 | 32 | 2 | 20 |

These mixed timings do not establish performance equivalence with the original
backend. No new feature or launch setting met the promotion gate. The
[final measurement audit](measurements/r9700/remaining-cuda/f3f746602ae5f7c2/final-measurement-audit.json)
verifies 3,504 selected raw logs, executable hashes, actual resources, and the
required longer comparisons. The [requirement audit](measurements/r9700/remaining-cuda/f3f746602ae5f7c2/requirements-completion-audit.json)
records the implementation, validation, and target limitations for every plan item.
CUDA NTT and FFT compile checks include SM80 and
SM89 images; NVIDIA runtime validation was unavailable.

Latency raw level-01 NMux at 128 streams lost 2.2–3.0% in all three original
longer runs. Its selected device kernel has identical normalized instructions
and captured resources to the original. In an isolated confirmation, the
original/candidate differences ranged from a 4.7% loss to a 4.1% gain, while
an identical-binary original/original control showed 3.0–3.9% differences.
The earlier loss did not reproduce consistently, and these controls do not
establish a code-caused slowdown or performance equivalence. Both the original
results and the [isolated confirmation](measurements/r9700/remaining-cuda/f3f746602ae5f7c2/latency-nmux-baseline-review/confirmation/summary.json)
are preserved. A separate consumer-schedule experiment failed its targeted
screens and was not adopted.

Prepared MUX key switching executes only when its descriptor rows fit in the
available workspace. With the current lvl02 parameters, the raw MUX pre-add
workspace is too small and uses the original fallback. The public post-rotation
MUX workspace can fit those rows. Performance reports must distinguish these
paths before attributing a timing change to a prepared-key schedule.
