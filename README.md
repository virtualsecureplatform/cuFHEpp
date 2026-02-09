# cuFHEpp
CUDA-accelerated Fully Homomorphic Encryption over the Torus Library.  
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
The cuFHEpp library is an open-source library for Fully Homomorphic Encryption (FHE) on CUDA-enabled GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in CUDA C++. Compared to the [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp), which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHEpp library yields almost the same performance per SM. Since the GPU has many SMs (128 in the A100), cuFHEpp delivers better performance when there are enough parallelizable tasks.

By default, cuFHEpp uses a negacyclic FFT over double-precision complex numbers powered by [GPU-FFT](https://github.com/Alisah-Ozcan/GPU-FFT). The half-size FFT trick packs N real coefficients into N/2 complex values, eliminating modular reduction overhead and leveraging native FMA instructions. GPU-FFT's FFNT class generates the root and twist tables, which are then used by custom shared-memory Cooley-Tukey/Gentleman-Sande butterfly kernels optimized for N=1024. An alternative FFT backend adapted from the [tfhe-rs](https://github.com/zama-ai/tfhe-rs) CUDA backend is available via `-DUSE_GPU_FFT=OFF`. A small-modulus NTT path (using [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT)) is also available via `-DUSE_FFT=OFF`.

Key bundle bootstrapping (`-DUSE_KEY_BUNDLE=ON`, default) processes 2 LWE bits per blind rotation step, reducing the number of iterations by half at the cost of a slightly more complex per-step computation. This yields a ~10-17% throughput improvement over the standard 1-bit blind rotation.

| [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp) | cuFHEpp (FFT+KeyBundle) | cuFHEpp (FFT) | cuFHEpp (NTT) |
|---|---|---|---|
| 10 ms | 12 ms | 14 ms | 18 ms |

### System Requirements
**The library has been tested on Ubuntu Desktop 24.04 & NVIDIA A100 only.**
GPU support requires NVIDIA Driver and NVIDIA CUDA Toolkit.

### Installation (Linux)
Do the standard CMake compilation process.
```
cd cufhepp
cmake -B build -DENABLE_TEST=ON
cd build
make
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
