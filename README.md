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
The cuFHEpp library is an open-source library for Fully Homomorphic Encryption (FHE) on CUDA-enabled GPUs. It implements the TFHE scheme [CGGI16][CGGI17] proposed by Chillotti et al. in CUDA C++. Compared to the [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp), which reports the fastest gate-by-gate bootstrapping performance on CPUs, the cuFHEpp library yields almost the same performance per SM. Since the GPU has many SMs (128 in the A100), cuFHEpp delivers better performance when there are enough parallelizable tasks. The cuFHEpp library benefits greatly from an improved CUDA implementation of the number-theoretic transform (NTT), [GPU-NTT](https://github.com/Alisah-Ozcan/GPU-NTT).

| [TFHEpp](https://github.com/virtualsecureplatform/TFHEpp) | cuFHE |
|---|---|
| 10 ms | 13 ms |

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
Use files in `cufhe/test/` as examples. To summarize, follow the following function calling procedures.
```c++
SetGPUNum(2); //Set number of gpu to use. Now use 2 GPU. If you do not specify GPU number, use only 1 GPU.
SetSeed(); // init random generator seed
PriKey pri_key;
PubKey pub_key;
KeyGen(pub_key, pri_key); // key generation
// alternatively, write / read key files
Ptxt pt[2];
pt[0] = 0; // 0 or 1, single bit
pt[1] = 1;

Ctxt ct[4];
Encrypt(ct[0], pt[0], pri_key);
Encrypt(ct[1], pt[1], pri_key);
Encrypt(ct[2], pt[0], pri_key);
Encrypt(ct[3], pt[1], pri_key);

Initialize(pub_key); // for GPU library

Stream stream_gpu_0(0); //Create Stream runs on GPU0
stream_gpu_0.Create();

Stream stream_gpu_1(1); //Create Stream runs on GPU1
stream_gpu_1.Create();

Nand(ct[0], ct[0], ct[1], stream_gpu_0); //Run Nand on GPU0
Nand(ct[2], ct[2], ct[3], stream_gpu_1); //Run Nand on GPU1

Synchronize(); //Synchronize All GPU

Decrypt(pt[0], ct[0], pri_key);

stream_gpu_0.Destroy(); //Destroy Stream
stream_gpu_1.Destory(); 

CleanUp(); // for GPU library
```

Currently implemented gates are `And, AndNY, AndYN, Or, OrNY, OrYN Nand, Nor, Xor, Xnor, Not, Mux, Copy`.

## Reference
[CGGI16]: Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2016, December). Faster fully homomorphic encryption: Bootstrapping in less than 0.1 seconds. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 3-33). Springer, Berlin, Heidelberg.

[CGGI17]: Chillotti, I., Gama, N., Georgieva, M., & Izabachène, M. (2017, December). Faster Packed Homomorphic Operations and Efficient Circuit Bootstrapping for TFHE. In International Conference on the Theory and Application of Cryptology and Information Security (pp. 377-408). Springer, Cham.
