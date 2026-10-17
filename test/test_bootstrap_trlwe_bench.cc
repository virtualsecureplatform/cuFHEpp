// Compile the shared host driver without reference kernels so device linking
// matches ordinary C++ applications using the library.
#define CUFHE_TRLWE_BENCH_ONLY
#include "test_bootstrap_trlwe.cu"
