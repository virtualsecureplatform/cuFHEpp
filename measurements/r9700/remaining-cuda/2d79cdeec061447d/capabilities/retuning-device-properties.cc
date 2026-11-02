#include <hip/hip_runtime_api.h>
#include <cstdio>
int main() { hipDeviceProp_t p{}; const auto error = hipGetDeviceProperties(&p, 0); if (error != hipSuccess) { std::fprintf(stderr, "%s\n", hipGetErrorString(error)); return 1; } std::printf("{\"multiProcessorCount\":%d,\"warpSize\":%d}\n", p.multiProcessorCount, p.warpSize); }
