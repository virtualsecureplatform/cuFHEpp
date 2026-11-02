#include <hip/hip_runtime.h>
extern "C" __global__ void ProbeAsync(int* result) {
#if __has_builtin(__builtin_amdgcn_global_load_async_to_lds_b128)
    result[0] = __builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_async_to_lds_b128);
#else
    result[0] = -1;
#endif
#if __has_builtin(__builtin_amdgcn_global_load_lds)
    result[1] = __builtin_amdgcn_is_invocable(__builtin_amdgcn_global_load_lds);
#else
    result[1] = -1;
#endif
}
