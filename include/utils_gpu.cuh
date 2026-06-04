/**
 * Copyright 2018 Wei Dai <wdai3141@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <stdint.h>

namespace cufhe {

#ifdef __CUDACC__

__device__ inline uint32_t ThisThreadRankInBlock()
{
    return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
}

__device__ inline uint32_t ThisBlockSize()
{
    return blockDim.x * blockDim.y * blockDim.z;
}

/**
 * Convert double to uint64_t with modular wrapping arithmetic.
 *
 * Equivalent to (uint64_t)(int64_t)round(d), but handles values outside the
 * signed 64-bit range by extracting IEEE 754 mantissa/exponent bits so the
 * result wraps mod 2^64.
 */
__device__ __forceinline__ uint64_t double_to_torus64(double d)
{
    uint64_t i = __double_as_longlong(d);
    int expo = ((int)(i >> 52)) & 0x7FF;
    if (expo == 0) return 0;
    uint64_t m = (i & 0x000FFFFFFFFFFFFFull) | 0x0010000000000000ull;
    int shift = expo - 1075;
    uint64_t val;
    if (shift >= 64)
        val = 0;
    else if (shift >= 0)
        val = m << shift;
    else if (shift > -64)
        val = m >> (-shift);
    else
        val = 0;
    return (i >> 63) ? -val : val;
}

#endif  // __CUDACC__

}  // namespace cufhe
