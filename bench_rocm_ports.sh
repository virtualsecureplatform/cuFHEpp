#!/usr/bin/env bash
# Build and benchmark CUDA variants of the ROCm-tuned transform paths.
#
# Usage:
#   ./bench_rocm_ports.sh build   # configure + build all variants (no GPU needed)
#   ./bench_rocm_ports.sh run     # run test_backend_gate_bench per variant, print summary
#   ./bench_rocm_ports.sh check   # run lvl1/lvl02 gate correctness tests per variant
#
# Variants are built into build-bench/<name> for CMAKE_CUDA_ARCHITECTURES=89.

set -u
cd "$(dirname "$0")"

ARCH=${ARCH:-89}
JOBS=${JOBS:-$(nproc)}
OUT=build-bench

# Pin every port flag explicitly so variant names stay meaningful regardless
# of the project's defaults.
ALL_OFF="-DUSE_CUDA_WARP_TRANSFORM=OFF -DUSE_CUDA_REGISTER_NTT_ACCUM=OFF -DUSE_CUDA_LOW_LDS_BOOTSTRAP=OFF -DUSE_CUDA_PAIRED_GPU_FFT=OFF -DUSE_CUDA_PAIRED_NTT=OFF -DUSE_CUDA_SHOUP_NTT=OFF -DUSE_CUDA_MIN_BLOCKS=OFF"
FFT="-DUSE_FFT=ON -DUSE_GPU_FFT=ON"
NTT="-DUSE_FFT=OFF -DUSE_GPU_FFT=OFF"
declare -A VARIANTS=(
    [fft-base]="$FFT $ALL_OFF"
    [fft-lowlds]="$FFT $ALL_OFF -DUSE_CUDA_LOW_LDS_BOOTSTRAP=ON"
    [fft-paired]="$FFT $ALL_OFF -DUSE_CUDA_PAIRED_GPU_FFT=ON"
    [fft-lowlds-paired]="$FFT $ALL_OFF -DUSE_CUDA_LOW_LDS_BOOTSTRAP=ON -DUSE_CUDA_PAIRED_GPU_FFT=ON"
    [ntt-base]="$NTT $ALL_OFF"
    [ntt-warp]="$NTT $ALL_OFF -DUSE_CUDA_WARP_TRANSFORM=ON"
    [ntt-regaccum]="$NTT $ALL_OFF -DUSE_CUDA_REGISTER_NTT_ACCUM=ON"
    [ntt-warp-regaccum]="$NTT $ALL_OFF -DUSE_CUDA_WARP_TRANSFORM=ON -DUSE_CUDA_REGISTER_NTT_ACCUM=ON"
    [ntt-regaccum-lowlds]="$NTT $ALL_OFF -DUSE_CUDA_REGISTER_NTT_ACCUM=ON -DUSE_CUDA_LOW_LDS_BOOTSTRAP=ON"
    [ntt-all]="$NTT $ALL_OFF -DUSE_CUDA_WARP_TRANSFORM=ON -DUSE_CUDA_REGISTER_NTT_ACCUM=ON -DUSE_CUDA_LOW_LDS_BOOTSTRAP=ON"
    [ntt-gen2]="$NTT $ALL_OFF -DUSE_CUDA_REGISTER_NTT_ACCUM=ON -DUSE_CUDA_LOW_LDS_BOOTSTRAP=ON -DUSE_CUDA_PAIRED_NTT=ON -DUSE_CUDA_SHOUP_NTT=ON -DUSE_CUDA_MIN_BLOCKS=ON"
)
ORDER="fft-base fft-lowlds fft-paired fft-lowlds-paired ntt-base ntt-warp ntt-regaccum ntt-warp-regaccum ntt-regaccum-lowlds ntt-all ntt-gen2"

cmd=${1:-build}

case "$cmd" in
build)
    for v in $ORDER; do
        echo "=== configure+build $v ==="
        cmake -S . -B "$OUT/$v" -DCMAKE_CUDA_ARCHITECTURES=$ARCH \
            -DENABLE_TEST=ON ${VARIANTS[$v]} > "$OUT-$v-cfg.log" 2>&1 \
            || { echo "!! configure failed: $v (see $OUT-$v-cfg.log)"; continue; }
        mv "$OUT-$v-cfg.log" "$OUT/$v/configure.log"
        cmake --build "$OUT/$v" -j"$JOBS" \
            --target test_backend_gate_bench test_gate_gpu test_gate_gpu_lvl02 \
            > "$OUT/$v/build.log" 2>&1 \
            || { echo "!! build failed: $v (see $OUT/$v/build.log)"; continue; }
        echo "ok: $v"
    done
    ;;
run)
    mkdir -p "$OUT/results"
    for v in $ORDER; do
        bin="$OUT/$v/test/test_backend_gate_bench"
        [ -x "$bin" ] || { echo "missing $bin (run '$0 build' first)"; continue; }
        echo "=== bench $v ==="
        "$bin" | tee "$OUT/results/$v.txt"
    done
    echo
    echo "=== summary (ms/gate) ==="
    printf '%-22s %-14s %s\n' variant throughput latency
    for v in $ORDER; do
        r="$OUT/results/$v.txt"
        [ -f "$r" ] || continue
        tp=$(grep -m1 '^Throughput' "$r" | awk '{print $2}')
        lat=$(grep -m1 '^Latency' "$r" | awk '{print $2}')
        printf '%-22s %-14s %s\n' "$v" "${tp:-?}" "${lat:-?}"
    done
    ;;
check)
    for v in $ORDER; do
        for t in test_gate_gpu test_gate_gpu_lvl02; do
            bin="$OUT/$v/test/$t"
            [ -x "$bin" ] || continue
            echo "=== $v / $t ==="
            "$bin" > "$OUT/$v/$t.out" 2>&1 && echo PASS || \
                { echo "FAIL (see $OUT/$v/$t.out)"; tail -3 "$OUT/$v/$t.out"; }
        done
    done
    ;;
*)
    echo "usage: $0 [build|run|check]"; exit 1 ;;
esac
