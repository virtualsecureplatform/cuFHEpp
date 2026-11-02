#!/usr/bin/env bash
# Reproducible R9700 checks and measurements. Run each GPU job exclusively.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${1:-} == remaining ]]; then
    shift
    exec python3 ./bench_r9700_remaining.py "$@"
fi
action=${1:-build}
profiles=("${@:2}")
if ((${#profiles[@]} == 0)); then
    profiles=(r9700-ntt-baseline r9700-ntt r9700-ntt-throughput r9700-ntt-latency r9700-fft-baseline r9700-fft)
fi
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib:${LD_LIBRARY_PATH:-}"
repeats=${REPEATS:-15}
run_benchmark() {
    local log=$1
    shift
    printf '%q ' "$@" > "${log%.log}.command"
    printf '\n' >> "${log%.log}.command"
    "$@" > "$log" 2>&1
}
for profile in "${profiles[@]}"; do
    case "$profile" in r9700-*) ;; *) echo "Expected an r9700 preset: $profile" >&2; exit 2 ;; esac
    build="build-$profile"
    result="measurements/r9700/$profile"
    mkdir -p "$result"
    targets=(test_bootstrap_bench test_bootstrap_bench_lvl02 test_bootstrap_bench_lvl01_br_iks test_bootstrap_bench_lvl02_br_iks test_bootstrap_batch test_bootstrap_public_batch test_bootstrap_binary test_bootstrap_public_binary test_bootstrap_mux test_bootstrap_public_mux test_keyswitch_gpu test_keyswitch_gpu_lvl20)
    if [[ "$profile" == *ntt* || "$profile" == r9700-remaining-* ]]; then
        targets+=(test_small_modulus)
        if [[ "$profile" != *baseline* ]]; then targets+=(test_bootstrap_cooperative test_bootstrap_public_cooperative); fi
    else targets+=(test_fft_roundtrip); fi
    case "$action" in
        build)
            cmake --preset "$profile" > "$result/configure.log" 2>&1
            cmake --build "$build" --target "${targets[@]}" -j"${JOBS:-8}" > "$result/build.log" 2>&1
            ;;
        check)
            for test in "${targets[@]}"; do
                [[ "$test" == test_bootstrap_bench* ]] && continue
                if [[ "$profile" == r9700-ntt-baseline && ( "$test" == test_bootstrap_mux || "$test" == test_bootstrap_public_mux ) ]]; then
                    echo "$profile $test skipped: original lvl02 MUX exceeds 64 KiB LDS"
                    continue
                fi
                "$build/test/$test" > "$result/$test.log" 2>&1
                echo "$profile $test PASS"
            done
            if [[ "$profile" != r9700-ntt-baseline ]]; then
                "$build/test/test_bootstrap_mux" graph > "$result/mux-graph.log" 2>&1
                "$build/test/test_bootstrap_public_mux" graph > "$result/public-mux-graph.log" 2>&1
            fi
            ;;
        run)
            for suffix in '' _lvl02 _lvl01_br_iks _lvl02_br_iks; do
                binary="$build/test/test_bootstrap_bench$suffix"
                for count in 1 2 3 4 6 7 10 11 16 17 32 64 128 512; do
                    run_benchmark "$result/batch${suffix}-b${count}.log" "$binary" 1 "$count" "$repeats" batch
                done
                for streams in 1 8 32 128; do
                    run_benchmark "$result/scalar${suffix}-s${streams}.log" "$binary" "$streams" 1 "$repeats" packed
                done
                run_benchmark "$result/graph${suffix}.log" "$binary" 1 17 "$repeats" batch graph
            done
            ;;
        *) echo "usage: $0 [build|check|run] [r9700-preset ...]" >&2; exit 2 ;;
    esac
    echo "$profile $action complete"
done
