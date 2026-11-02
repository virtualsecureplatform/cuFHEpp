#!/usr/bin/env python3
"""Build, validate and compare independently selectable R9700 CUDA ports.

GPU commands hold an exclusive advisory lock. Results are append-only per source
fingerprint: changing the source never reuses a preceding validation receipt.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NEW = '''MULTIOUT_KEYSWITCH WAVE_KEYSWITCH OUTPUT_EXPONENT_CACHE
SCALAR_MONOMIAL_CACHE64 REGISTER_NTT_IO64 REGISTER_MONOMIALS31 NATIVE_NTT64
FAST_TORUS64 SCALED_INVERSE64 PACKED_PAIRED_ROOTS UNREDUCED_INVERSE_DIFF
MONOMIAL_PREFETCH'''.split()
VARIANTS = {
    'baseline': {},
    'prepared': {'PREPARED_KEYSWITCH': 'ON'},
    'multi': {'MULTIOUT_KEYSWITCH': 'ON'},
    'wave': {'WAVE_KEYSWITCH': 'ON'},
    'multi-wave': {'MULTIOUT_KEYSWITCH': 'ON', 'WAVE_KEYSWITCH': 'ON'},
    'exponents': {'OUTPUT_EXPONENT_CACHE': 'ON'},
    'scalar-cache': {'SCALAR_MONOMIAL_CACHE64': 'ON'},
    'register-io': {'SCALAR_MONOMIAL_CACHE64': 'ON', 'REGISTER_NTT_IO64': 'ON'},
    'register31': {'REGISTER_MONOMIALS31': 'ON'},
    'native64': {'NATIVE_NTT64': 'ON'},
    'torus64': {'FAST_TORUS64': 'ON'},
    'inverse64': {'SCALED_INVERSE64': 'ON'},
    'roots31': {'PACKED_PAIRED_ROOTS': 'ON'},
    'difference31': {'UNREDUCED_INVERSE_DIFF': 'ON'},
    'staging': {'MONOMIAL_STAGING': 'ON'},
    'prefetch': {'MONOMIAL_STAGING': 'ON', 'MONOMIAL_PREFETCH': 'ON'},
    'all': {name: 'ON' for name in NEW if name != 'MONOMIAL_PREFETCH'},
}
BENCHES = ['test_bootstrap_bench' + x for x in ('', '_lvl02', '_lvl01_br_iks', '_lvl02_br_iks')]
TESTS = ['test_keyswitch_gpu', 'test_keyswitch_gpu_lvl20', 'test_bootstrap_binary',
         'test_bootstrap_public_binary', 'test_bootstrap_mux', 'test_bootstrap_public_mux',
         'test_bootstrap_batch', 'test_bootstrap_public_batch']
COUNTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 17, 32, 64, 128, 512]


def applicable(profile, variant):
    if 'fft' in profile:
        return variant in ('baseline', 'prepared', 'multi', 'wave', 'multi-wave')
    if variant == 'prepared':
        return False  # already enabled by the NTT presets
    if variant in ('scalar-cache', 'register-io'):
        return profile == 'r9700-ntt-throughput'
    if variant in ('staging', 'prefetch'):
        return profile != 'r9700-ntt-throughput'
    return True


def control(profile, variant):
    if variant == 'register-io':
        return 'scalar-cache'
    if variant == 'prefetch':
        return 'staging'
    if 'fft' in profile and variant in ('multi', 'wave', 'multi-wave'):
        return 'prepared'
    return 'baseline'


def source_manifest():
    paths = subprocess.check_output(['git', 'ls-files', 'include', 'src', 'test', 'CMakeLists.txt', 'CMakePresets.json'], cwd=ROOT, text=True).splitlines()
    paths += ['bench_r9700_remaining.py']
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    return dict(files=hashes, tfhepp=subprocess.check_output(['git', '-C', 'thirdparties/TFHEpp', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


def execute(argv, path, env):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix('.command').write_text(shlex.join(map(str, argv)) + '\n')
    with path.open('w') as out:
        subprocess.run(argv, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)


def parse(path, repeats):
    body = path.read_text()
    samples = [float(x) for x in re.findall(r'^repeat=\d+ ms/gate=([\d.e+-]+)', body, re.M)]
    result = re.search(r'dispatch=(\S+) streams=(\d+) gates=(\d+) median_ms/gate=([\d.e+-]+) errors=(\d+)', body)
    if result is None or len(samples) != repeats or int(result[5]) or 'timed_transfers=0' not in body.splitlines()[0]:
        raise RuntimeError(f'Incomplete, non-resident, or failing measurement: {path}')
    return dict(median_ms_per_gate=float(result[4]), samples_ms_per_gate=samples,
                dispatch=result[1], streams=int(result[2]), gates=int(result[3]), errors=0)


def cases(full):
    result = [('batch', 1, n, False) for n in (COUNTS if full else [1, 17, 512])]
    result += [('packed', n, 1, False) for n in ([1, 8, 32, 128] if full else [1, 128])]
    result += [(mode, n, 1, False) for mode in ['packed-mux', 'packed-nmux'] for n in ([1, 8, 32, 128] if full else [1])]
    result += [('batch', 1, 17, True)]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['build', 'check', 'screen', 'run', 'report'])
    ap.add_argument('--profiles', nargs='+', default=['r9700-ntt', 'r9700-ntt-throughput', 'r9700-ntt-latency', 'r9700-fft'])
    ap.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--jobs', type=int, default=8)
    ap.add_argument('--focused', action='store_true', help='Run arithmetic, key-switch and gate tests; reserve exhaustive batch/cooperative checks for final configurations')
    args = ap.parse_args()
    if args.runs < 1 or args.repeats < 1:
        ap.error('runs and repeats must be positive')
    os.chdir(ROOT)
    manifest = source_manifest()
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    result_root = ROOT / 'measurements/r9700/remaining-cuda' / fingerprint
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / 'source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    env = dict(os.environ)
    env.setdefault('HIP_VISIBLE_DEVICES', '0')
    env.setdefault('OMP_NUM_THREADS', '8')
    env['CUFHE_BENCH_RESOURCES'] = '1'
    env['LD_LIBRARY_PATH'] = str(Path(env.get('ROCM_PATH', '/opt/rocm')) / 'lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    print(f'results={result_root}', flush=True)
    configs = [(p, v) for p in args.profiles for v in args.variants if applicable(p, v)]
    # Include each candidate's matched control in build/check requests too.
    expanded = []
    def add_config(p, v):
        if v not in ('baseline', 'prepared'):
            add_config(p, control(p, v))
        if (p, v) not in expanded:
            expanded.append((p, v))
    for p, v in configs:
        add_config(p, v)
    configs = expanded
    lock = open('/tmp/cufhepp-r9700-gpu.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    for p, v in configs:
        options = {'USE_HIP_' + n: 'OFF' for n in NEW}
        options.update({'USE_HIP_' + n: value for n, value in VARIANTS[v].items()})
        if 'fft' in p and v != 'baseline':
            options['USE_HIP_PREPARED_KEYSWITCH'] = 'ON'
        build = ROOT / f'build-remaining-{p}-{v}'
        dest = result_root / p / v
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'options.json').write_text(json.dumps(options, indent=2) + '\n')
        tests = TESTS + (['test_fft_roundtrip'] if 'fft' in p else ['test_small_modulus', 'test_bootstrap_cooperative', 'test_bootstrap_public_cooperative'])
        if args.focused:
            tests = [name for name in tests if name not in ('test_bootstrap_batch', 'test_bootstrap_public_batch', 'test_bootstrap_cooperative', 'test_bootstrap_public_cooperative')]
        if args.action == 'build':
            execute(['cmake', '--preset', p, '-B', str(build)] + [f'-D{k}={value}' for k, value in options.items()], dest / 'configure.log', env)
            execute(['cmake', '--build', str(build), '--target'] + BENCHES + tests + [f'-j{args.jobs}'], dest / 'build.log', env)
            cache = {line.split('=', 1)[0]: line.split('=', 1)[1] for line in (build / 'CMakeCache.txt').read_text().splitlines() if '=' in line and not line.startswith(('//', '#'))}
            (dest / 'effective-cache.json').write_text(json.dumps(cache, indent=2) + '\n')
            (dest / 'built.json').write_text(json.dumps(dict(source=fingerprint)) + '\n')
        elif args.action == 'check':
            if not (dest / 'built.json').exists():
                raise RuntimeError(f'Build this source first: {p}/{v}')
            for name in tests:
                execute([str(build / 'test' / name)], dest / f'{name}.log', env)
            for name in ['test_bootstrap_mux', 'test_bootstrap_public_mux']:
                execute([str(build / 'test' / name), 'graph'], dest / f'{name}-graph.log', env)
            (dest / 'checked.json').write_text(json.dumps(dict(source=fingerprint, tests=tests, exhaustive=not args.focused)) + '\n')
        elif args.action in ('screen', 'run'):
            if not (dest / 'checked.json').exists():
                raise RuntimeError(f'Validate this source first: {p}/{v}')
            for name in BENCHES:
                for mode, streams, count, graph in cases(args.action == 'run'):
                    for run in range(args.runs):
                        pair = [v] if v in ('baseline', 'prepared') else [control(p, v), v]
                        # Reverse alternating pairs to reduce ordering bias.
                        if run % 2:
                            pair.reverse()
                        for member in pair:
                            member_dest = result_root / p / member
                            member_build = ROOT / f'build-remaining-{p}-{member}'
                            if not (member_dest / 'checked.json').exists():
                                raise RuntimeError(f'Validate this source first: {p}/{member}')
                            against = v if member != v else ''
                            suffix = f'-against-{against}' if against else ''
                            log = member_dest / args.action / f'{name}-{mode}-s{streams}-b{count}-g{int(graph)}-r{run}-n{args.repeats}{suffix}.log'
                            if not log.exists():
                                argv = [str(member_build / 'test' / name), str(streams), str(count), str(args.repeats), mode] + (['graph'] if graph else [])
                                execute(argv, log, env)
                            data = parse(log, args.repeats)
                            log.with_suffix('.json').write_text(json.dumps(dict(profile=p, variant=member, against=against, phase=args.action, benchmark=name, mode=mode, streams=streams, count=count, graph=graph, run=run, **{k: value for k, value in data.items() if k != 'streams'}), indent=2) + '\n')
        print(f'{p}/{v} {args.action} complete', flush=True)
    if args.action == 'report':
        records = [json.loads(path.read_text()) for path in result_root.glob('*/*/*/*.json')]
        index = {(r['profile'], r['variant'], r['benchmark'], r['mode'], r['streams'], r['count'], r['graph'], r['run'], r.get('against', ''), r['phase']): r for r in records}
        outcomes = []
        for p, v in configs:
            if v in ('baseline', 'prepared'):
                continue
            comparisons = []
            for key, r in index.items():
                if key[:2] != (p, v) or key[-2]:
                    continue
                ref = index.get((p, control(p, v), *key[2:-2], v, key[-1]))
                if ref:
                    comparisons.append(dict(phase=r['phase'], benchmark=r['benchmark'], mode=r['mode'], streams=r['streams'], count=r['count'], graph=r['graph'], run=r['run'], improvement=1-r['median_ms_per_gate']/ref['median_ms_per_gate']))
            grouped = {}
            for r in comparisons:
                key = (r['phase'], r['benchmark'], r['mode'], r['streams'], r['count'], r['graph'])
                grouped.setdefault(key, []).append(r['improvement'])
            repeatable = [key for key, values in grouped.items() if len(values) >= 3 and min(values) >= .03]
            regressions = [key for key, values in grouped.items() if len(values) >= 3 and statistics.median(values) < -.02]
            outcomes.append(dict(profile=p, variant=v, control=control(p,v), repeatable_wins=repeatable, regressions=regressions,
                                 decision='candidate-for-full-validation' if repeatable and not regressions else 'experimental-opt-in', comparisons=comparisons))
        (result_root / 'comparison.json').write_text(json.dumps(outcomes, indent=2) + '\n')
        print(f'{len(records)} measurements; {len(outcomes)} comparisons')
    return 0


if __name__ == '__main__':
    sys.exit(main())
