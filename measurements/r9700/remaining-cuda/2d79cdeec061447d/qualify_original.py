#!/usr/bin/env python3
"""Qualify selected ports against the original b0818d2 backend, same harness.

Build original-baseline/build-original.py first and validate candidate variants
with bench_r9700_remaining.py. This artifact is separate from exploratory
feature-disabled controls. Timings alternate fresh original/candidate keys.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

SNAPSHOT = Path(__file__).resolve().parent
REPO = SNAPSHOT.parents[3]
sys.path.insert(0, str(REPO))
import bench_r9700_remaining as bench


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profiles', nargs='+', required=True,
                    choices=['r9700-ntt', 'r9700-ntt-throughput', 'r9700-ntt-latency', 'r9700-fft'])
    ap.add_argument('--variants', nargs='+', required=True, choices=bench.VARIANTS)
    ap.add_argument('--screen', action='store_true')
    ap.add_argument('--repeats', type=int, default=15)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--case', action='append', help='One benchmark:mode:streams:count:graph(0/1), repeatable; otherwise use screen/full grid')
    args = ap.parse_args()
    if args.repeats < 1 or args.runs < 1:
        ap.error('runs and repeats must be positive')
    manifest = bench.source_manifest()
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    if fingerprint != SNAPSHOT.name:
        raise RuntimeError('Source changed; qualify the matching source snapshot')
    if args.case:
        points = []
        for item in args.case:
            binary, mode, streams, count, graph = item.split(':')
            if binary not in bench.BENCHES or mode not in ('batch', 'packed', 'packed-mux', 'packed-nmux'):
                ap.error(f'Unsupported point: {item}')
            streams, count, graph = int(streams), int(count), int(graph)
            if min(streams, count) < 1 or graph not in (0, 1):
                ap.error(f'Invalid dimensions: {item}')
            points.append((binary, mode, streams, count, bool(graph)))
    else:
        points = [(name, *case) for name in bench.BENCHES for case in bench.cases(not args.screen)]
    env = dict(os.environ)
    env.setdefault('HIP_VISIBLE_DEVICES', '0')
    env.setdefault('OMP_NUM_THREADS', '8')
    env['CUFHE_BENCH_RESOURCES'] = '1'
    env['LD_LIBRARY_PATH'] = env.get('ROCM_PATH', '/opt/rocm') + '/lib:' + env.get('LD_LIBRARY_PATH', '')
    handles = []
    for profile in sorted(set(args.profiles)):
        handle = open('/tmp/cufhepp-' + profile + '-experiment.lock', 'w')
        fcntl.flock(handle, fcntl.LOCK_EX)
        handles.append(handle)
    gpu = open('/tmp/cufhepp-r9700-gpu.lock', 'w')
    fcntl.flock(gpu, fcntl.LOCK_EX)
    phase = 'selected' if args.case else 'screen' if args.screen else 'full'
    for profile in args.profiles:
        original_receipt = json.loads((SNAPSHOT / 'original-baseline' / profile / 'built.json').read_text())
        original_dir = REPO / 'build-remaining-original' / profile / 'test'
        for variant in args.variants:
            if not bench.applicable(profile, variant):
                continue
            candidate_receipt = json.loads((SNAPSHOT / profile / variant / 'checked.json').read_text())
            if candidate_receipt['source'] != fingerprint:
                raise RuntimeError('Candidate receipt source mismatch')
            candidate_dir = REPO / f'build-remaining-{profile}-{variant}' / 'test'
            for name in {point[0] for point in points}:
                if digest(original_dir / name) != original_receipt['binaries'][name]:
                    raise RuntimeError('Original binary changed: ' + name)
                if digest(candidate_dir / name) != candidate_receipt['binaries'][name]:
                    raise RuntimeError('Candidate binary changed: ' + name)
            dest = SNAPSHOT / 'qualification' / profile / variant / phase
            dest.mkdir(parents=True, exist_ok=True)
            for name, mode, streams, count, graph in points:
                for run in range(args.runs):
                    members = [('original', original_dir), ('candidate', candidate_dir)]
                    if run % 2:
                        members.reverse()
                    for member, directory in members:
                        path = dest / f'{name}-{mode}-s{streams}-b{count}-g{int(graph)}-r{run}-n{args.repeats}-{member}.log'
                        argv = [str(directory / name), str(streams), str(count), str(args.repeats), mode]
                        if graph:
                            argv.append('graph')
                        if not path.exists():
                            path.with_suffix('.command').write_text(shlex.join(argv) + '\n')
                            with path.open('w') as out:
                                subprocess.run(argv, cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
                        data = bench.parse(path, args.repeats)
                        record = dict(profile=profile, variant=variant, member=member,
                                      source=fingerprint if member == 'candidate' else original_receipt['backend_commit'],
                                      binary_sha256=digest(directory / name), benchmark=name, mode=mode,
                                      streams=streams, count=count, graph=graph, run=run,
                                      repetitions=args.repeats, phase=phase,
                                      **{k: v for k, v in data.items() if k != 'streams'})
                        path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
                print(profile, variant, name, mode, streams, count, 'PASS', flush=True)
            records = [json.loads(path.read_text()) for path in dest.glob('*.json')]
            index = {(r['benchmark'], r['mode'], r['streams'], r['count'], r['graph'], r['run'], r['member']): r
                     for r in sorted(records, key=lambda r: r['repetitions'])}
            comparisons = []
            for key, candidate in index.items():
                if key[-1] != 'candidate':
                    continue
                original = index.get((*key[:-1], 'original'))
                if original:
                    comparisons.append(dict(benchmark=key[0], mode=key[1], streams=key[2], count=key[3],
                                            graph=key[4], run=key[5],
                                            improvement=1-candidate['median_ms_per_gate']/original['median_ms_per_gate'],
                                            original_ms_per_gate=original['median_ms_per_gate'],
                                            candidate_ms_per_gate=candidate['median_ms_per_gate']))
            (dest.parent / (phase + '-comparison.json')).write_text(json.dumps(comparisons, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
