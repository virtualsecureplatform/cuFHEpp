#!/usr/bin/env python3
"""Exploratory screens by applicable torus level; final qualification is full-grid.

The selection is fixed before any NTT timings are collected. All configurations
already passed both levels' correctness checks. The final selected configuration
must still be qualified on both levels and orders against the original backend.
"""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench

profile=sys.argv[1]
if profile=='r9700-ntt-throughput':
    variants=['all','multi','wave','multi-wave','exponents','scalar-cache','register-io','register31','native64','torus64','inverse64','roots31','difference31']
elif profile=='r9700-ntt':
    variants=['staging','prefetch','native64','torus64','inverse64','multi','wave','multi-wave']
else:raise SystemExit('Expected an NTT profile')
all_benches=list(bench.BENCHES)
scopes={}
for variant in variants:
    if variant in ('native64','torus64','inverse64','scalar-cache','register-io'):
        names=[name for name in all_benches if '_lvl02' in name]
        reason='Only 64-bit NTT operations change; 31-bit transforms and key switching are unchanged.'
    elif variant in ('register31','roots31','difference31'):
        names=[name for name in all_benches if '_lvl02' not in name]
        reason='Only 31-bit NTT operations change; 64-bit transforms and key switching are unchanged.'
    elif variant=='exponents':
        names=['test_bootstrap_bench_lvl02_br_iks']
        reason='The cache is enabled only for public lvl02 single blind-rotation binary gates.'
    else:
        names=all_benches
        reason='Both levels and orders are retained in this exploratory screen.'
    scopes[variant]=dict(benchmarks=names,reason=reason)
(ROOT/(profile+'-exploratory-scopes.json')).write_text(json.dumps(scopes,indent=2)+'\n')
for variant in variants:
    bench.BENCHES=scopes[variant]['benchmarks']
    sys.argv=[__file__,'screen','--profiles',profile,'--variants',variant]
    bench.main()
