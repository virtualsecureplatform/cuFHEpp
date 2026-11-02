#!/usr/bin/env python3
"""Reproduce composed candidates/tuning without changing the frozen C++ source.

JSON: {profile: {variant: {flags: {SHORT_HIP_NAME: ON/OFF},
cache: {CMAKE_CACHE_NAME: value}, control: baseline, benchmarks: [...],
cases: [[mode,streams,count,graph],...]}}}. Scopes apply only to exploratory
screen/run actions. Validation and original-backend qualification remain broad.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

SNAPSHOT=Path(__file__).resolve().parent
REPO=SNAPSHOT.parents[3]
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=['build','check','screen','run','report','qualify'])
parser.add_argument('definitions',type=Path)
parser.add_argument('--profiles',nargs='+',required=True)
parser.add_argument('--variants',nargs='+',required=True)
args,extra=parser.parse_known_args()
definitions=json.loads(args.definitions.read_text())
all_benchmarks=list(bench.BENCHES)
all_cases=bench.cases
original_control=bench.control
original_applicable=bench.applicable
bench.control=lambda p,v:definitions.get(p,{}).get(v,{}).get('control',original_control(p,v))
bench.applicable=lambda p,v:v in definitions.get(p,{}) or original_applicable(p,v)

if args.action=='qualify':
    spec=importlib.util.spec_from_file_location('original_qualification',SNAPSHOT/'qualify_original.py')
    qualification=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualification)

for profile in args.profiles:
    entries=definitions[profile]
    for name,entry in entries.items():
        if not re.fullmatch(r'[a-z0-9-]+',name):raise ValueError('Invalid variant name')
        for value in entry.get('flags',{}).values():
            if value not in ('ON','OFF'):raise ValueError('HIP flags must be ON/OFF strings')
        bench.VARIANTS[name]=entry.get('flags',{})
        bench.CACHE_OPTIONS[name]=entry.get('cache',{})
    for name in entries:
        chain=[]
        while name not in ('baseline','prepared'):
            if name in chain:raise ValueError('Cyclic controls: '+str(chain))
            chain.append(name)
            name=bench.control(profile,name)
            if name not in bench.VARIANTS:raise ValueError('Unknown control '+name)
    # Build/check all requested variants together so the main runner deduplicates
    # shared controls. Per-variant timing scopes apply only to screen/run.
    if args.action in ('build','check'):
        bench.BENCHES=all_benchmarks
        bench.cases=all_cases
        sys.argv=[__file__,args.action,'--profiles',profile,'--variants',*args.variants,*extra]
        bench.main()
        continue
    for variant in args.variants:
        entry=entries[variant]
        bench.BENCHES=all_benchmarks
        bench.cases=all_cases
        if args.action in ('screen','run'):
            bench.BENCHES=entry.get('benchmarks',all_benchmarks)
            if 'cases' in entry:
                points=[tuple(point) for point in entry['cases']]
                bench.cases=lambda full,points=points:points
        if args.action=='qualify':
            sys.argv=[__file__,'--profiles',profile,'--variants',variant,*extra]
            qualification.main()
        else:
            sys.argv=[__file__,args.action,'--profiles',profile,'--variants',variant,*extra]
            bench.main()
