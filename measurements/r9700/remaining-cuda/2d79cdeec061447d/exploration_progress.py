#!/usr/bin/env python3
"""Summarize completed exploratory pairs without waiting for the GPU lock.

This is progress evidence, not original-backend qualification or a preset gate.
"""
import collections
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
sys.path.insert(0, str(REPO))
import bench_r9700_remaining as bench

control_overrides = {}
for definitions_path in ROOT.glob('*definitions.json'):
    for profile, entries in json.loads(definitions_path.read_text()).items():
        for variant, definition in entries.items():
            control_overrides[(profile, variant)] = definition.get('control', 'baseline')

records = []
for path in ROOT.glob('r9700-*/*/screen/*.json'):
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue  # a writer may be between write and close; retry next invocation
    records.append(record)
index = {}
for r in sorted(records, key=lambda r: r['repetitions']):
    key = (r['profile'], r['variant'], r['benchmark'], r['mode'], r['streams'], r['count'], r['graph'], r['run'], r['against'])
    index[key] = r
pairs = collections.defaultdict(list)
counts = collections.Counter((r['profile'], r['variant'], r['against']) for r in records)
for key, r in index.items():
    if key[-1] or r['variant'] in ('baseline', 'prepared'):
        continue
    control = control_overrides.get((r['profile'], r['variant']), bench.control(r['profile'], r['variant']))
    other = index.get((r['profile'], control, *key[2:-1], r['variant']))
    if other:
        pair_key = (r['profile'], r['variant'], r['benchmark'], r['mode'], r['streams'], r['count'], r['graph'])
        pairs[pair_key].append((r['run'], 1-r['median_ms_per_gate']/other['median_ms_per_gate']))
summary = []
for key, values in sorted(pairs.items()):
    values = sorted(values)
    if len(values) != 3 or [r[0] for r in values] != [0, 1, 2]:
        continue
    improvements = [r[1] for r in values]
    summary.append(dict(profile=key[0], variant=key[1], benchmark=key[2], mode=key[3],
                        streams=key[4], count=key[5], graph=key[6],
                        min_improvement=min(improvements), median_improvement=statistics.median(improvements),
                        max_improvement=max(improvements), improvements=improvements))
output = dict(status='exploratory-only-not-qualified', records=len(records), complete_three_key_points=summary,
              counts=[dict(profile=k[0], variant=k[1], against=k[2], count=v) for k,v in sorted(counts.items())])
(ROOT/'exploration-progress.json').write_text(json.dumps(output, indent=2)+'\n')
print('records',len(records),'complete three-key comparisons',len(summary))
for r in summary:
    if r['min_improvement'] >= .03 or r['median_improvement'] <= -.02:
        print(r['profile'],r['variant'],r['benchmark'],r['mode'],r['streams'],r['count'],
              'gain %',*[round(100*x,2) for x in r['improvements']])
