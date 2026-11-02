#!/usr/bin/env python3
"""Read-only full-grid qualification audit; partial runs never qualify presets."""
from pathlib import Path
from collections import defaultdict
import json
import os
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
import bench_r9700_remaining as bench
EXPECTED={(name,*case) for name in bench.BENCHES for case in bench.cases(True)}
reports=[]
for directory in sorted((ROOT/'qualification').glob('*/*')):
    index={}
    for phase in ['full','selected']:
        for path in (directory/phase).glob('*.json'):
            try:r=json.loads(path.read_text())
            except json.JSONDecodeError:continue  # A live writer may not have closed this record.
            key=(r['benchmark'],r['mode'],r['streams'],r['count'],r['graph'],r['run'],r['member'])
            if key not in index or r['repetitions']>index[key]['repetitions']:
                index[key]=r
    groups=defaultdict(list)
    for key,r in index.items():
        if key[-1]!='candidate':continue
        ref=index.get((*key[:-1],'original'))
        if ref is None or r['repetitions']!=ref['repetitions']:continue
        groups[key[:5]].append(dict(run=key[5],repetitions=r['repetitions'],gain=1-r['median_ms_per_gate']/ref['median_ms_per_gate']))
    points=[]
    for key,rows in sorted(groups.items()):
        complete={r['run'] for r in rows}>={0,1,2}
        gains=[r['gain'] for r in rows]
        median=statistics.median(gains)
        win=complete and min(gains)>=.03
        regression=complete and median<-.02
        # Close thresholds or discordant key runs warrant 30-repetition checks.
        uncertain=complete and min(r['repetitions'] for r in rows)<30 and (abs(median-.03)<=.01 or abs(median+.02)<=.01 or (min(gains)<.03<=max(gains)) or (min(gains)<-.02<=max(gains)) or regression)
        unresolved_regression=complete and min(r['repetitions'] for r in rows)>=30 and min(gains)<-.02<=max(gains)
        points.append(dict(benchmark=key[0],mode=key[1],streams=key[2],count=key[3],graph=key[4],complete=complete,rows=rows,median_gain=median,repeatable_win=win,regression=regression,needs_30=uncertain,unresolved_regression=unresolved_regression))
    covered={key for key,rows in groups.items() if {r['run'] for r in rows}>={0,1,2}}
    resources=[]
    seen=set()
    for path in sorted((directory/'full').glob('*.log')):
        # Keep unique actual launch resource shapes, not thousands of repeated graph nodes.
        for match in re.finditer(r'resource_node=\d+ registers=(\d+) static_lds=(\d+) dynamic_lds=(\d+) local_bytes=(\d+) threads=(\d+) resident_blocks=(\d+) grid=(\d+),(\d+),(\d+)',path.read_text()):
            values=tuple(map(int,match.groups()))
            member='candidate' if path.stem.endswith('-candidate') else 'original'
            identity=(member,*values[:6])
            if identity in seen:continue
            seen.add(identity)
            resources.append(dict(member=member,registers=values[0],static_lds=values[1],dynamic_lds=values[2],local_bytes=values[3],threads=values[4],resident_blocks=values[5],example_log=str(path.relative_to(ROOT))))
    reports.append(dict(profile=directory.parent.name,variant=directory.name,full_grid_complete=EXPECTED<=covered,expected_points=len(EXPECTED),complete_points=len(covered&EXPECTED),wins=sum(p['repeatable_win'] for p in points),regressions=sum(p['regression'] for p in points),unresolved_regressions=sum(p['unresolved_regression'] for p in points),needs_30=sum(p['needs_30'] for p in points),points=points,resources=resources))
temporary=ROOT/f'.qualification-audit-{os.getpid()}.json'
temporary.write_text(json.dumps(reports,indent=2)+'\n')
temporary.replace(ROOT/'qualification-audit.json')
for r in reports:
    print(r['profile'],r['variant'],f"{r['complete_points']}/{r['expected_points']} full-grid points",f"wins={r['wins']} regressions={r['regressions']} rerun30={r['needs_30']} unresolved_after30={r['unresolved_regressions']}")
