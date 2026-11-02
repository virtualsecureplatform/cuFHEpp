#!/usr/bin/env python3
"""Inventory actual HIP graph launch resources from completed measurements."""
from pathlib import Path
import json
import re
ROOT=Path(__file__).resolve().parent
PATTERN=re.compile(r'^resource_node=(\d+) registers=(\d+) static_lds=(\d+) dynamic_lds=(\d+) local_bytes=(\d+) threads=(\d+) resident_blocks=(\d+) grid=(\d+),(\d+),(\d+)$',re.M)
fields=['registers','static_lds','dynamic_lds','local_bytes','threads','resident_blocks']
shapes={}
completed=0
for record_path in sorted(ROOT.glob('r9700-*/*/screen/*.json')):
    try:r=json.loads(record_path.read_text())
    except json.JSONDecodeError:continue
    log=record_path.with_suffix('.log')
    if r.get('errors')!=0:continue
    rows=PATTERN.findall(log.read_text())
    if not rows:raise RuntimeError('Completed HIP resource measurement has no kernel attributes: '+str(log))
    completed+=1
    for row in rows:
        values=tuple(map(int,row[1:7]))
        if min(values[4:])<=0:raise RuntimeError('Invalid launch/occupancy: '+str(log))
        key=(r['profile'],r['variant'],r['benchmark'],*values)
        if key not in shapes:
            shapes[key]=dict(profile=key[0],variant=key[1],benchmark=key[2],**dict(zip(fields,values)),observed_nodes=0,example_log=str(log.relative_to(ROOT)),example_mode=r['mode'],example_grid=list(map(int,row[7:])))
        shapes[key]['observed_nodes']+=1
output=dict(source=ROOT.name,status='observed-completed-measurements',completed_logs=completed,shapes=list(shapes.values()))
(ROOT/'resource-inventory.json').write_text(json.dumps(output,indent=2)+'\n')
print('Completed logs:',completed,'unique actual kernel resource shapes:',len(shapes))
