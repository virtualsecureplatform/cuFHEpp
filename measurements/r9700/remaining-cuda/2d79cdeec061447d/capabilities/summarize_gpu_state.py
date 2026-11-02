#!/usr/bin/env python3
"""Summarize active timed samples in a read-only GPU trace."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('trace',type=Path)
a=p.parse_args()
groups=defaultdict(list)
metadata=None
for line in a.trace.read_text().splitlines():
 try:r=json.loads(line)
 except json.JSONDecodeError:continue
 if r['type']=='metadata':metadata=r;continue
 if int(r['metrics'].get('gpu_busy_percent','0'))<90:continue
 for child in r['children']:
  if child['last_repeat'] is not None and 1<=child['last_repeat']<29:
   groups[(child['pid'],child['log'])].append(r['metrics'])
rows=[]
for (pid,log),samples in groups.items():
 stats={}
 for name in ['freq1_input','freq2_input','temp2_input','power1_average']:
  values=[int(s[name]) for s in samples if name in s]
  if values:stats[name]=dict(min=min(values),median=statistics.median(values),max=max(values))
 row=dict(pid=pid,log=log,active_timed_samples=len(samples),raw_sysfs_statistics=stats)
 record=Path(log).with_suffix('.json')
 if record.exists():
  try:row['median_ms_per_gate']=json.loads(record.read_text())['median_ms_per_gate']
  except json.JSONDecodeError:pass
 rows.append(row)
out=dict(status='observational-window-not-causal-proof',metadata=metadata,samples_filtered='GPU busy >=90%; last completed repetition1..28',rows=rows)
output=a.trace.with_suffix('.summary.json')
output.write_text(json.dumps(out,indent=2)+'\n')
print('Summarized',len(rows),'benchmark processes:',output)
