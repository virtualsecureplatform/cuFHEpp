#!/usr/bin/env python3
"""Check exact initial and diagnostic screen coverage; never change presets."""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
import bench_r9700_remaining as bench
progress=json.loads((ROOT/'exploration-progress.json').read_text())
rows=progress['complete_three_key_points']
key=lambda r:(r['benchmark'],r['mode'],r['streams'],r['count'],r['graph'])
scopes={p:json.loads((ROOT/(p+'-exploratory-scopes.json')).read_text()) for p in ('r9700-ntt','r9700-ntt-throughput')}
scopes['r9700-fft']={v:dict(benchmarks=bench.BENCHES) for v in ('multi','wave','multi-wave')}
scopes['r9700-ntt-latency']={'all':dict(benchmarks=bench.BENCHES)}
diagnostics={}
for path in ROOT.glob('*diagnostics-definitions.json'):
 for profile,entries in json.loads(path.read_text()).items():
  for variant,entry in entries.items():
   diagnostics.setdefault((profile,variant),set()).update((name,*case) for name in entry['benchmarks'] for case in entry['cases'])
try:qualified=json.loads((ROOT/'qualification-audit.json').read_text())
except FileNotFoundError:qualified=[]
reports=[]
for profile,entries in scopes.items():
 for variant,entry in entries.items():
  expected={(name,*case) for name in entry['benchmarks'] for case in bench.cases(False)}
  extra=diagnostics.get((profile,variant),set())-expected
  measurements=[r for r in rows if r['profile']==profile and r['variant']==variant]
  actual={key(r) for r in measurements}
  initial_complete=expected<=actual
  diagnostics_complete=extra<=actual
  regressions=[r for r in measurements if r['median_improvement']<-.02]
  wins=[r for r in measurements if r['min_improvement']>=.03]
  full=next((r for r in qualified if r['profile']==profile and r['variant']==variant),None)
  decision='awaiting-initial-screen'
  if initial_complete:
   decision='awaiting-diagnostics' if not diagnostics_complete else 'experimental-opt-in'
   if diagnostics_complete and wins and not regressions:decision='candidate-awaits-full-qualification'
   if regressions:decision='screen-regression-remains-opt-in'
  if full and full['full_grid_complete']:
   if full['regressions'] or full.get('unresolved_regressions',0):decision='full-grid-regression-or-uncertainty-remains-opt-in'
   elif full['needs_30']:decision='awaiting-longer-full-grid-points'
   elif full['wins']:decision='recorded-gates-pass-requires-profile-review-and-tuning'
  receipt=ROOT/profile/variant/'checked.json'
  reports.append(dict(profile=profile,variant=variant,control=bench.control(profile,variant),initial_expected=len(expected),initial_complete=len(expected&actual),diagnostic_expected=len(extra),diagnostic_complete=len(extra&actual),complete_initial=initial_complete,complete_diagnostics=diagnostics_complete,repeatable_screen_wins=len(wins),median_screen_regressions=len(regressions),decision=decision,validation_receipt=str(receipt.relative_to(ROOT)) if receipt.exists() else None,missing_initial=[list(x) for x in sorted(expected-actual)],missing_diagnostics=[list(x) for x in sorted(extra-actual)]))
(ROOT/'exploration-coverage.json').write_text(json.dumps(dict(status='progress-only-no-preset-promotion',source=ROOT.name,reports=reports),indent=2)+'\n')
for profile in scopes:
 selected=[r for r in reports if r['profile']==profile]
 print(profile,'initial complete',sum(r['complete_initial'] for r in selected),'/',len(selected),'decisions',dict(Counter(r['decision'] for r in selected)))
