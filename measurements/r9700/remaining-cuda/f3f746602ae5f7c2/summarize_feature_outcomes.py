#!/usr/bin/env python3
"""Reconcile immutable feature screens with completed full-grid qualifications."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'2d79cdeec061447d'
read=lambda p:json.loads(p.read_text())
coverage=read(OLD/'exploration-coverage.json')['reports']
rows={}
for item in coverage:
    assert item['complete_initial'] and item['complete_diagnostics']
    row=dict(item)
    row['screen_source']=OLD.name
    row['screen_evidence']='../'+OLD.name+'/exploration-coverage.json'
    rows[item['profile'],item['variant']]=row
full=[(OLD,r) for r in read(OLD/'qualification-audit.json') if r['full_grid_complete']]
for filename in ['paired-multi-qualification-summary.json','paired-multi-wave-qualification-summary.json','throughput-register31-qualification-summary.json','fft-multi-wave-qualification-summary.json']:
    r=read(ROOT/filename)
    assert r['complete_points']==116 and not r['needs_30']
    full.append((ROOT,r))
for source,r in full:
    assert not r['needs_30']
    assert r['regressions'] or r['unresolved_regressions'], 'Review a potential promotion explicitly'
    key=r['profile'],r['variant']
    row=rows.setdefault(key,dict(profile=key[0],variant=key[1]))
    row['decision']='full-grid-regression-or-uncertainty-remains-opt-in'
    row['qualification']=dict(source=source.name,points=116,runs=3,required_longer_runs_complete=True,wins=r['wins'],median_regressions=r['regressions'],unresolved_regressions=r['unresolved_regressions'])
    row['qualification_evidence']=('' if source==ROOT else '../'+source.name+'/')+'qualification-audit.json'
assert all('await' not in r['decision'] for r in rows.values())
result=dict(scope='Completed feature experiments only; launch retuning and final configuration qualification are separate',status='complete-no-new-feature-promotion',source=ROOT.name,earlier_cpp_identical_source=OLD.name,production_new_flags='OFF',outcomes=list(rows.values()))
(ROOT/'completed-feature-outcomes.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'Reconciled {len(rows)} feature experiments; {len(full)} full-grid qualifications; no pending feature decisions')
