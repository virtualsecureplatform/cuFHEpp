"""Retune qualified feature settings, confirm uncertain points, qualify useful launch settings."""
import collections,json,os,statistics,subprocess,sys,time
from pathlib import Path
repo=Path('/home/nimda/sources/cuFHEpp');root=Path(Path('/tmp/r9700-final-snapshot').read_text().strip())
sys.path.insert(0,str(repo));import bench_r9700_remaining as bench

def wait(pid):
 path=Path(f'/proc/{pid}/stat')
 try:start=path.read_text().split()[21]
 except FileNotFoundError:return
 while True:
  try:
   fields=path.read_text().split()
   if fields[21]!=start or fields[2]=='Z':return
  except FileNotFoundError:return
  time.sleep(10)
def run(argv,label):
 print('START',label,flush=True)
 with open('/tmp/r9700-retuning-'+label+'.log','w') as out:subprocess.run(argv,cwd=repo,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('PASS',label,flush=True)
def experiment(action,definitions,profile,variants,extra=()):
 run(['python3','-B',str(root/'experiment_configs.py'),action,str(definitions),'--profiles',profile,'--variants',*variants,*extra],profile+'-'+action+'-'+definitions.stem)
def audit():
 run(['python3','-B',str(root/'qualification_audit.py')],'qualification-audit')
 return json.loads((root/'qualification-audit.json').read_text())
def qualify(profile,variant,definitions=None):
 command=['python3','-B',str(root/'qualify_original.py')] if definitions is None else ['python3','-B',str(root/'experiment_configs.py'),'qualify',str(definitions)]
 args=['--profiles',profile,'--variants',variant]
 run(command+args,profile+'-'+variant+'-full15')
 # Include all additional neighbours of actual cooperative/fallback transitions.
 extras=[]
 if 'fft' not in profile:
  for name in bench.BENCHES:
   for count in sorted(set(range(1,23))-set(bench.COUNTS)):extras+=['--case',f'{name}:batch:1:{count}:0']
  run(command+args+extras,profile+'-'+variant+'-transition15')
 report=next(r for r in audit() if r['profile']==profile and r['variant']==variant)
 assert report['full_grid_complete']
 uncertain=[p for p in report['points'] if p['needs_30']]
 if uncertain:
  extra=['--repeats','30']
  for p in uncertain:extra+=['--case',f"{p['benchmark']}:{p['mode']}:{p['streams']}:{p['count']}:{int(p['graph'])}"]
  run(command+args+extra,profile+'-'+variant+'-longer30')
  report=next(r for r in audit() if r['profile']==profile and r['variant']==variant)
 assert not report['needs_30']
 return report

def screen_points(profile,variant,control):
 index={}
 for member in [variant,control]:
  for path in (root/profile/member/'screen').glob('*.json'):
   r=json.loads(path.read_text())
   if r.get('against','') != (variant if member==control else ''):continue
   key=(r['benchmark'],r['mode'],r['streams'],r['count'],r['graph'],r['run'],member)
   if key not in index or r['repetitions']>index[key]['repetitions']:index[key]=r
 groups=collections.defaultdict(list)
 for key,r in index.items():
  if key[-1]!=variant:continue
  ref=index.get((*key[:-1],control))
  if ref and r['repetitions']==ref['repetitions']:groups[key[:5]].append((r['run'],r['repetitions'],1-r['median_ms_per_gate']/ref['median_ms_per_gate']))
 points=[]
 for key,rows in groups.items():
  assert {r[0] for r in rows}=={0,1,2}
  gains=[r[2] for r in rows];med=statistics.median(gains);n=min(r[1] for r in rows)
  uncertain=n<30 and (abs(med-.03)<=.01 or abs(med+.02)<=.01 or min(gains)<.03<=max(gains) or min(gains)<-.02<=max(gains) or med<-.02)
  points.append(dict(key=list(key),gains=gains,repetitions=n,win=min(gains)>=.03,regression=med<-.02,unresolved=n>=30 and min(gains)<-.02<=max(gains),needs30=uncertain))
 return points

wait(3608049)
assert 'QUALIFICATION CAMPAIGN COMPLETE' in Path('/tmp/r9700-final-qualification-controller.log').read_text(),'Qualification controller failed'
wait(3613593)
assert 'LATENCY RETUNING SCREEN COMPLETE' in Path('/tmp/r9700-final-latency-retune-controller.log').read_text(),'Latency controller failed'
# The original latency sweep is immutable; supplement its missing capacity
# neighbours before selection. Both preceding GPU campaigns are terminal here.
followup=root/'latency-residency-boundary-followup-definitions.json'
experiment('screen',followup,'r9700-ntt-latency',list(json.loads(followup.read_text())['r9700-ntt-latency']))
reports=audit(); selection={};final=[]
for profile in ['r9700-ntt','r9700-ntt-throughput','r9700-ntt-latency','r9700-fft']:
 valid=[r for r in reports if r['profile']==profile and r['full_grid_complete'] and r['wins'] and not(r['regressions'] or r['unresolved_regressions'] or r['needs_30'])]
 chosen=max(valid,key=lambda r:r['wins'])['variant'] if valid else 'baseline'
 selection[profile]=chosen
 flags=dict(bench.VARIANTS[chosen])
 if 'fft' in profile and chosen!='baseline':flags['PREPARED_KEYSWITCH']='ON'
 entries={}
 for n in [2,3,4]:entries['selected-waves'+str(n)]=dict(flags=flags,control=chosen,cache={'CUFHE_HIP_MIN_WAVES_PER_EU':str(n),'CUFHE_HIP_COOPERATIVE_GROUPS':'0'})
 if 'fft' not in profile:
  cases=[['batch',1,n,False] for n in range(1,23)]+[['batch',1,17,True]]
  for n in [2,3,5,9]:entries['selected-groups'+str(n)]=dict(flags=flags,control=chosen,cache={'CUFHE_HIP_MIN_WAVES_PER_EU':'1','CUFHE_HIP_COOPERATIVE_GROUPS':str(n)},benchmarks=[n for n in bench.BENCHES if '_lvl02' in n],cases=cases)
 if profile=='r9700-ntt-latency':
  # Reuse the completed sweep; its selected features are the retained baseline.
  assert chosen=='baseline'
  definitions=root/'latency-retuning-definitions.json';entries=json.loads(definitions.read_text())[profile]
 else:
  definitions=root/(profile+'-selected-retuning-definitions.json');definitions.write_text(json.dumps({profile:entries},indent=2)+'\n')
  for action in ['build','check','screen']:experiment(action,definitions,profile,list(entries))
 experiment('report',definitions,profile,list(entries))
 summaries=[];useful=[]
 for variant,entry in entries.items():
  points=screen_points(profile,variant,entry['control']);assert points
  if any(p['win'] for p in points):
   uncertain=[p for p in points if p['needs30']]
   if uncertain:
    longer=dict(entry);longer['benchmarks']=sorted({p['key'][0] for p in uncertain});longer['cases']=[list(c) for c in sorted({tuple(p['key'][1:]) for p in uncertain})]
    path=root/(profile+'-'+variant+'-uncertain-definitions.json');path.write_text(json.dumps({profile:{variant:longer}},indent=2)+'\n')
    experiment('screen',path,profile,[variant],['--repeats','30'])
    points=screen_points(profile,variant,entry['control'])
  accepted=any(p['win'] for p in points) and not any(p['regression'] or p['unresolved'] or p['needs30'] for p in points)
  summaries.append(dict(variant=variant,points=points,screen_candidate=accepted))
  if accepted:useful.append(variant)
 (root/(profile+'-retuning-screen-summary.json')).write_text(json.dumps(summaries,indent=2)+'\n')
 qualified=[]
 for variant in useful:
  # Exhaustive tests also replace focused latency receipts before final qualification.
  for action in ['build','check']:experiment(action,definitions,profile,[variant])
  report=qualify(profile,variant,definitions)
  if report['wins'] and not(report['regressions'] or report['unresolved_regressions']):qualified.append(report)
 if qualified:
  winner=max(qualified,key=lambda r:r['wins'])['variant'];entry=entries[winner]
 else:
  winner=chosen;entry=dict(flags=flags,cache={},control='baseline')
  # The retained setting still receives a final complete grid and transition checks.
  for action in ['build','check']:
   run(['python3','-B','bench_r9700_remaining.py',action,'--profiles',profile,'--variants',winner],profile+'-'+winner+'-final-'+action)
  qualify(profile,winner)
 final.append(dict(profile=profile,selected_features=chosen,recommended_variant=winner,configuration=entry,qualified_tuning_variants=[r['variant'] for r in qualified],promotion='Pending final review; no presets changed by this controller'))
 (root/'final-retuning-recommendations.json').write_text(json.dumps(final,indent=2)+'\n')
(root/'retuning-feature-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print('RETUNING AND FINAL QUALIFICATION COMPLETE',flush=True)
