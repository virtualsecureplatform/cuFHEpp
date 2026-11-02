import hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/home/nimda/sources/cuFHEpp')
old=repo/'measurements/r9700/remaining-cuda/2d79cdeec061447d'
pid=3141153
try: started=Path(f'/proc/{pid}/stat').read_text().split()[21]
except FileNotFoundError: started=None
while started:
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().split()
  if fields[21]!=started or fields[2]=='Z': break
 except FileNotFoundError: break
 time.sleep(10)
def run(argv, log):
 print('START',log,flush=True)
 with Path(log).open('w') as out: subprocess.run(argv,cwd=repo,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('PASS',log,flush=True)
run(['python3','-B',str(old/'exploration_progress.py')],'/tmp/r9700-final-old-progress.log')
run(['python3','-B',str(old/'exploration_coverage.py')],'/tmp/r9700-final-old-coverage.log')
coverage=json.loads((old/'exploration-coverage.json').read_text())['reports']
assert all(r['complete_initial'] and r['complete_diagnostics'] for r in coverage), 'Old screens incomplete'
subprocess.run(['git','apply','--check',str(old/'runner-final-fixes.patch')],cwd=repo,check=True)
subprocess.run(['git','apply',str(old/'runner-final-fixes.patch')],cwd=repo,check=True)
sys.path.insert(0,str(repo)); import bench_r9700_remaining as bench
manifest=bench.source_manifest(); fp=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
new=old.parent/fp; new.mkdir(exist_ok=True)
for name in ['qualify_original.py','qualification_audit.py','experiment_configs.py','tune_after_qualification.py','resource_inventory.py']:
 shutil.copy2(old/name,new/name)
base=new/'original-baseline'; base.mkdir(exist_ok=True)
for name in ['provenance.json','reuse-validation.json']: shutil.copy2(old/'original-baseline'/name,base/name)
for profile in ['r9700-ntt','r9700-ntt-throughput','r9700-ntt-latency','r9700-fft']:
 (base/profile).mkdir(exist_ok=True); shutil.copy2(old/'original-baseline'/profile/'built.json',base/profile/'built.json')
selection=[(r['profile'],r['variant']) for r in coverage if r['decision']=='candidate-awaits-full-qualification']
(new/'migration.json').write_text(json.dumps(dict(previous=old.name,source=fp,reason='Runner report scoping and independent FFT wave controls; C++ unchanged',candidate_selection=selection),indent=2)+'\n')
Path('/tmp/r9700-final-snapshot').write_text(str(new)+'\n')
for profile,variant in selection:
 prefix=f'/tmp/r9700-final-{profile}-{variant}'
 for action in ['build','check']:
  run(['python3','-B','bench_r9700_remaining.py',action,'--profiles',profile,'--variants',variant],prefix+'-'+action+'.log')
 run(['python3','-B',str(new/'qualify_original.py'),'--profiles',profile,'--variants',variant],prefix+'-full15.log')
 run(['python3','-B',str(new/'qualification_audit.py')],prefix+'-audit15.log')
 reports=json.loads((new/'qualification-audit.json').read_text())
 report=next(r for r in reports if r['profile']==profile and r['variant']==variant)
 assert report['full_grid_complete']
 uncertain=[p for p in report['points'] if p['needs_30']]
 if uncertain:
  argv=['python3','-B',str(new/'qualify_original.py'),'--profiles',profile,'--variants',variant,'--repeats','30']
  for p in uncertain: argv+=['--case',f"{p['benchmark']}:{p['mode']}:{p['streams']}:{p['count']}:{int(p['graph'])}"]
  run(argv,prefix+'-longer30.log')
  run(['python3','-B',str(new/'qualification_audit.py')],prefix+'-audit30.log')
print('QUALIFICATION CAMPAIGN COMPLETE',new,flush=True)
