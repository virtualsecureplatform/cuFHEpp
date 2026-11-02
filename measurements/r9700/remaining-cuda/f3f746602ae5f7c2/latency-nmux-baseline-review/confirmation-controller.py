"""Confirm one baseline timing regression after the main GPU campaign is terminal."""
import fcntl,hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
REPO=Path('/home/nimda/sources/cuFHEpp');ROOT=REPO/'measurements/r9700/remaining-cuda/f3f746602ae5f7c2'
sys.path.insert(0,str(REPO));import bench_r9700_remaining as bench
OUT=ROOT/'latency-nmux-baseline-review/confirmation';OUT.mkdir(parents=True,exist_ok=True)
parent=Path('/proc/3775684/stat')
try:started=parent.read_text().split()[21]
except FileNotFoundError:started=None
print('WAIT main controller 3775684; no GPU work before its terminal success',flush=True)
while started is not None:
    try:
        fields=parent.read_text().split()
        if fields[2]=='Z' or fields[21]!=started:break
    except FileNotFoundError:break
    time.sleep(10)
assert 'RETUNING AND FINAL QUALIFICATION COMPLETE' in Path('/tmp/r9700-final-retuning-controller.log').read_text()
assert bench.source_manifest()==json.loads((ROOT/'source.json').read_text())
original=REPO/'build-remaining-original/r9700-ntt-latency/test/test_bootstrap_bench'
candidate=REPO/'build-remaining-r9700-ntt-latency-baseline/test/test_bootstrap_bench'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for binary,receipt in [(original,ROOT/'original-baseline/r9700-ntt-latency/built.json'),(candidate,ROOT/'r9700-ntt-latency/baseline/checked.json')]:
    assert sha(binary)==json.loads(receipt.read_text())['binaries']['test_bootstrap_bench']
def hardware():
    result={}
    for d in Path('/sys/class/drm').glob('card[0-9]*/device'):
        for name in ['vendor','device','pp_dpm_sclk','pp_dpm_mclk','gpu_busy_percent']:
            try:result[str(d/name)]=(d/name).read_text().strip()
            except (OSError,UnicodeDecodeError):pass
    return result
records=[]
env=os.environ.copy()
env.setdefault('HIP_VISIBLE_DEVICES','0')
env.setdefault('OMP_NUM_THREADS','8')
env['CUFHE_BENCH_RESOURCES']='1'
env['LD_LIBRARY_PATH']=env.get('ROCM_PATH','/opt/rocm')+'/lib:'+env.get('LD_LIBRARY_PATH','')
with open('/tmp/cufhepp-r9700-gpu.lock','a') as gpu_lock:
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    assert bench.source_manifest()==json.loads((ROOT/'source.json').read_text())
    print('START isolated confirmation and original/original self-control',flush=True)
    for run in range(3):
        for phase in ['original-candidate','original-original']:
            members=[('first',original),('second',candidate if phase=='original-candidate' else original)]
            if run%2:members.reverse()
            for member,binary in members:
                prefix=OUT/f'{phase}-r{run}-{member}-n30';argv=[str(binary),'128','1','30','packed-nmux']
                command=dict(argv=argv,environment={k:env[k] for k in ['HIP_VISIBLE_DEVICES','OMP_NUM_THREADS','CUFHE_BENCH_RESOURCES','LD_LIBRARY_PATH']},binary_sha256=sha(binary),before=hardware())
                prefix.with_suffix('.command.json').write_text(json.dumps(command,indent=2)+'\n')
                with prefix.with_suffix('.log').open('w') as log:subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                parsed=bench.parse(prefix.with_suffix('.log'),30);assert parsed['errors']==0
                record=dict(phase=phase,run=run,member=member,repetitions=30,binary_sha256=sha(binary),after=hardware(),**parsed)
                prefix.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
                print('PASS',phase,run,member,flush=True)
    summaries=[]
    for phase in ['original-candidate','original-original']:
        gains=[]
        for run in range(3):
            pair={r['member']:r for r in records if r['phase']==phase and r['run']==run}
            gains.append(1-pair['second']['median_ms_per_gate']/pair['first']['median_ms_per_gate'])
        summaries.append(dict(phase=phase,gains=gains,median_gain=statistics.median(gains),loss_above_two_percent_in_all_runs=max(gains)<-.02))
    result=dict(scope='Isolated diagnostic only; preserves the original qualification data and does not establish profile-wide performance equivalence',source=ROOT.name,profile='r9700-ntt-latency',benchmark='test_bootstrap_bench',mode='packed-nmux',streams=128,runs=3,repetitions=30,invocations=len(records),errors=0,results=summaries)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CONFIRMATION COMPLETE',json.dumps(summaries),flush=True)
