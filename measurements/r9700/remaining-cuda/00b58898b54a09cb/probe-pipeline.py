#!/usr/bin/env python3
"""One isolated checkout pipeline; its binaries never share old-root controls."""
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import bench_r9700_remaining as bench
root=Path(__file__).resolve().parent
assert bench.ROOT==root
manifest=bench.source_manifest()
fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
snapshot=root/'measurements/r9700/remaining-cuda'/fingerprint
snapshot.mkdir(parents=True,exist_ok=True)
real_flock=fcntl.flock
profile_lock=Path('/tmp/cufhepp-r9700-ntt-throughput-experiment.lock')
isolated_lock_path=Path(f'/tmp/cufhepp-policy-probe-{fingerprint}-ntt-throughput.lock')
isolated_profile_lock=isolated_lock_path.open('w')
def isolated_flock(fd,operation):
    descriptor=fd if isinstance(fd,int) else fd.fileno()
    if Path(f'/proc/self/fd/{descriptor}').resolve()==profile_lock:
        return real_flock(isolated_profile_lock,operation)
    return real_flock(fd,operation)
fcntl.flock=isolated_flock
policy=dict(source=fingerprint,isolated_root=str(root),single_pipeline=True,shared_gpu_lock='/tmp/cufhepp-r9700-gpu.lock',profile_lock_redirected_from=str(profile_lock),isolated_profile_lock=str(isolated_lock_path),phases=['build-exhaustive','check-exhaustive','screen-critical30'],variants=['multi','multi-mux'])
(snapshot/'pipeline-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
for action in ['build','check','screen']:
    if action=='screen':
        cases=[('batch',1,1,False),('batch',1,512,False),('packed-mux',8,1,False),('packed-mux',32,1,False),('packed-nmux',32,1,False)]
        bench.cases=lambda full:cases
        (snapshot/'critical-cases.json').write_text(json.dumps(dict(benchmarks=bench.BENCHES,cases=cases,repetitions=30,fresh_keys=3),indent=2)+'\n')
    sys.argv=[__file__,action,'--profiles','r9700-ntt-throughput','--variants','multi','multi-mux','--repeats','30']
    print('policy probe',fingerprint,action,flush=True)
    bench.main()
