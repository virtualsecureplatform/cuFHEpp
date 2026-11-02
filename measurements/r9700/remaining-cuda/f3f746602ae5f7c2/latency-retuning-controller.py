import json,subprocess
from pathlib import Path
repo=Path('/home/nimda/sources/cuFHEpp'); snapshot=Path('/tmp/r9700-final-snapshot').read_text().strip();root=Path(snapshot)
definitions=root/'latency-retuning-definitions.json'; variants=list(json.loads(definitions.read_text())['r9700-ntt-latency'])
for action in ['build','check','screen','report']:
 argv=['python3','-B',str(root/'experiment_configs.py'),action,str(definitions),'--profiles','r9700-ntt-latency','--variants',*variants]
 if action in ['build','check']:argv+=['--focused']
 print('START',action,flush=True)
 with open('/tmp/r9700-final-latency-retune-'+action+'.log','w') as out:subprocess.run(argv,cwd=repo,stdout=out,stderr=subprocess.STDOUT,check=True)
 print('PASS',action,flush=True)
print('LATENCY RETUNING SCREEN COMPLETE',flush=True)
