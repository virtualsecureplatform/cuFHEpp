#!/usr/bin/env python3
"""Run a tuning sweep only after its original-backend qualification finishes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench

def qualification_argv(profile, candidate, definitions=None):
    if definitions is None:
        command = ['python3', '-B', str(ROOT/'qualify_original.py')]
    else:
        command = ['python3', '-B', str(ROOT/'experiment_configs.py'),
                   'qualify', str(definitions)]
    return [*command, '--profiles', profile, '--variants', candidate,
            '--repeats', '30']

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--profile',required=True)
p.add_argument('--candidate',required=True)
p.add_argument('--definitions',type=Path,required=True)
p.add_argument('--candidate-definitions',type=Path,
               help='Register a composed candidate when rerunning uncertain points')
p.add_argument('--variants',nargs='+',required=True)
p.add_argument('--log-prefix',required=True)
a=p.parse_args()
if a.candidate_definitions:
    entries=json.loads(a.candidate_definitions.read_text()).get(a.profile,{})
    if a.candidate not in entries:
        p.error('Candidate is absent from --candidate-definitions')
stat=Path(f'/proc/{a.pid}/stat')
try:started=stat.read_text().split()[21]
except FileNotFoundError:started=None
while started:
    try:
        fields=stat.read_text().split()
        if fields[21]!=started or fields[2]=='Z':break
    except FileNotFoundError:break
    time.sleep(10)
path=ROOT/'qualification'/a.profile/a.candidate/'full-comparison.json'
rows=json.loads(path.read_text())
expected={(name,*case,run) for name in bench.BENCHES for case in bench.cases(True) for run in range(3)}
actual={(r['benchmark'],r['mode'],r['streams'],r['count'],r['graph'],r['run']) for r in rows}
if actual!=expected:raise RuntimeError('Preceding qualification did not complete the full grid: '+str(path))
def audit():
    subprocess.run(['python3','-B',str(ROOT/'qualification_audit.py')],cwd=REPO,check=True)
    reports=json.loads((ROOT/'qualification-audit.json').read_text())
    report=next(r for r in reports if r['profile']==a.profile and r['variant']==a.candidate)
    if not report['full_grid_complete']:raise RuntimeError('Qualification coverage is incomplete')
    return report
report=audit()
uncertain=[point for point in report['points'] if point['needs_30']]
if uncertain:
    argv=qualification_argv(a.profile,a.candidate,a.candidate_definitions)
    for point in uncertain:
        case=f"{point['benchmark']}:{point['mode']}:{point['streams']}:{point['count']}:{int(point['graph'])}"
        argv.extend(['--case',case])
    Path(a.log_prefix+'-rerun30.command.json').write_text(json.dumps(argv,indent=2)+'\n')
    with open(a.log_prefix+'-rerun30.log','w') as out:
        subprocess.run(argv,cwd=REPO,stdout=out,stderr=subprocess.STDOUT,check=True)
    report=audit()
if report['needs_30']:raise RuntimeError('Required reruns did not complete')
if report['regressions'] or report.get('unresolved_regressions', 0) or not report['wins']:
    decision=dict(profile=a.profile,candidate=a.candidate,status='experimental-opt-in',reason='Full original qualification has regressions, unresolved variation, or no repeatable improvement; tuning not launched',regressions=report['regressions'],unresolved_regressions=report.get('unresolved_regressions', 0),wins=report['wins'])
    (ROOT/(a.profile+'-'+a.candidate+'-tuning-decision.json')).write_text(json.dumps(decision,indent=2)+'\n')
    print(decision,flush=True)
    sys.exit(0)
for action in ['build','check','screen']:
    argv=['python3','-B',str(ROOT/'experiment_configs.py'),action,str(a.definitions),'--profiles',a.profile,'--variants',*a.variants]
    log=Path(a.log_prefix+'-'+action+'.log')
    with log.open('w') as out:
        subprocess.run(argv,cwd=REPO,stdout=out,stderr=subprocess.STDOUT,check=True)
