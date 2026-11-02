#!/usr/bin/env python3
"""Compose the two independently promising paired-profile candidates."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
pid = 3141159
stat = Path(f'/proc/{pid}/stat')
try:
    started = stat.read_text().split()[21]
except FileNotFoundError:
    started = None
while started:
    try:
        fields = stat.read_text().split()
        if fields[21] != started or fields[2] == 'Z':
            break
    except FileNotFoundError:
        break
    time.sleep(10)

subprocess.run(['python3', '-B', str(ROOT/'exploration_progress.py')], cwd=REPO,
               stdout=subprocess.DEVNULL, check=True)
points = json.loads((ROOT/'exploration-progress.json').read_text())['complete_three_key_points']
sys.path.insert(0, str(REPO))
import bench_r9700_remaining as bench
for variant in ('multi', 'native64'):
    names = bench.BENCHES if variant == 'multi' else [n for n in bench.BENCHES if '_lvl02' in n]
    expected = {(n, *case) for n in names for case in bench.cases(False)}
    rows = [p for p in points if p['profile'] == 'r9700-ntt' and p['variant'] == variant]
    actual = {(p['benchmark'], p['mode'], p['streams'], p['count'], p['graph']) for p in rows}
    if not expected <= actual:
        raise RuntimeError('Missing individual feature screen: '+variant)
    if any(p['median_improvement'] < -.02 for p in rows) or not any(p['min_improvement'] >= .03 for p in rows):
        raise RuntimeError('Individual feature no longer meets exploratory combination gate: '+variant)

base = ['python3', '-B', str(ROOT/'experiment_configs.py')]
options = [str(ROOT/'paired-combination-definitions.json'), '--profiles', 'r9700-ntt', '--variants', 'multi-native64']
for action in ('build', 'check', 'screen'):
    print('paired multi-native64 '+action, flush=True)
    subprocess.run([*base, action, *options], cwd=REPO, check=True)

subprocess.run(['python3', '-B', str(ROOT/'exploration_progress.py')], cwd=REPO,
               stdout=subprocess.DEVNULL, check=True)
rows = [p for p in json.loads((ROOT/'exploration-progress.json').read_text())['complete_three_key_points']
        if p['profile'] == 'r9700-ntt' and p['variant'] == 'multi-native64']
if len(rows) != len(bench.BENCHES)*len(bench.cases(False)):
    raise RuntimeError('Combination screen incomplete')
decision = dict(profile='r9700-ntt', variant='multi-native64',
                wins=sum(p['min_improvement'] >= .03 for p in rows),
                regressions=sum(p['median_improvement'] < -.02 for p in rows))
decision['status'] = 'proceed-to-original-full-grid' if decision['wins'] and not decision['regressions'] else 'experimental-opt-in'
(ROOT/'paired-combination-screen-decision.json').write_text(json.dumps(decision, indent=2)+'\n')
print(decision, flush=True)
if decision['status'] == 'proceed-to-original-full-grid':
    subprocess.run([*base, 'qualify', *options], cwd=REPO, check=True)
