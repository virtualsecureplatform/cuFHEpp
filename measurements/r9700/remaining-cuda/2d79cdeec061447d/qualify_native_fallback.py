#!/usr/bin/env python3
"""Qualify native arithmetic separately if the prepared/native bundle fails."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]

def requires_native_qualification(report):
    if not report['full_grid_complete'] or report['needs_30']:
        raise RuntimeError('Bundle qualification has not resolved the required grid')
    return bool(report['regressions'] or report.get('unresolved_regressions', 0)
                or not report['wins'])

pid = 3463048
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

sys.path.insert(0, str(REPO))
import bench_r9700_remaining as bench
assert bench.source_manifest() == json.loads((ROOT/'source.json').read_text())
subprocess.run(['python3', '-B', str(ROOT/'qualification_audit.py')],
               cwd=REPO, check=True)
report = next(r for r in json.loads((ROOT/'qualification-audit.json').read_text())
              if r['profile'] == 'r9700-ntt' and r['variant'] == 'multi-native64')
required = requires_native_qualification(report)
decision = dict(profile='r9700-ntt', candidate='native64',
                status='qualify-individual-native-path' if required else 'skip-qualified-bundle-retained',
                bundle_regressions=report['regressions'],
                bundle_unresolved_regressions=report.get('unresolved_regressions', 0))
(ROOT/'native-fallback-decision.json').write_text(json.dumps(decision, indent=2)+'\n')
print(decision, flush=True)
if not required:
    sys.exit(0)

for action in ('build', 'check'):
    print('native-only exhaustive '+action, flush=True)
    subprocess.run(['python3', '-B', str(REPO/'bench_r9700_remaining.py'), action,
                    '--profiles', 'r9700-ntt', '--variants', 'native64'],
                   cwd=REPO, check=True)
print('native-only original full qualification', flush=True)
qualification = subprocess.Popen(
    ['python3', '-B', str(ROOT/'qualify_original.py'), '--profiles', 'r9700-ntt',
     '--variants', 'native64'], cwd=REPO)
if qualification.wait():
    raise RuntimeError('Native-only full qualification did not finish')

variants = [f'native64-waves{n}' for n in (2, 3, 4)]
variants += [f'native64-groups{n}' for n in (2, 3, 5, 9)]
subprocess.run(
    ['python3', '-B', str(ROOT/'tune_after_qualification.py'),
     '--pid', str(qualification.pid), '--profile', 'r9700-ntt',
     '--candidate', 'native64', '--definitions', str(ROOT/'native-tuning-definitions.json'),
     '--variants', *variants, '--log-prefix', '/tmp/r9700-remaining-native-tuning'],
    cwd=REPO, check=True)
