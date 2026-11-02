#!/usr/bin/env python3
"""Collect measured benchmark output; incomplete or failing runs are rejected."""
import json
import re
import shlex
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
commands = json.loads((root / 'commands.json').read_text()) if (root / 'commands.json').exists() else {}
records = []
for path in sorted(root.rglob('*.log')):
    text = path.read_text(errors='replace')
    header = re.search(r'^GPU=(.*?) n=(\d+) N=(\d+) k=(\d+) l=(\d+) torus_bits=(\d+) order=(\S+) gate=(\S+) backend=(\S+) block_binary=(\d+)', text, re.M)
    if header is None:
        continue
    result = re.search(r'dispatch=(\S+) streams=(\d+) gates=(\d+) median_ms/gate=([\d.e+-]+) errors=(\d+)', text)
    if result is None:
        raise SystemExit(f'Incomplete benchmark: {path.relative_to(root)}')
    if int(result[5]):
        raise SystemExit(f'Failed decryption: {path.relative_to(root)}')
    samples = [float(x) for x in re.findall(r'^repeat=\d+ ms/gate=([\d.e+-]+)', text, re.M)]
    command = commands.get(str(path.relative_to(root)))
    command_file = path.with_suffix('.command')
    if command is None and command_file.exists():
        argv = shlex.split(command_file.read_text())
        command = dict(argv=argv, mode=argv[4])
    explicit_mode = re.search(r' mode=(\S+)', text.splitlines()[0])
    mode = command['mode'] if command else explicit_mode[1] if explicit_mode else None
    transfer_tag = re.search(r' timed_transfers=([01])', text.splitlines()[0])
    timed_transfers = bool(int(transfer_tag[1])) if transfer_tag else (mode not in ('batch', 'packed', 'packed-mux', 'packed-nmux') and result[1] == 'streams' and header[7] == 'br-iks' and header[8] in ('nand', 'mux', 'nmux'))
    dispatch = 'packed' if mode in ('packed', 'packed-mux', 'packed-nmux') and result[1] != 'graph' else result[1]
    records.append(dict(command=command['argv'] if command else None, reported_dispatch=result[1], timed_transfers=timed_transfers, log=str(path.relative_to(root)), gpu=header[1], n=int(header[2]), N=int(header[3]), k=int(header[4]), l=int(header[5]), torus_bits=int(header[6]), order=header[7], gate=header[8], backend=header[9], dispatch=dispatch, streams=int(result[2]), gates=int(result[3]), median_ms_per_gate=float(result[4]), repetitions=len(samples), min_ms_per_gate=min(samples), max_ms_per_gate=max(samples), errors=int(result[5])))
(root / 'summary.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'{len(records)} complete benchmarks; all decryptions passed')
