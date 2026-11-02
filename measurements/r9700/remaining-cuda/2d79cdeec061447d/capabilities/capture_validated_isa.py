#!/usr/bin/env python3
"""Capture selected ISA from binaries tied to current validation receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
SNAPSHOT=Path(__file__).resolve().parents[1]
REPO=SNAPSHOT.parents[3]
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench
assert bench.source_manifest()==json.loads((SNAPSHOT/'source.json').read_text())
objdump='/opt/rocm/llvm/bin/llvm-objdump'
nm='/opt/rocm/llvm/bin/llvm-nm'
math=['TestMadd2Kernel','TestMadd4WideKernel','TestGoldilocksAddKernel','TestGoldilocksSubKernel','TestGoldilocksTorusKernel']
configs=[('baseline-arithmetic','r9700-ntt-throughput','baseline',math),('all-arithmetic','r9700-ntt-throughput','all',math),('prefetch-fixtures','r9700-ntt','prefetch',['TestScalarMonomialCacheKernelILb1ELb1ELb1ELb0E','TestPairedMonomialCache31KernelILb0ELb0ELb1E'])]
for label,profile,variant,patterns in configs:
    directory=SNAPSHOT/profile/variant
    receipt_path=directory/'checked.json'
    receipt=json.loads(receipt_path.read_text())
    binary=REPO/f'build-remaining-{profile}-{variant}/test/test_small_modulus'
    digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    assert receipt['source']==SNAPSHOT.name and receipt['binaries']['test_small_modulus']==digest
    extracted=subprocess.check_output([objdump,'--offloading',str(binary)],cwd=REPO,text=True)
    (Path(__file__).parent/f'validated-{label}-offloading.log').write_text(extracted)
    objects=[Path(line.split('Extracting offload bundle: ',1)[1]) for line in extracted.splitlines() if 'Extracting offload bundle: ' in line and 'gfx1201' in line]
    assert len(objects)==1
    symbols=[]
    for line in subprocess.check_output([nm,'--defined-only',str(objects[0])],text=True).splitlines():
        parts=line.split()
        if len(parts)==3 and parts[1] in ('T','W') and any(pattern in parts[2] for pattern in patterns):symbols.append(parts[2])
    assert len(symbols)==len(patterns),(label,symbols)
    argv=[objdump,'-d','--disassemble-symbols='+','.join(symbols),str(objects[0])]
    isa=subprocess.check_output(argv,cwd=REPO,text=True)
    output=Path(__file__).parent/f'validated-{label}.isa'
    output.write_text(isa)
    cache=json.loads((directory/'effective-cache.json').read_text())
    metadata=dict(source=SNAPSHOT.name,binary=str(binary.relative_to(REPO)),sha256=digest,validated_receipt=str(receipt_path.relative_to(SNAPSHOT)),command=argv,symbols=symbols,hip_options={k:v for k,v in cache.items() if k.startswith('USE_HIP_')},isa_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(label,len(symbols),'kernels captured; receipt hash matches')
