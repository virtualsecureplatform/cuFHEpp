#!/usr/bin/env python3
"""Verify focused implementation evidence; final exhaustive/performance audit is separate."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
OLD=ROOT.parent/'2d79cdeec061447d'
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
previous=read(OLD/'source.json');current=bench.source_manifest()
relevant=[p for p in previous['files'] if p.startswith(('include/','src/','test/')) or p in ['CMakeLists.txt','CMakePresets.json']]
assert all(current['files'][p]==previous['files'][p] for p in relevant)
allpath=OLD/'r9700-ntt-throughput/all'
options=read(allpath/'options.json')
cache={}
for line in (REPO/'build-remaining-r9700-ntt-throughput-all/CMakeCache.txt').read_text().splitlines():
    if line and not line.startswith(('#','//')) and '=' in line and ':' in line.split('=',1)[0]:
        key,value=line.split('=',1);cache[key.split(':')[0]]=value
assert all(cache[k]==v for k,v in options.items())
fixtures=[
 ('prepared-key-schedules','test_keyswitch_gpu_lvl20.log',['key-switch exercised policy=258 threads=512 extracted=1: PASS','key-switch exercised policy=261 threads=512 extracted=0: PASS','aliased scratch']),
 ('output-exponent-cache','test_small_modulus.log',['Output exponent cache (4762800 exponents / body-padding / both aliases): PASS','Full BR original / output-cache / MUX signs (172032 coefficients / exact-forward aliases): PASS']),
 ('scalar64-cache-register-io-and-reuse','test_small_modulus.log',['scalar64 exercised policy=7 PASS','scalar64 delayed peer-readers / workspace reuse: PASS','64-bit scalar cached / uncached external product (49152 coefficients): PASS']),
 ('register31-consumer-schedules','test_small_modulus.log',['register31 exercised policy=1 PASS','register31 exercised policy=3 PASS','31-bit register schedules / external product (65536 coefficients): PASS']),
 ('native-arithmetic','test_small_modulus.log',['64-bit host/device modular addition (300256 cases): PASS','64-bit host/device modular subtraction (200144 cases): PASS','64-bit host/device full uint128 reduction (1000400 cases): PASS','64-bit host/device madd2 (266384 cases): PASS','64-bit host/device madd4 (396611 cases): PASS']),
 ('exact-torus-scaled-inverse','test_small_modulus.log',['64-bit host/device exact torus conversion (500200 cases): PASS','64-bit host/device fixed inverse scaling: PASS']),
 ('packed-roots-inverse-difference','test_small_modulus.log',['31-bit inverse butterflies (100125 cases): PASS','31-bit paired roundtrip / negacyclic square: PASS'])]
rows=[]
receipt=read(allpath/'checked.json');assert receipt['source']==OLD.name
for requirement,filename,expected in fixtures:
    path=allpath/filename;body=path.read_text()
    assert all(text in body for text in expected),requirement
    rows.append(dict(requirement=requirement,status='focused-reference-evidence-verified',log='../'+str(path.relative_to(ROOT.parent)),log_sha256=sha(path),verified_markers=expected))
prefetch=OLD/'r9700-ntt/prefetch'
assert read(prefetch/'options.json')['USE_HIP_MONOMIAL_PREFETCH']=='ON'
path=prefetch/'test_small_modulus.log';marker='64-bit staged paired / independent scalar external product: PASS';assert marker in path.read_text()
rows.append(dict(requirement='software-monomial-prefetch',status='focused-reference-evidence-verified',log='../'+str(path.relative_to(ROOT.parent)),log_sha256=sha(path),verified_markers=[marker]))
binary_checks=[]
for directory,build in [(allpath,'build-remaining-r9700-ntt-throughput-all'),(prefetch,'build-remaining-r9700-ntt-prefetch')]:
    checked=read(directory/'checked.json');assert checked['source']==OLD.name
    for name,digest in checked['binaries'].items():assert sha(REPO/build/'test'/name)==digest,(build,name)
    binary_checks.append(dict(configuration=build,binary_hashes_verified=len(checked['binaries'])))
stress=read(OLD/'inverse-reuse-regression/validation.json')
assert next(r for r in stress['results'] if r['variant']=='corrected')['exits']==[0,0,0]
assert next(r for r in stress['results'] if r['variant']=='barrier-removed')['exits']==[1,1,1]
result=dict(scope='Focused exact-reference/policy evidence only; not exhaustive configuration coverage, performance qualification or whole-plan completion',source=ROOT.name,evidence_source=OLD.name,unchanged_cpp_build_and_harness_files=len(relevant),all_combined_options=options,combined_effective_cache_matches=True,combined_test_receipt_exhaustive=receipt['exhaustive'],requirements=rows,inverse_reuse_negative_control_verified=True)
result['current_evidence_binaries']=binary_checks
(ROOT/'focused-implementation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(f'PASS {len(rows)} focused implementation requirements; {len(relevant)} current source files match tested snapshot; exhaustive final audit remains separate')
