#!/usr/bin/env python3
"""Verify final measurement evidence. This does not assert whole-plan completion."""
import hashlib,json,math,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[3]
sys.path.insert(0,str(REPO))
import bench_r9700_remaining as bench
OLD=ROOT.parent/'2d79cdeec061447d'
PROFILES=['r9700-ntt','r9700-ntt-throughput','r9700-ntt-latency','r9700-fft']
BINARIES=['test_bootstrap_bench'+suffix for suffix in ['', '_lvl02', '_lvl01_br_iks', '_lvl02_br_iks']]
COUNTS=[1,2,3,4,5,6,7,9,10,11,16,17,32,64,128,512]
CASES=[('batch',1,n,False) for n in COUNTS]
CASES += [(mode,n,1,False) for mode in ['packed','packed-mux','packed-nmux'] for n in [1,8,32,128]]
CASES += [('batch',1,17,True)]
EXPECTED={(name,*case) for name in BINARIES for case in CASES}
assert len(EXPECTED)==116

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise RuntimeError(message)

def main():
    recommendations=read(ROOT/'final-retuning-recommendations.json')
    require({r['profile'] for r in recommendations}==set(PROFILES),'Four final configurations required')
    previous=read(OLD/'source.json');current=bench.source_manifest()
    require(current['tfhepp']=='1dd9a1addef4cae01da9bb6bfff0d22b55bec683','Dependency pin changed')
    dependency_changes=subprocess.check_output(['git','-C','thirdparties/TFHEpp','status','--porcelain','--untracked-files=no'],cwd=REPO,text=True)
    require(not dependency_changes.strip(),'TFHEpp has uncommitted source changes')
    # CMakePresets may change only for separately reviewed performance promotions.
    relevant=[p for p in previous['files'] if p.startswith(('include/','src/','test/')) or p=='CMakeLists.txt']
    require(all(current['files'][p]==previous['files'][p] for p in relevant),'C++/harness/build source changed since qualification')
    output=dict(scope='Final configuration measurements, exhaustive receipts, dependency pin and compile-only CUDA compatibility; documentation and requirement review remain separate',source=ROOT.name,tfhepp=current['tfhepp'],profiles=[])
    provenance=read(ROOT/'original-baseline'/'provenance.json')
    require(provenance['backend_commit']=='b0818d28c5eab44dda1592225a132a4abd97517f','Wrong original backend')
    require(provenance['tfhepp']==current['tfhepp'],'Original dependency differs')
    require(provenance['harness_sha256']==current['files']['test/test_bootstrap_bench.cc'],'Original harness differs')
    audits=read(ROOT/'qualification-audit.json')
    for recommendation in recommendations:
        profile=recommendation['profile'];variant=recommendation['recommended_variant']
        receipt=read(ROOT/profile/variant/'checked.json')
        require(receipt['source']==ROOT.name and receipt['exhaustive'],f'Exhaustive validation required: {profile}/{variant}')
        tests=bench.TESTS+(['test_fft_roundtrip'] if profile=='r9700-fft' else ['test_small_modulus','test_bootstrap_cooperative','test_bootstrap_public_cooperative'])
        require(set(receipt['tests'])==set(tests),f'Incomplete exhaustive test list: {profile}/{variant}')
        for name,digest in receipt['binaries'].items():
            require(sha(REPO/f'build-remaining-{profile}-{variant}'/'test'/name)==digest,f'Changed selected binary: {profile}/{variant}/{name}')
        for name in [*tests,'test_bootstrap_mux-graph','test_bootstrap_public_mux-graph']:
            require((ROOT/profile/variant/(name+'.log')).is_file(),f'Missing test log: {profile}/{variant}/{name}')
        original=read(ROOT/'original-baseline'/profile/'built.json')
        for name in BINARIES:
            require(sha(REPO/'build-remaining-original'/profile/'test'/name)==original['binaries'][name],f'Changed original binary: {profile}/{name}')
        directory=ROOT/'qualification'/profile/variant
        index={}
        for phase in ['full','selected']:
            for path in (directory/phase).glob('*.json'):
                record=read(path)
                key=(record['benchmark'],record['mode'],record['streams'],record['count'],record['graph'],record['run'],record['member'])
                if key not in index or record['repetitions']>index[key][0]['repetitions']:index[key]=(record,path)
        required=set(EXPECTED)
        if profile!='r9700-fft':required|={(name,'batch',1,n,False) for name in BINARIES for n in range(1,23)}
        audit=next(r for r in audits if r['profile']==profile and r['variant']==variant)
        points={(p['benchmark'],p['mode'],p['streams'],p['count'],p['graph']):p for p in audit['points']}
        logs=0;resources=0;longer=0
        for key in sorted(required):
            point=points.get(key)
            require(point is not None and point['complete'] and not point['needs_30'],f'Incomplete/uncertain point: {profile}/{variant}/{key}')
            rows={r['run']:r for r in point['rows']}
            for run in [0,1,2]:
                pair=[]
                for member in ['original','candidate']:
                    record,path=index[(*key,run,member)]
                    require(record['profile']==profile and record['variant']==variant and record['errors']==0,f'Wrong/failing record: {path}')
                    expected_digest=(original if member=='original' else receipt)['binaries'][key[0]]
                    require(record['binary_sha256']==expected_digest,f'Unmatched measured executable: {path}')
                    require(record['repetitions']>=15,f'Too few repetitions: {path}')
                    log=path.with_suffix('.log');parsed=bench.parse(log,record['repetitions'])
                    require(parsed['samples_ms_per_gate']==record['samples_ms_per_gate'],f'Samples differ from raw log: {path}')
                    require(parsed['median_ms_per_gate']==record['median_ms_per_gate'],f'Median differs from raw log: {path}')
                    require(path.with_suffix('.command').is_file(),f'Missing invocation: {path}')
                    body=log.read_text()
                    require(re.search(r'resource_node=\d+ registers=\d+ static_lds=\d+ dynamic_lds=\d+ local_bytes=\d+ threads=\d+ resident_blocks=\d+',body) is not None,f'Missing actual kernel resources: {log}')
                    pair.append(record);logs+=1;resources+=1
                require(pair[0]['repetitions']==pair[1]['repetitions'],f'Unmatched repetitions: {key}')
                gain=1-pair[1]['median_ms_per_gate']/pair[0]['median_ms_per_gate']
                require(math.isclose(gain,rows[run]['gain'],abs_tol=1e-12),f'Audit disagrees with raw pair: {key}')
                if pair[0]['repetitions']>=30:longer+=1
        output['profiles'].append(dict(profile=profile,variant=variant,standard_points=116,transition_extra_points=len(required-EXPECTED),independent_runs=3,verified_raw_logs=logs,resource_records_present=resources,longer_comparisons=longer,exhaustive_tests=len(tests),graph_tests=2,wins=audit['wins'],median_regressions=audit['regressions'],unresolved_regressions=audit['unresolved_regressions']))
    cuda=read(OLD/'cuda-compatibility'/'validation.json')
    require({r['backend'] for r in cuda}=={'ntt','fft'},'Both CUDA backends required')
    for row in cuda:
        require(row['compile']=='passed' and sha(REPO/row['binary'])==row['sha256'],'CUDA compile executable changed')
        images=(OLD/'cuda-compatibility'/(row['backend']+'-images.log')).read_text()
        require('sm_80.cubin' in images and 'sm_89.cubin' in images,'SM80/SM89 images required')
        require(row['runtime']=='not-tested-no-NVIDIA-device','Unexpected CUDA runtime claim')
    stress=read(OLD/'inverse-reuse-regression'/'validation.json')
    for name,exits in [('corrected',[0,0,0]),('barrier-removed',[1,1,1])]:
        row=next(r for r in stress['results'] if r['variant']==name)
        require(row['exits']==exits,'Inverse workspace reuse regression control failed')
        for run in range(3):require((OLD/'inverse-reuse-regression'/f'{name}-{run}.log').is_file(),'Missing stress raw log')
    output.update(status='PASS',cuda='NTT+FFT SM80/SM89 compile/images verified; NVIDIA runtime unavailable',stress='Three corrected passes and three expected barrier-removed failures')
    (ROOT/'final-measurement-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
