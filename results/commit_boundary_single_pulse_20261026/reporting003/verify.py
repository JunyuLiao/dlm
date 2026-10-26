"""Final source, model, binding, native-accounting and completion verification."""
from pathlib import Path
import hashlib
import json
import time
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parents[1]
REPO=R.parents[1]
C=R/'panel/attempt004'
B=R/'boundary/attempt001'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    out=R/'verification001'
    out.mkdir(exist_ok=False)
    inputs={}
    def read(p):
        inputs[str(p.relative_to(R))]=sha(p)
        return json.loads(p.read_text())
    pins=[]
    for root in [B,C]:
        protocol=read(root/'protocol.json')
        checks={f:sha(REPO/f)==h for f,h in protocol['sources'].items()}
        assert all(checks.values())
        external=json.loads((root/'private/external_sources.json').read_text())
        assert all(sha(Path(f))==h for f,h in external.items())
        binding=read(root/'binding_receipt.json')
        assert binding['protocol_sha256']==sha(root/'protocol.json')
        pins.append(dict(attempt=str(root.relative_to(R)),source_checkpoint=binding['source_checkpoint'],protocol_sha256=sha(root/'protocol.json'),source_files=len(checks),all_passed=True,external_source_files=len(external)))
        assert read(root/'execution_complete.json')['passed']
    assert read(C/'qualification_complete.json')['passed']
    assert read(C/'qualification_controls.json')['passed']
    assert read(C/'online_legacy_qualification.json')['passed']
    assert read(C/'fairness001/summary.json')['passed']
    assert read(C/'paired_readout001/final_complete.json')['passed']
    assert read(R/'reporting_execution002/final_complete.json')['passed']
    model=read(C/'model_fingerprints.json')
    paths=json.loads((R/'private/runtime_paths.json').read_text())
    verified=[]
    for entry in model['files']:
        p=Path(paths['model_path'])/entry['file']
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        verified.append({k:entry[k] for k in ['file','bytes','sha256']})
        (out/'progress.json').write_text(json.dumps(dict(verified_model_files=len(verified),total=len(model['files']),unix=time.time())))
    installed=Path(paths['python']).parents[1]/'lib/python3.12/site-packages'
    runtime=[]
    for f,h in read(C/'protocol.json')['runtime_source_pins'].items():
        p=installed/('vllm/'+f if not f.startswith('vllm_flash_attn/') else f)
        assert sha(p)==h
        runtime.append(dict(logical_file=f,sha256=h))
    gold=read(R/'scorer_verification001.json')
    assert sha(Path(paths['gold_path']))==gold['files'][0]['sha256']
    manifest=read(C/'manifest_identity.json')
    binding=read(C/'manifest_binding.json')
    assert sha(C/'manifest_identity.json')==binding['manifest_sha256']
    assert sha(C/'private/manifest.json')==manifest['private_manifest_sha256']
    counts=dict(qualification_pairs=0,panel_pairs=0,valid_panel_timings=0,exact_work_rows=0,uncommitted_tail_calls=0)
    for phase,number in [('qualify',3),('panel',24)]:
        for arm in read(C/'protocol.json')['arms']:
            leaf=C/f'runs/{phase}/{arm}'
            assert read(leaf/'final_complete.json')['passed']
            for i in range(number):
                audit=read(leaf/f'audit{i:03d}.json')
                clean=read(leaf/f'clean{i:03d}.json')
                pair=read(leaf/f'pair{i:03d}.json')
                assert pair['passed']
                assert pair['audit_sha256']==sha(leaf/f'audit{i:03d}.json') and pair['clean_sha256']==sha(leaf/f'clean{i:03d}.json')
                if phase=='panel':
                    counts['panel_pairs']+=1
                    counts['valid_panel_timings']+=clean['timing_qualified']
                    counts['uncommitted_tail_calls']+=audit['uncommitted_tail']
                    assert (audit['id'],audit['seed'])==(manifest['identities'][i//2]['id'],[42,43][i%2])
                else:counts['qualification_pairs']+=1
                work=audit['work']
                ge=work['global_eligible_tiles'];gs=work['global_skipped_tiles'];le=work['native_local_eligible_rectangles'];ls=work['native_local_skipped_tiles']
                assert work['global_kept_tiles']==ge-gs and ls==0
                assert work['overall_eligible_rectangles']==ge+le
                assert work['overall_skipped_rectangles']==gs+ls
                assert work['overall_rectangle_sparsity']==(gs+ls)/(ge+le)
                assert audit['N']==sum(not x['commit'] for x in audit['native_trace'])
                assert audit['C']==sum(x['commit'] for x in audit['native_trace'])
                counts['exact_work_rows']+=1
    tests=ET.parse(C/'tests/prelaunch002.xml').getroot()
    suites=list(tests.iter('testsuite'))
    assert sum(int(x.attrib['tests']) for x in suites)==23
    assert all(int(x.attrib['failures'])==int(x.attrib['errors'])==0 for x in suites)
    assert read(B/'final_complete.json')['passed'] and read(R/'cpu_addendum001/final_complete.json')['passed']
    result=dict(passed=True,completed_unix=time.time(),source_pins=pins,model_revision=model['model_revision'],model_files=verified,runtime_files=runtime,official_gold_reverified=True,CPU_tests=23,counts=counts,
        no_native_forwards_in_verifier=True,phase_and_numerical_conventions_sha256=sha(R/'ANALYSIS_CONVENTIONS.json'))
    for name,value in [('summary.json',result),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True)
    print(json.dumps(dict(passed=True,counts=counts,model_files=len(verified),runtime_files=len(runtime),source_pins=pins)))


if __name__=='__main__':
    main()
