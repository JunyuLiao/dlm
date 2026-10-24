"""Review frozen identities, complete matched coverage, invariants and privacy."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / 'closeout001'
    out.mkdir(exist_ok=False)
    protocol = read(root / 'e2e_protocol.json')
    score = read(root / 'scoring001/official_scores.json')
    analysis = read(root / 'analysis001/summary.json')
    checks = dict(
        sources=all(sha(p) == h for p,h in protocol['sources'].items()),
        scorer_sources=all(sha(p) == h for p,h in protocol['scorer_sources'].items()),
        private_inputs=sha(root/'private/tokenized_manifest.json') == protocol['private_manifest_sha256'],
        public_identity=sha(root/'manifest_identity.json') == protocol['manifest_identity_sha256'],
        model_identity=sha(root/'model_fingerprints.json') == protocol['model_fingerprints_sha256'],
        runtime_identity=sha(root/'runtime_fingerprints.json') == protocol['runtime_fingerprints_sha256'],
        exact_25x2_scope=len(protocol['panel']['short']) == 40 and len(protocol['panel']['long']) == 10,
        nine_original_blocks=len(protocol['blocked_ids']) == 9,
        nine_arms=len(protocol['arms']) == 9,
        official_score_coverage=len(score['outcomes']) == 450,
        official_sandbox=score['human_eval_sandbox_preflight'] == 'passed',
        analysis_protocol=analysis['protocol_sha256'] == sha(root/'e2e_protocol.json'),
        analysis_inputs=all(sha(root/p) == h for p,h in read(root/'analysis001/input_hashes.json').items()),
    )
    cpu = ET.parse(root/'tests/cpu002.xml').getroot().find('testsuite').attrib
    checks['CPU32']=cpu['tests']=='32' and cpu['errors']==cpu['failures']==cpu['skipped']=='0'
    pins=[];resources=[];invalid_timing=[];scope_errors=[];cells=0
    for suite in ['short','long']:
        planned={(x['id'],x['seed']) for x in protocol['panel'][suite]}
        for arm in protocol['arms']:
            directory_binding=read(root/'result_directory_bindings.json')
            leaf=root/directory_binding['result_paths'].get(suite,{}).get(arm,'runs/panel/'+suite+'/'+arm)
            launch_root=root/directory_binding['launch_paths'].get(suite,{}).get(arm,'launches/'+suite+'/'+arm)
            marker=read(leaf/'final_complete.json')
            launch=read(launch_root/'launch_receipt.json')
            exit_receipt=read(launch_root/'exit_receipt.json')
            assert marker['passed'] and marker['cells']==len(planned) and exit_receipt['returncode']==0
            assert marker['protocol_sha256']==launch['protocol_sha256']==sha(root/'e2e_protocol.json')
            assert launch['gpu_idle_before'] and launch['campaign_budget'] is None
            resources.append(dict(suite=suite,arm=arm,allocated_worker_seconds=exit_receipt['allocated_worker_seconds'],
                engine_executions=marker['model_executions'],startup_forwards=marker['startup_model_forwards']))
            observed=set()
            for f in sorted(leaf.glob('clean[0-9][0-9][0-9].json')):
                b=read(f);a=read(leaf/f.name.replace('clean','audit'))
                observed.add((b['id'],b['seed']));cells+=1
                import math
                identity=next(x for x in read(root/'manifest_identity.json')['identities'] if x['id']==b['id'])
                assert b['model_forwards']==math.ceil(identity['prompt_token_count']/16384)+b['N']+b['C']
                private=read(root/'private/generations/panel'/suite/arm/f.name)
                assert private['id']==b['id'] and private['seed']==b['seed'] and private['output_sha256']==b['output_sha256']
                assert all(a[k]==b[k] for k in ['output_sha256','N','C','T'])
                assert a['N']==sum(not x['commit'] for x in a['native_trace'])
                assert a['C']==sum(x['commit'] for x in a['native_trace'])
                native_canvas_steps=[];tail=0
                for event in a['native_trace']:
                    if event['commit']:native_canvas_steps.append(tail);tail=0
                    else:tail+=1
                assert len(native_canvas_steps)==b['C'] and sum(native_canvas_steps)+tail==b['N']
                if not b['timing_valid']:
                    invalid_timing.append(dict(arm=arm,id=b['id'],seed=b['seed'],captures=b['cuda_capture_events'],jit=b['request_jit_events']))
                work=a['work']
                expected_decode_mode='FULL' if arm=='dense_full_fix51994' else 'PIECEWISE'
                assert work['decode_cudagraph_modes']=={expected_decode_mode:a['N']}
                assert work['native_local_skipped_tiles']==0
                assert work['global_kept_tiles']<=work['global_eligible_tiles']
                selector=a['selector']
                if selector:
                    assert selector['selector_extra_model_forwards']==0 and selector['current_qkv']
                    assert not selector['cached_old_kv'] and not selector['cached_attention_outputs']
                    assert selector['target_skipped_tiles']==work['global_skipped_tiles']
                    assert selector['carried_mask_calls']==0
                    for call in selector['physical_calls']:
                        if call['canvas_skipped'] or call['local_skipped']:
                            scope_errors.append(dict(arm=arm,id=b['id'],call=call))
                    for bank in selector['first_two_candidate_banks']:
                        for unit in bank['units']:
                            chosen=unit['chosen']
                            if chosen:
                                assert chosen['feasible'] and chosen['current_error']<=.1 and chosen['deleted_mass']<=.2
                                assert len(chosen['tiles'])==unit['deletion_budget']
                pins.append(dict(suite=suite,arm=arm,id=b['id'],seed=b['seed'],audit_clean_exact=True,
                    clean_timing_valid=b['timing_valid'],source_checkpoint=launch['source_checkpoint']))
            assert observed==planned
    checks['450_exact_pairs']=cells==450
    checks['selector_scope_and_guards']=not scope_errors
    # Invalid timings are explicitly listed. They cannot support any speed claim.
    checks['timing_availability_recorded']=True
    forbidden={'prompt','gold','answer','completion','token_ids','prompt_token_ids','generated_text','credentials','api_key','password'}
    hits=[];public=[]
    def scan(x,path):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in forbidden:hits.append([path,k])
                if k=='stop_reason' and isinstance(v,int):hits.append([path,'raw_stop_token_id'])
                scan(v,path)
        elif isinstance(x,list):
            for v in x:scan(v,path)
    for folder,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in {'private','cache','runtime','__pycache__'}]
        for name in files:
            p=Path(folder)/name;public.append(p)
            if p.suffix=='.json':scan(read(p),str(p.relative_to(root)))
    checks['public_json_privacy']=not hits
    checks['private_inputs_ignored']=subprocess.run(['git','check-ignore',str(root/'private/tokenized_manifest.json')],capture_output=True).returncode==0
    if not all(checks.values()):raise RuntimeError(str({k:v for k,v in checks.items() if not v})+' '+str(hits))
    result=dict(passed=True,checks=checks,paired_cells=pins,resources=resources,
        allocated_worker_seconds=sum(x['allocated_worker_seconds'] for x in resources),
        counted_model_forwards=sum(x['engine_executions']+x['startup_forwards'] for x in resources),
        invalid_clean_timings=invalid_timing,all_timing_qualified=not invalid_timing,
        protocol_sha256=sha(root/'e2e_protocol.json'),verified_unix=time.time(),method_promoted=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    inventory=[dict(path=str(p.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size) for p in public]
    (out/'publication_files.json').write_text(json.dumps(dict(files=inventory,private_excluded=True),indent=2)+'\n')
    (root/'final_complete.json').write_text(json.dumps(dict(passed=True,scope='25questions x2seeds x9arms',
        protocol_sha256=sha(root/'e2e_protocol.json'),verification_sha256=sha(out/'verification.json'),
        official_scores_sha256=sha(root/'scoring001/official_scores.json'),summary_sha256=sha(root/'analysis001/summary.json'),
        method_promoted=False,all_timing_qualified=not invalid_timing),indent=2)+'\n')
    print(json.dumps(dict(passed=True,cells=cells,checks=len(checks),invalid_clean_timings=len(invalid_timing))))


if __name__=='__main__':
    main()
