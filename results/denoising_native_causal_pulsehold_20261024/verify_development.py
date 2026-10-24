"""Close out only the narrowed pulse/hold development leaf."""
import argparse,hashlib,json,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();r=args.root.resolve();out=r/'closeout001';repo=r.parents[2]
    protocol=read(r/'protocol.json');marker=read(r/'execution001/final_complete.json');analysis=read(r/'analysis001/analysis_complete.json');summary=read(r/'analysis001/summary.json');exit=read(r/'exit_receipt.json')
    checks=dict(source_pins_match=all(sha(Path(f))==h for f,h in protocol['sources'].items()),
        private_manifest_identity=sha(r/'private/tokenized_manifest.json')==protocol['private_manifest_sha256'],
        manifest_public_identity=sha(r/'manifest_identity.json')==protocol['public_manifest_sha256'],
        model16_files=read(r/'model_fingerprints.json')['passed'] and len(read(r/'model_fingerprints.json')['files'])==16,
        runtime_identity=sha(r/'runtime_fingerprints.json')==protocol['runtime_fingerprints_sha256'],
        execution_complete=marker['passed'] and marker['protocol_sha256']==sha(r/'protocol.json') and exit['returncode']==0,
        bounded_allocation=exit['allocated_worker_seconds']+exit['previous_allocated_seconds']<=7200,
        bounded_forwards=exit['forwards']+exit['previous_forwards']<=10000,
        analysis_identity=analysis['passed'] and analysis['summary_sha256']==sha(r/'analysis001/summary.json') and analysis['protocol_sha256']==sha(r/'protocol.json'),
        analysis_inputs=all(sha(r/f)==h for f,h in read(r/'analysis001/input_hashes.json').items()),
        four_development_anchors=summary['anchor_count']==4,
        all_no_promotion=summary['method_promoted'] is False and summary['official_task_outcomes'] is None)
    xml=ET.parse(r/'tests/cpu004.xml').getroot().find('testsuite').attrib
    checks['cpu24_pass']=xml['tests']=='24' and xml['errors']==xml['failures']=='0'
    qualification=[];branches=0;first_call_checks=0;acceptance={'threshold_ties':0,'threshold_near':0,'cpu_threshold_discrepancies':0,'native_winner_mismatches':0,'native_winner_ties':0}
    for item in sorted((r/'execution001').glob('item*')):
        lm=read(item/'reference_landmarks.json')
        expected=sorted(set(x for x in [max(2,lm['L']//2),max(2,lm['L']-2)] if x<lm['L']))
        if lm['anchors']!=expected:raise RuntimeError('anchor algorithm drift')
        for folder in sorted(item.glob('anchor*')):
            controls=read(folder/'controls.json');snap=read(folder/'snapshot.json');bank=read(folder/'candidate_bank.json');choices=read(folder/'selections.json');receipt=read(folder/'receipt.json')
            if not snap['passed'] or snap['metadata_mismatches'] or snap['storage_mismatches']:raise RuntimeError('snapshot metadata/bytes')
            keys=['fingerprint_sha256','remaining_denoise','rng_cpu','rng_cuda','forwards','steps','committed']
            for name in ['native_repeat','new_measurement_disabled','post_native']:
                if any(controls['native'][k]!=controls[name][k] for k in keys):raise RuntimeError('native replay or measurement mismatch')
            for name in ['allkept_repeat','post_allkept']:
                if any(controls['allkept'][k]!=controls[name][k] for k in keys):raise RuntimeError('allkept replay mismatch')
            if not controls['gqa_qualification']['passed'] or max(controls['gqa_qualification']['worst_relative_output_error'].values())>.02:raise RuntimeError('GQA gate')
            if controls['full_qualification']:
                q=controls['full_qualification']
                if not q['passed'] or any(x!=q['unrestored_reference'] for x in q['branches']):raise RuntimeError('full-request replay')
            valid={x['support_hash'] for x in bank['candidates'] if x['feasible']}
            if any(x is not None and x not in valid for x in choices['selections'].values()):raise RuntimeError('infeasible score selection')
            if len(valid)<4 and choices['finalist_hashes']:raise RuntimeError('nonidentifying selection')
            if any(x['physical_eligible']-x['physical_kept']!=1 for x in bank['candidates']):raise RuntimeError('short equal-work budget')
            selection_time=(folder/'selections.json').stat().st_mtime_ns
            aa=controls['allkept']
            same_first_by_support={}
            for bfile in sorted(folder.glob('branch*.json')):
                branches+=1
                if bfile.stat().st_mtime_ns<selection_time:raise RuntimeError('outcome firewall timestamp')
                branch=read(bfile);ex=branch['execution'];trace=ex['trace'];n=ex['remaining_denoise']
                if len(trace)!=n or ex['steps']!=ex['forwards']:raise RuntimeError('missing events/hidden forward')
                if n==aa['remaining_denoise'] and any(ex[k]!=aa[k] for k in ['rng_cpu','rng_cuda']):raise RuntimeError('equal-call RNG mismatch')
                if branch['schedule']=='AS':
                    if trace[0]['state_hashes']!=aa['trace'][0]['state_hashes']:raise RuntimeError('AS first A differs from AA')
                    first_call_checks+=1
                else:
                    key=branch['support_hash'];state=trace[0]['state_hashes']
                    if key in same_first_by_support and state!=same_first_by_support[key]:raise RuntimeError('SA/SS first S mismatch')
                    same_first_by_support[key]=state
                    if state['history']!=bank['first_pulse_decisions'][key]['history_digest']:raise RuntimeError('score pulse history mismatch')
                    first_call_checks+=1
                if not ex['committed'] or not trace[-1]['converged']:raise RuntimeError('native first canvas stop missing')
                if len(ex['physical'])!=n:raise RuntimeError('wrong target call count')
                if ex['global_kept_tiles']+ex['global_skipped_tiles']!=ex['global_eligible_tiles'] or ex['local_skipped_tiles']!=0:raise RuntimeError('work accounting')
                if any(x['current_canvas_skipped'] or x['local_skipped'] for x in ex['physical']):raise RuntimeError('scope')
                schedule={'SA':['S','A'],'AS':['A','S'],'SS':['S','S']}[branch['schedule']]
                for i,call in enumerate(ex['physical']):
                    expected=schedule[i] if i<2 else 'A'
                    if call['action']!=expected or call['skipped']!=(1 if expected=='S' else 0):raise RuntimeError('pulse/hold exposure')
            for key in ['native','native_repeat','allkept','allkept_repeat','post_native','post_allkept']:
                for call in controls[key]['trace']:
                    wm=call['winner_measurement'];am=call['acceptance_measurement']
                    acceptance['native_winner_mismatches']+=wm['native_winner_mismatches'];acceptance['native_winner_ties']+=wm['tie_positions']
                    acceptance['threshold_ties']+=am['threshold_tie_positions'];acceptance['threshold_near']+=am['threshold_near_positions']
                    acceptance['cpu_threshold_discrepancies']+=am['cpu_threshold_discrepancies_given_device_entropy']
            qualification.append(dict(id=receipt['id'],anchor=receipt['anchor'],storages=snap['storages'],tensors=snap['tensors'],alias_groups=len(snap['alias_groups']),native_and_allkept_exact=True,measurement_parity=True,full_request_qualified=controls['full_qualification'] is not None,GQA=controls['gqa_qualification']))
    checks['all_native_winners_exact']=acceptance['native_winner_mismatches']==0
    checks['all_four_anchor_certificates']=len(qualification)==4
    checks['HE_two_full_request_certificates']=sum(x['full_request_qualified'] for x in qualification)==2
    checks['factorial_count_matches_analysis']=branches==summary['branch_count']
    checks['factorial_first_calls_and_aligned_RNG']=first_call_checks==branches
    checks['nine_original_long_items_stay_blocked']=len(protocol['blocked_ids'])==9
    checks['final_gpu_idle']=read(out/'gpu_status.json')['idle']
    if not all(checks.values()):raise RuntimeError(str(checks))
    # No prompt/gold/token arrays or generation text may enter publication JSON.
    forbidden={'prompt','gold','answer','completion','token_ids','prompt_token_ids','generated_text','credentials','api_key','password'}
    public=[f for f in r.rglob('*') if f.is_file() and not any(x in {'private','cache','runtime','__pycache__'} for x in f.relative_to(r).parts) and f.suffix in {'.json','.md','.py','.xml','.pdf','.png'} and f.name not in {'cpu001.xml','cpu002.xml'}]
    hits=[]
    def inspect(x,f):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in forbidden:hits.append([str(f.relative_to(repo)),k])
                inspect(v,f)
        elif isinstance(x,list):
            for v in x:inspect(v,f)
    for f in public:
        if f.suffix=='.json':inspect(read(f),f)
    if hits:raise RuntimeError('publication privacy fields '+str(hits))
    write(out/'qualification_summary.json',dict(anchors=qualification,acceptance_measurement='reconstructed only, not compiled acceptance-buffer capture',measurement_totals_over_repeated_controls=acceptance))
    write(out/'privacy_audit.json',dict(passed=True,forbidden_field_hits=hits,private_manifest_ignored=subprocess.run(['git','check-ignore',str(r/'private/tokenized_manifest.json')],cwd=repo,capture_output=True).returncode==0))
    write(out/'verification.json',dict(passed=True,checks=checks,protocol_sha256=sha(r/'protocol.json'),source_checkpoint=read(r/'launch_receipt.json')['source_checkpoint'],verified_unix=time.time(),method_promoted=False,confirmation_complete=False,official_task_scoring=False))
    write(out/'artifact_inventory.json',dict(files=[dict(path=str(f.relative_to(repo)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(public)],scope='public diagnostic artifacts only; inventory excludes its own newly-created closeout files'))
    write(out/'development_leaf_complete.json',dict(passed=True,scope='prospectively narrowed four-anchor short development diagnostic only',protocol_sha256=sha(r/'protocol.json'),verification_sha256=sha(out/'verification.json'),analysis_sha256=sha(r/'analysis001/analysis_complete.json'),method_promoted=False,full_grid_complete=False))
    print(json.dumps(dict(passed=True,checks=len(checks),anchors=len(qualification),factorial_branches=branches,measurement_totals=acceptance)))
if __name__=='__main__':main()
