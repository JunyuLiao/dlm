"""Verify bounded native diagnostics and inventory sanitized artifacts."""
from pathlib import Path
import hashlib,json,subprocess,time,xml.etree.ElementTree as ET
REPO=Path('/home/exouser/ljy/dlm')
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
BASE=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def main():
    p=read(ROOT/'protocol.json');a=read(ROOT/'analysis001/summary.json');am=read(ROOT/'analysis001/analysis_complete.json')
    restore=BASE/'attempt004/qualification';rr=read(restore/'receipt.json');rm=read(restore/'final_complete.json')
    checks=dict(sources_match=all(sha(Path(f))==h for f,h in p['sources'].items()),private_manifest_match=sha(ROOT/'private/tokenized_manifest.json')==p['private_manifest_sha256'],restore_marker_identity=rm['passed'] and rm['receipt_sha256']==sha(restore/'receipt.json') and rm['protocol_sha256']==sha(BASE/'attempt004/protocol.json'),restore_all_native_fields_equal=all(all(x[k]==rr['branches'][0][k] for k in rr['branches'][0] if k!='label') for x in rr['branches']),restore_storage235=all(read(restore/(label+'_restore.json'))['passed'] and read(restore/(label+'_restore.json'))['storages']==235 for label in ['dense_restored','dense_restored_repeat']),restore_no_hidden_forwards=rr['zero_hidden_forwards'] and rr['request_cuda_captures']==0,analysis_identity=am['passed'] and am['protocol_sha256']==sha(ROOT/'protocol.json') and am['summary_sha256']==sha(ROOT/'analysis001/summary.json') and am['script_sha256']==sha(ROOT/'analysis001/analyze.py'),analysis_inputs_unchanged=all(sha(ROOT/f)==h for f,h in read(ROOT/'analysis001/input_hashes.json').items()))
    allrows=[];groups=[];item_count=0
    for group in ['smoke','short','long']:
        marker=read(ROOT/group/'final_complete.json');groups.append(marker)
        checks[group+'_complete']=marker['passed'] and marker['protocol_sha256']==sha(ROOT/'protocol.json') and read(ROOT/(group+'_exit.json'))['returncode']==0
        for folder in sorted((ROOT/group).glob('item*')):
            item_count+=1;control=read(folder/'controls.json')
            checks[str(folder.relative_to(ROOT))+'_native_restore']=control['restore_native_equal'] and control['dense']['steps']==control['dense']['forwards']==control['repeated']['forwards']==2
            for f in folder.glob('candidate*.json'):
                x=read(f);allrows.append(x);ex=x['execution'];deleted=x['prefix_blocks']-x['kept_prefix_blocks']
                if ex['steps']!=ex['forwards'] or ex['steps']!=2:raise RuntimeError('hidden or missing forward')
                if any(ex[k]!=control['dense'][k] for k in ['rng_cpu','rng_cuda']):raise RuntimeError('RNG order')
                if len(ex['physical'])!=2:raise RuntimeError('missing physical receipt')
                for row in ex['physical']:
                    if row['kept_tiles']+row['skipped_tiles']!=row['eligible_tiles'] or row['skipped_tiles']!=deleted or row['current_canvas_deleted_tiles'] or row['local_skipped_tiles']:raise RuntimeError('physical scope')
                if not deleted and any(v['logit_l2']!=0 for v in x['native_vs_allkept']):raise RuntimeError('allkept mismatch')
    checks['all23_candidate_forward_rng_physical_gates']=len(allrows)==23 and item_count==6
    checks['exact_candidate_counts']=len(allrows)==a['candidates']==23 and sum(x['kept_prefix_blocks']<x['prefix_blocks'] for x in allrows)==a['sparse_candidates']==18
    checks['no_request_captures']=all(x['cuda_captures']==0 for x in groups)
    checks['explicit3779_forwards']=sum(x['explicit_native_forwards'] for x in groups)==3779
    checks['jit3_recorded']=sum(x['jit_events'] for x in groups)==3
    checks['heldout_no_selection_advantage']=a['heldout_response_error_distinct_selections']==0
    checks['no_early_stop_changes']=a['zero_stop_changes']
    checks['nine_blocked_original_ids']=len(p['blocked_ids'])==9 and not set(p['blocked_ids'])&set(p['short_ids']+p['long_ids']+[p['smoke_item']])
    for name in ['cpu001.xml','cpu002.xml']:
        xml=ET.parse(ROOT/'tests'/name).getroot().find('testsuite').attrib
        checks[name+'_19_pass']=xml['tests']=='19' and xml['errors']==xml['failures']=='0'
    original=read(OUT/'initial_document_hashes.json')
    checks['agents_prior_changes_untouched']=sha(REPO/'AGENTS.md')==original['AGENTS.md']
    checks['legacy_state_untouched']=sha(REPO/'STATE.json')==original['STATE.json']
    gpu=read(OUT/'gpu_status.json');checks['gpu_idle']=gpu['idle']
    checks['git_diff_check']=subprocess.run(['git','diff','--check'],cwd=REPO,capture_output=True).returncode==0
    if not all(checks.values()):raise RuntimeError(str(checks))
    excluded={'private','cache','runtime','__pycache__'}
    public=[f for f in sorted(BASE.rglob('*')) if f.is_file() and not any(x in excluded for x in f.relative_to(BASE).parts) and f.parent!=OUT and f.suffix in {'.py','.json','.md','.xml','.png','.pdf'}]
    forbidden={'prompt','gold','answer','completion','token_ids','prompt_token_ids','generated_text','credentials','api_key','password'};hits=[]
    def inspect(x,f):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in forbidden or (k=='prompt_tokens' and not isinstance(v,int)):hits.append(dict(file=str(f.relative_to(REPO)),key=k))
                inspect(v,f)
        elif isinstance(x,list):
            for v in x:inspect(v,f)
    for f in public:
        if f.suffix=='.json':inspect(read(f),f)
    if hits:raise RuntimeError('privacy hits '+str(hits))
    docs=[REPO/x for x in ['HANDOFF.md','docs/DECISIONS.md','docs/RESULTS_LEDGER.md','docs/DENOISING_NATIVE_CAUSAL_20261024.md']]
    write(OUT/'privacy_audit.json',dict(passed=True,structured_public_json_forbidden_field_hits=hits,private_manifest_ignored=subprocess.run(['git','check-ignore',str(ROOT/'private/tokenized_manifest.json')],cwd=REPO,capture_output=True).returncode==0,scope='Public structured JSON field audit; code/report manually inspected. Inputs, logs, caches, tensor values and generated text excluded.'))
    reservations=[dict(path=str(f.relative_to(REPO)),**read(f)) for f in sorted(BASE.glob('attempt*/*exit*.json'))]
    total=sum(x['seconds'] for x in reservations)
    write(OUT/'verification.json',dict(passed=True,checks=checks,verified_unix=time.time(),branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),protocol_sha256=sha(ROOT/'protocol.json'),reservations=reservations,reserved_worker_seconds=total,resource_definition='Conservative launched-worker reservation including startup, warmup, failures and shutdown; not active GPU seconds or clean latency',whole_study_complete=False,bounded_native_diagnostics_complete=True,method_promoted=False))
    inventory=[dict(path=str(f.relative_to(REPO)),bytes=f.stat().st_size,sha256=sha(f)) for f in public+docs]
    write(OUT/'artifact_inventory.json',dict(files=inventory,exclusions='private/cache/runtime/__pycache__; closeout files inventoried separately'))
    with (OUT/'changed_files.txt').open('x') as f:f.write('\n'.join(str(x.relative_to(REPO)) for x in public+docs)+'\n')
    write(OUT/'diagnostic_stage_complete.json',dict(passed=True,scope='Native first-canvas restoration and bounded current-QKV support diagnostics only; whole acceleration plan partial',protocol_sha256=sha(ROOT/'protocol.json'),verification_sha256=sha(OUT/'verification.json'),summary_sha256=sha(ROOT/'summary.md'),analysis_marker_sha256=sha(ROOT/'analysis001/analysis_complete.json'),method_promoted=False))
    print(json.dumps(dict(passed=True,checks=len(checks),candidate_records=len(allrows),publication_files=len(public+docs),reserved_worker_seconds=total)))
if __name__=='__main__':main()
