"""Final independent input, source, accounting and privacy checks; no model calls."""
from pathlib import Path
import hashlib,importlib.util,json,math,re,shutil,subprocess
R=Path(__file__).resolve().parent;S=R.parent;REPO=S.parents[1];E=S/'experiment/attempt006';P=S/'profiling/attempt002'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    assert load(E/'development_complete.json')['passed'] and not load(E/'development_complete.json')['any_rule_advances']
    sources={}
    for p,h in load(E/'protocol.json')['sources'].items():assert sha(REPO/p)==h,p;sources[p]=h
    pauses=load(S/'initial_audit.json')['pause_hold_marker_hashes']
    for p,h in pauses.items():assert sha(REPO/p)==h,p
    meta={x['id']:x for x in load(P/'roster_identity.json')['identities']};records=[];inputs={}
    leaves=[(P/'runs','**/*.json'),(S/'experiment/attempt003/runs/references','*.json'),(S/'experiment/attempt004/runs/references','*.json'),(S/'experiment/attempt005/runs/early','*.json'),(S/'experiment/attempt005/runs/treatments','**/*.json'),(E/'runs/treatments','**/*.json')]
    for base,pattern in leaves:
        for p in sorted(base.glob(pattern)):
            v=load(p)
            if not all(k in v for k in ['N','C','T','model_forwards','output_sha256']):continue
            assert v['model_forwards']==math.ceil(meta[v['id']]['prompt_token_count']/16384)+v['N']+v['C'];assert math.ceil(v['T']/256)==v['C'];assert v['startup_model_calls_in_request']==v['extra_inside_execute_Python_forwards']==0
            if v['audit']:
                assert sum(not x['commit'] for x in v['native_trace'])==v['N'];assert sum(x['commit'] for x in v['native_trace'])==v['C'];assert v['uncommitted_tail']==1
            inputs[str(p.relative_to(S))]=sha(p);records.append(v)
    assert len(records)==522
    # One completed attempt003 request was printed and saved privately, but its
    # original post-request comparison failed before its public receipt saved.
    lost=dict(id='aime26/5',seed=42,N=141,C=11,T=2666,model_forwards=153,evidence_sha256=sha(S/'failure_review001/verification.json'),not_in_scientific_aggregate=True)
    resources=load(S/'report001/resources.json');init=sum(x.get('initialization_engine_forwards',6 if x.get('failed_attempt') else 0) for x in resources['explicit_forward_receipts']);startup=sum(x['startup_model_calls_outside_execute'] for x in resources['explicit_forward_receipts'])
    assert sum(v['model_forwards'] for v in records)+lost['model_forwards']+init+startup==resources['total_explicit_model_forwards']
    public=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(S.relative_to(REPO))],cwd=REPO,text=True).splitlines();public=sorted(set(public));assert not any('/private/' in p or '/cache/' in p or '/runtime/' in p or p.endswith(('.pt','.npy','.safetensors')) for p in public)
    violations=[];private_values=[v for v in load(S/'private/runtime_paths.json').values() if isinstance(v,str) and v.startswith('/')]
    for name in public:
        p=REPO/name
        if p.suffix not in ['.json','.md','.py','.xml']:continue
        text=p.read_text()
        if any(value in text for value in private_values):violations.append(name)
        if p.suffix=='.json' and re.search(r'"(?:prompt|prompt_tokens|token_ids|expected|gold|completion|raw_completion)"\s*:',text):violations.append(name)
    assert not violations,violations
    save(R/'verification.json',dict(passed=True,protocol_sha256=sha(E/'protocol.json'),source_checkpoint=load(E/'binding.json')['source_checkpoint'],pinned_sources=len(sources),source_drift=[],paused_markers_unchanged=len(pauses),direct_complete_request_receipts=len(records),additional_completed_failed_comparison_request=lost,total_completed_native_requests=len(records)+1,total_explicit_model_forwards=resources['total_explicit_model_forwards'],initialization_engine_forwards=init,startup_model_calls_outside_execute=startup,zero_extra_in_request_forwards=True,public_payload_scan_passed=True,private_payloads_ignored=True,input_hashes=inputs,no_new_model_calls=True))
    # Re-read actual weights/runtime/scorer bytes after GPU work has ended.
    for name in ['protocol.json','model_fingerprints.json']:shutil.copyfile(E/name,R/name)
    spec=importlib.util.spec_from_file_location('final_environment_verifier',E/'verify_environment.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.R=R;module.S=S;module.main()
    save(R/'final_complete.json',dict(passed=True,verification_sha256=sha(R/'verification.json'),model_runtime_verification_sha256=sha(R/'model_runtime_verification001.json'),source_sha256=sha(Path(__file__))))
if __name__=='__main__':main()
