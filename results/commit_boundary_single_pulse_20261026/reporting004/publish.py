"""Freeze the public inventory/completion after independent verification."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,subprocess
R=Path(__file__).resolve().parents[1];REPO=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((R/'verification002/summary.json').read_text())['passed']
    assert json.loads((R/'closeout001/preservation.json').read_text())['passed']
    assert json.loads((R/'closeout001/operator_qualification.json').read_text())['passed']
    assert json.loads((R/'panel/attempt004/final_complete.json').read_text())['passed']
    rel=str(R.relative_to(REPO))
    paths=sorted(set(subprocess.check_output(['git','ls-files','-co','--exclude-standard','--',rel],cwd=REPO,text=True).splitlines()))
    future=[str((R/n).relative_to(REPO)) for n in ['closeout001/publication_files.json','closeout001/publication_receipt.json','final_complete.json']]
    assert all(not (REPO/p).exists() for p in future)
    patterns=[r'/home/exouser/(?:dyh|\.cache|\.codex|\.local|\.huggingface)',r'(?:sk-[a-zA-Z0-9]{20}|ghp_[a-zA-Z0-9]{20}|Bearer [a-zA-Z0-9]{20})']
    def data_check(v):
        if isinstance(v,dict):
            for k,x in v.items():
                assert not(k in ['prompt','prompt_tokens','token_ids','completion','gold','expected','raw_generation'] and x is not None),k
                assert k!='input_ids' or isinstance(x,dict),'raw input_ids'
                data_check(x)
        elif isinstance(v,list):
            for x in v:data_check(x)
    inventory=[]
    for path in paths:
        p=REPO/path
        assert p.is_file() and not p.is_symlink(),path
        assert not any(x in p.relative_to(R).parts for x in ['private','cache','runtime','__pycache__']),path
        assert p.suffix not in ['.pt','.bin','.safetensors','.so','.pyc','.log'],path
        if p.suffix in ['.json','.md','.py','.xml','.jsonl']:
            text=p.read_text()
            assert not any(re.search(pattern,text) for pattern in patterns),path
            if p.suffix=='.json':data_check(json.loads(text))
        inventory.append(dict(path=path,bytes=p.stat().st_size,sha256=sha(p)))
    for path in ['HANDOFF.md','docs/DECISIONS.md','docs/RESULTS_LEDGER.md']:
        data=subprocess.check_output(['git','show',':'+path],cwd=REPO)
        inventory.append(dict(path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),scope='staged committed base plus task delta; original unrelated worktree edits excluded'))
    inventory.extend(dict(path=p,self_referential_metadata=True,integrity='Git tree and final marker bind these metadata files; no circular self hash') for p in future)
    def save(p,x):
        with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)
    save(R/'closeout001/publication_files.json',dict(files=sorted(inventory,key=lambda x:x['path']),study_public_files=len(paths)+len(future),shared_documents=3,all_files_changed_since_start_are_task_owned=True,source_start='748e22b328e167cd64c64a46afda9bec1ff06d32'))
    receipt=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),public_file_count=len(paths)+len(future),shared_document_count=3,public_data_bytes=sum(x['bytes'] for x in inventory if 'bytes' in x),
        no_raw_prompts_answers_token_ids_tensors_or_completions=True,no_private_host_paths_or_credentials_found=True,private_directories_ignored=True,original_dirty_documents_preserved=True,
        source_verification_sha256=sha(R/'verification002/summary.json'),inventory_sha256=sha(R/'closeout001/publication_files.json'),publication_reader_sha256=sha(Path(__file__)),model_forwards=0)
    save(R/'closeout001/publication_receipt.json',receipt)
    c=R/'panel/attempt004';b=R/'boundary/attempt001'
    save(R/'final_complete.json',dict(passed=True,status='complete_bounded_study',scientific_advancement=False,no_Stage_BC=True,no_push=True,no_campaign_resource_ceilings=True,
        start_commit='748e22b328e167cd64c64a46afda9bec1ff06d32',branch='denoising_trajectory_research',
        boundary_protocol_sha256=sha(b/'protocol.json'),panel_protocol_sha256=sha(c/'protocol.json'),panel_manifest_sha256=sha(c/'manifest_identity.json'),
        boundary_complete_sha256=sha(b/'final_complete.json'),panel_complete_sha256=sha(c/'final_complete.json'),CPU_addendum_complete_sha256=sha(R/'cpu_addendum001/final_complete.json'),
        verification_sha256=sha(R/'verification002/summary.json'),summary_sha256=sha(R/'summary.md'),method_decisions_sha256=sha(R/'closeout001/method_decisions.json'),resources_sha256=sha(R/'closeout001/resources.json'),
        publication_receipt_sha256=sha(R/'closeout001/publication_receipt.json'),inventory_sha256=sha(R/'closeout001/publication_files.json'),hardware_status_sha256=sha(R/'closeout001/hardware_final.json')))
    print(json.dumps(dict(passed=True,public_files=len(paths)+len(future),shared_documents=3,final_complete=True)))
if __name__=='__main__':main()
