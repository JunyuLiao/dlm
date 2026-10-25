"""Sanitized publication inventory and final completion marker."""
import hashlib
import json
from pathlib import Path
import re
import time

S=Path(__file__).resolve().parents[1];REPO=S.parents[1];OUT=S/'closeout001'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)


def public(p):return not any(x in ['private','cache','runtime','__pycache__','.pytest_cache'] for x in p.relative_to(S).parts) and p.suffix!='.pyc'


def main():
    start=time.time();sensitive={'prompt','prompt_tokens','token_ids','completion','completions','gold','expected','raw_generation','raw_text','credentials'}
    hits=[];files=[p for p in sorted(S.rglob('*')) if p.is_file() and public(p)]
    def check(x,file):
        if isinstance(x,dict):
            for k,v in x.items():
                if k in sensitive:hits.append(dict(file=file,key=k))
                check(v,file)
        elif isinstance(x,list):
            for v in x:check(v,file)
        elif isinstance(x,str) and re.search(r'/home/exouser/(?:dyh|\.cache|\.codex)|/data/',x):hits.append(dict(file=file,key='private_host_path'))
    count=0
    for p in files:
        if p.suffix=='.json':check(json.loads(p.read_text()),str(p.relative_to(S)));count+=1
    save(OUT/'privacy_audit.json',dict(passed=not hits,JSON_files_checked=count,sensitive_payload_hits=hits,
        private_cache_runtime_and_tensors_excluded=True,source='Exact sensitive data keys and private-host paths; source-code variable names are not dataset payloads.'))
    assert not hits,'publication data-key audit requires review'
    # Existing shared-file edits are preserved; publish only our delta identity.
    changes=json.loads((S/'private/shared_own_changes.json').read_text());shared=[]
    for name,x in changes.items():
        current=(REPO/name).read_text();text=x['text']
        base=current[len(text):] if x['mode']=='prepend' else current[:-len(text)]
        if name=='HANDOFF.md':assert current.startswith(text) and base==(S/'private/HANDOFF.md').read_text()
        if name=='docs/DECISIONS.md':assert current.endswith(text) and base==(S/'private/DECISIONS.md').read_text()
        shared.append(dict(file=name,ownership=x['mode']+' this task text only',task_delta_sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert sha(REPO/'AGENTS.md')==sha(S/'private/AGENTS.md')
    for root in [S/'track_r/attempt001',S/'track_h/attempt001']:
        protocol=json.loads((root/'protocol.json').read_text());assert all(sha(REPO/p)==v for p,v in protocol['sources'].items())
        assert json.loads((root/'final_complete.json').read_text())['passed']
    inventory=[]
    for p in sorted(S.rglob('*')):
        if p.is_file() and public(p) and p not in [OUT/'publication_files.json',S/'final_complete.json']:
            inventory.append(dict(file=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=sha(p)))
    save(OUT/'publication_files.json',dict(files=inventory,shared_file_task_deltas=shared,AGENTS_not_task_owned=True,
        excluded='All private/cache/runtime/pycache data, and self/final marker to avoid recursive hashes.',exact_task_owned_changed_files_from_starting_HEAD=True))
    verification=json.loads((OUT/'verification.json').read_text());resources=json.loads((OUT/'resources.json').read_text())
    save(S/'final_complete.json',dict(passed=True,Track_R_complete=True,Track_H_gated_leaf_complete=True,H_confirmation_launched=False,
        protocols=verification['sources'],summary_sha256=sha(S/'summary.md'),verification_sha256=sha(OUT/'verification.json'),
        publication_inventory_sha256=sha(OUT/'publication_files.json'),privacy_sha256=sha(OUT/'privacy_audit.json'),
        resources={k:resources[k] for k in ['complete_native_requests','explicit_model_forwards','allocated_GPU_worker_hours']},
        model_forwards_in_publication=0,CPU_publication_seconds=time.time()-start,accuracy_noninferiority_established=False,clean_speedup_established=False,
        Stage_BC_launched=False,production_kernel_changes=False,pushed=False))
    print(json.dumps(dict(public_files=len(inventory),JSON_checked=count,privacy_passed=True,new_model_forwards=0)))


if __name__=='__main__':main()
