"""Independent CPU isolation of attempt003's unready-history comparison failure."""
from pathlib import Path
import hashlib,json,sys
import torch
R=Path(__file__).resolve().parent;S=R.parent;P=S/'profiling/attempt002';E=S/'experiment/attempt003'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    torch.set_num_threads(1);inputs={};refpath=next(p for p in (P/'runs/profile/matched_allkept_l5_l17_l29').glob('audit*.json') if (lambda r:r['id']=='aime26/5' and r['seed']==42)(json.loads(p.read_text())));ref=json.loads(refpath.read_text());a=P/'private/generations'/Path(ref['label']+'.pt');b=E/'private/generations/references/9af43b20a706_seed42.pt';x=torch.load(a,weights_only=False);y=torch.load(b,weights_only=False);diff=[]
    for p in [refpath,a,b,E/'runs/references/failure.json',E/'launches/references/exit_receipt.json',E/'code/study_worker.py']:inputs[str(p.relative_to(S))]=sha(p)
    assert x.keys()==y.keys() and len(x)==141
    for call in x:
        for field in x[call]:
            if not torch.equal(x[call][field],y[call][field]):diff.append(dict(canvas=call[0],call=call[1],field=field,positions=int((x[call][field]!=y[call][field]).sum())))
    assert diff==[dict(canvas=0,call=1,field='stable_positions',positions=7)]
    public_first=next(r for r in ref['native_trace'] if not r['commit']);assert public_first['history_ready'] is False
    sys.path.insert(0,str(E/'code'));from numerical_verify import verify_bank
    numerical=[]
    for p in sorted((E/'banks').rglob('*.json')):
        site=json.loads(p.read_text());inputs[str(p.relative_to(S))]=sha(p);operand=None
        if site.get('operands'):
            op=E/site['operands']['relative'];assert sha(op)==site['operands']['sha256'];inputs[str(op.relative_to(S))]=sha(op);operand=torch.load(op,weights_only=False)
        for bank in site['banks']:
            if 'alias_of' in bank:continue
            path=E/bank['private_relative'];assert sha(path)==bank['private_sha256'];inputs[str(path.relative_to(S))]=sha(path)
            numerical.append(dict(site=str(p.relative_to(E)),drop_count=bank['drop_count'],**verify_bank(bank,torch.load(path,weights_only=False),operand)))
    save(R/'verification.json',dict(passed=True,reference_calls=141,all_winner_logit_margin_entropy_tensors_exact=True,differences=diff,unready_history_only=True,native_history_at_intervention_not_exempted=True,output_NCT_RNG_forward_checks_passed_before_trace_assert='Control-flow witness in retained traceback; failed implementation did not persist the full public comparison record. Corrected attempt004 does.',numerical_banks=numerical,input_hashes=inputs,model_forwards=0,source_sha256=sha(Path(__file__))))
    save(R/'final_complete.json',dict(passed=True,scope='CPU failure isolation and numerical checks; not a sparse trajectory result',verification_sha256=sha(R/'verification.json')))
if __name__=='__main__':main()
