"""Descriptive readout forecast audit; future pair alignment is evaluation-only."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent;E=R.parent/'experiment/attempt006'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
def main():
    assert load(E/'analysis001/final_complete.json')['passed'];plan=load(E/'branch_manifest.json');inputs={str(p.relative_to(R.parent)):sha(p) for p in [E/'branch_manifest.json',E/'analysis001/final_complete.json']};out=[];bankcache={}
    for cell in plan['cells']:
        label=cell['reference_label'];private=E/'private/generations'/Path(label+'.pt');a=torch.load(private,map_location='cpu',weights_only=False);inputs[str(private.relative_to(R.parent))]=sha(private)
        cid=Path(label).name
        for j,b in enumerate(cell['branches']):
            if b['layer']!=29:continue
            rp=E/f'runs/treatments/{cid}/s{j:03d}.json';s=load(rp);inputs[str(rp.relative_to(R.parent))]=sha(rp);sitepath=E/b['bank_relative']
            if str(sitepath) not in bankcache:bankcache[str(sitepath)]=load(sitepath);inputs[str(sitepath.relative_to(R.parent))]=sha(sitepath)
            bank=next(v for v in bankcache[str(sitepath)]['banks'] if v.get('drop_count')==b['drop_count'] and 'alias_of' not in v);candidate=bank['candidates'][b['candidate_index']]
            previous=a[(b['canvas'],b['step']-1)];current=a[(b['canvas'],b['step'])];row=next(r for r in s['native_trace'] if not r['commit'] and (r['canvas'],r['native_step'])==(b['canvas'],b['step']))
            aligned=(previous['winner']==current['winner'])&(previous['competitor']==current['competitor']);protected=previous['stable_positions'];pred=np.asarray(candidate['projected_margin_change_by_position'],dtype=float);actual=np.asarray(row['reference_pair_margin'],dtype=float)-current['margin'].double().numpy();mask=aligned.numpy();eligible=mask&(np.abs(pred)>=1e-6)&(np.abs(actual)>=1e-6)
            def forecast(m):
                x=pred[m];y=actual[m]
                if not len(x):return dict(positions=0,cosine=None,relative_residual=None)
                nx=float(np.linalg.norm(x));ny=float(np.linalg.norm(y));return dict(positions=len(x),predicted_L2=nx,actual_L2=ny,cosine=float(np.dot(x,y)/(nx*ny)) if min(nx,ny)>=1e-6 else None,relative_residual=float(np.linalg.norm(x-y))/ny if ny>=1e-6 else None)
            out.append(dict(id=cell['id'],seed=cell['seed'],identity=b['identity'],benchmark=cell['benchmark'],anchor=b['stratum'],fractions=b['fractions'],families=b['families'],previous_current_pair_alignment_fraction=float(aligned.float().mean()),forecast_on_aligned_pairs=forecast(mask),forecast_on_aligned_protected_pairs=forecast(mask&protected.numpy()),sign_positions=int(eligible.sum()),sign_agreement=float(np.mean(np.sign(pred[eligible])==np.sign(actual[eligible]))) if eligible.any() else None,actual_entropy_delta=row['entropy_mean']-float(current['entropy'].mean()),worst_fixed_A_winner_erosion=row['worst_margin_erosion'],fixed_A_winner_crossings=row['reference_winner_crossings'],actual_native_winner_changes=row['actual_native_winner_changes'],actual_postpulse_state_is_outcome_only=True))
    byq=defaultdict(list)
    for r in out:
        c=r['forecast_on_aligned_pairs']['cosine']
        if c is not None:byq[r['id']].append(c)
    summary=dict(completed=True,branches=len(out),questions=len({r['id'] for r in out}),question_mean_aligned_pair_cosines={q:float(np.mean(v)) for q,v in byq.items()},mean_previous_current_pair_alignment=float(np.mean([r['previous_current_pair_alignment_fraction'] for r in out])),future_pair_filter_is_evaluation_only=True,not_used_for_selection_or_advancement=True,limitation='Unaligned previous-pair logits were not saved. This is a conditional descriptive audit of the fixed linear component, not full Jacobian validation or task correctness.')
    save(R/'branches.json',out);save(R/'summary.json',summary);save(R/'input_hashes.json',inputs);save(R/'final_complete.json',dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(E/'protocol.json'),summary_sha256=sha(R/'summary.json')))
if __name__=='__main__':main()
