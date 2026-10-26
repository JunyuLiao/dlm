"""Independent CPU actual S-A and full lexicographic initial-bank qualification."""
from pathlib import Path
import json,hashlib,math,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R/'code'))
from op_bank import matched
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b):return math.isclose(float(a),float(b),rel_tol=2e-5,abs_tol=2e-7)
def main():
    torch.set_num_threads(1);inputs={};rows=[];banks=[]
    for p in sorted((R/'banks/references').rglob('initial_d*.json')):
        inputs[str(p.relative_to(R))]=sha(p);b=json.loads(p.read_text());pp=R/b['private_relative'];assert sha(pp)==b['private_sha256'];inputs[b['private_relative']]=sha(pp);priv=torch.load(pp,map_location='cpu',weights_only=False);rs=b['candidates']
        if not rs:
            banks.append(dict(path=str(p.relative_to(R)),candidates=0,reason=b.get('reason'),selectors={}))
            assert not b['all_selections']
            continue
        for i,(r,v) in enumerate(zip(rs,priv['candidates'])):
            x=v['sparse'].float()-priv['dense'].float();assert torch.equal(x,v['actual_bias']);x=x.double();dense=priv['dense'].double();ref=v['FP32_bias'].double();floor=max(1e-12,float(dense.norm())*1e-6)
            err=float((x.norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-8)).max());res=float((x-ref).norm())/max(float(ref.norm()),floor)
            assert near(err,r['current_error']) and near(res,r['actual_vs_FP32_relative_residual'])
            assert near(float(v['mass'].max()),r['deleted_mass']) and r['feasible']==(err<=.1 and r['math_current_error']<=.1 and r['deleted_mass']<=.2)
            proj=(x*priv['readout_probe'].double()).sum(-1);pos=torch.cat([proj[0]+proj[2],proj[1]+proj[3]])
            base=torch.cat([priv['base_margin'][0],priv['base_margin'][1]]);stable=torch.cat([priv['protected'][0],priv['protected'][1]]);ew=torch.cat([priv['entropy_weight'][0],priv['entropy_weight'][1]]).double()
            assert int(((base+pos<0)&stable).sum())==r['projected_stable_crossings'];assert near(float((pos*ew).sum()),r['projected_entropy_weighted_gain'])
            rows.append(dict(bank=str(p.relative_to(R)),candidate=i,actual_error=err,relative_FP32_residual=res,cosine=r['actual_vs_FP32_cosine'],roundoff_or_tiny=r['roundoff_or_tiny_bias']))
        ids=[i for i,r in enumerate(rs) if r['feasible']]
        if ids:
            ce=min(ids,key=lambda i:(rs[i]['current_error'],rs[i]['support_sha256']));pool=[i for i in ids if matched(rs[i],rs[ce])];readout=min(pool,key=lambda i:(rs[i]['projected_stable_crossings'],-rs[i]['projected_entropy_weighted_gain'],rs[i]['current_error'],rs[i]['support_sha256']))
            assert ce==b['all_selections']['current_error'] and readout==b['all_selections']['readout_margin'];assert all(i in ids for i in b['all_selections'].values())
        banks.append(dict(path=str(p.relative_to(R)),candidates=len(rs),selectors=b['all_selections'],selection_uses_only_current_past=True))
    assert rows,'No initial candidate bank qualified'
    out=R/'numerical_qualification002';out.mkdir(exist_ok=False)
    for n,v in [('candidates.json',rows),('banks.json',banks),('input_hashes.json',inputs),('final_complete.json',dict(passed=True,banks=len(banks),candidates=len(rows),residual_quantiles=np.quantile([r['relative_FP32_residual'] for r in rows],[0,.25,.5,.75,1]).tolist(),residual_gt_half=sum(r['relative_FP32_residual']>.5 for r in rows),residual_gt_one=sum(r['relative_FP32_residual']>1 for r in rows),native_forwards=0,scope='Saved actual consumer difference and FP32 reference, not a full denoising response certificate'))]:
        with (out/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    print(json.dumps(dict(passed=True,banks=len(banks),candidates=len(rows))))
if __name__=='__main__':main()
