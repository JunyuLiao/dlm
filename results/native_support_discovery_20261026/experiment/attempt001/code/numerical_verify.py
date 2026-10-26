"""Independent CPU reader of saved native differences and operands; no model calls."""
from pathlib import Path
import itertools,json,math,sys
import torch
from pulse_runtime import ROOT,save,sha,fingerprint
from op_bank import matched,UNITS
def load(p):return json.loads(p.read_text())
def close(a,b):return math.isclose(float(a),float(b),rel_tol=2e-5,abs_tol=2e-7)
def verify_bank(bank,private,operand=None):
    dense=private['dense'].float();rows=[];pairs=[]
    for i,(rec,priv) in enumerate(zip(bank['candidates'],private['candidates'])):
        actual=priv['sparse'].float()-dense;assert torch.equal(actual,priv['actual_bias'])
        err=float((actual.norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-8)).max());assert close(err,rec['current_error']);assert close(actual.norm(),rec['bias_Frobenius'])
        assert close(priv['mass'].max(),rec['deleted_mass']);assert fingerprint(priv['sparse'])==rec['sparse_sha256']
        assert close((priv['mass']/private['prefix_attention_mass'].clamp_min(1e-12)).max(),rec['prefix_only_deleted_mass_max'])
        projection=(actual*private['readout_probe']).sum(-1);position=torch.cat([projection[0]+projection[2],projection[1]+projection[3]])
        base=torch.cat([private['base_margin'][0],private['base_margin'][1]]);protect=torch.cat([private['protected'][0],private['protected'][1]]);ew=torch.cat([private['entropy_weight'][0],private['entropy_weight'][1]])
        assert int(((base+position<0)&protect).sum())==rec['projected_stable_crossings'];assert close((position*ew).sum(),rec['projected_entropy_weighted_gain'])
        assert all(close(a,b) for a,b in zip(position.tolist(),rec['projected_margin_change_by_position']))
        rows.append(dict(candidate=i,actual_bias_sha256=fingerprint(actual),actual_bias_numeric_reproduced=True,projected_score_reproduced=True,roundoff_or_tiny_bias=rec['roundoff_or_tiny_bias'],actual_vs_FP32_cosine=rec['actual_vs_FP32_cosine']))
    for i,j in itertools.combinations(range(len(bank['candidates'])),2):
        a=bank['candidates'][i];b=bank['candidates'][j]
        if not(a['feasible'] and b['feasible'] and matched(a,b)):continue
        x=private['candidates'][i]['actual_bias'];y=private['candidates'][j]['actual_bias'];sep=float((x-y).norm())/max(float(x.norm()),float(y.norm()),1e-12)
        pairs.append(dict(i=i,j=j,relative_bias_separation=sep,meaningful=a['bias_direction_available'] and b['bias_direction_available'] and sep>=.25,selected_both=i in bank['screen'] and j in bank['screen']))
    math_checks=[]
    if operand:
        # Four fixed rows/unit, full key/value length, independent CPU softmax.
        for u,(h,qb) in enumerate(UNITS):
            ids=torch.tensor([0,31,63,127]);q=operand['q'][qb*128+ids,h].float();k=operand['k'][:,h//8].float();v=operand['v'][:,h//8].float();p=(q@k.T*operand['scale']).softmax(-1);A=p@v
            residual=float((A-private['FP32_dense'][u,ids]).norm()/private['FP32_dense'][u,ids].norm().clamp_min(1e-8));assert residual<.002
            maxres=0.
            for i,rec in enumerate(bank['candidates']):
                tiles=rec['units'][u]['tiles'];deleted=(torch.tensor(tiles)[:,None]*64+torch.arange(64)).flatten();mass=p[:,deleted].sum(-1);bias=(mass[:,None]*A-p[:,deleted]@v[deleted])/(1-mass)[:,None]
                ref=private['candidates'][i]['FP32_bias'][u,ids];res=float((bias-ref).norm())/max(float(ref.norm()),1e-6*float(A.norm()),1e-12);assert res<.02;maxres=max(maxres,res)
            math_checks.append(dict(unit=u,rows=ids.tolist(),full_key_length=len(k),dense_relative_residual=residual,maximum_FP32_bias_relative_residual=maxres))
    return dict(passed=True,candidates=rows,error_mass_matched_pairs=pairs,independent_full_key_math=math_checks)
def main():
    torch.set_num_threads(1);results=[];inputs={}
    for path in sorted((ROOT/'banks/references').rglob('*.json')):
        site=load(path);inputs[str(path.relative_to(ROOT))]=sha(path);operand=None
        if site.get('operands'):
            o=site['operands'];p=ROOT/o['relative'];assert sha(p)==o['sha256'];inputs[o['relative']]=o['sha256'];operand=torch.load(p,map_location='cpu',weights_only=False)
        for bank in site['banks']:
            if 'alias_of' in bank or not bank['candidates']:continue
            p=ROOT/bank['private_relative'];assert sha(p)==bank['private_sha256'];inputs[bank['private_relative']]=bank['private_sha256'];priv=torch.load(p,map_location='cpu',weights_only=False)
            results.append(dict(site=str(path.relative_to(ROOT)),drop_count=bank['drop_count'],**verify_bank(bank,priv,operand)))
    save(ROOT/'numerical_qualification.json',dict(passed=all(r['passed'] for r in results),banks=len(results),results=results,input_hashes=inputs,model_forwards=0))
if __name__=='__main__':main()
