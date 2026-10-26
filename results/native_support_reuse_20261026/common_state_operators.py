"""Held/fresh attention at the first support difference, on identical operands."""
from pathlib import Path
from functools import lru_cache
import json,hashlib,itertools,math
import torch
S=Path(__file__).resolve().parent;R=S/'attempt002';OUT=S/'common_state_operators001';INPUTS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
@lru_cache(maxsize=64)
def read(relative):
    p=R/relative;INPUTS[relative]=sha(p);return json.loads(p.read_text())
@lru_cache(maxsize=16)
def data(relative,expected):
    p=R/relative;assert sha(p)==expected;INPUTS[relative]=expected;return torch.load(p,map_location='cpu',weights_only=False)
def main():
    torch.set_num_threads(1);assert read('runs/treatments/final_complete.json')['passed'];manifest=read('branch_manifest.json');rows=[]
    for c in manifest['cells']:
        labels={(l['fraction'],l['selector'],l['lifecycle']):l['branch'] for l in c['labels']}
        for f,selector,horizon in itertools.product([.05,.1],['current_error','mass','support_sham','readout_margin','response_qpast','margin_weighted','signed_high','signed_low'],[2,4]):
            ids=[labels[f,selector,f'{l}{horizon}'] for l in ['held','refresh']];base=dict(id=c['id'],seed=c['seed'],benchmark=c['benchmark'],fraction=f,selector=selector,horizon=horizon)
            if not all(ids):rows.append(dict(base,available=False,reason='initial_bank_fallback'));continue
            h,r=[read('branch_records/'+i+'.json') for i in ids];he={e['offset']:e for e in h['events'] if e.get('intervention')};re={e['offset']:e for e in r['events'] if e.get('intervention')}
            common=sorted(set(he)&set(re));differences=[o for o in common if he[o]['actual_support_sha256']!=re[o]['actual_support_sha256']];offset=differences[0] if differences else 1 if 1 in common else None
            if offset is None:rows.append(dict(base,available=False,reason='no_second_deleting_call_in_both',early_stop_or_refresh_fallback=True));continue
            # At the first differing support, every previous physical operation is equal.
            assert all(he[o]['actual_support_sha256']==re[o]['actual_support_sha256'] for o in common if o<offset)
            def choice(record,event):return read('choices/'+record['label']+f'/s{event["step"]:02d}.json')
            hc=choice(h,he[offset]);rc=choice(r,re[offset]);logical=lambda g:{k:v for k,v in g.items() if k!='page_table_sha256'}
            assert logical(hc['geometry'])==logical(rc['geometry']),'First differing operator has different current native QKV'
            hv=data(hc['private_relative'],hc['private_sha256']);rv=data(rc['private_relative'],rc['private_sha256']);assert torch.equal(hv['dense'],rv['dense'])
            previous=offset-1
            if previous not in he:rows.append(dict(base,available=False,reason='previous_deleting_operator_not_available'));continue
            pc=choice(h,he[previous]);pv=data(pc['private_relative'],pc['private_sha256']);past=pv['sparse'].double();held=hv['sparse'].double();fresh=rv['sparse'].double();hb=hv['actual_bias'].double();rb=rv['actual_bias'].double()
            scale=max(float(past.norm()),1e-12);sep=float((hb-rb).norm())/max(float(hb.norm()),float(rb.norm()),1e-12)
            e1=hc['measurement']['current_error'];e2=rc['measurement']['current_error'];m1=hc['measurement']['deleted_mass'];m2=rc['measurement']['deleted_mass'];match=abs(e1-e2)/max(e1,e2,1e-6)<=.05 and abs(m1-m2)<=.01
            rows.append(dict(base,available=True,compared_offset=offset,first_differing_offset=differences[0] if differences else None,support_alias=not differences,current_QKV_byte_equal=True,allkept_current_output_equal=True,held_relative_motion=float((held-past).norm())/scale,refresh_relative_motion=float((fresh-past).norm())/scale,relative_bias_separation=sep,error_mass_matched=match,meaningful=match and sep>=.25 and min(hc['measurement']['bias_relative_Frobenius'],rc['measurement']['bias_relative_Frobenius'])>=1e-6,held_current_error=e1,refresh_current_error=e2,held_deleted_mass=m1,refresh_deleted_mass=m2,held_remaining=h['local_endpoint']['remaining_calls'],refresh_remaining=r['local_endpoint']['remaining_calls'],held_correct=h['correct'],refresh_correct=r['correct'],interpretation='Common-current-operand local comparison; nonlinear norm changes are not a causal mediation decomposition'))
    OUT.mkdir(exist_ok=False)
    for n,v in [('comparisons.json',rows),('input_hashes.json',INPUTS),('final_complete.json',dict(passed=True,labels=len(rows),common_operand_comparisons=sum(r['available'] for r in rows),nonalias_comparisons=sum(r.get('available',False) and not r['support_alias'] for r in rows),matched_meaningful=sum(r.get('meaningful',False) for r in rows),native_forwards_added=0,mechanistic_mediation_certified=False))]:
        with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    print(json.dumps(dict(passed=True,comparisons=sum(r['available'] for r in rows))))
if __name__=='__main__':main()
