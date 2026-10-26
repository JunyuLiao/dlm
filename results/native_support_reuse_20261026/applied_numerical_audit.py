"""Independent applied-vector/current-support/SC audit; no model or CUDA calls."""
from pathlib import Path
from collections import Counter
import json,hashlib,math,sys
import numpy as np
import torch
S=Path(__file__).resolve().parent;R=S/'attempt002';sys.path.insert(0,str(R/'code'))
from reuse_selectors import active_schedule
from op_bank import support_hash
OUT=S/'applied_numerical_audit001';INPUTS={}
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while x:=f.read(4*1024*1024):h.update(x)
    return h.hexdigest()
def read(p):INPUTS[str(p.relative_to(S))]=digest(p);return json.loads(p.read_text())
def fp(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def near(a,b):return math.isclose(float(a),float(b),rel_tol=2e-5,abs_tol=2e-7)
def main():
    torch.set_num_threads(1);assert read(R/'runs/treatments/final_complete.json')['passed'];plan=read(R/'branch_manifest.json');branches={b['identity']:(c,b) for c in plan['cells'] for b in c['branches']};vector_rows=[];native=[];scope=[]
    refs={}
    for p in (R/'runs/references').glob('*seed*.json'):
        v=read(p)
        if not v['label'].endswith(('_off','_repeat')):refs[v['id'],v['seed']]=v
    for p in sorted((R/'branch_records').glob('*.json')):
        r=read(p);c,b=branches[r['branch_identity']];anchor=c['anchor'];events=[e for e in r['events'] if e.get('intervention')];rows=[x for x in r['consumer_physical'] if x['skipped']]
        assert len(events)==len(rows)==r['intervention_calls']
        for row in rows:
            offset=row['native_step']-anchor['after_completed_calls']-1
            assert row['layer']==29 and row['canvas']==anchor['canvas'] and active_schedule(b['lifecycle'],offset)
            assert row['skipped']==4*b['drop_count'] and row['LOCAL_skipped']==row['protected_skipped']==0
            event=next(e for e in events if e['step']==row['native_step']);assert support_hash(row['actual_support'])==event['actual_support_sha256']
        for e in events:
            path=R/'choices'/r['label']/f's{e["step"]:02d}.json';choice=read(path);private=R/choice['private_relative'];assert digest(private)==choice['private_sha256'];INPUTS[str(private.relative_to(S))]=choice['private_sha256'];v=torch.load(private,map_location='cpu',weights_only=False)
            actual=v['sparse'].float()-v['dense'].float();assert torch.equal(actual,v['actual_bias']);assert fp(v['sparse'])==e['actual_output_sha256']
            x=actual.double();dense=v['dense'].double();ref=v['FP32_bias'].double();floor=max(1e-12,1e-6*float(dense.norm()));res=float((x-ref).norm())/max(float(ref.norm()),floor);err=float((x.norm(dim=-1)/dense.norm(dim=-1).clamp_min(1e-8)).max())
            assert near(err,e['measurement']['current_error']) and near(res,e['measurement']['actual_vs_FP32_relative_residual'])
            assert near(float(v['mass'].max()),e['measurement']['deleted_mass'])
            vector_rows.append(dict(identity=b['identity'],id=r['id'],seed=r['seed'],lifecycle=b['lifecycle'],offset=e['offset'],current_error=err,deleted_mass=float(v['mass'].max()),relative_FP32_residual=res,cosine=e['measurement']['actual_vs_FP32_cosine'],later_guard_violation=e['offset']>0 and not e['measurement']['feasible'],bias_relative_Frobenius=e['measurement']['bias_relative_Frobenius']))
        base=refs[r['id'],r['seed']];first=events[0] if events else None
        state=None
        if first:
            t=str(first['step']);a=base['window_native_states'].get(t);z=r['window_native_states'].get(t)
            if a and z:
                state={k:dict(byte_changed=a[k]['byte_sha256']!=z[k]['byte_sha256'],numerically_changed=a[k].get('numerical_sha256',a[k]['byte_sha256'])!=z[k].get('numerical_sha256',z[k]['byte_sha256'])) for k in ['stochastic_canvas','native_winner','self_condition','history','confidence','converged']}
        term=r['local_endpoint']['terminal'];adaptive=term['history_ready'] and term['stable'] and term['confidence']
        native.append(dict(identity=b['identity'],id=r['id'],seed=r['seed'],lifecycle=b['lifecycle'],initial_pulse_state_changes=state,target_adaptive=adaptive,target_capped=term['cap_reached'],full_48call_events=sum(x.get('cap_reached',False) for x in r['native_trace'] if not x['commit']),complete_correct=r['correct'],output_capped=r['output_capped'],missing_final=r['no_final_response'],actual_acceptance_counts_qualified=False))
        scope.append(dict(identity=b['identity'],all_interventions_in_frozen_scope=True,scheduled_lifecycle=b['lifecycle'],actual_interventions=len(events),protected_and_LOCAL_deletions=0,native_forwards=r['model_forwards'],source_protocol_sha256=r['protocol_sha256']))
    OUT.mkdir(exist_ok=False);sc=[r['initial_pulse_state_changes']['self_condition'] for r in native if r['initial_pulse_state_changes']]
    summary=dict(passed=True,branches=len(native),applied_operators=len(vector_rows),FP32_residual_quantiles=np.quantile([r['relative_FP32_residual'] for r in vector_rows],[0,.25,.5,.75,1]).tolist(),residual_gt_half=sum(r['relative_FP32_residual']>.5 for r in vector_rows),residual_gt_one=sum(r['relative_FP32_residual']>1 for r in vector_rows),SC_comparable=len(sc),SC_byte_changes=sum(x['byte_changed'] for x in sc),SC_numeric_changes=sum(x['numerically_changed'] for x in sc),SC_byte_only=sum(x['byte_changed'] and not x['numerically_changed'] for x in sc),target_adaptive=sum(r['target_adaptive'] for r in native),target_caps=sum(r['target_capped'] for r in native),native_forwards_added=0,timing_qualification=False)
    for n,v in [('vectors.json',vector_rows),('native_states.json',native),('scope.json',scope),('input_hashes.json',INPUTS),('final_complete.json',summary)]:
        with (OUT/n).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
    print(json.dumps(summary))
if __name__=='__main__':main()
