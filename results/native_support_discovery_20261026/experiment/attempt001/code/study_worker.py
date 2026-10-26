"""One GPU worker; every treatment is a fresh, officially scored native request."""
from pathlib import Path
import argparse,json,sys
from pulse_runtime import ROOT,PROFILE,NativeHarness,PulseStudy,save,sha,site_key,exact_request,core_trace_equal
LIVE={}

def load(p):return json.loads(p.read_text())
def cell_id(row,seed):
    import hashlib
    return hashlib.sha256(row['id'].encode()).hexdigest()[:12]+f'_seed{seed}'
def profile_record(row,seed):
    matches=[]
    for p in (PROFILE/'runs/profile/matched_allkept_l5_l17_l29').glob('audit*.json'):
        v=load(p)
        if (v['id'],v['seed'])==(row['id'],seed):matches.append((p,v))
    assert len(matches)==1;return matches[0]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=['references','early','treatments'],required=True);a=ap.parse_args()
    root=ROOT;manifest=load(PROFILE/'private/manifest.json');rows=[r for r in manifest if r['partition']=='development'];leaf=root/'runs'/a.job;leaf.mkdir(parents=True,exist_ok=False);LIVE.update(root=root,leaf=leaf)
    ext=PulseStudy(root,'warm',{});h=NativeHarness.__new__(NativeHarness);LIVE['harness']=h;h.__init__(root,'matched_allkept_l5_l17_l29',ext)
    fixtures=[next(r for r in rows if r['benchmark']==task) for task in ['aime26','humaneval','longbench_v2']]
    for i,row in enumerate(fixtures):
        ext.mode='warm';ext.config={};v=h.request(row,42,f'{a.job}/warm{i:03d}',audit=False);save(leaf/f'warm{i:03d}.json',v)
    if a.job=='references':
        for row in rows:
            for seed in [42,43]:
                cid=cell_id(row,seed);p,baseline=profile_record(row,seed);anchors=baseline['anchors'];chosen=anchors.get('instability',anchors.get('confidence'))
                operand=site_key(chosen,29) if chosen and row in fixtures and seed==42 else None
                ext.mode='bank_reference';ext.config=dict(anchors=anchors,operand_anchor=operand,commit_layers=[29]);v=h.request(row,seed,'references/'+cid,audit=True)
                checks=exact_request(v,baseline);assert all(checks.values()),'Fresh bank observer changed reference output/state/RNG/counts'
                assert core_trace_equal(v['native_trace'],baseline['native_trace']),'Bank observer changed native trajectory'
                v['profiling_reference']=dict(relative=str(p.relative_to(PROFILE)),sha256=sha(p),checks=checks);save(leaf/(cid+'.json'),v)
    elif a.job=='early':
        plans=load(root/'early_maps.json')
        for plan in plans['cells']:
            row=next(r for r in rows if r['id']==plan['id']);cid=cell_id(row,42);ref=load(root/'runs/references'/(cid+'.json'))
            ext.mode='early_reference';ext.config=dict(anchors=plan['anchors'],maps=plan['maps'],operand_anchor=plan.get('operand_anchor'),commit_layers=[5,17],reference_label=ref['label']);v=h.request(row,42,'early/'+cid,audit=True)
            checks=exact_request(v,ref);assert all(checks.values());v['reference_checks']=checks;save(leaf/(cid+'.json'),v)
    else:
        plan=load(root/'branch_manifest.json');assert plan['frozen_before_sparse_outcomes'];qualified_tasks=set()
        for cell in plan['cells']:
            row=next(r for r in rows if r['id']==cell['id']);seed=cell['seed'];cid=cell_id(row,seed);ref=load(root/'runs/references'/(cid+'.json'));branches=cell['branches']
            if not branches:continue
            for j,b in enumerate(branches):
                ext.mode='branch';ext.config=dict(b,anchors=ref['anchors'],reference_before=ref['pre_intervention'],reference_label=ref['label'],commit_layers=[b['layer']]);label=f'treatments/{cid}/s{j:03d}'
                v=h.request(row,seed,label,audit=True);v['branch_identity']=b['identity'];save(leaf/cid/f's{j:03d}.json',v)
                if j==0:
                    if seed==42 and row['benchmark'] not in qualified_tasks:
                        parity=[v]
                        for mode,audit in [('off',False),('repeat',True)]:
                            ext.mode='branch';ext.config=dict(b,anchors=ref['anchors'],reference_before=ref['pre_intervention'],reference_label=ref['label'],commit_layers=[b['layer']]);w=h.request(row,seed,f'treatments/{cid}/{mode}',audit=audit);save(leaf/cid/(mode+'.json'),w);parity.append(w)
                        checks=[exact_request(parity[0],p) for p in parity[1:]];assert all(all(c.values()) for c in checks)
                        assert parity[0]['native_trace']==parity[2]['native_trace']
                        save(leaf/cid/'qualification.json',dict(passed=True,checks=checks,policy=b['identity'],same_policy_support_exact=all(p['pulse_calls']==1 for p in parity)));qualified_tasks.add(row['benchmark'])
                    run_control(h,ext,row,seed,cid,'after_perturbation',ref,leaf)
            run_control(h,ext,row,seed,cid,'after_group',ref,leaf)
    h.verify_sources();save(leaf/'final_complete.json',dict(passed=True,job=a.job,protocol_sha256=sha(root/'protocol.json'),**h.totals()))

def run_control(h,ext,row,seed,cid,kind,ref,leaf):
    ext.mode='control';ext.config=dict(anchors=ref['anchors'],reference_before=ref['pre_intervention'],reference_label=ref['label'],commit_layers=[29]);v=h.request(row,seed,f'treatments/{cid}/{kind}',audit=True);checks=exact_request(v,ref);assert all(checks.values()),'Bracketed all-kept reference changed'
    assert core_trace_equal(v['native_trace'],ref['native_trace'])
    v['repeated_control_checks']=checks;save(leaf/cid/(kind+'.json'),v)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        h=LIVE.get('harness');s=getattr(h,'state',{})
        if LIVE:save(LIVE['leaf']/'failure.json',dict(passed=False,error_class=type(e).__name__,reason='Exact traceback is retained only in the ignored worker log.',engine_execute_forwards=s.get('executions',0),startup_model_calls_outside_execute=s.get('startup',0),extra_model_calls_inside_execute=s.get('extra_inside_execute',0)))
        raise
