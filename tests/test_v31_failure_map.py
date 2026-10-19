from types import SimpleNamespace
import pytest
import torch
from scripts import v31_failure_map_study as s
from scripts.v31_failure_map_repair import RepairRuntime, install_budget_audit
from experiments.numerical_qk_reuse.v31_refresh_diagnostic import GLOBAL_LAYERS, FACTORIAL


def cell(i,length):return dict(dataset='lb',index=i,prompt_tokens=length,seed=42)


def test_metadata_selection_no_outcomes_and_warmup_disjoint():
    c=[cell(10*j+i,n) for j,n in enumerate((20000,40000,80000,110000)) for i in range(7)]
    a,w=s.choose(c)
    b,z=s.choose([dict(x,accuracy=x['index']%2,N=10000-x['index']) for x in reversed(c)])
    assert [x['index'] for x in a]==[x['index'] for x in b]
    assert len(a)==16 and not {x['index'] for x in a}&{x['index'] for x in w}
    assert [x['index'] for x in w]==[x['index'] for x in z]


def test_screen_schedule_counts_seed_pairing_and_controls():
    c=[cell(i,20000) for i in range(16)];w=[cell(99,20000),cell(100,110000)]
    total=0
    for fam,k,fixed in [('shared',8192,True),('shared',4096,False),('shared',2048,False)]+[(f,8192,False) for f in s.REFS]:
        rows,arms=s.schedule(c,w,fam,[1,42],0,fixed)
        assert len(arms)==len(rows)+1 and sum(a['warmup'] for a in arms)==4
        total+=sum(not a['warmup'] for a in arms)
        for r,a in zip(rows,arms[1:]):
            if not a['warmup']:assert r['seed'] in (1,42)
    assert total==320


def test_budget_charges_orphans_and_never_borrows():
    assert s.available('screen',[])==5400
    assert s.available('screen',[dict(phase='screen',reserved_s=5400,elapsed_s=None)])==0
    assert s.available('repair',[dict(phase='screen',reserved_s=5400,elapsed_s=10)])==3600
    with pytest.raises(ValueError):s.available('screen',[dict(phase='screen',reserved_s=1,elapsed_s=-1)])


def row(score=1,n=24,c=2,finish='stop'):
    return dict(score=score,denoise_forwards=n,canvases=c,finish_reason=finish)


def test_case_gate_quality_caps_drift_and_missing_controls():
    assert s.case_type([row(n=24),row(n=24)],[row(n=20),row(n=20)])=='convergence'
    assert s.case_type([row(0),row(0)],[row(),row()])=='accuracy'
    assert s.case_type([row(0),row(1)],[row(),row()]) is None
    assert s.case_type([row(n=24),row(n=30)],[row(n=20)]) is None
    assert s.case_type([row(n=24,finish='length'),row(n=24)],[row(n=20)]) is None
    assert s.case_type([row(n=24),row(n=24)],[]) is None
    records=[]
    for seed in [1,42]:
        for label in ['control_before','control_after','dense_full_fix51994','allkept_fa4']:
            v=row(0 if label.startswith('control') else 1)
            v.update(index=2,panel_seed=seed,engine_repeat=0,label=label,k=8192,
                     family='shared' if label.startswith('control') else label)
            records.append(v)
    assert s.cases(records,8192,[1,42],[0],['dense_full_fix51994','allkept_fa4'])[0]['failure_type']=='accuracy'
    assert not s.cases(records[:-1],8192,[1,42],[0],['dense_full_fix51994','allkept_fa4'])


def adapter():
    a=SimpleNamespace(arm='mage',mage_select='fa4',mage_granularity='qblock_max',mage_carry_first=True,
        mage_k=128,mage_select_step=1,mage_reselect=None,mage_reselect_trigger=(.15,),mage_sticky=1.386,bound=False,
        canvas_id=1,_trig_at=None,calls={'mage_reused_calls':0},_mage_count={},mage_state={})
    for layer in GLOBAL_LAYERS:
        mask=torch.zeros(1,2,2,20,dtype=torch.bool);mask[...,:2]=True;mask[...,16:]=True
        a.mage_state[layer]=dict(canvas=1,nk=1280,prefix=1024,kept=mask,lists=mask.clone())
        a._mage_count[layer]=[1,1280,1]
    def original(layer,q,b,scale,prefix,n):
        a._mage_count[layer][2]+=1;a.calls['mage_reused_calls']+=1
        return q.transpose(1,2)+2
    def observe(layer,q,b,scale,prefix,n):
        mask=a.mage_state[layer]['kept'].clone();mask[...,:16]=False;mask[...,2:4]=True
        a.calls['extra_selection']=1
        return q.transpose(1,2)+7,mask
    a._mage=original;a._select_initial_refresh=observe;a.begin_request=lambda:None;a._value_receipt=lambda:{}
    return a


@pytest.mark.parametrize('policy',FACTORIAL)
def test_repair_separates_output_support_preserves_native_calls_rng_and_budget(policy):
    a=adapter();r=RepairRuntime(a,list_fn=lambda x:x.clone());r.set_policy(policy)
    r.before_event_callback=lambda *args:dict(prefix_signature='same')
    q=torch.zeros(1,2,256,3);b=dict(k=q,v=q);rng=torch.get_rng_state().clone()
    for layer in GLOBAL_LAYERS:
        out=r.mage(layer,q,b,.1,1024,256)
        assert torch.all(out==(7 if policy.startswith('dense') else 2))
        mask=a.mage_state[layer]['kept']
        assert torch.all(mask[...,:16].sum(-1)==2) and torch.all(mask[...,16:])
        assert bool(mask[...,2:4].all())==policy.endswith('fresh')
    assert a.calls=={'mage_reused_calls':5}
    assert all(x[2]==2 for x in a._mage_count.values())
    assert torch.equal(rng,torch.get_rng_state())
    receipt=r.receipt()['refresh_diagnostic']
    assert receipt['complete_scope'] and receipt['extra_diagnostic_work']['dense_observer_calls']==5


def test_repair_control_no_added_observation_and_skips_native_refresh():
    a=adapter();r=RepairRuntime(a);r.set_policy('control');r.before_event_callback=lambda *x:dict(prefix_signature='same')
    q=torch.zeros(1,2,256,3);r.mage(5,q,dict(k=q,v=q),.1,1024,256)
    assert r.control_anchor and not r.events and 'extra_selection' not in a.calls
    a=adapter();a._trig_at=(1,2);r=RepairRuntime(a);r.set_policy('sparse_fresh')
    r.mage(5,q,dict(k=q,v=q),.1,1024,256)
    assert r.episode is None and not r.events


def test_actual_mask_audit_detects_budget_violation():
    a=adapter();install_budget_audit(a);a.begin_request()
    q=torch.zeros(1,2,256,3)
    a._select_initial_refresh(5,q,{},.1,1024,256)
    assert a._value_receipt()['failure_map_budget_errors']==0
    a.mage_k=64
    a._select_initial_refresh(5,q,{},.1,1024,256)
    assert a._value_receipt()['failure_map_budget_errors']>0


def test_repair_causal_gate_rejects_prefix_and_sham_drift():
    records=[]
    for label in ('control_before','control_after',*FACTORIAL):
        r=row();r.update(index=1,panel_seed=43,engine_repeat=0,label=label,
            output_hash='same',output_tokens=300,native_receipt=dict(failure_map_budget_errors=0,
            refresh_diagnostic=dict(exposed=True,complete_scope=True,prefix_signature='prefix',
                added_model_forwards=0,added_sampler_calls=0)))
        records.append(r)
    assert s.repair_qualification(records)['qualified_groups']==1
    records[-1]['output_hash']='different'
    assert s.repair_qualification(records)['qualified_groups']==0
    records[-1]['output_hash']='same'
    records[2]['native_receipt']['refresh_diagnostic']['prefix_signature']='different'
    assert s.repair_qualification(records)['qualified_groups']==0
