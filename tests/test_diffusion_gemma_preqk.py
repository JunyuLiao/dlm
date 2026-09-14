from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from experiments.diffusion_gemma_preqk.config import RouterConfig, screening_configs
from experiments.diffusion_gemma_preqk.state import History, RequestState, fixed_budget, protected_set, cover_rows, smooth


def test_unknown_and_current_proxy():
    e=torch.ones(1,2,3,4,dtype=torch.bool);h=History.empty(e,())
    p=torch.arange(24.).reshape_as(e)
    assert torch.equal(h.predict(RouterConfig(estimator='proxy'),e,proxy=p),p)
    assert (h.predict(RouterConfig(),e)>0).all()


def test_frequency_uses_importance_and_eligible_count():
    e=torch.tensor([[[[True,True,False,False]]]])
    h=History.empty(e,());mass=torch.tensor([[[[.4,.6,0.,0.]]]])
    h.update(mass,mass,e,e,0,eligible=e)
    assert h.frequency.flatten().tolist()==[0,1,0,0]
    h.update(mass,mass,e,torch.zeros_like(e),1,eligible=e)
    assert h.frequency.flatten().tolist()==[0,2,0,0]
    with pytest.raises(ValueError):h.update(mass,mass,e,e,3,eligible=e)


def test_state_resets_step_cache_and_length():
    e=torch.ones(1,2,3,4,dtype=torch.bool);s=RequestState('x')
    def get(step=0,cache=7,prefix=128):
        return s.get(0,e,step=step,prefix_length=prefix,query_length=192,cache_identity=cache)
    a=get();a.update(e.float(),e.float(),e,e,0,eligible=e)
    assert get(1) is a
    assert get(2) is not a
    b=get();b.update(e.float(),e.float(),e,e,0,eligible=e)
    assert get(1,cache=8) is not b
    c=get();c.update(e.float(),e.float(),e,e,0,eligible=e)
    assert get(1,prefix=129) is not c


def test_fixed_budget_random_protection_rng_and_infeasibility():
    e=torch.ones(1,2,3,8,dtype=torch.bool);score=torch.arange(48.).reshape_as(e)
    p=protected_set(e,192,512,'beginning');rng=torch.random.get_rng_state()
    a,am=fixed_budget(score,e,p,.5)
    b,bm=fixed_budget(score,e,p,.5,random_protection=True,salt=42)
    assert torch.equal(torch.random.get_rng_state(),rng)
    assert (a.sum(-1)==4).all() and (b.sum(-1)==4).all()
    assert torch.equal(am['protected'],bm['protected'])
    c,cm=fixed_budget(score,e,e,.9)
    assert c.all() and (cm['budget_excess']==7).all()


def test_protection_regions_and_no_ineligible_tiles():
    e=torch.ones(1,1,2,5,dtype=torch.bool);e[...,0]=False
    assert not protected_set(e,129,257,'beginning').any()
    assert protected_set(e,129,257,'prefix_end')[...,2].all()
    assert protected_set(e,129,257,'canvas_end')[...,4].all()
    for name in screening_configs():
        config=screening_configs()[name];p=protected_set(e,129,257,config.protection)
        keep,_=fixed_budget(torch.ones_like(e,dtype=torch.float),e,p,.5)
        assert not (keep & ~e).any()


def test_row_rescue_is_geometry_only():
    v=torch.zeros(1,1,2,3,64,dtype=torch.bool)
    v[...,0,0,0]=True;v[...,1,2,63]=True
    keep=torch.tensor([[[True,False,False]]]);out,count=cover_rows(keep,v)
    assert out.tolist()==[[[True,False,True]]] and count.item()==1


def test_neighbor_axes_and_partial_mask():
    v=torch.tensor([[[[0.,3.,6.],[9.,12.,15.]]]]);e=torch.ones_like(v,dtype=torch.bool)
    assert torch.equal(smooth(v,e,-1,'mean'),torch.tensor([[[[1.5,3.,4.5],[10.5,12.,13.5]]]]))
    assert torch.equal(smooth(v,e,-2,'max'),torch.tensor([[[[9.,12.,15.],[9.,12.,15.]]]]))
    e[...,2]=False
    assert smooth(v,e,-1,'mean')[0,0,0,1]==1.5


def test_sparse_unknown_age_and_renormalization():
    e=torch.ones(1,1,1,2,dtype=torch.bool);h=History.empty(e,())
    mass=torch.tensor([[[[.2,.8]]]]);h.update(mass,mass,e,e,0,eligible=e)
    observed=torch.tensor([[[[True,False]]]])
    h.update(torch.ones_like(mass),torch.ones_like(mass),observed,observed,1,eligible=e,conditional=True)
    assert torch.allclose(h.mean,mass)
    assert h.age.flatten().tolist()==[0,1]
    assert h.observations.flatten().tolist()==[2,1]


def test_geometry_gqa_partial_tiles_and_dense_no_rng_change(tmp_path,monkeypatch):
    from dllm.attention.blasst.core import _prepare_attention_scores,_finish_eager_attention
    from experiments.diffusion_gemma_preqk.routing import ScreeningAttention,geometry
    from experiments.diffusion_gemma_preqk.summaries import block_norms
    import experiments.diffusion_gemma_preqk.routing as routing
    torch.manual_seed(3)
    q=torch.randn(1,2,65,4);k=torch.randn(1,1,130,4);v=torch.randn_like(k)
    valid=torch.ones(1,2,65,130,dtype=torch.bool);valid[...,0,:64]=False
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    e,blocks=geometry(valid)
    assert e.shape==(1,2,2,3) and blocks.shape==(1,2,2,64,3,64)
    class Producer:
        bytes=0
        def get(self,module,heads):return block_norms(v.float().square().mean(-1).sqrt().repeat_interleave(heads,1)),('cache',1)
    router=ScreeningAttention('x','aime24',producer=Producer(),steps=2)
    _,vv,ss,vm=_prepare_attention_scores(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=None)
    reference=_finish_eager_attention(q,vv,ss,vm,0,False)[0]
    original_fixed=routing.fixed_budget;original_matmul=torch.matmul;selected=[0]
    def counted(*args,**kwargs):selected[0]+=1;return original_fixed(*args,**kwargs)
    def guarded(a,b,*args,**kwargs):
        # The only torch.matmul in collector is full current token QK.
        if a.shape[-2]==65:assert selected[0]==40
        return original_matmul(a,b,*args,**kwargs)
    monkeypatch.setattr(routing,'fixed_budget',counted);monkeypatch.setattr(torch,'matmul',guarded)
    rng=torch.random.get_rng_state();out=router(module,q,k,v,valid,scaling=.5,is_causal=False)[0]
    assert torch.equal(out,reference) and torch.equal(torch.random.get_rng_state(),rng)
    data=router.records[0]['metrics'];assert data.shape==(48,1,2,2,21)
    assert torch.tensor(data).isfinite().all()
    # Every physical-tile skip is in the eligible population, including partials.
    assert (data[...,1]<=data[...,0]).all()
    assert (data[...,15]+data[...,17]+data[...,19]==data[...,0]).all()
    assert (data[...,16]+data[...,18]+data[...,20]==data[...,1]).all()


def test_producer_value_norms_partial_gqa():
    from experiments.diffusion_gemma_preqk.summaries import block_norms
    x=torch.cat((torch.ones(1,2,64),torch.full((1,2,1),3.)),-1)
    summaries=block_norms(x)
    assert summaries['rms'].tolist()==[[[1.,3.],[1.,3.]]]
    assert torch.equal(summaries['rms'],summaries['max'])


def test_deterministic_disjoint_selection():
    import json
    from pathlib import Path
    from experiments.diffusion_gemma_preqk.run import heldout_selection
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _rows
    dev=json.loads(Path('results/diffusion_gemma_oracle/setup.json').read_text())['samples']
    pool=_rows(Path('results/diffusion_gemma_solattn_blasst_multibench_controlled/manifest.jsonl'))
    aime=_rows(Path('/home/exouser/ljy/Skills/nemo_skills/dataset/aime24/test.txt'))
    a=heldout_selection(dev,pool,aime);b=heldout_selection(dev,list(reversed(pool)),list(reversed(aime)))
    assert a==b and len(a)==20


def test_online_dense_parity_sparse_observation_and_refresh():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention,observations
    from dllm.attention.blasst.core import _prepare_attention_scores,_finish_eager_attention
    q=torch.randn(1,2,64,4);k=torch.randn(1,1,256,4);v=torch.randn_like(k)
    valid=torch.ones(1,2,64,256,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    config=RouterConfig(sparsity=.5,refresh_interval=2,exploration_tiles=0)
    router=OnlineAttention('x',config,Producer(),diagnostics=True)
    _,vv,ss,vm=_prepare_attention_scores(module,q,k,v,valid,scaling=.5,is_causal=False,sliding_window=None)
    reference=_finish_eager_attention(q,vv,ss,vm,0,False)[0]
    out=router(module,q,k,v,valid,scaling=.5,is_causal=False)[0]
    assert torch.equal(out,reference) and router.last_decisions.all()
    counts=router.state.layers[0].observations.clone()
    module._blasst_2d_runtime.current_denoising_iteration=1
    out=router(module,q,k,v,valid,scaling=.5,is_causal=False)[0]
    keep=router.last_decisions.clone()
    assert (keep.sum(-1)==2).all() and out.isfinite().all()
    assert torch.equal(router.state.layers[0].observations,counts+keep.to(counts.dtype))
    assert (router.state.layers[0].age[~keep]==1).all()
    module._blasst_2d_runtime.current_denoising_iteration=2
    assert torch.equal(router(module,q,k,v,valid,scaling=.5,is_causal=False)[0],reference)
    assert router.last_decisions.all() and router.costs['refresh_tiles']==8
    assert router.costs['retained_tiles']==20 and router.costs['eligible_tiles']==24


def test_online_current_logits_cannot_change_history_mask():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention
    q=torch.randn(1,1,64,4);k=torch.randn(1,1,256,4);v=torch.randn_like(k)
    valid=torch.ones(1,1,64,256,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    routers=[OnlineAttention('x',RouterConfig(refresh_interval=0,exploration_tiles=0),Producer()) for _ in range(2)]
    for router in routers:router(module,q,k,v,valid,scaling=.5,is_causal=False)
    module._blasst_2d_runtime.current_denoising_iteration=1
    routers[0](module,q,k,v,valid,scaling=.5,is_causal=False)
    routers[1](module,-q,2*k,-v,valid,scaling=.5,is_causal=False)
    assert torch.equal(routers[0].last_decisions,routers[1].last_decisions)


def test_observations_ignore_dropped_scores():
    from experiments.diffusion_gemma_preqk.online import observations
    scores=torch.zeros(1,1,64,128);valid=torch.ones_like(scores,dtype=torch.bool);valid[...,64:]=False
    # Dropped logits are -inf before normalization, regardless of diagnostic values.
    scores=scores.masked_fill(~valid,-torch.inf)
    mean,peak=observations(scores,valid)
    assert mean.tolist()==[[[[1.,0.]]]] and torch.equal(mean,peak)


def test_report_weights_tile_counts_and_pairs_prompts():
    import numpy as np
    from experiments.diffusion_gemma_preqk.report import ratios,paired_ablations
    from experiments.diffusion_gemma_preqk.routing import FIELDS
    a=np.zeros(len(FIELDS));a[0]=10;a[1]=5;a[2]=8;a[3]=10;a[4]=1;a[5]=100;a[11]=5;a[12]=4
    b=a.copy();b[0]=100;b[1]=10
    assert ratios(a+b)['sparsity']==15/110
    table={}
    for ident in ('a','b'):
        table[ident,'longbench','last_mass_s50','local']=a.copy()
        worse=a.copy();worse[4]=4
        table[ident,'longbench','proxy_s50','local']=worse
    result=paired_ablations(table)
    row=next(r for r in result if r['candidate']=='last_mass_s50' and r['attention_type']=='overall')
    assert row['paired_prompts']==2 and row['mean_error_delta']==pytest.approx(-.1)


def test_direct_online_diagnostic_matches_block_contribution_oracle():
    from experiments.diffusion_gemma_preqk.online import direct_diagnostics
    from experiments.diffusion_gemma_oracle.routing import signals,masked_diagnostics
    scores=torch.randn(1,2,64,130);values=torch.randn(1,2,130,7)
    valid=torch.ones_like(scores,dtype=torch.bool);valid[...,0,:]=False
    scores=scores.masked_fill(~valid,-torch.inf)
    keep=torch.tensor([[[True,False,True],[True,True,False]]])
    _,e,mass,contrib,has,_=signals(scores,valid,values)
    ref=masked_diagnostics(keep,mass,contrib,has)
    token_keep=keep.repeat_interleave(64,-1)[...,:130][...,None,:].expand_as(valid)
    direct=direct_diagnostics(scores,valid,token_keep,values)
    for key in ref:assert torch.allclose(ref[key].float(),direct[key].float(),atol=1e-5,rtol=1e-5)


def test_online_diagnostics_do_not_affect_rng_masks_or_outputs():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention
    q=torch.randn(1,1,64,4);k=torch.randn(1,1,130,4);v=torch.randn_like(k)
    valid=torch.ones(1,1,64,130,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    routers=[OnlineAttention('x',RouterConfig(refresh_interval=0,exploration_tiles=0),Producer(),diagnostics=d) for d in (True,False)]
    for step in range(3):
        module._blasst_2d_runtime.current_denoising_iteration=step
        rng=torch.random.get_rng_state()
        outputs=[r(module,q,k,v,valid,scaling=.5,is_causal=False)[0] for r in routers]
        assert torch.equal(outputs[0],outputs[1]) and torch.equal(routers[0].last_decisions,routers[1].last_decisions)
        assert torch.equal(rng,torch.random.get_rng_state())


def test_online_shortlist_is_bounded_paired_and_excludes_heldout_benchmarks():
    from experiments.diffusion_gemma_preqk.online_run import build_conditions
    selection=dict(hypothesis='test',decision_rationale='fixture, not real advancement',entries=[
        dict(benchmark=b,predictor='last_mass',target_sparsity=.5) for b in ('aime24','longbench')])
    conditions=build_conditions(selection);assert len(conditions)==4
    for benchmark in ('aime24','longbench'):
        pair=[v['config'] for v in conditions.values() if v['benchmark']==benchmark]
        assert {p['history_source'] for p in pair}=={'dense_reference','sparse'}
        assert all(pair[0][k]==pair[1][k] for k in pair[0] if k!='history_source')
    selection['entries'][0]['benchmark']='ruler16k'
    with pytest.raises(ValueError):build_conditions(selection)


def test_online_freeze_refuses_incomplete_screen(tmp_path,monkeypatch):
    import json
    import experiments.diffusion_gemma_preqk.report as reporting
    from experiments.diffusion_gemma_preqk.online_run import freeze
    monkeypatch.setattr(reporting,'report',lambda root:None)
    (tmp_path/'screen_audit.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(RuntimeError,match='complete development screening'):
        freeze(tmp_path,tmp_path/'online',tmp_path/'selection.json')
    assert not (tmp_path/'online'/'freeze.json').exists()


def test_online_quality_treats_seeds_within_prompt_and_weights_tokens():
    from experiments.diffusion_gemma_preqk.online_report import quality_summary
    def row(ident,seed,score,tokens):
        return dict(id=ident,task='x',seed=seed,dense_score=1.,score=score,matching_tokens=tokens//2,
            compared_tokens=tokens,sequence_exact_match=False,output_tokens=tokens,dense_output_tokens=tokens,
            budget_hit=False,denoising_steps=10)
    rows=[row('a',1,0.,10),row('a',2,1.,10),row('b',1,1.,100)]
    summary=quality_summary(rows)
    assert summary['prompts']==2 and summary['generations']==3
    assert summary['score']==.75 and summary['accuracy_delta']==-.25
    assert summary['matching_tokens']==60 and summary['compared_tokens']==120 and summary['token_agreement']==.5


def test_online_count_weighting_and_refresh_denominator():
    from experiments.diffusion_gemma_preqk.online_report import routing_summary
    summary=routing_summary([dict(eligible=100,skipped=50,sparse_state_rows=20,sparse_state_mass_sum=10),
        dict(eligible=400,skipped=0,sparse_state_rows=100,sparse_state_mass_sum=100,refresh_calls=1)])
    assert summary['sparsity']==.1 and summary['retained_mass']==pytest.approx(110/120)


def test_raw_trace_audit_checks_masks_budgets_and_cache_alignment(tmp_path):
    import numpy as np
    from experiments.diffusion_gemma_preqk.audit import validate_trace,validate_consecutive_coverage
    from experiments.diffusion_gemma_preqk.routing import FIELDS
    data=np.zeros((2,1,2,2,len(FIELDS)),dtype=np.float32)
    data[...,0]=4;data[...,1]=2;data[...,15]=2;data[...,17]=2;data[...,18]=2
    eligible=np.ones((1,2,4),dtype=bool)
    masks=np.broadcast_to(np.array([True,True,False,False]),(2,1,2,4)).copy()
    path=tmp_path/'trace.npz';np.savez(path,metrics=data,eligible=eligible,selected_q0=masks)
    record=dict(fields=FIELDS,names=['last_mass_s50','oracle_mass_s50'],path=str(path),kv_length=256,prefix_length=128)
    assert validate_trace(record,heads=2)['mask_checks']==8
    records=[dict(record,layer=layer,step=step,history_available=bool(step),attention_type=kind)
        for step in (0,1) for layer,kind in enumerate(('local','global'))]
    assert validate_consecutive_coverage(records,layers=2,heads=2)['steps']==[0,1]
    records[-1]['history_available']=False
    with pytest.raises(AssertionError,match='cache alignment'):validate_consecutive_coverage(records,layers=2,heads=2)
    masks[0,0,0,0]=False
    np.savez(path,metrics=data,eligible=eligible,selected_q0=masks)
    with pytest.raises(AssertionError):validate_trace(record,heads=2)


def test_online_report_regenerates_and_rejects_prompt_mismatch(tmp_path,monkeypatch):
    import hashlib
    import json
    import experiments.diffusion_gemma_preqk.online_report as reporting
    from experiments.diffusion_gemma_preqk.online_run import build_conditions
    from experiments.diffusion_gemma_preqk.run import digest,sample_key
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    monkeypatch.setattr(reporting,'score',lambda row,prediction:float(prediction==row['expected']))
    entries=[dict(benchmark=b,predictor='last_mass',target_sparsity=.5) for b in ('aime24','longbench')]
    conditions=build_conditions(dict(hypothesis='fixture',decision_rationale='test only',entries=entries))
    rows=[];cache={}
    for b in ('aime24','longbench'):
        prompt='question '+b
        row=dict(id=b+'/test',benchmark=b,task='fixture_'+b,prompt=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),seed=42,generation_budget=8,expected='correct')
        rows.append(row);path=tmp_path/(b+'_dense.json')
        _write(path,dict(row,prediction='correct',completion_tokens=[1,2,3]))
        cache[row['id']]=dict(path=str(path),sha256=digest(path))
    _write(tmp_path/'freeze.json',dict(samples=rows,conditions=conditions,cached_dense=cache))
    _write(tmp_path/'smoke.json',dict(passed=True,fingerprint='fixture'))
    for name,condition in conditions.items():
        row=next(r for r in rows if r['benchmark']==condition['benchmark']);records=[]
        for layer,kind in enumerate(('local','global')):
            records.append(dict(layer=layer,head=0,step=1,calls=3,attention_type=kind,eligible=10,skipped=4,
                prefix_eligible=6,prefix_skipped=2,canvas_eligible=4,canvas_skipped=2,mixed_eligible=0,mixed_skipped=0,
                sparse_state_rows=192,sparse_state_mass_sum=144,sparse_state_error_sq=1,sparse_state_dense_sq=10))
        path=tmp_path/'conditions'/name/'shards'/f'{sample_key(row)}.json'
        _write(path,dict(row,condition=name,config=condition['config'],fingerprint='fixture',prediction='correct',
            completion_tokens=[1,4,3,5],termination_reason='eos',records=records,attention_calls=6,model_evaluations=3,
            max_state_bytes=128,costs=dict(eligible_tiles=20,retained_tiles=12),prefix_value_elements=0,
            canvas_value_elements=0,diagnostic_e2e_seconds=1.))
    reporting.report(tmp_path)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['audit']['complete'] and summary['audit']['completed_generations']==4
    for row in summary['conditions']:
        assert row['token_agreement']==.5 and row['overall_sparsity']==.4
        assert row['overall_retained_mass']==.75 and row['score']==1.
    assert (tmp_path/'cost_breakdown.csv').exists() and (tmp_path/'report.md').exists()
    bad=json.loads(path.read_text());bad['prompt_hash']='bad';_write(path,bad)
    with pytest.raises(AssertionError):reporting.report(tmp_path)


def test_online_last_mask_matches_previous_importance_not_dense_execution():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention
    q=torch.ones(1,1,64,4);k=torch.zeros(1,1,256,4);k[...,128:192,:]=4;v=torch.ones_like(k)
    valid=torch.ones(1,1,64,256,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    config=RouterConfig(estimator='last_mask',history_source='dense_reference',refresh_interval=0,exploration_tiles=0)
    router=OnlineAttention('x',config,Producer())
    router(module,q,k,v,valid,scaling=.5,is_causal=False)
    assert router.last_decisions.all() # paid initialization executes every tile
    predicted=router.state.layers[0].last_mask.clone()
    assert predicted.tolist()==[[[[True,False,True,False]]]]
    module._blasst_2d_runtime.current_denoising_iteration=1
    router(module,-q,k,v,valid,scaling=.5,is_causal=False)
    assert torch.equal(router.last_decisions,predicted) # current QK does not change it
    assert router.costs['history_mask_selections']==2


def test_value_rank_reconstruction_and_vectorized_spearman():
    import numpy as np
    from scipy.stats import spearmanr
    from experiments.diffusion_gemma_preqk.report import predictor_scores,rank_correlations
    raw={'last_mass':np.array([[[.1,.5,.2,.2],[.4,.1,.4,.1]]],dtype=np.float32),
        'value_rms':np.array([[[1.001,.999,1.,1.],[1.,1.,.999,1.]]],dtype=np.float32)}
    actual=predictor_scores(raw,'value_rms')[0]
    assert np.allclose(actual,raw['last_mass'][0]*raw['value_rms'][0])
    target=np.array([[.2,.3,.3,.2],[.3,.4,.1,.2]])
    e=np.array([[True,True,True,False],[False,True,True,True]])
    expected=[spearmanr(actual[h,e[h]],target[h,e[h]]).statistic for h in range(2)]
    assert np.allclose(rank_correlations(actual,target,e),expected)


def test_trajectory_work_uses_all_shapes_and_checks_layer_counts():
    from experiments.diffusion_gemma_preqk.work_volume import baseline_work
    calls=[dict(layer=layer,attention_type=kind,head_ids=[0,1],query_length=64,sequence_length=length,
        eligible_tiles=2*(length//64)) for layer,kind,length in
        ((0,'local',128),(1,'global',128),(0,'local',256),(1,'global',512))]
    result=baseline_work(calls)
    assert result['steps']==2 and result['eligible']==32
    assert result['qk_flops']==2*2*64*((128+256)*256+(128+512)*512)
    with pytest.raises(ValueError):baseline_work(calls[:-1])


def test_paired_allocation_balances_each_type_head_and_query_row():
    from experiments.diffusion_gemma_preqk.allocation import PairedLayerAllocation
    policy=dict(layer_kinds={'0':'local','1':'global','2':'local','3':'global'},
        groups={'local':{'low':[0],'high':[2]},'global':{'low':[1],'high':[3]}},shift_tiles=1)
    allocator=PairedLayerAllocation(policy)
    e=torch.ones(1,2,3,8,dtype=torch.bool);e[...,0,:]=False;e[...,1,2:]=False
    for active in (False,True):
        for layer in range(4):
            kind=policy['layer_kinds'][str(layer)]
            delta,base=allocator.shift(layer,kind,e,.5,active)
            if active:
                keep,_=fixed_budget(e.float(),e,torch.zeros_like(e),.5,budget_delta=delta)
            else:keep=e.clone()
            allocator.observe(layer,kind,keep,base)
    assert len(allocator.audit)==4
    assert all(a['max_abs_row_delta']==0 and a['quota_delta']==0 for a in allocator.audit)
    # Delta zero is the original selector exactly, including exploration.
    score=torch.rand(e.shape);age=torch.zeros_like(e,dtype=torch.int32)
    old,_=fixed_budget(score,e,torch.zeros_like(e),.5,explore=1,age=age)
    new,_=fixed_budget(score,e,torch.zeros_like(e),.5,explore=1,age=age,budget_delta=torch.zeros_like(e.sum(-1)))
    assert torch.equal(old,new)


def test_isolated_sparse_call_fires_only_once_across_canvases():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention
    q=torch.randn(1,1,64,4);k=torch.randn(1,1,256,4);v=torch.randn_like(k)
    valid=torch.ones(1,1,64,256,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=1,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    router=OnlineAttention('x',RouterConfig(exploration_tiles=0),Producer(),isolate=(0,1))
    skips=[]
    for step in (0,1,2,0,1,2):
        module._blasst_2d_runtime.current_denoising_iteration=step
        router(module,q,k,v,valid,scaling=.5,is_causal=False)
        skips.append(int((~router.last_decisions).sum()))
    assert skips==[0,2,0,0,0,0] and router.costs['isolated_sparse_calls']==1


def test_allocated_shadow_diagnostics_do_not_change_generation():
    from experiments.diffusion_gemma_preqk.online import OnlineAttention
    policy=dict(layer_kinds={'0':'global','1':'global'},groups={'global':{'low':[0],'high':[1]}},shift_tiles=1)
    class Producer:
        bytes=0
        def get(self,module,heads):return {},('cache',1)
    config=RouterConfig(allocation='paired_layer_shift',refresh_interval=0,exploration_tiles=1)
    routers=[OnlineAttention('x',config,Producer(),diagnostics=d,allocation_policy=policy) for d in (False,True)]
    q=torch.randn(1,2,64,4);k=torch.randn(1,1,256,4);v=torch.randn_like(k)
    valid=torch.ones(1,2,64,256,dtype=torch.bool)
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,is_sliding=False,
        _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    for step in range(3):
        for layer in range(2):
            module.layer_idx=layer;module._blasst_2d_runtime.current_denoising_iteration=step
            rng=torch.random.get_rng_state()
            outputs=[r(module,q,k,v,valid,scaling=.5,is_causal=False)[0] for r in routers]
            assert torch.equal(outputs[0],outputs[1]) and torch.equal(rng,torch.random.get_rng_state())
    router=routers[1]
    assert all(a['actual_delta']==0 for a in router.allocator.audit)
    assert sum(r['eligible']-r['skipped'] for r in router.records)==sum(r['shadow_uniform_retained'] for r in router.records)


def test_refinement_rejects_unproven_or_unbounded_cases():
    from experiments.diffusion_gemma_preqk.refinement import build_conditions
    selection=dict(cases=[dict(benchmark='aime24',target_sparsity=.4,label='last_mass',sources=['sparse','dense_reference'])])
    with pytest.raises(ValueError,match='evidence'):build_conditions(selection)
    selection['parent_evidence']={'fixture':'test'}
    assert len(build_conditions(selection))==2
    selection['cases'][0]['benchmark']='heldout'
    with pytest.raises(ValueError):build_conditions(selection)


def test_performance_component_wrapper_restores_functions(monkeypatch):
    from experiments.diffusion_gemma_preqk.performance import ComponentSpans
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):pass
        def elapsed_time(self,other):return 2.
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    owner=SimpleNamespace(compute=lambda x:x+1);original=owner.compute
    collector=ComponentSpans();collector.wrap(owner,'compute','fixture')
    assert owner.compute(4)==5
    collector.stack.close()
    assert owner.compute is original
    row=collector.summary()[0]
    assert row['calls']==1 and row['cuda_stream_span_ms']==2. and row['host_wall_seconds']>=0


def test_allocation_prompt_deltas_require_identical_work():
    from experiments.diffusion_gemma_preqk.refinement_report import paired_error_deltas
    record=dict(attention_type='local',eligible=20,skipped=10,shadow_uniform_retained=10,
        sparse_state_error_sq=1.,sparse_state_dense_sq=100.,shadow_uniform_error_sq=4.,shadow_uniform_dense_sq=100.,
        sparse_state_rows=64,shadow_uniform_rows=64,sparse_state_mass_sum=60.,shadow_uniform_mass_sum=50.)
    shards=[dict(id='fixture',records=[record,dict(record,attention_type='global')])]
    rows=paired_error_deltas(shards)
    assert rows[0]['error_delta']==pytest.approx(-.1) and rows[0]['allocated_retained']==20
    record['shadow_uniform_retained']=11
    with pytest.raises(AssertionError):paired_error_deltas(shards)


def test_performance_supervisor_reuses_queue_with_explicit_targets():
    from pathlib import Path
    from experiments.diffusion_gemma_preqk.launch import pipeline_command
    with pytest.raises(ValueError):pipeline_command('performance',Path('out'))
    command=pipeline_command('performance',Path('out'),stage=Path('source'),conditions=['plain'])
    assert command[-6:]==['--stage','source','--output','out','--conditions','plain']


def test_performance_summary_sums_prompt_medians(tmp_path):
    import json
    from experiments.diffusion_gemma_preqk.performance import summarize
    from experiments.diffusion_gemma_preqk.run import sample_key
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    samples=[dict(id=i,benchmark='fixture') for i in ('short','long')]
    _write(tmp_path/'setup.json',dict(samples=samples,conditions={'plain':{'benchmark':'fixture'}},repeats=3,code_fingerprint='test'))
    for sample,scale in zip(samples,(1.,10.)):
        for method,factor in (('dense',1.),('plain',2.)):
            for repeat,noise in enumerate((1.,.5,1.5)):
                _write(tmp_path/'timings'/method/f'{sample_key(sample)}_{repeat}.json',dict(
                    id=sample['id'],repeat=repeat,method=method,code_fingerprint='test',exact_reference_parity=True,
                    diagnostics=False,e2e_seconds=scale*factor*noise,peak_allocated_bytes=1024,state_bytes=64))
    summarize(tmp_path)
    rows=json.loads((tmp_path/'summary.json').read_text())['conditions']
    plain=next(r for r in rows if r['method']=='plain')
    assert plain['summed_prompt_median_seconds']==22. and plain['reference_e2e_time_ratio_vs_dense']==2.


def test_heldout_storage_ids_keep_prompt_clusters_and_reserved_seeds():
    from experiments.diffusion_gemma_preqk.validation import reserved_rows
    protocol=dict(development=[dict(id='old',prompt_hash='oldhash')],heldout_reserved=[
        dict(id='new',prompt_hash='newhash',benchmark='longbench',paired_seeds=[314159,271828])])
    rows=reserved_rows(protocol,{'longbench'})
    assert len(rows)==2 and len({r['id'] for r in rows})==2
    assert {r['prompt_id'] for r in rows}=={'new'} and [r['seed'] for r in rows]==[314159,271828]
    protocol['heldout_reserved'][0]['prompt_hash']='oldhash'
    with pytest.raises(ValueError,match='contamination'):reserved_rows(protocol,{'longbench'})


def test_heldout_quality_bootstrap_clusters_storage_replicates():
    from experiments.diffusion_gemma_preqk.online_report import quality_summary
    def row(ident,prompt,score):
        return dict(id=ident,prompt_id=prompt,task='x',seed=42,dense_score=1.,score=score,
            matching_tokens=1,compared_tokens=2,sequence_exact_match=False,output_tokens=2,dense_output_tokens=2,
            budget_hit=False,denoising_steps=3)
    result=quality_summary([row('a:seed1','a',0.),row('a:seed2','a',1.),row('b:seed1','b',1.)])
    assert result['prompts']==2 and result['generations']==3 and result['score']==.75


def test_validation_queue_uses_existing_supervisor():
    from pathlib import Path
    from experiments.diffusion_gemma_preqk.launch import pipeline_command
    assert pipeline_command('validation',Path('heldout'))[-3:]==['pipeline','--stage','heldout']


def test_final_record_audit_validates_counts_heads_and_layer_types():
    from experiments.diffusion_gemma_preqk.final_report import audit_records
    records=[dict(layer=layer,head=head,calls=1,attention_type=kind,eligible=10,skipped=5,
        prefix_eligible=6,canvas_eligible=4,mixed_eligible=0,prefix_skipped=3,canvas_skipped=2,mixed_skipped=0)
        for layer,kind in enumerate(('local','global')) for head in range(16)]
    data=dict(records=records,costs=dict(eligible_tiles=320,retained_tiles=160),
        metadata={'sampling':{'native_temperature_schedule':True}},executed_qk_saved=0,executed_pv_saved=0)
    audit_records(data,['sliding_attention','full_attention'])
    records[0]['attention_type']='global'
    with pytest.raises(AssertionError):audit_records(data,['sliding_attention','full_attention'])


def test_existing_physical_reference_is_not_native_numerical_drop_in(tmp_path):
    from experiments.diffusion_gemma_preqk.backend_audit import audit
    result=audit(tmp_path)
    assert result['exists'] and not result['compatible_drop_in']
    assert result['no_skip_cpu_fixture']['different_elements']>0
    assert not result['gpu_latency_measured']
