import math
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from dllm.attention.blasst.core import Blasst2DConfig, Blasst2DRuntime, apply_blasst_2d, dense_eager_attention_forward
from experiments.diffusion_gemma_value_aware.operators import *
from experiments.diffusion_gemma_value_aware.routing import Attention, screen_configs


def example(dtype=torch.float32):
    torch.manual_seed(123)
    s = torch.randn(1, 2, 7, 150, dtype=dtype)
    valid = torch.ones_like(s, dtype=torch.bool)
    valid[..., 0, :] = False; valid[..., 1, :64] = False
    v = torch.randn(1, 2, 150, 8, dtype=dtype)
    s = s.masked_fill(~valid, -torch.inf)
    meta = value_summaries(v, valid.any(-2))
    return s, valid, v, meta, block_state(s, valid, v)


def test_pooling_padding_and_cancellation():
    v = torch.tensor([[[[2., 0.], [-2., 0.], [0., 1.], [999., 0.]]]])
    mask = torch.tensor([[[True, True, True, False]]])
    a = value_summaries(v, mask)
    assert a['mean'].item() == pytest.approx(5/3)
    assert a['vector_mean'].item() == pytest.approx(1/3)
    assert a['max'].item() == 2
    assert a['rms'].item() == pytest.approx(math.sqrt(3))
    assert a['p95'].item() == 2
    assert a['counts'].item() == 3
    assert a['ref'].item() == pytest.approx(math.sqrt(3))


def test_prefix_reuse_and_canvas_refresh():
    cache = ValueCache(); torch.manual_seed(0)
    v = torch.randn(1, 1, 150, 4); mask = torch.ones(1, 1, 150, dtype=torch.bool)
    a = cache.get(0, v, mask, 100)
    v2 = v.clone(); v2[..., 130, :] *= 10
    b = cache.get(0, v2, mask, 100)
    assert cache.reused_blocks == 1
    assert torch.equal(a['vectors'][..., :1, :], b['vectors'][..., :1, :])
    assert not torch.equal(a['vectors'], b['vectors'])
    v2[..., 0, :] *= 3
    c = cache.get(0, v2, mask, 100)
    assert cache.reused_blocks == 1
    for key, value in value_summaries(v2, mask).items(): assert torch.allclose(c[key], value)


def test_previous_state_and_stability():
    s, valid, v, meta, state = example()
    m, z, out = dense_previous(state)
    assert torch.isneginf(z[..., 0]).all()
    for j in (1, 2):
        active = valid[..., :j*64].any(-1)
        expected = torch.logsumexp(s[..., :j*64], -1)
        assert torch.allclose(z[..., j][active], expected[active], atol=1e-6)
        p = torch.softmax(torch.where(active[..., None], s[..., :j*64], 0.), -1) * valid[..., :j*64]
        assert torch.allclose(out[..., j, :], p @ v[..., :j*64, :], atol=1e-6)
    huge = block_state(s+10000, valid, v)
    for config in screen_configs().values():
        risk, _ = screen_risks(huge, meta, config)
        assert not torch.isnan(risk).any()


def test_previous_output_cannot_underflow_from_future_logits():
    s=torch.zeros(1,1,1,128);s[...,64:]=1000.
    v=torch.ones(1,1,128,1)*3
    state=block_state(s,torch.ones_like(s,dtype=torch.bool),v)
    assert dense_previous(state)[2][...,1,:].item()==pytest.approx(3.)


@pytest.mark.parametrize('lam', [.01, .5, 1., 2., 100.])
def test_blasst_previous_max_matches_repository(lam):
    s, valid, v, meta, state = example()
    c = Config(method='blasst', log_threshold=math.log(lam))
    got = streaming_mask(state, meta, c)
    _, d = apply_blasst_2d(s, valid, None, Blasst2DConfig(blasst_lambda=lam,
        allow_lambda_above_one=True, q_tile_size=128, kv_tile_size=64))
    assert torch.equal(got, d.skip_mask[..., 0, :])
    if lam > 1:
        # Updated-max semantics erase positive record margins and are NOT used.
        before = torch.nn.functional.pad(state['b'].cummax(-1).values[..., :-1], (1,0), value=-torch.inf)
        assert torch.isposinf((state['b']-before)[..., 0][state['active'][..., 0]]).all()


@pytest.mark.parametrize('method', ['value','mass','mass_value','risk','aligned','centered','compensate','zero_pv'])
def test_streaming_unpruned_and_nonempty(method):
    s, valid, v, meta, state = example()
    low = streaming_mask(state, meta, Config(method=method, log_threshold=-math.inf))
    assert not low.any()
    high = streaming_mask(state, meta, Config(method=method, log_threshold=100))
    assert (state['active'] & ~high[..., None, :]).any(-1).equal(valid.any(-1))
    out = reference_output(s, valid, v, high, Config(method=method), meta['vectors'])
    assert torch.isfinite(out).all()


def test_constant_value_norms_do_not_isolate_value_effect_from_running_max():
    # With threshold exp(1), the middle record-high block is skipped. Legacy
    # BLASST nevertheless updates seen_m to .5; new risks keep retained_m=0.
    # Thus the final block differs even though every value/norm/ref is exactly1.
    scores=torch.zeros(1,1,Q_TILE,3*KV_TILE)
    scores[...,KV_TILE:2*KV_TILE]=.5
    scores[...,2*KV_TILE:]=1.2
    valid=torch.ones_like(scores,dtype=torch.bool)
    values=torch.ones(1,1,3*KV_TILE,1)
    meta=value_summaries(values,valid.any(-2));state=block_state(scores,valid,values)
    assert torch.equal(meta['rms'],torch.ones_like(meta['rms']))
    assert torch.equal(meta['token_norms'],torch.ones_like(meta['token_norms']))
    legacy=streaming_mask(state,meta,Config(method='blasst',log_threshold=1.))
    assert legacy.tolist()==[[[False,True,True]]]
    for method in ('value','aligned'):
        mask=streaming_mask(state,meta,Config(method=method,pooling='rms',log_threshold=1.))
        assert mask.tolist()==[[[False,True,False]]]
    # At lambda<=1 the skipped-new-maximum mechanism cannot occur in this
    # equal-norm case; the distinction is not a change of retained softmax.
    for method in ('blasst','value','aligned'):
        assert not streaming_mask(state,meta,Config(method=method,log_threshold=0.)).any()


@pytest.mark.parametrize('method', ['mass','compensate','zero_pv'])
def test_explicit_reference_and_diagnostic(method):
    s, valid, v, meta, state = example()
    changed = torch.tensor([[[False, False, True], [False, False, True]]])
    c = Config(method=method)
    actual = reference_output(s, valid, v, changed, c, meta['vectors'])
    keep = ~changed.repeat_interleave(64,-1)[..., :150][..., None, :]
    densep = torch.softmax(torch.where(valid.any(-1,keepdim=True), s, 0.), -1) * valid
    if method == 'mass':
        probs = densep * keep; expected = probs @ v / probs.sum(-1,keepdim=True).clamp_min(1e-30)
    else:
        expected = (densep * keep) @ v
        if method == 'compensate':
            expected += densep[...,128:].sum(-1,keepdim=True) * meta['vectors'][...,2,None,:]
    assert torch.allclose(actual, expected, atol=2e-6)
    d = diagnostics(state, changed, c, meta)
    error = (actual - densep@v).square().sum((-1,-2))
    assert torch.allclose(d['error_sq'], error, atol=2e-6)


def test_constant_values_compensation_exact_but_zero_pv_not():
    s, valid, v, _, _ = example(); v[:] = 2
    meta = value_summaries(v, valid.any(-2)); state = block_state(s,valid,v)
    mask = streaming_mask(state,meta,Config(method='compensate',log_threshold=0.))
    assert mask.any()
    compensated = diagnostics(state,mask,Config(method='compensate'),meta)
    zero = diagnostics(state,mask,Config(method='zero_pv'),meta)
    assert compensated['error_sq'].max() < 1e-10
    assert zero['error_sq'].min() > 0


def test_mass_estimate_uses_retained_not_skipped_state():
    s = torch.zeros(1,1,1,192);s[...,64:128]=-5.;s[...,128:]=-2.
    valid=torch.ones_like(s,dtype=torch.bool);v=torch.ones(1,1,192,1)
    meta=value_summaries(v,valid.any(-2));state=block_state(s,valid,v)
    # block 1 is deleted; block 2 bound denominator contains block 0 only.
    expected=math.exp(-2)/(1+math.exp(-2))
    c=Config(method='mass',log_threshold=math.log(expected)-.0001)
    assert streaming_mask(state,meta,c).tolist()==[[[False,True,False]]]


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_adapter_dense_parity_gqa_and_layer_threshold(dtype):
    torch.manual_seed(0)
    runtime=Blasst2DRuntime(Blasst2DConfig());runtime.current_denoising_iteration=0
    module=SimpleNamespace(num_key_value_groups=2,layer_idx=0,training=False,_blasst_2d_runtime=runtime)
    q=torch.randn(1,2,129,8,dtype=dtype);k=torch.randn(1,1,199,8,dtype=dtype);v=torch.randn_like(k)
    mask=torch.ones(1,1,129,199,dtype=torch.bool)
    dense=dense_eager_attention_forward(module,q,k,v,mask,is_causal=False)[0]
    a=Attention();assert torch.equal(a(module,q,k,v,mask,is_causal=False)[0],dense)
    unpruned=Attention(dict(method='mass',log_threshold=-math.inf))
    assert torch.equal(unpruned(module,q,k,v,mask,is_causal=False)[0],dense)
    policy={'local':{'log_threshold':-100.},'global':{'log_threshold':100.}}
    a=Attention(dict(method='mass'),policy);a(module,q,k,v,mask,is_causal=False)
    assert sum(r['skipped'] for r in a.records)>0
    module.is_sliding=True
    a=Attention(dict(method='mass'),policy);a(module,q,k,v,mask,is_causal=False)
    assert sum(r['skipped'] for r in a.records)==0
    assert sum(r['eligible'] for r in a.records)==16


def test_gaussian_direction_and_nonnegative_topp():
    assert list(BETAS.values()) == pytest.approx([-.674490,0,.674490,1.281552],abs=1e-6)
    p=torch.tensor([[[-3.,-1.,1.,3.]]]);e=torch.ones_like(p,dtype=torch.bool)
    keep,z,_=proxy_keep(p,e,Config(method='sol',mode='gaussian',amount=.5))
    assert keep.tolist()==[[[False,False,True,True]]]
    keep,_,_=proxy_keep(p-100,e,Config(method='sol',mode='topp',amount=.9))
    assert keep.tolist()==[[[False,False,True,True]]]


def test_calibration_count_weights_ties_and_unattainable():
    import numpy as np
    from experiments.diffusion_gemma_value_aware.calibration import quantile_threshold, aggregate
    q=quantile_threshold([np.array([0.,1.,np.inf]),np.array([0.]*7+[1.,np.inf])],.5)
    assert q['eligible']==12
    assert q['achieved_dense_sparsity']==pytest.approx(8/12)
    assert q['log_threshold']>0
    q=quantile_threshold([np.array([0.,1.,np.inf])],.9)
    assert q['unattainable'] and q['achieved_dense_sparsity']==pytest.approx(2/3)
    rows=[dict(eligible=4,skipped=2,softmax_skipped=2,pv_omitted=2,compensated=0,mass_sum=1.,rows=2,error_sq=1,dense_sq=4),
          dict(eligible=12,skipped=3,softmax_skipped=3,pv_omitted=3,compensated=0,mass_sum=8.,rows=8,error_sq=0,dense_sq=5)]
    a=aggregate(rows);assert a['physical_sparsity']==5/16 and a['mass']==.9


def test_dataset_audit_excludes_heldout_from_calibration():
    from experiments.diffusion_gemma_value_aware.protocol import audit,sha
    def row(benchmark,task,i,cal=False):
        text=f'{benchmark}/{task}/{i}'
        return dict(id=text,source_id=i,benchmark=benchmark,task=task,prompt=text,prompt_hash=sha(text),calibration=cal)
    aime=[row('aime26','AIME26',i,i in (2,8,14,20,23,30)) for i in range(1,31)]
    lb=[row('longbench',t,i) for t in ('qasper','hotpotqa','gov_report','trec','passage_retrieval_en') for i in range(10)]
    cal=[row('longbench',t,i,True) for t in ('qasper','hotpotqa','gov_report','trec','passage_retrieval_en') for i in (10,11)]
    dev=[row('longbench',t,12) for t in ('qasper','hotpotqa','gov_report','trec','passage_retrieval_en')]
    setup=dict(final=aime+lb,calibration=[r for r in aime if r['calibration']]+cal,development=dev)
    assert audit(setup)['passed']
    setup['development'][0]=dict(setup['final'][-1],split='development')
    with pytest.raises(AssertionError): audit(setup)


def test_compensation_accounting_is_not_deletion():
    from experiments.diffusion_gemma_value_aware.calibration import aggregate
    s,valid,v,meta,state=example()
    router=Attention()
    mask=torch.tensor([[[False,False,True],[False,False,True]]])
    router.record(state,mask,Config(method='compensate'),meta,0,0,'local','test',70)
    records=router.records
    assert all(r['skipped']==r['softmax_skipped']==0 for r in records)
    assert all(r['prefix_skipped']==r['boundary_skipped']==r['canvas_skipped']==0 for r in records)
    a=aggregate(records)
    assert a['pv_omission']>0 and a['physical_sparsity']==0
    assert a['denominator_mass']==1 and a['mass']<1


def test_worst_row_safeguard_and_query_inflation():
    r=torch.tensor([[[[0.,-3.],[-3.,-3.]]]])
    active=torch.ones_like(r,dtype=torch.bool)
    c=Config(method='mass')
    assert aggregate_risk(r,active,c).tolist()==[[[0.,-3.]]]
    a=aggregate_risk(r,active,replace(c,aggregation='mean_safeguard',safeguard_factor=2.))
    assert a[0,0,0]>=-math.log(2.)
    assert a[0,0,1]==pytest.approx(-3.)


def test_dense_reuse_preserves_generation_and_source(tmp_path):
    import json
    from experiments.diffusion_gemma_value_aware.evaluate import reuse_dense
    row=dict(id='a',prompt_hash='h',seed=42,generation_budget=32)
    data=dict(row,fingerprint='fp',config={},thresholds=None,screen=True,completion_tokens=[1,2,3],
        records=[dict(probe='execution',eligible=8),dict(probe='diagnostic',eligible=8)],distributions=[1])
    src=tmp_path/'source.json';src.write_text(json.dumps(data))
    dest=tmp_path/'dest.json';reuse_dense(src,dest,row,'fp')
    copied=json.loads(dest.read_text())
    assert copied['completion_tokens']==[1,2,3] and not copied['screen']
    assert copied['records']==[dict(probe='execution',eligible=8)]
    assert copied['reused_dense_source']['source_screen']
    assert json.loads(src.read_text())==data
    with pytest.raises(AssertionError):reuse_dense(src,dest,dict(row,seed=99),'fp')
