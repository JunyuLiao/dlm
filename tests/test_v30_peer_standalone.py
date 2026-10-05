from types import SimpleNamespace as NS
import math
import pytest
import torch
from experiments.numerical_qk_reuse.v30_peer_standalone import QueryWeights,validate_policy,check_standalone_receipt,StandalonePeerAdapter,GLOBAL_LAYERS


def test_confidence_is_previous_only_resets_and_does_not_change_logits():
    w=QueryWeights('confidence');torch.testing.assert_close(w.begin(0,(1,2),'cpu'),torch.full((1,2),4.))
    x=torch.log(torch.tensor([[[.5,.25,.25],[.8,.1,.1]]]))
    old=x.clone();w.observe(x);torch.testing.assert_close(x,old,rtol=0,atol=0)
    torch.testing.assert_close(w.begin(1,(1,2),'cpu'),1+3*torch.sqrt(torch.tensor([[.5,.2]])))
    w.observe(x);torch.testing.assert_close(w.begin(0,(1,3),'cpu'),torch.full((1,3),4.))


def test_unit_reads_no_logits_and_temporal_uses_only_prior_flips():
    w=QueryWeights('unit');assert w.begin(0,(1,2),'cpu') is None;w.observe(object())
    t=QueryWeights('temporal');assert t.begin(0,(1,2),'cpu') is None
    t.observe(torch.tensor([[[1.,0.],[1.,0.]]]));torch.testing.assert_close(t.begin(1,(1,2),'cpu'),torch.ones(1,2))
    t.observe(torch.tensor([[[0.,1.],[1.,0.]]]));torch.testing.assert_close(t.begin(2,(1,2),'cpu'),torch.tensor([[2.5,1.]]))


def test_clock_and_policy_fail_closed():
    w=QueryWeights('confidence')
    with pytest.raises(RuntimeError):w.observe(torch.zeros(1,2,3))
    with pytest.raises(ValueError):w.begin(2,(1,2),'cpu')
    w.begin(0,(1,2),'cpu')
    with pytest.raises(RuntimeError):w.begin(1,(1,2),'cpu')
    with pytest.raises(ValueError):w.observe(torch.zeros(1,3,3))
    w.observe(torch.zeros(1,2,3))
    with pytest.raises(ValueError):w.begin(1,(1,3),'cpu')
    assert validate_policy('value_allkept',None)==-math.inf
    with pytest.raises(ValueError):validate_policy('value_unit',None)
    with pytest.raises(ValueError):validate_policy('kernel_dense',{'global_log_threshold':-3})


class FakeCache:
    def __init__(self):self.projected_tokens=0;self.reused_tokens=0;self.closed=False
    def close(self):self.closed=True


class FakeRouter:
    def __init__(self,adapter,library,thresholds,**kw):
        self.cache=FakeCache();self.pending=[];self.calls=0;self.invocations=[];self.query_sensitivity=None;self.geometry={}
    def __call__(self,module,q,k,v,mask,**kwargs):
        self.calls+=1;self.invocations.append(dict(layer=module.layer_idx,kwargs=kwargs,shape=tuple(k.shape)))
        return q.transpose(1,2).contiguous()+2,None
    def records(self):return [dict(layer=5,skipped=0,eligible=1)]


def adapter(variant='value_allkept'):
    types=['full_attention' if i in GLOBAL_LAYERS else 'sliding_attention' for i in range(30)]
    return StandalonePeerAdapter(types,peer_module=NS(Attention=FakeRouter),library='unused',variant=variant)


def test_no_m3_install_forward_identity_scale_and_native_layer_exclusion(monkeypatch):
    from experiments.numerical_qk_reuse import v21
    monkeypatch.setattr(v21,'install',lambda *a,**kw:pytest.fail('M3 must not be installed'))
    a=adapter();a.begin_request();assert a.runtime is None
    a.canvas=torch.zeros(1,2,dtype=torch.long);a.on_prepare(False,0,4,2)
    assert a.active_for('model.layers.5.self_attn.attn')==5
    assert a.active_for('model.layers.4.self_attn.attn') is None
    q=torch.randn(2,2,4);out=torch.empty_like(q);kv=torch.randn(1,1,4,8)
    impl=NS(head_size=4,scale=1.,alibi_slopes=None,logits_soft_cap=0)
    metadata=NS(num_actual_tokens=2,block_table=torch.tensor([[0]]))
    assert a.forward(impl,5,q,kv,metadata,out) is out
    torch.testing.assert_close(out,q+2)
    assert a.router.invocations[0]['kwargs']==dict(scaling=1.,is_causal=False,sliding_window=None)
    assert a.router.invocations[0]['shape']==(1,1,4,4)
    with pytest.raises(ValueError):a.forward(impl,4,q,kv,metadata,out)
    impl.scale=.5
    with pytest.raises(ValueError):a.forward(impl,5,q,kv,metadata,out)
    a.on_sample(torch.zeros(1,4,3));r=a.end_request()
    assert r['standalone_peer']['m3_installed'] is False and r['standalone_peer']['weight_observes']==1
    assert not a.bound and a.runtime is None and a.router is None and not a.buffers


def test_cleanup_even_when_allkept_receipt_detects_skips():
    a=adapter();a.begin_request();router=a.router;router.records=lambda:[dict(skipped=1)]
    with pytest.raises(RuntimeError,match='skipped'):a.end_request()
    assert not a.bound and a.router is None and router.cache.closed and a.stub is None


def test_bare_dense_calls_dense_kernel_without_value_router_and_releases_canvas_storage():
    a=adapter('kernel_dense');geometry_calls=[]
    a.peer_module.geometry=lambda *args,**kwargs:(geometry_calls.append(args) or object(),None)
    a.begin_request();a.canvas=torch.zeros(1,2,dtype=torch.long);a.on_prepare(False,0,4,2)
    invocations=[]
    def kernel(q,k,v,z,ref,**kwargs):
        invocations.append(kwargs);return NS(output=q+3,skipped=torch.zeros(1,dtype=torch.bool),eligible=torch.ones(1,dtype=torch.bool))
    a.router.kernel=kernel
    q=torch.randn(2,2,4);out=torch.empty_like(q);kv=torch.randn(1,1,4,8)
    impl=NS(head_size=4,scale=1.,alibi_slopes=None,logits_soft_cap=0)
    metadata=NS(num_actual_tokens=2,block_table=torch.tensor([[0]]))
    for layer in (5,11):a.forward(impl,layer,q,kv,metadata,out)
    torch.testing.assert_close(out,q+3)
    assert not a.router.invocations and len(geometry_calls)==1
    assert all(x['mode']=='dense' and x['log_threshold']==-math.inf and x['scale']==1. for x in invocations)
    a.on_sample(torch.zeros(1,2,3));a.on_prepare(True,1,4,2);assert not a.dense_dummy
    r=a.end_request()['standalone_peer'];assert r['calls_by_layer'][5]==r['calls_by_layer'][11]==1
    assert r['projected_tokens']==0


def test_shape_and_scope_configuration_rejected():
    with pytest.raises(ValueError):StandalonePeerAdapter(['full_attention'],peer_module=None,library='x',variant='kernel_dense')
    with pytest.raises(ValueError):QueryWeights('cgate')


def test_policy_requires_public_git_provenance():
    p=dict(global_log_threshold=-4.,source_commit='b'*40,source_file='results/policy.json',calibration_scope='historical development only')
    assert validate_policy('value_unit',p)==-4.
    for delta in (dict(source_commit='main'),dict(source_file='../../private.json'),
                  dict(source_file='E:/private.json'),dict(global_log_threshold=math.nan)):
        with pytest.raises(ValueError):validate_policy('value_unit',dict(p,**delta))


def test_receipt_rejects_fallback_missing_layer_missing_sample_and_unexpected_projection():
    from copy import deepcopy
    r=dict(method=None,adapter=dict(global_calls=10,begins=2,observes=2,order_errors=0),standalone_peer=dict(
        variant='value_allkept',consumer='Junyu_SM90_C_ABI',scope='five_GLOBAL_decoder_layers_only',
        m3_installed=False,dp=False,carry0=False,score_cache=False,calls=10,weight_begins=2,weight_observes=2,
        calls_by_layer={str(k):2 for k in GLOBAL_LAYERS},query_weights='unit',kernel_dense_mode=False,
        projected_tokens=20,tile_counts=[dict(layer=k,skipped=0,eligible=4) for k in GLOBAL_LAYERS],threshold=None))
    check_standalone_receipt(r,2,'value_allkept')
    mutations=[('consumer','FA4'),('m3_installed',True),('calls_by_layer',{'5':10}),
               ('weight_observes',1),('projected_tokens',0),('query_weights','confidence'),
               ('tile_counts',[dict(layer=k,skipped=1,eligible=4) for k in GLOBAL_LAYERS])]
    for key,value in mutations:
        bad=deepcopy(r);bad['standalone_peer'][key]=value
        with pytest.raises(ValueError):check_standalone_receipt(bad,2,'value_allkept')
    r['standalone_peer'].update(variant='value_unit',threshold=-4.)
    check_standalone_receipt(r,2,'value_unit')
    # Zero observed sparsity is a valid negative result, not a routing failure.
    r['standalone_peer'].update(variant='kernel_dense',threshold=None,kernel_dense_mode=True,projected_tokens=0)
    check_standalone_receipt(r,2,'kernel_dense')


def test_cleanup_when_peer_close_itself_fails():
    a=adapter();a.begin_request()
    def fail():raise RuntimeError('close failed')
    a.router.cache.close=fail
    with pytest.raises(RuntimeError,match='close failed'):a.end_request()
    assert not a.bound and a.router is None and a.stub is None and not a.buffers
