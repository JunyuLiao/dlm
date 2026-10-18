import pytest
import torch
from scripts.v31_refresh_operator_replay import tensor_meta,restore,compare,compare_outputs,outputs
from scripts.v31_refresh_followup import available,attribute,union


def test_layout_roundtrip_preserves_offset_and_stride():
    source=torch.arange(200,dtype=torch.float32).as_strided((2,3,4),(50,4,1),7)
    result=restore(source.clone(),tensor_meta(source),'cpu')
    assert result.stride()==source.stride() and result.storage_offset()==7
    assert torch.equal(source,result)


def test_difference_reports_magnitude_and_nonfinite_patterns():
    a=torch.tensor([1.,2.,float('-inf')]);b=torch.tensor([1.,2.01,float('-inf')])
    r=compare(a,b);assert not r['exact'] and r['max_abs']==pytest.approx(.01,abs=1e-6)
    assert r['nonfinite_pattern_equal']
    assert not compare(a,torch.ones(3))['nonfinite_pattern_equal']


def test_inactive_partial_storage_is_excluded_from_semantic_comparison():
    lse=torch.tensor([[[[0.,float('-inf')]]]])
    a={'out_partial':torch.tensor([[[[[1.],[99.]]]]]).reshape(1,1,2,1,1),'lse_partial':lse}
    b={k:v.clone() for k,v in a.items()};b['out_partial'][0,0,1,0,0]=-999
    assert compare_outputs(outputs(a),outputs(b))['out_partial_active']['exact']


def test_budget_includes_failures_and_orphan_reservations():
    assert available('capture',[])==900
    rows=[dict(phase='capture',reserved_s=900,elapsed_s=800,status='failed')]
    assert available('capture',rows)==100
    rows[0]['elapsed_s']=None;assert available('capture',rows)==0
    assert available('recovery',rows)==300


def test_attribution_links_launch_not_gpu_timestamp_and_preserves_ancestry():
    ranges=[dict(id=0,start=10,end=50,thread=7),dict(id=1,start=20,end=40,thread=7)]
    metadata={0:dict(request=3,label='denoise/global_refresh',parent=None),1:dict(request=3,label='denoise/observation',parent=0)}
    apis=[dict(start=25,end=30,thread=7,correlation=5)]
    activities=[dict(start=100,end=200,correlation=5),dict(start=180,end=230,correlation=6)]
    result=attribute(ranges,apis,activities,metadata)
    assert result['unlinked_activities']==1
    r={x['label']:x for x in result['rows']}
    assert r['denoise/global_refresh']['gpu_sum_ns']==100
    assert r['denoise/observation']['gpu_sum_ns']==100
    assert r['denoise/global_refresh']['cpu_ns']==40


def test_union_does_not_double_count_overlap():
    assert union([(0,10),(5,15),(30,35)])==20
    with pytest.raises(ValueError):union([(2,1)])
