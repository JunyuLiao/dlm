import json
import pytest
from scripts.v31_value_investigation_cost_report import trace
from scripts.v31_value_pause_analysis import summaries


def test_profiler_excludes_annotation_spans_from_activity(tmp_path):
    events=[dict(cat='kernel',name='a',ts=1000,dur=100),
            dict(cat='kernel',name='b',ts=1200,dur=100),
            dict(cat='gpu_memcpy',name='c',ts=1050,dur=75),
            dict(cat='gpu_user_annotation',name='stage:x',ts=1000,dur=300),
            dict(cat='user_annotation',name='stage:x',ts=0,dur=500)]
    path=tmp_path/'trace.json';path.write_text(json.dumps(dict(traceEvents=events)))
    result=trace(path)
    assert result['kernels']==2
    assert result['activity_union_ms']==pytest.approx(.225)
    assert result['activity_gap_ms']==pytest.approx(.075)
    assert result['stages']['stage:x']['gpu_activity_union_ms']==pytest.approx(.225)


def test_trajectory_product_decomposition_uses_ratio_of_totals():
    cells=[dict(control=dict(N=20,C=2,T=500,S=1,W=2,accuracy=1,cap=0),
                candidate=dict(N=60,C=3,T=750,S=3,W=4,accuracy=1,cap=0)),
           dict(control=dict(N=80,C=8,T=2000,S=4,W=5,accuracy=0,cap=0),
                candidate=dict(N=20,C=2,T=500,S=1,W=2,accuracy=0,cap=0))]
    result=summaries(cells)
    assert result['calls_per_canvas']==[10,16]
    assert result['N_decomposition']['canvas']==-32.5
    assert result['N_decomposition']['within_canvas']==22.5
    assert sum(result['N_decomposition'].values())==result['delta']['N']==-10
