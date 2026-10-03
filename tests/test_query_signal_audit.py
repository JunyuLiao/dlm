import json
from pathlib import Path
import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from experiments.value_direction_hopper.query_signal_audit import _derive, load_condition, row_metrics
from experiments.value_direction_hopper.query_adaptive import State


def test_vector_metrics_match_sklearn_with_ties_and_mask():
    rng=np.random.default_rng(7)
    score=rng.integers(0,5,size=(10,16))/4
    y=rng.random((10,16,3))<.35
    mask=rng.random((10,16))<.8
    result=row_metrics(y,score,mask)
    for i in range(10):
        for j in range(3):
            yy=y[i,mask[i],j]; ss=score[i,mask[i]]
            if len(np.unique(yy))==2:
                assert result[i,j,0]==pytest.approx(roc_auc_score(yy,ss))
            if yy.any():
                assert result[i,j,1]==pytest.approx(average_precision_score(yy,ss))
            assert result[i,j,5]==pytest.approx(np.mean((ss-yy)**2))


def test_constant_scores_have_random_precision_and_auc():
    y=np.array([[[True],[False],[False],[True]]])
    m=row_metrics(y,np.ones((1,4)))
    np.testing.assert_allclose(m[0,0,:5],[.5,.5,.5,.5,.5])
    empty=row_metrics(y,np.ones((1,4)),np.zeros((1,4),bool))
    assert np.isnan(empty).all()


@pytest.mark.parametrize('method', ['M_prior','C_prior','T_prior','T_smooth','T_hybrid','T_run','M_gate','C_gate','T_gate','H_anchor','C_soft','C_tail','C_gate_soft'])
def test_reconstruction_matches_production_next_call(method):
    torch.manual_seed(7)
    q=8; steps=6
    logits=torch.randn(steps,1,q,5)
    accepted=torch.rand(steps,1,q)>.3
    gamma=.65 if method.endswith('gate') or method in ('H_anchor','C_soft','C_tail','C_gate_soft') else .8 if method=='T_smooth' else .5
    state=State(method,None,m_ref=14.258454322814941,trajectory_gamma=gamma,
                gate_tau=3.5 if method=='T_gate' else 2.5,diagnostics=False)
    arr={k:[] for k in ('accepted','top','confidence','raw_confidence','margin','entropy','previous_winner_confidence')}
    canvas=torch.zeros((1,q),dtype=torch.long)
    state.begin(48,canvas)
    expected=[]
    for i in range(steps):
        x=logits[i]; temp=.4+.4*(48-i)/48
        state.observe_logits(x,accepted[i],48-i)
        for k,v in dict(accepted=accepted[i],top=x.argmax(-1),confidence=state.confidence,
              raw_confidence=(x*temp).softmax(-1).max(-1).values,margin=state.margin,
              entropy=torch.distributions.Categorical(logits=x).entropy(),
              previous_winner_confidence=state.confidence).items():
            arr[k].append(v.numpy().reshape(q))
        state.begin(47-i,canvas)
        expected.append((state.used_weights.numpy().reshape(q)-1)/3)
    derived=_derive({k:np.stack(v) for k,v in arr.items()})
    np.testing.assert_allclose(derived[method],expected,atol=2e-7)


def test_canvas_reset_and_complete_horizons(tmp_path):
    root=tmp_path/'traces/dense/seed42'; root.mkdir(parents=True)
    shape=(6,2)
    arr=dict(canvas=np.array([0,0,0,1,1,1]),call=np.array([1,2,3,1,2,3]),
        accepted=np.ones(shape,bool),top=np.array([[0,0],[0,1],[0,1],[8,8],[8,8],[8,9]]),
        confidence=np.full(shape,.9),margin=np.ones(shape),entropy=np.ones(shape),
        raw_confidence=np.full(shape,.8),previous_winner_confidence=np.full(shape,.9))
    (root/'1.json').write_text(json.dumps(dict(seed=42,id='aime26/1')))
    np.savez(root/'1.npz',**arr)
    data=load_condition(tmp_path,'dense',horizon=2)
    assert data['meta'].tolist()==[[1,42,0,2],[1,42,1,2]]
    # First observation in each canvas resets q to .5 for an accepted token.
    from experiments.value_direction_hopper.query_signal_audit import FEATURES
    np.testing.assert_array_equal(data['x'][:,:,FEATURES.index('q_g05')],.5)
    assert data['y'][:,:,1].tolist()==[[False,True],[False,True]]
    assert not data['y'][:,:,0].any()
    with pytest.raises(ValueError,match='one canvas'):
        _derive(arr)
