from experiments.diffusion_gemma_aime26_modes.protocol import final_response, MODES, CONDITIONS, BUDGET, numeric_score
from experiments.diffusion_gemma_aime26_modes.run import request


def test_thought_is_not_answer():
    text='<|channel>thought intermediate \\boxed{42}'
    assert final_response(text,True)==''
    assert not numeric_score(final_response(text,True),'42')['correct']


def test_final_channel_only():
    text='<|channel>thought \\boxed{42}<channel|>The answer is \\boxed{17}<turn|>'
    assert numeric_score(final_response(text,True),'17')['correct']
    assert not numeric_score(final_response(text,True),'42')['correct']


def test_non_thinking_answer():
    assert numeric_score(final_response('\\boxed{007}<turn|>',False),'7')['correct']


def test_full_grid_and_native_thinking_request():
    assert len(MODES)*len(CONDITIONS)*30==600 and BUDGET==32768
    for mode in MODES:
        thinking=mode.endswith('_think')
        r=request(dict(prompt='question',thinking=thinking,generation_budget=BUDGET,seed=42))
        assert r.extra['thinking']==thinking and r.max_new_tokens==32768 and r.temperature==0.


def test_dense_counters_do_not_count_counterfactual_skips(monkeypatch):
    import torch
    from types import SimpleNamespace
    from experiments.diffusion_gemma_aime26_modes import run
    q=torch.randn(1,2,65,4);k=torch.randn(1,2,130,4);v=k.clone()
    scores=q@k.transpose(-1,-2);valid=torch.ones_like(scores,dtype=torch.bool)
    monkeypatch.setattr(run,'_prepare_attention_scores',lambda *a,**kw:(k,v,scores,valid))
    monkeypatch.setattr(run,'_finish_eager_attention',lambda *a,**kw:(q,None))
    monkeypatch.setattr(run,'_attention_type',lambda *a:'global')
    module=SimpleNamespace(layer_idx=0,training=False,_blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    stat=run.DenseCounter();stat(module,q,k,v,None)
    c=stat.calls[0]
    assert c['eligible_tiles']==12 and c['skipped_tiles']==0
    assert c['valid_rows']==130 and c['retained_dense_attention_mass']==1.
    assert c['region_counts']['prefix']['eligible_tiles']==8
    assert c['region_counts']['canvas']['eligible_tiles']==4
