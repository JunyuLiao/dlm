from scripts.v31_refresh_replay_bootstrap import boot_observer


def test_patch_precedes_observer_import():
    steps=[]
    def patch():steps.append('patch');return 'source.py'
    def load():
        assert steps==['patch'];steps.append('import');return 'observer'
    assert boot_observer(patch,load)==('source.py','observer')
    assert steps==['patch','import']


def test_fast_metrics_match_original_exact_and_nonexact(monkeypatch):
    import torch
    from scripts import v31_refresh_operator_replay as old
    from scripts import v31_refresh_replay_metrics_fast as fast
    monkeypatch.setattr(fast,'_original_compare',old.compare)
    pairs=[(torch.tensor([0.,1.,float('nan'),float('-inf')]),torch.tensor([0.,1.,float('nan'),float('-inf')])),
           (torch.tensor([0.,1.]),torch.tensor([-0.,1.])),
           (torch.tensor([1.,2.]),torch.tensor([1.,2.01])),
           (torch.tensor([True,False]),torch.tensor([True,False])),
           (torch.ones(1),torch.ones(2))]
    for a,b in pairs:assert fast.compare_fast(a,b)==old.compare(a,b)


def test_fast_partial_extraction_matches_active_and_inactive_slots():
    import torch
    from scripts import v31_refresh_operator_replay as old
    from scripts import v31_refresh_replay_metrics_fast as fast
    for lse in (torch.zeros(1,1,1,2),torch.tensor([[[[0.,float('-inf')]]]])):
        tensors=dict(lse_partial=lse,out_partial=torch.arange(4.).reshape(1,1,2,1,2),out=torch.ones(2))
        a,b=old.outputs(tensors),fast.outputs_fast(tensors)
        assert a.keys()==b.keys()
        for k in a:assert torch.equal(a[k],b[k])
