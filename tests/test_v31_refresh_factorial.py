"""Factor bindings, reset clocks, and an independently derived selector oracle."""
import numpy as np
import pytest
import torch

from experiments.numerical_qk_reuse.v31_stable_refresh import StableState
from scripts.v31_refresh_factor_analysis import contrast, factorial
from scripts.v31_refresh_factorial_replay import ReplayClock, configurations, NumpyOracle


SELECTOR = dict(mode='stable', hysteresis=0., persistence=2, event_cap=2,
                cumulative_cap=8, removal_tolerance=.025, updates_enabled=True, probes=0)


def test_factorial_is_complete_and_unconfounded():
    grid = factorial()
    assert len({r['arm'] for r in grid}) == 16
    assert {(r['mode'], r['progress_dense'], r['interval']) for r in grid} == {
        (mode, progress, period) for mode in ('raw', 'stable')
        for progress in (True, False) for period in (1, 2, 4, 8)}
    configs = configurations(SELECTOR)
    assert len(configs) == 18
    for entry in grid:
        cfg, period, progress = configs[entry['arm']]
        assert cfg.mode == entry['mode'] and period == entry['interval'] and progress == entry['progress_dense']
        assert cfg.probes == 0 and cfg.hysteresis == 0


@pytest.mark.parametrize('progress',[True,False])
@pytest.mark.parametrize('period',[1,2,4,8])
def test_frequency_counts_reuse_steps_and_only_progress_resets(progress, period):
    clock = ReplayClock(period, progress)
    assert clock.step(False) == ('preanchor', False)
    assert clock.step(True) == ('initial', False)
    due = []
    for step in range(1,18):
        reason, refresh = clock.step(step == 6)
        if refresh:
            due.append(step)
        if step == 6:
            assert reason == ('progress' if progress else 'reuse')
    expected = [step for step in range(1,18)
                if (step != 6 and (step if step < 6 else step - 6) % period == 0)] if progress else list(range(period,18,period))
    assert due == expected


@pytest.mark.parametrize('arm', [r['arm'] for r in factorial()])
def test_each_factor_binding_matches_independent_numpy_oracle(arm):
    cfg, period, progress = configurations(SELECTOR)[arm]
    rng = np.random.default_rng(73)
    anchor = rng.normal(size=(3,15,128)).astype(np.float32)
    selected = np.zeros((3,15),dtype=bool); selected[:,:4] = True
    gpu = StableState(torch.from_numpy(anchor),torch.from_numpy(selected),cfg)
    oracle = NumpyOracle(anchor,selected,cfg)
    clock = ReplayClock(period,progress); clock.step(True)
    for step in range(1,26):
        kind,due = clock.step(step == 12)
        raw = anchor + rng.normal(0,1,anchor.shape).astype(np.float32)
        if kind == 'progress':
            gpu = StableState(torch.from_numpy(raw),gpu.selected,cfg)
            oracle = NumpyOracle(raw,oracle.selected,cfg)
        elif due:
            fresh = np.where(gpu.selected.numpy()[...,None],raw,-np.inf).astype(np.float32)
            tail = np.full((3,128),5.,dtype=np.float32)
            got,info = gpu.observe(torch.from_numpy(fresh),torch.from_numpy(tail))
            expected,score = oracle.observe(fresh,tail)
            assert np.array_equal(got.numpy(),expected)
            np.testing.assert_allclose(info['score'].numpy(),score,atol=2e-6,rtol=1e-6)
            assert (got.sum(1)==4).all()
            if cfg.mode == 'stable':
                assert info['swaps'].max() <= 2 and gpu.cumulative.max() <= 8


def test_contrast_requires_same_cells_and_clusters_seeds_by_problem():
    a = {'a':dict(index=0,denoise_forwards=2,canvases=1,output_tokens=10),
         'b':dict(index=0,denoise_forwards=2,canvases=1,output_tokens=10)}
    b = {key:dict(row,denoise_forwards=4) for key,row in a.items()}
    result = contrast(a,b,dict(a=True,b=False),dict(a=False,b=False))
    assert result['problem_clusters'] == 1
    assert result['accuracy_delta_pp'] == 50 and result['N_ratio'] == .5
    with pytest.raises(ValueError):
        contrast(a,{'a':b['a']},dict(a=True,b=False),dict(a=False))
