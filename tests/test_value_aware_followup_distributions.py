import numpy as np
import pytest

from experiments.diffusion_gemma_value_aware_followup.distribution_diagnostics import (
    pool_moments, quantile_summary, guarded_sol_rows)


def bucket(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=x.mean(), std=x.std(),
        skew=(x**3).mean(), fourth_moment=(x**4).mean())


def test_proxy_moments_pool_counts_before_centralization():
    a = np.array([-2., -1., 0., 1., 2.]*20)
    b = np.array([12., 15.])
    out = pool_moments([bucket(a), bucket(b)])
    x = np.concatenate([a, b])
    centered = (x-x.mean())/x.std()
    assert out['count'] == 102
    assert out['mean'] == pytest.approx(x.mean())
    assert out['std'] == pytest.approx(x.std())
    assert out['central_skew'] == pytest.approx((centered**3).mean())
    assert out['excess_kurtosis'] == pytest.approx((centered**4).mean()-3.)
    assert out['mean'] != pytest.approx((a.mean()+b.mean())/2)


def test_degenerate_proxy_moments_are_not_gaussian_claims():
    out = pool_moments([bucket([0.]*10)])
    assert out['central_skew'] is None and out['excess_kurtosis'] is None
    with pytest.raises(ValueError, match='positive'):
        pool_moments([])
    with pytest.raises(ValueError, match='nonfinite'):
        pool_moments([dict(bucket([1.]), fourth_moment=float('nan'))])


def test_quantile_averages_explicitly_are_not_pooled_quantiles():
    rows = [dict(count=100, p10=0., p50=1., p90=2., p99=3.),
        dict(count=1, p10=10., p50=20., p90=30., p99=40.)]
    out = quantile_summary(rows)
    assert out['count_weighted_bucket_p50'] == pytest.approx(120/101)
    assert out['min_bucket_p50'] == 1. and out['max_bucket_p50'] == 20.
    assert 'NOT pooled quantiles' in out['interpretation']
    with pytest.raises(ValueError, match='positive-count'):
        quantile_summary([])


def test_sol_screen_uses_only_corrected_masks_and_exact_raw_counts():
    common = dict(benchmark='longbench_v2', split='development', attention_type='overall',
        eligible=1000., skipped=750., rows=100., mass=.9, relative_error=.1, physical_sparsity=.75)
    probe = 'guarded_sol/plain/gaussian/s75'
    counts = {('longbench_v2', 'development', probe, 'overall'):
        dict(eligible=1000., skipped=750., rows=100., rescued_rows=2.)}
    rows = [dict(common, probe=probe), dict(common, probe='sol/plain/gaussian/s75')]
    out = guarded_sol_rows(rows, counts)
    assert len(out) == 1
    assert out[0]['beta'] == pytest.approx(.6744897501960817)
    assert out[0]['rescued_row_fraction'] == .02
    assert out[0]['actual'] == .75
    rows[0]['skipped'] = 749
    with pytest.raises(ValueError, match='raw counters'):
        guarded_sol_rows(rows, counts)


def test_distribution_diagnosis_refuses_final_split():
    with pytest.raises(ValueError, match='final examples'):
        guarded_sol_rows([dict(split='final')], {})
