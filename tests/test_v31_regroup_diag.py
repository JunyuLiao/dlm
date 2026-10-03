"""v31 regroup-potential diagnostic: kept fractions at 128 rows, 64-row halves, regrouped halves and per row."""
from types import SimpleNamespace

import torch


def test_regroup_kept_fractions_on_a_constructed_need_matrix():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='allkept', regroup_diag=True)
    a._regroup_acc, a._regroup_n = None, 0
    b, h, qb, pt = 1, 2, 1, 4
    need = torch.zeros(b, h, qb, 128, pt, dtype=torch.bool)
    rows_a = torch.arange(0, 128, 2)                     # interleaved rows: even rows need tile 0, odd rows tile 1
    need[:, :, :, rows_a, 0] = True
    need[:, :, :, rows_a + 1, 1] = True
    need[:, :, :, rows_a[:3] + 1, 1] = True
    lognorm = torch.where(need.permute(0, 1, 2, 4, 3), torch.tensor(0.0), torch.tensor(-10.0))
    state = SimpleNamespace(lognorm=lognorm, eligible=torch.ones(b, h, qb, pt, dtype=torch.int8))
    a._regroup_account(state, torch.ones(b, 1), None, -1.0, 128)
    r = a._regroup_receipt()
    assert r['kept128'] == 0.5                           # tiles 0 and 1 of 4
    assert r['kept64'] == 0.5                            # both halves still need both tiles (interleaved rows)
    # sorting by need count cannot separate (all rows need one tile); the random-projection order groups even and
    # odd rows apart, so each half needs one tile
    assert r['kept64_regroup_count'] == 0.5
    assert r['kept64_regroup_proj'] == 0.25
    assert abs(r['kept_per_row'] - 0.25) < 1e-6
