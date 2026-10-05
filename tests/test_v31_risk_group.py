"""v31 group-shared method selection (RISK_GROUP=kv): one top-k decision per (KV head, block) on the max of the group's
worst-row values (CPU; needs the deployment's torch + triton import, no GPU)."""
import torch

from experiments.numerical_qk_reuse import v27_dense_prefix as dp
from experiments.numerical_qk_reuse.v31_group_select import grouped_topk_skip

B, H, HK, QB, PT, NQ = 1, 16, 2, 2, 40, 256
G = H // HK


def _inputs(seed):
    g = torch.Generator().manual_seed(seed)
    lognorm = torch.randn(B, H, QB, PT, 128, generator=g) - 5.0
    eligible = torch.ones(B, H, QB, PT, dtype=torch.int8)
    sensitivity = torch.ones(B, NQ)
    reference = torch.rand(B, HK, generator=g) + 0.5
    return lognorm, eligible, sensitivity, reference


def test_identical_heads_reproduce_the_per_head_rule():
    lognorm, eligible, sensitivity, reference = _inputs(1)
    lognorm = lognorm[:, ::G].repeat_interleave(G, dim=1)          # every head of a group sees the same values
    per_head = dp.topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    grouped = grouped_topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    assert torch.equal(per_head, grouped)


def test_one_shared_set_per_group_with_the_per_head_budget():
    lognorm, eligible, sensitivity, reference = _inputs(2)
    skip = grouped_topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    per_head = dp.topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    n_drop = PT - (PT - int((1 - 0.12) * PT + 1e-9))
    for grp in range(HK):
        assert torch.equal(skip[:, grp * G:(grp + 1) * G], skip[:, grp * G:grp * G + 1].expand(-1, G, -1, -1))
    assert bool((skip.sum(-1) == n_drop).all()) and bool((per_head.sum(-1) == n_drop).all())   # same work
    assert not torch.equal(skip, per_head)                          # independent heads: the sets do differ


def test_a_needle_of_one_head_is_kept_for_its_group():
    lognorm, eligible, sensitivity, reference = _inputs(3)
    lognorm[0, 5, 1, 17, 77] = 10.0                                 # head 5 (group 0), block 1, one row: tile 17
    skip = grouped_topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    assert not bool(skip[0, :G, 1, 17].any())


def test_a_tile_any_head_must_keep_is_never_dropped_for_the_group():
    lognorm, eligible, sensitivity, reference = _inputs(4)
    lognorm[:, :G, :, 9] = -50.0                                    # tile 9 is the least useful for group 0 ...
    eligible[0, 3, 0, 9] = 0                                        # ... but head 3 may not drop it in block 0
    lognorm[0, 6, 1, 9, 0] = float('inf')                           # and head 6 marks it first-support in block 1
    skip = grouped_topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)
    assert not bool(skip[0, :G, 0, 9].any()) and not bool(skip[0, :G, 1, 9].any())
    assert bool(dp.topk_skip(lognorm, eligible, sensitivity, reference, NQ, 0.12)[0, 0, 0, 9])


def test_padded_rows_do_not_decide():
    lognorm, eligible, sensitivity, reference = _inputs(5)
    nq = 200                                                        # block 1 has 72 real rows
    lognorm[0, 2, 1, 23, 100] = 10.0                                # a padded row of head 2, block 1
    lognorm[0, :G, 1, 23] = -50.0
    lognorm[0, 2, 1, 23, 100] = 10.0
    skip = grouped_topk_skip(lognorm, eligible, torch.ones(B, nq), reference, nq, 0.12)
    assert bool(skip[0, :G, 1, 23].all())


def test_adapter_refuses_configs_it_would_silently_ignore():
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    bad = [dict(arm='mage', risk_group='kv'),
           dict(arm='method', config={'decision_interval': 6}, condition='x', risk_group='kv'),
           dict(arm='method', config={'risk_topk': 'k12', 'q_block': 64}, condition='x', risk_group='kv'),
           dict(arm='method', config={'risk_topk': 'k12'}, condition='x', risk_group='qhead')]
    for kw in bad:
        try:
            VllmMethodAdapter(types, **kw)
            raise AssertionError(f'accepted {kw}')
        except ValueError:
            pass
