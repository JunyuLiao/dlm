"""Qualification for the pre-QK current-output consumer.

The claim under test is narrow and checkable: on the SAME Q/K/V and the SAME
retained support, the consumer (a) produces the current attention output, (b)
is no further from an independent FP32 reference than the incumbent
materialized-score path is, and (c) physically issues no QK dot and no K/V
load for a dropped tile. Errors are asserted against the incumbent's own
distance to FP32, not against a hand-tuned tolerance.
"""
import math

import pytest
import torch

from scripts.preqk_qualification import expected_programs, fp32_reference, one_case, relative

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

LOCAL = dict(h=16, hk=8, d=256)     # sliding layers, head_dim 256, GQA 2:1
GLOBAL = dict(h=16, hk=2, d=512)    # full layers, global_head_dim 512, GQA 8:1


@pytest.mark.parametrize('name,kwargs', [
    ('local', dict(**LOCAL, nq=256, nk=1024, window=1024, threshold=-1.0099318265914916,
                   seed=41, drop_fraction=.4)),
    ('global', dict(**GLOBAL, nq=200, nk=1536, window=None, threshold=-3.1366905212402343,
                    seed=43, drop_fraction=.9)),
])
def test_dropped_tiles_issue_no_qk_dot_and_no_kv_load(name, kwargs):
    row = one_case(name, **kwargs)
    assert row['skipped_tiles'] > 0, 'case must actually drop tiles to prove anything'
    # Exact: every retained tile once per 16-row program, dropped tiles never.
    assert row['qk_tile_dots'] == row['expected_tile_programs']
    assert row['kv_tile_loads'] == row['expected_tile_programs']
    assert row['executed_tile_programs'] == row['expected_tile_programs']


@pytest.mark.parametrize('name,kwargs', [
    ('local_window', dict(**LOCAL, nq=256, nk=384, window=1024, threshold=-1.0, seed=11)),
    ('local_narrow', dict(**LOCAL, nq=256, nk=512, window=64, threshold=-1.0, seed=14)),
    ('local_partial', dict(**LOCAL, nq=200, nk=421, window=1024, threshold=-1.0, seed=13)),
    ('global_long', dict(**GLOBAL, nq=256, nk=2048, window=None, threshold=-3.0, seed=22)),
    ('global_dropped', dict(**GLOBAL, nq=256, nk=1024, window=None, threshold=-3.0,
                            seed=42, drop_fraction=.6)),
    ('d64_gqa1', dict(h=4, hk=4, d=64, nq=128, nk=192, window=None, threshold=-1.0, seed=31)),
])
def test_preqk_is_not_further_from_fp32_than_the_incumbent_path(name, kwargs):
    row = one_case(name, **kwargs)
    # The incumbent BF16 path is not ground truth. What must hold is that
    # swapping in the pre-QK consumer does not move us further from FP32.
    assert row['preqk_vs_fp32'] <= row['incumbent_vs_fp32'] + 1e-6, row
    # And the two BF16 paths agree to BF16 accumulation-order scale.
    assert row['preqk_vs_incumbent'] < 1e-3, row


def test_all_kept_agrees_with_declared_dense_mathematics():
    """With nothing dropped the consumer must be plain masked attention."""
    row = one_case('all_kept', **LOCAL, nq=256, nk=384, window=1024,
                   threshold=-1.0, seed=15, all_kept=True)
    assert row['retained_tiles'] == row['eligible_tiles']
    assert row['preqk_vs_fp32'] <= row['incumbent_vs_fp32'] + 1e-6


def test_window_semantics_match_attention_validity_exactly():
    """The in-kernel window must reproduce _attention_validity's bound."""
    from dllm.attention.blasst.core import _attention_validity
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention, route_only
    from experiments.numerical_qk_reuse.integration import Attention

    generator = torch.Generator(device='cuda').manual_seed(7)
    h, hk, d, nq, nk, window = 8, 2, 128, 256, 640, 300
    q = (torch.randn(1, nq, h, d, generator=generator, device='cuda', dtype=torch.bfloat16)
         .transpose(1, 2) * .3)
    k = (torch.randn(1, nk, hk, d, generator=generator, device='cuda', dtype=torch.bfloat16)
         .transpose(1, 2) * .3)
    v = torch.randn(1, nk, hk, d, generator=generator, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    scores = Attention.observe_scores(q, k, None, 1., False, window, 0).contiguous()

    # Producer legality, straight from the shared validity helper.
    valid = _attention_validity(None, q, k, is_causal=False, sliding_window=window)
    assert torch.equal(torch.isfinite(scores), valid.expand_as(scores))

    z = torch.randn(1, hk, nk, 32, generator=generator, device='cuda', dtype=torch.float32).contiguous()
    ref = (torch.rand(1, hk, generator=generator, device='cuda', dtype=torch.float32) + .5).contiguous()
    routing = route_only(scores, z, ref, log_threshold=-math.inf)
    keep = torch.zeros_like(routing.skipped)
    consumed = preqk_attention(q, k, v, keep, routing.eligible, scale=1., window=window)
    exact = fp32_reference(q, k, v, keep, routing.eligible, scale=1., window=window)
    # A wrong in-kernel bound would attend illegal keys and blow this up far
    # beyond the BF16 floor the qualification table reports (~1e-2).
    assert relative(consumed.output, exact) < 2e-2


def test_consumer_rejects_inputs_outside_its_qualified_contract():
    from experiments.numerical_qk_reuse.cached_executor import preqk_attention

    generator = torch.Generator(device='cuda').manual_seed(2)
    q = torch.randn(1, 4, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, 2, 192, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, 2, 192, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
    shape = (1, 4, 1, 3)
    keep = torch.zeros(shape, dtype=torch.bool, device='cuda')
    elig = torch.ones(shape, dtype=torch.bool, device='cuda')

    with pytest.raises(ValueError):   # the decoder is bidirectional; causal is unqualified
        preqk_attention(q, k, v, keep, elig, scale=1., is_causal=True)
    with pytest.raises(ValueError):   # head dimension must be contiguous
        preqk_attention(q.transpose(-1, -2).transpose(-1, -2)[..., ::2], k, v, keep, elig, scale=1.)
    with pytest.raises(ValueError):   # bitmap geometry must match the tiling
        preqk_attention(q, k, v, keep[..., :1], elig, scale=1.)
    with pytest.raises(ValueError):   # float bitmaps are not silently cast
        preqk_attention(q, k, v, keep.float(), elig, scale=1.)


def test_expected_programs_accounts_for_a_short_final_query_tile():
    retained = torch.zeros((1, 1, 2, 4), dtype=torch.bool)
    retained[0, 0, 0, :] = True      # full 128-row tile -> 8 programs each
    retained[0, 0, 1, :2] = True     # 72-row tail tile  -> 5 programs each
    assert expected_programs(retained, nq=200) == 4 * 8 + 2 * 5
