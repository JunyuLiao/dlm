"""Qualification for the exact prefix-block-summary selector.

The claim is exactness, not "close enough": for KV tiles lying wholly inside
the immutable prefix, the per-row block log mass and weighted projected value
depend only on the frozen cached scores and the frozen prefix projected V, so
caching them at the real score anchor and reading them back must leave every
bitmap, eligibility flag and malformed flag bit-identical -- while alpha,
risk, the retained scan state and the drop decision stay live.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

LOCAL_THRESHOLD = -1.0099318265914916


def geometry(seed, *, h=16, hk=8, d=256, nq=256, prefix=1023, canvas=256, window=1024):
    from experiments.numerical_qk_reuse.integration import Attention
    generator = torch.Generator(device='cuda').manual_seed(seed)
    nk = prefix + canvas
    q = torch.randn(1, nq, h, d, generator=generator, device='cuda',
                    dtype=torch.bfloat16).transpose(1, 2) * .3
    k = torch.randn(1, nk, hk, d, generator=generator, device='cuda',
                    dtype=torch.bfloat16).transpose(1, 2) * .3
    scores = Attention.observe_scores(q, k, None, 1., False, window, 0)
    z = torch.randn(1, hk, nk, 32, generator=generator, device='cuda',
                    dtype=torch.float32).contiguous()
    reference = (torch.rand(1, hk, generator=generator, device='cuda',
                            dtype=torch.float32) + .5).contiguous()
    return generator, scores, z, reference, nq, nk, h, hk, prefix


def summary_for(scores, z, reference, prefix, h, nq, nk, identity=('r', 0, 'a'),
                threshold=LOCAL_THRESHOLD, sensitivity=None):
    from experiments.numerical_qk_reuse.cached_executor import allocate_summary, route_only
    qb, kt = (nq + 127) // 128, (nk + 63) // 64
    summary = allocate_summary(1, h, qb, kt, prefix // 64, 32, scores.device, identity)
    route_only(scores, z, reference, sensitivity=sensitivity, log_threshold=threshold,
               summary=summary, store_summary=True)
    return summary


def agree(left, right):
    return (torch.equal(left.skipped, right.skipped)
            and torch.equal(left.eligible, right.eligible)
            and torch.equal(left.invalid_tiles, right.invalid_tiles))


def test_1_identical_bitmaps_across_changing_canvas_v_sensitivity_and_reference():
    """Spec test 1: same cached scores + prefix V, live T/ref/canvas V."""
    from experiments.numerical_qk_reuse.cached_executor import route_only
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(11)
    summary = summary_for(scores, z, reference, prefix, h, nq, nk)
    for _ in range(6):
        live = z.clone()
        # Only the CURRENT CANVAS portion of the projected V may move.
        live[:, :, prefix:, :] = torch.randn_like(live[:, :, prefix:, :])
        sensitivity = (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                            dtype=torch.float32)).contiguous()
        live_reference = (torch.rand(1, hk, generator=generator, device='cuda',
                                     dtype=torch.float32) + .5).contiguous()
        legacy = route_only(scores, live, live_reference, sensitivity=sensitivity,
                            log_threshold=LOCAL_THRESHOLD)
        optimized = route_only(scores, live, live_reference, sensitivity=sensitivity,
                               log_threshold=LOCAL_THRESHOLD, summary=summary)
        assert agree(legacy, optimized)


def test_2_decisions_are_not_frozen_by_the_summary():
    """Spec test 2: same summary, changed T or ref, must be able to change a
    decision. Catches accidentally caching risk/alpha/the bitmap."""
    from experiments.numerical_qk_reuse.cached_executor import route_only
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(12)
    summary = summary_for(scores, z, reference, prefix, h, nq, nk)
    low = route_only(scores, z, reference, log_threshold=LOCAL_THRESHOLD, summary=summary)
    hot = (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                dtype=torch.float32)).contiguous()
    changed_t = route_only(scores, z, reference, sensitivity=hot,
                           log_threshold=LOCAL_THRESHOLD, summary=summary)
    changed_ref = route_only(scores, z, (reference * 40.).contiguous(),
                             log_threshold=LOCAL_THRESHOLD, summary=summary)
    assert not torch.equal(low.skipped, changed_t.skipped), 'T is not live'
    assert not torch.equal(low.skipped, changed_ref.skipped), 'reference is not live'
    # ...and each still matches its own legacy recomputation exactly.
    for sensitivity, ref, observed in ((hot, reference, changed_t),
                                       (None, (reference * 40.).contiguous(), changed_ref)):
        legacy = route_only(scores, z, ref, sensitivity=sensitivity,
                            log_threshold=LOCAL_THRESHOLD)
        assert agree(legacy, observed)


def test_3_identity_changes_invalidate_and_pointers_do_not_authorize_reuse():
    """Spec test 3: prefix mutation, score epoch, canvas/request, boundary."""
    from experiments.numerical_qk_reuse.cached_executor import PrefixSummary
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(13)
    summary = summary_for(scores, z, reference, prefix, h, nq, nk, identity=('r', 0, 'a'))
    assert summary.matches(('r', 0, 'a'), prefix // 64)
    for wrong in (('r', 1, 'a'), ('r', 0, 'b'), ('other', 0, 'a')):
        assert not summary.matches(wrong, prefix // 64)
    assert not summary.matches(('r', 0, 'a'), prefix // 64 - 1)   # boundary moved
    # Same buffers, recycled object: identity is what authorizes reuse.
    recycled = PrefixSummary(summary.z, summary.mu, summary.active, summary.bad,
                             summary.prefix_tiles, ('r', 9, 'a'))
    assert not recycled.matches(('r', 0, 'a'), prefix // 64)


def test_3b_router_invalidates_summaries_on_commit_and_new_canvas():
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda n, m: False)
    router = Attention(adapter, {k: {'log_threshold': -.5} for k in ('local', 'global')},
                       decision_interval=1, selector='prefix_block_summary')
    try:
        router.summaries[3] = object()
        model.encoder(torch.zeros(1, device='cuda'))       # encoder commit
        assert not router.summaries
        router.summaries[3] = object()
        router.begin_step(1, 0)                             # new canvas
        assert not router.summaries
    finally:
        router.close()


@pytest.mark.parametrize('case', ['all_kept', 'near_threshold', 'all_masked',
                                  'nonfinite', 'tail', 'gqa1'])
def test_4_edge_cases_stay_bit_identical(case):
    """Spec test 4: all-kept, near-threshold, all-masked, nonfinite, GQA, tails."""
    from experiments.numerical_qk_reuse.cached_executor import route_only
    settings = dict(seed=14)
    if case == 'gqa1':
        settings.update(h=4, hk=4, d=64)
    if case == 'tail':
        settings.update(prefix=1000, canvas=200)      # neither is tile-aligned
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(**settings)
    threshold = LOCAL_THRESHOLD
    if case == 'all_kept':
        threshold = -math.inf
    if case == 'all_masked':
        scores = scores.clone()
        scores[:, :, :5, :] = -math.inf
        scores = scores.contiguous()
    if case == 'nonfinite':
        scores = scores.clone()
        scores[0, 0, 7, 11] = float('nan')
        scores[0, 1, 9, 130] = float('inf')
        scores = scores.contiguous()
    if case == 'near_threshold':
        # Put the threshold where risks actually concentrate, so ties and
        # borderline comparisons are genuinely exercised.
        probe = route_only(scores, z, reference, log_threshold=-math.inf)
        del probe
        threshold = -2.0
    summary = summary_for(scores, z, reference, prefix, h, nq, nk, threshold=threshold)
    sensitivity = (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                        dtype=torch.float32)).contiguous()
    legacy = route_only(scores, z, reference, sensitivity=sensitivity, log_threshold=threshold)
    optimized = route_only(scores, z, reference, sensitivity=sensitivity,
                           log_threshold=threshold, summary=summary)
    assert agree(legacy, optimized), case


def test_5_reused_prefix_tiles_do_not_reread_scores_or_redo_the_sketch_dot():
    """Spec test 5: the avoided work must be observable, not asserted.

    Corrupting the cached scores and the projected V *inside the summarized
    prefix range* after the summary is built must not change the optimized
    result -- it could only do so if those inputs were still being read.
    """
    from experiments.numerical_qk_reuse.cached_executor import route_only
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(15)
    summary = summary_for(scores, z, reference, prefix, h, nq, nk)
    sensitivity = (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                        dtype=torch.float32)).contiguous()
    before = route_only(scores, z, reference, sensitivity=sensitivity,
                        log_threshold=LOCAL_THRESHOLD, summary=summary)

    summarized_keys = (prefix // 64) * 64
    poisoned_scores = scores.clone()
    poisoned_scores[:, :, :, :summarized_keys] = -7.5
    poisoned_z = z.clone()
    poisoned_z[:, :, :summarized_keys, :] = 99.
    after = route_only(poisoned_scores.contiguous(), poisoned_z.contiguous(), reference,
                       sensitivity=sensitivity, log_threshold=LOCAL_THRESHOLD, summary=summary)
    assert agree(before, after), 'summarized tiles still read their scores/sketch'

    # The same poisoning MUST change the legacy path, or the probe is vacuous.
    legacy_before = route_only(scores, z, reference, sensitivity=sensitivity,
                               log_threshold=LOCAL_THRESHOLD)
    legacy_after = route_only(poisoned_scores.contiguous(), poisoned_z.contiguous(), reference,
                              sensitivity=sensitivity, log_threshold=LOCAL_THRESHOLD)
    assert not agree(legacy_before, legacy_after)

    # Mutable (canvas / boundary) tiles must still be recomputed: poisoning
    # beyond the summarized range has to move the optimized result too.
    tail_poisoned = scores.clone()
    tail_poisoned[:, :, :, summarized_keys:] = -7.5
    moved = route_only(tail_poisoned.contiguous(), z, reference, sensitivity=sensitivity,
                       log_threshold=LOCAL_THRESHOLD, summary=summary)
    assert not agree(before, moved), 'mutable tiles were not recomputed'


def test_6_anchor_rebuild_tracks_a_new_score_observation():
    """A fresh anchor must rebuild: a summary from the old scores is not valid
    for new ones, and rebuilding must reproduce the new legacy result."""
    from experiments.numerical_qk_reuse.cached_executor import route_only
    generator, scores, z, reference, nq, nk, h, hk, prefix = geometry(16)
    summary = summary_for(scores, z, reference, prefix, h, nq, nk, identity=('r', 0, 'anchor0'))
    new_scores = (scores + .5).contiguous()               # a later real observation
    stale = route_only(new_scores, z, reference, log_threshold=LOCAL_THRESHOLD, summary=summary)
    legacy = route_only(new_scores, z, reference, log_threshold=LOCAL_THRESHOLD)
    assert not agree(stale, legacy), 'stale summary silently agreed; probe is vacuous'
    rebuilt = summary_for(new_scores, z, reference, prefix, h, nq, nk,
                          identity=('r', 1, 'anchor1'))
    fresh = route_only(new_scores, z, reference, log_threshold=LOCAL_THRESHOLD, summary=rebuilt)
    assert agree(fresh, legacy)
