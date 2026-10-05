"""routing_only_current_output: Phase D successor selected by the phaseAB
diagnostic (results/numerical_qk_reuse_recovery_20260924/phaseAB_report.md).

The routing decision (which KV64 tiles are retained) must still come from
the stale cached score, unchanged from production M1/M3 -- but the final
softmax/PV output must always reflect CURRENT, not cached, scores. This is
a real extra QK cost, not a relabeling of M1's cheaper reuse.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_router(output_mode):
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda name, module: False)
    router = Attention(adapter, {k: {'log_threshold': -.5} for k in ('local', 'global')},
                       decision_interval=1, output_mode=output_mode)
    module = SimpleNamespace(layer_idx=1, training=False, is_sliding=False, layer_type='full_attention')
    return router, module, model


def inputs(seed, prefix_k=None, prefix_v=None, cache=None):
    """Same-canvas calls must reuse the SAME prefix/cache object (only the
    canvas portion of q/k/v evolves as denoising proceeds); a brand new
    prefix object is a new canvas/commit, which correctly forces a fresh
    score+decision (identity.prefix_owner changes)."""
    generator = torch.Generator(device='cuda').manual_seed(seed)
    canvas_k = torch.randn(1, 2, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
    canvas_v = torch.randn(1, 2, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(1, 4, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
    if prefix_k is None:
        prefix_k = torch.randn(1, 2, 64, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
        prefix_v = torch.randn(1, 2, 64, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
        cache = SimpleNamespace(layers={1: SimpleNamespace(keys=prefix_k, values=prefix_v)},
                                is_compileable=False, get_seq_length=lambda: 64)
    k = torch.cat([prefix_k, canvas_k], dim=2)
    v = torch.cat([prefix_v, canvas_v], dim=2)
    return q, k, v, cache, prefix_k, prefix_v


def test_routing_decision_unchanged_but_output_reflects_current_scores():
    cached, module, _ = make_router('cached_scores')
    routed, _, _ = make_router('routing_only_current_output')
    prefixes = {}
    try:
        for name, router in (('cached_scores', cached), ('routing_only_current_output', routed)):
            q, k, v, cache, pk, pv = inputs(11)
            prefixes[name] = (pk, pv, cache)
            router.begin_step(0, 0)
            router.identify(module, (), {'past_key_values': cache})
            router(module, q, k, v, None, scaling=1., is_causal=False)

        outputs = {}
        for name, router in (('cached_scores', cached), ('routing_only_current_output', routed)):
            # Same canvas (same prefix object), different canvas content --
            # score_age=1 with score_period=8, so this call reuses the stale
            # score for routing but the canvas itself really did change.
            pk, pv, cache = prefixes[name]
            q, k, v, cache, _, _ = inputs(97, pk, pv, cache)
            router.begin_step(0, 1)
            router.identify(module, (), {'past_key_values': cache})
            outputs[name] = router(module, q, k, v, None, scaling=1., is_causal=False)[0]

        # The retained-tile bitmap must be identical: routing_only_current_output
        # does not change WHICH tiles are kept, only what scores fill them.
        assert torch.equal(cached.cache.entries[1].decision.skipped,
                           routed.cache.entries[1].decision.skipped)
        # But the numeric output must differ, since cached_scores fills the
        # softmax with 8-call-old scores while routing_only recomputes fresh
        # QK for the same retained support.
        assert not torch.allclose(outputs['cached_scores'], outputs['routing_only_current_output'])
        # routing_only_current_output must have actually spent a fresh QK
        # matmul on this held-score call, not silently reused it.
        assert routed.routing_only_extra_qk_elements > 0
        assert cached.routing_only_extra_qk_elements == 0
        torch.cuda.synchronize()
    finally:
        cached.close()
        routed.close()


def test_routing_only_output_matches_reference_oracle_on_retained_support():
    """Direct proof against the Torch oracle: routing_only's output must equal
    softmax(current scores restricted to the stale-score-derived support) @ v,
    the exact Cell C construction from the phaseAB diagnostic."""
    from experiments.numerical_qk_reuse import reference as oracle
    from experiments.numerical_qk_reuse.integration import Attention

    routed, module, _ = make_router('routing_only_current_output')
    try:
        q0, k0, v0, cache, pk, pv = inputs(5)
        routed.begin_step(0, 0)
        routed.identify(module, (), {'past_key_values': cache})
        routed(module, q0, k0, v0, None, scaling=1., is_causal=False)

        q1, k1, v1, cache1, _, _ = inputs(6, pk, pv, cache)
        routed.begin_step(0, 1)
        routed.identify(module, (), {'past_key_values': cache1})
        output = routed(module, q1, k1, v1, None, scaling=1., is_causal=False)[0]

        current_scores = Attention.observe_scores(q1, k1, None, 1., False, None, 0)
        bitmap = routed.cache.entries[1].decision.skipped
        expected = oracle.attention(current_scores, v1, skipped=bitmap)
        torch.testing.assert_close(output, expected.output.transpose(1, 2).contiguous(),
                                   atol=2e-2, rtol=2e-2)
        torch.cuda.synchronize()
    finally:
        routed.close()
