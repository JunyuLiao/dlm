"""Bounded actual CUDA dispatch/lifecycle checks, no model/answer data."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_router(interval):
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda name, module: False)
    router = Attention(adapter, {k: {'log_threshold': -math.inf} for k in ('local', 'global')},
                       decision_interval=interval)
    module = SimpleNamespace(layer_idx=5, training=False, is_sliding=False, layer_type='full_attention')
    return router, module, model


def inputs():
    generator = torch.Generator(device='cuda').manual_seed(5901)
    q = torch.randn(1, 4, 129, 256, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    k = torch.randn(1, 2, 194, 256, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    v = torch.randn(1, 2, 194, 256, generator=generator, device='cuda', dtype=torch.bfloat16)
    prefix = SimpleNamespace(keys=k[..., :65, :].clone(), values=v[..., :65, :].clone())
    cache = SimpleNamespace(layers={5: prefix}, is_compileable=False, get_seq_length=lambda: 65)
    return q, k, v, cache


def test_reuse_never_calls_current_qk_and_updates_current_v():
    router, module, model = make_router(2)
    q, k, v, cache = inputs()
    try:
        router.begin_step(0, 0)
        router.identify(module, (), {'past_key_values': cache})
        first = router(module, q, k, v, None, scaling=1., is_causal=False)[0]
        router.begin_step(0, 1)
        with patch.object(router, 'observe_scores', side_effect=AssertionError('Reuse executed current QK')):
            second = router(module, q*7, k*9, v*2, None, scaling=1., is_causal=False)[0]
        torch.testing.assert_close(second.float(), first.float()*2, atol=.01, rtol=.02)
        assert router.score_calls == 1 and router.decision_calls == 1 and router.held_calls == 1
        assert router.call_metadata[-1]['current_qk_elements'] == 0
        # M1 decision refresh at2 still does NOT refresh numerical scores.
        router.begin_step(0, 2)
        with patch.object(router, 'observe_scores', side_effect=AssertionError('Decision ran QK')):
            router(module, q*7, k*9, v, None, scaling=1., is_causal=False)
        assert router.score_calls == 1 and router.decision_calls == 2
        # Encoder entry invalidates BEFORE even an identity-preserving commit.
        model.encoder(torch.zeros(1, device='cuda'))
        assert not router.cache.entries
        router.begin_step(1, 0)
        router.identify(module, (), {'past_key_values': cache})
        router(module, q, k, v, None, scaling=1., is_causal=False)
        assert router.score_calls == 2
        torch.cuda.synchronize()
    finally:
        router.close()


def test_r1_same_executor_equals_m1_and_actual_refresh_at_eight():
    one, module, _ = make_router(1)
    two, _, _ = make_router(1)
    q, k, v, cache = inputs()
    try:
        for step in (0, 1, 2, 7, 8):
            outputs = []
            for router in (one, two):
                router.begin_step(0, step)
                router.identify(module, (), {'past_key_values': cache})
                outputs.append(router(module, q*(step+1), k, v, None, scaling=1., is_causal=False)[0])
            assert torch.equal(*outputs)
        assert one.score_calls == two.score_calls == 2
        assert one.decision_calls == two.decision_calls == 5
        assert one.call_metadata[-1]['score_anchor'] == 8
        torch.cuda.synchronize()
    finally:
        one.close()
        two.close()
