"""Phase A1: native_mask must recover the native-legal support, unlike legacy.

Legacy ('legacy_junyu_mask', the default and only mode used by every prior
result) preserves Junyu's query-relative local window even though the
installed native SDPA ignores that window entirely once a layer's stored
prefix already exceeds window-length-minus-canvas. native_mask removes that
extra narrowing so the numerical reuse adapter's legal-key support matches
what native dense actually attends, on the SAME cached scores.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def make_router(support):
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda name, module: False)
    router = Attention(adapter, {k: {'log_threshold': -math.inf} for k in ('local', 'global')},
                       decision_interval=1, support=support)
    module = SimpleNamespace(layer_idx=2, training=False, is_sliding=True, layer_type='sliding_attention')
    return router, module, model


def test_native_mask_recovers_prefix_keys_legacy_incorrectly_drops():
    # window=8, canvas=4: legacy's bound k_pos >= q_pos-window+1 excludes early
    # prefix keys once prefix exceeds window-canvas=4, matching the analytic
    # audit in scripts/native_reuse_mask_audit.py at realistic (1024, 256) scale.
    window, canvas, prefix, hk, d = 8, 4, 7, 1, 64
    generator = torch.Generator(device='cuda').manual_seed(7)
    k = torch.randn(1, hk, prefix + canvas, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, hk, prefix + canvas, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(1, hk, canvas, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    prefix_tensor = k[..., :prefix, :].clone()
    cache = SimpleNamespace(layers={2: SimpleNamespace(keys=prefix_tensor, values=v[..., :prefix, :].clone())},
                            is_compileable=False, get_seq_length=lambda: prefix)

    legal_counts = {}
    for support in ('legacy_junyu_mask', 'native_mask'):
        router, module, _ = make_router(support)
        try:
            router.begin_step(0, 0)
            router.identify(module, (), {'past_key_values': cache})
            router.sketches.identify(module, (), {'past_key_values': cache})
            router(module, q, k, v, None, scaling=1., is_causal=False, sliding_window=window)
            scores = router.cache.entries[2].scores
            legal_counts[support] = int(torch.isfinite(scores).sum().item())
        finally:
            router.close()

    assert legal_counts['native_mask'] > legal_counts['legacy_junyu_mask']
