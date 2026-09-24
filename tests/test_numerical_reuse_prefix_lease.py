"""Phase C repair proof: leased prefix V/norm sketches must not change output.

Restoring Junyu's Sketches lease (value_direction_hopper.integration.Sketches)
into the numerical decision-refresh step is a pure engineering optimization.
This asserts output/phase invariance against the pre-repair brute-force
formula across an unaligned prefix boundary, a changing canvas, and a
changing validity mask -- not just that the reuse counters moved.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def brute_force_projected_ref(v, valid, projections, layer, hk, d):
    """The exact pre-repair formula: full recompute over the entire tensor."""
    matrix = projections.get(layer, hk, d, 'gaussian', 32, 1729, v.device)
    current = v.float()
    projected = torch.matmul(current, matrix).contiguous()
    ref = (current.square().sum(-1).masked_fill(~valid, 0.).sum(-1) /
           valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12).contiguous()
    return projected, ref


def make_router():
    from experiments.numerical_qk_reuse.integration import Attention

    class DiffusionGemmaEncoderModel(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.add_module('encoder', DiffusionGemmaEncoderModel())
    adapter = SimpleNamespace(model=model, is_blasst_attention_module=lambda name, module: False)
    router = Attention(adapter, {k: {'log_threshold': -math.inf} for k in ('local', 'global')},
                       decision_interval=1)
    module = SimpleNamespace(layer_idx=3, training=False, is_sliding=False, layer_type='full_attention')
    return router, module, model


def test_leased_prefix_matches_brute_force_across_unaligned_boundary_and_changing_canvas():
    """Prefix length 65 is deliberately not a multiple of 64 (KV64 alignment)."""
    router, module, model = make_router()
    generator = torch.Generator(device='cuda').manual_seed(4242)
    hk, d, prefix_len, canvas_len = 2, 128, 65, 40
    total = prefix_len + canvas_len
    k = torch.randn(1, hk, total, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    prefix_v = torch.randn(1, hk, prefix_len, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    cache = SimpleNamespace(layers={3: SimpleNamespace(keys=k[..., :prefix_len, :].clone(), values=prefix_v)},
                            is_compileable=False, get_seq_length=lambda: prefix_len)
    try:
        for call in range(3):
            q = torch.randn(1, 4, canvas_len, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
            canvas_v = torch.randn(1, hk, canvas_len, d, generator=generator, device='cuda', dtype=torch.bfloat16)
            v = torch.cat([prefix_v, canvas_v], dim=2)
            router.begin_step(0, call)
            router.identify(module, (), {'past_key_values': cache})
            # Simulates the real forward_pre_hook a live model would fire.
            router.sketches.identify(module, (), {'past_key_values': cache})
            output = router(module, q, k, v, None, scaling=1., is_causal=False)[0]

            valid = router.valid_keys[3]
            expected_projected, expected_ref = brute_force_projected_ref(v, valid, router.projections, 3, hk, d)
            from experiments.numerical_qk_reuse.cached_executor import attention
            expected = attention(router.cache.get(router.cache.entries[3].identity).scores, v,
                                 expected_projected, expected_ref,
                                 log_threshold=-math.inf, trace=False)
            torch.testing.assert_close(output, expected.output.transpose(1, 2).contiguous())
            if call >= 1:
                # Once the encoder-owned prefix source has been seen twice with
                # an unchanged version, the aligned prefix block (positions
                # 0:64) must be a genuine lease, not a relabeled recompute.
                assert router.sketches.reused_tokens > 0
        torch.cuda.synchronize()
    finally:
        router.close()


def test_commit_invalidates_the_lease_even_with_identical_content():
    """A new encoder forward must force start=0, even if values happen to repeat."""
    router, module, model = make_router()
    generator = torch.Generator(device='cuda').manual_seed(99)
    hk, d, prefix_len, canvas_len = 2, 64, 64, 16
    k = torch.randn(1, hk, prefix_len + canvas_len, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    prefix_v = torch.randn(1, hk, prefix_len, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    canvas_v = torch.randn(1, hk, canvas_len, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    v = torch.cat([prefix_v, canvas_v], dim=2)
    cache = SimpleNamespace(layers={3: SimpleNamespace(keys=k[..., :prefix_len, :].clone(), values=prefix_v)},
                            is_compileable=False, get_seq_length=lambda: prefix_len)
    q = torch.randn(1, 4, canvas_len, d, generator=generator, device='cuda', dtype=torch.bfloat16) * .1
    try:
        router.begin_step(0, 0)
        router.identify(module, (), {'past_key_values': cache})
        router.sketches.identify(module, (), {'past_key_values': cache})
        router(module, q, k, v, None, scaling=1., is_causal=False)
        reused_before = router.sketches.reused_tokens
        # A fresh encoder forward (even with bit-identical prefix values, a new
        # tensor object) must invalidate the lease rather than infer equality
        # from content.
        model.encoder(torch.zeros(1, device='cuda'))
        new_prefix_v = prefix_v.clone()
        new_cache = SimpleNamespace(layers={3: SimpleNamespace(keys=k[..., :prefix_len, :].clone(), values=new_prefix_v)},
                                    is_compileable=False, get_seq_length=lambda: prefix_len)
        new_v = torch.cat([new_prefix_v, canvas_v.clone()], dim=2)
        router.begin_step(1, 0)
        router.identify(module, (), {'past_key_values': new_cache})
        router.sketches.identify(module, (), {'past_key_values': new_cache})
        router(module, q, k, new_v, None, scaling=1., is_causal=False)
        assert router.sketches.reused_tokens == reused_before
        torch.cuda.synchronize()
    finally:
        router.close()
