"""route_only must be the SAME selector, minus the discarded PV.

v6's routing_only_current_output derived its support with the full
route+PV ``attention`` entry point and then threw the PV output away. These
tests pin that the new selector-only path (a) reproduces the old bitmap
exactly under nonuniform T, masked rows, nonfinite scores and both head
widths, (b) issues no PV launch, and (c) leaves the final attention output
bit-identical.
"""
import math
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def tensors(seed, *, nq=256, nk=384, h=8, hk=2, d=128, masked=0, nonfinite=None):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    scores = torch.randn(1, h, nq, nk, generator=generator, device='cuda', dtype=torch.float32)
    if masked:
        scores[..., :masked] = -math.inf          # structurally illegal keys
        scores[:, :, :2, :] = -math.inf           # fully masked query rows
    if nonfinite is not None:
        scores[0, 0, 3, 7] = nonfinite
    v = torch.randn(1, hk, nk, d, generator=generator, device='cuda', dtype=torch.bfloat16)
    z = torch.randn(1, hk, nk, 32, generator=generator, device='cuda', dtype=torch.float32)
    reference = torch.rand(1, hk, generator=generator, device='cuda', dtype=torch.float32) + .5
    return scores.contiguous(), v.contiguous(), z.contiguous(), reference.contiguous()


def sensitivity_for(nq, seed, uniform):
    if uniform:
        return None
    generator = torch.Generator(device='cuda').manual_seed(seed)
    # Production T weights are in [1, 1+beta]; never uniform, never <=0.
    return (1. + 3. * torch.rand(1, nq, generator=generator, device='cuda',
                                 dtype=torch.float32)).contiguous()


@pytest.mark.parametrize('uniform_t', [True, False])
@pytest.mark.parametrize('d', [64, 128])
@pytest.mark.parametrize('masked,nonfinite', [(0, None), (67, None), (0, float('nan')),
                                              (0, float('inf')), (67, float('nan'))])
def test_route_only_bitmap_matches_the_previous_route_plus_pv_api(uniform_t, d, masked, nonfinite):
    from experiments.numerical_qk_reuse.cached_executor import attention, route_only

    scores, v, z, reference = tensors(31, d=d, masked=masked, nonfinite=nonfinite)
    sensitivity = sensitivity_for(scores.shape[2], 77, uniform_t)
    threshold = -1.0099318265914916

    old = attention(scores, v, z, reference, sensitivity=sensitivity, log_threshold=threshold)
    new = route_only(scores, z, reference, sensitivity=sensitivity, log_threshold=threshold)

    assert torch.equal(old.skipped, new.skipped)
    assert torch.equal(old.eligible, new.eligible)
    # The selector's own malformed-score flag must agree with what the PV
    # pass reported per row, aggregated to its tile's query block.
    per_row = old.invalid_scores  # [B,H,Q]
    qb = new.invalid_tiles.shape[2]
    rows = torch.zeros_like(new.invalid_tiles[..., 0])
    for tile in range(qb):
        block = per_row[:, :, tile * 128:(tile + 1) * 128]
        rows[:, :, tile] = block.any(-1)
    assert torch.equal(new.invalid_tiles.any(-1), rows)


def test_route_only_issues_no_pv_launch():
    """The discarded PV must not merely be unused -- it must not be issued."""
    from experiments.numerical_qk_reuse import cached_executor

    scores, v, z, reference = tensors(5)
    launches = {'pv': 0, 'route': 0}

    class Spy:
        def __init__(self, inner, key):
            self.inner, self.key = inner, key

        def __getitem__(self, grid):
            launches[self.key] += 1
            return self.inner[grid]

    original_pv, original_route = cached_executor._pv, cached_executor._route
    cached_executor._pv = Spy(original_pv, 'pv')
    cached_executor._route = Spy(original_route, 'route')
    try:
        cached_executor.route_only(scores, z, reference, log_threshold=-1.)
        assert launches == {'pv': 0, 'route': 1}
        cached_executor.attention(scores, v, z, reference, log_threshold=-1.)
        assert launches == {'pv': 1, 'route': 2}
    finally:
        cached_executor._pv, cached_executor._route = original_pv, original_route
    torch.cuda.synchronize()


def test_route_only_does_not_allocate_a_discarded_output():
    from experiments.numerical_qk_reuse.cached_executor import Routing, route_only

    scores, _, z, reference = tensors(9)
    result = route_only(scores, z, reference, log_threshold=-1.)
    assert isinstance(result, Routing)
    assert not hasattr(result, 'output') and not hasattr(result, 'log_normalizer')
    # Only the three tile-shaped bitmaps, nothing per-query or per-dimension.
    for field in (result.skipped, result.eligible, result.invalid_tiles):
        assert field.shape == (1, scores.shape[1], 2, 6) and field.dtype == torch.bool


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
    module = SimpleNamespace(layer_idx=1, training=False, is_sliding=False,
                             layer_type='full_attention')
    return router, module, model


def canvas_inputs(seed, prefix_k=None, prefix_v=None, cache=None):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    canvas_k = torch.randn(1, 2, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
    canvas_v = torch.randn(1, 2, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
    q = torch.randn(1, 4, 128, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
    if prefix_k is None:
        prefix_k = torch.randn(1, 2, 64, 64, generator=generator, device='cuda', dtype=torch.bfloat16) * .3
        prefix_v = torch.randn(1, 2, 64, 64, generator=generator, device='cuda', dtype=torch.bfloat16)
        cache = SimpleNamespace(layers={1: SimpleNamespace(keys=prefix_k, values=prefix_v)},
                                is_compileable=False, get_seq_length=lambda: 64)
    return (q, torch.cat([prefix_k, canvas_k], dim=2), torch.cat([prefix_v, canvas_v], dim=2),
            cache, prefix_k, prefix_v)


def drive(router, module, steps):
    """One canvas, several decoder calls with genuinely changing canvas content."""
    outputs = []
    state = None
    for index, seed in enumerate(steps):
        if state is None:
            q, k, v, cache, pk, pv = canvas_inputs(seed)
            state = (pk, pv, cache)
        else:
            q, k, v, cache, _, _ = canvas_inputs(seed, *state)
        router.begin_step(0, index)
        router.identify(module, (), {'past_key_values': cache})
        router.sketches.identify(module, (), {'past_key_values': cache})
        outputs.append(router(module, q, k, v, None, scaling=1., is_causal=False)[0])
    return outputs


def test_refactor_leaves_routing_only_output_and_bitmaps_bit_identical():
    """Same live inputs and history -> the refactor may not move any bit."""
    from experiments.numerical_qk_reuse import integration

    steps = [11, 97, 23, 44]
    routed, module, _ = make_router('routing_only_current_output')
    try:
        new_outputs = drive(routed, module, steps)
        new_bitmap = routed.cache.entries[1].decision.skipped.clone()
    finally:
        routed.close()

    # Re-run with the pre-refactor shape: the bitmap comes from the full
    # route+PV entry point, whose PV output is then discarded. ``_route``
    # never reads V, so a stand-in V reproduces the old bitmap exactly while
    # still paying the discarded PV the old path paid.
    original = integration.route_only

    def fused_route_only(scores, z, reference, *, sensitivity=None, log_threshold=-math.inf,
                         summary=None, store_summary=False):
        from experiments.numerical_qk_reuse.cached_executor import Routing, attention
        stand_in = torch.zeros((z.shape[0], z.shape[1], z.shape[2], 64),
                               device=z.device, dtype=torch.bfloat16)
        legacy = attention(scores, stand_in, z, reference, sensitivity=sensitivity,
                           log_threshold=log_threshold)
        return Routing(legacy.skipped, legacy.eligible, torch.zeros_like(legacy.skipped))

    legacy_router, legacy_module, _ = make_router('routing_only_current_output')
    integration.route_only = fused_route_only
    try:
        legacy_outputs = drive(legacy_router, legacy_module, steps)
        legacy_bitmap = legacy_router.cache.entries[1].decision.skipped.clone()
    finally:
        integration.route_only = original
        legacy_router.close()

    assert torch.equal(new_bitmap, legacy_bitmap)
    for new, legacy in zip(new_outputs, legacy_outputs):
        assert torch.equal(new, legacy)
    torch.cuda.synchronize()


def test_cached_scores_mode_still_uses_its_fused_output():
    """cached_scores needs the route's PV output; it must not be starved."""
    from experiments.numerical_qk_reuse import cached_executor

    cached, module, _ = make_router('cached_scores')
    launches = {'pv': 0}

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def __getitem__(self, grid):
            launches['pv'] += 1
            return self.inner[grid]

    original = cached_executor._pv
    cached_executor._pv = Spy(original)
    try:
        drive(cached, module, [11, 97])
        # One PV per call, none discarded.
        assert launches['pv'] == 2
    finally:
        cached_executor._pv = original
        cached.close()
    torch.cuda.synchronize()


def test_score_anchor_step_does_not_observe_the_same_qk_twice():
    from experiments.numerical_qk_reuse import cached_executor
    from experiments.numerical_qk_reuse.integration import Attention

    routed, module, _ = make_router('routing_only_current_output')
    observed = {'count': 0}
    original_observe = Attention.observe_scores
    launches = {'pv': 0}

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def __getitem__(self, grid):
            launches['pv'] += 1
            return self.inner[grid]

    def counting(*args, **kwargs):
        observed['count'] += 1
        return original_observe(*args, **kwargs)

    original_pv = cached_executor._pv
    cached_executor._pv = Spy(original_pv)
    Attention.observe_scores = staticmethod(counting)
    try:
        drive(routed, module, [11])           # step 0 == score anchor
        assert observed['count'] == 1         # QK observed exactly once
        assert launches['pv'] == 1            # fused route+PV, nothing discarded
    finally:
        Attention.observe_scores = staticmethod(original_observe)
        cached_executor._pv = original_pv
        routed.close()
    torch.cuda.synchronize()
