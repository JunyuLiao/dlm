from dataclasses import replace
import pytest
import torch

from experiments.diffusion_gemma_value_aware.operators import Config, block_state, value_summaries, streaming_mask, aligned_logits
from experiments.diffusion_gemma_value_aware_gpu.kernels import block_statistics, query_state, route

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel validation')


@pytest.mark.parametrize('shape', [(128, 192), (17, 130), (129, 257), (128, 2048)])
@pytest.mark.parametrize('method,pool,threshold', [
    ('blasst', 'rms', 1.2), ('value', 'vector_mean', -.2), ('mass', 'rms', -.4),
    ('risk', 'mean', .2), ('aligned', 'rms', 1.2)])
def test_fused_statistics_and_streaming_match_reference(shape, method, pool, threshold):
    torch.manual_seed(71)
    nq, nk = shape
    scores = torch.randn(1, 2, nq, nk, device='cuda', dtype=torch.bfloat16)*3
    values = torch.randn(1, 2, nk, 32, device='cuda', dtype=torch.bfloat16)
    qi = torch.arange(nq, device='cuda')[:, None]+nk-nq
    ki = torch.arange(nk, device='cuda')[None, :]
    valid = ((ki <= qi) & (ki >= qi-255))[None, None].expand_as(scores).clone()
    valid[..., 0, :] = False
    valid[..., -3:] = False
    scores = scores.masked_fill(~valid, -torch.inf)
    meta = value_summaries(values, valid.any(-2))
    config = Config(method=method, pooling=pool, log_threshold=threshold)
    lognorm = (meta['token_norms']/meta['ref'][..., None]).clamp_min(1e-30).log() if method == 'aligned' else None
    stats = block_statistics(scores, valid, lognorm)
    masks, _ = route(stats, config, meta, threshold)
    for qb, start in enumerate(range(0, nq, 128)):
        state = block_state(scores[..., start:start+128, :], valid[..., start:start+128, :], values)
        got = query_state(stats, 1, 2, qb, min(128, nq-start))
        assert torch.equal(state['count'], got['count'])
        assert torch.equal(state['b'], got['b'])
        torch.testing.assert_close(state['logz'], got['logz'], atol=2e-5, rtol=2e-6)
        expected = streaming_mask(state, meta, config)
        assert torch.equal(expected, masks[:, qb, :].reshape_as(expected))
        kept = state['active'] & ~expected[..., None, :]
        assert torch.equal(kept.any(-1), state['active'].any(-1))


def test_strict_threshold_ties_and_blasst_preceding_seen_maximum():
    scores = torch.zeros(1, 1, 128, 192, device='cuda', dtype=torch.bfloat16)
    valid = torch.ones_like(scores, dtype=torch.bool)
    stats = block_statistics(scores, valid)
    config = Config(method='blasst')
    tied, _ = route(stats, config, {}, 0.)
    aggressive, _ = route(stats, config, {}, .1)
    assert not tied.any()
    assert aggressive[0, 0].tolist() == [False, True, True]


@pytest.mark.parametrize('method,pool,threshold', [
    ('dense', 'rms', 0.), ('blasst', 'rms', 1.2), ('value', 'vector_mean', -.2),
    ('mass', 'rms', -.4), ('risk', 'mean', .2), ('aligned', 'rms', 1.2)])
def test_gpu_adapter_gqa_output_and_counters_match_reference(method, pool, threshold):
    from types import SimpleNamespace
    from experiments.diffusion_gemma_value_aware.routing import Attention as Reference
    from experiments.diffusion_gemma_value_aware_gpu.routing import Attention as Accelerated
    torch.manual_seed(73)
    q = torch.randn(1, 4, 129, 32, device='cuda', dtype=torch.bfloat16)
    k, v = [torch.randn(1, 2, 270, 32, device='cuda', dtype=torch.bfloat16) for _ in range(2)]
    mask = torch.ones((1, 1, 129, 270), device='cuda', dtype=torch.bool)
    mask[..., 0, :] = False
    mask[..., -1] = False
    module = SimpleNamespace(num_key_value_groups=2, layer_idx=0, training=False,
        is_sliding=True, _blasst_2d_runtime=SimpleNamespace(current_denoising_iteration=0))
    config = dict(method=method, pooling=pool, log_threshold=threshold)
    ref, fast = Reference(config), Accelerated(config, validate=True)
    expected = ref(module, q, k, v, mask, scaling=.5, is_causal=False, sliding_window=256)[0]
    actual = fast(module, q, k, v, mask, scaling=.5, is_causal=False, sliding_window=256)[0]
    assert torch.equal(expected, actual)
    for a, b in zip(sorted(ref.records, key=lambda r:r['head']), sorted(fast.records, key=lambda r:r['head'])):
        for field in ('eligible', 'skipped', 'prefix_eligible', 'prefix_skipped', 'canvas_eligible', 'canvas_skipped', 'boundary_eligible', 'boundary_skipped', 'rows'):
            assert a[field] == b[field]
        for field in ('mass_sum', 'error_sq', 'dense_sq'):
            assert a[field] == pytest.approx(b[field], rel=3e-5, abs=3e-4)
