"""CPU checks for the v25 aligned16 route-storage variant (no GPU, no model)."""
import math

import pytest
import torch

from experiments.numerical_qk_reuse import integration
from experiments.numerical_qk_reuse.integration import Attention, PREQK_MODE


def _owner(storage='aligned16', precision='fp32_scores_bf16_pv', mode=PREQK_MODE):
    owner = Attention.__new__(Attention)
    owner.route_storage, owner.output_score_precision, owner.output_mode = storage, precision, mode
    owner.aligned_score_copies = owner.aligned_pad_bytes = owner.aligned_sketch_pads = 0
    owner.aligned_extra_pad_bytes = owner.aligned_copy_bytes = 0
    return owner


def test_logical_storage_is_the_same_tensor():
    score = torch.randn(1, 4, 8, 37)
    assert _owner('logical')._store_scores(score) is score


def test_aligned16_copies_exact_bits_with_minus_inf_tail():
    score = torch.randn(1, 4, 8, 37)
    score[..., 3] = -math.inf
    score[0, 1, 2, 5] = float('nan')
    owner = _owner()
    stored = owner._store_scores(score)
    assert stored.shape == (1, 4, 8, 48) and stored.is_contiguous()
    assert torch.equal(stored[..., :37].contiguous().view(torch.int32), score.view(torch.int32))
    assert bool(torch.isneginf(stored[..., 37:]).all())
    assert owner.aligned_score_copies == 1 and owner.aligned_pad_bytes == 1 * 4 * 8 * 48 * 4


def test_already_aligned_extent_is_not_copied():
    score = torch.randn(1, 2, 8, 64)
    owner = _owner()
    assert owner._store_scores(score) is score and owner.aligned_score_copies == 0


@pytest.mark.parametrize('precision,mode', [('legacy_bf16_scores', PREQK_MODE),
                                            ('fp32_scores_bf16_pv', 'cached_scores')])
def test_aligned16_refuses_paths_that_read_cache_as_logical_k(precision, mode):
    with pytest.raises(ValueError):
        _owner(precision=precision, mode=mode)._store_scores(torch.randn(1, 2, 4, 37))


def test_route_pads_only_the_sketch_and_keeps_reference(monkeypatch):
    seen = {}
    def fake(scores, z, ref, **kwargs):
        seen.update(scores=scores, z=z, ref=ref, kwargs=kwargs)
        return 'routed'
    monkeypatch.setattr(integration, 'route_only', fake)
    owner = _owner()
    stored = owner._store_scores(torch.randn(1, 4, 8, 37))
    z = torch.randn(1, 2, 37, 32)
    ref = torch.rand(1, 2) + .5
    assert owner._route(stored, z, ref, log_threshold=-3.) == 'routed'
    assert seen['z'].shape == (1, 2, 48, 32)
    assert torch.equal(seen['z'][:, :, :37], z) and not seen['z'][:, :, 37:].any()
    assert torch.equal(seen['ref'], ref) and seen['kwargs'] == {'log_threshold': -3.}
    assert owner.aligned_sketch_pads == 1


def test_route_rejects_pitch_that_changes_tiles_or_is_not_aligned16(monkeypatch):
    monkeypatch.setattr(integration, 'route_only', lambda *a, **k: None)
    owner = _owner()
    ref = torch.ones(1, 2)
    for pitch, nk in ((80, 60), (48, 30), (70, 60)):   # >=16 pad, >=16 pad, not multiple of 16
        with pytest.raises(ValueError):
            owner._route(torch.zeros(1, 4, 8, pitch), torch.zeros(1, 2, nk, 32), ref)
    with pytest.raises(ValueError):
        _owner('logical')._route(torch.zeros(1, 4, 8, 48), torch.zeros(1, 2, 37, 32), ref)


def test_v21_config_rejects_aligned16_without_fp32_output():
    from experiments.numerical_qk_reuse import v21
    with pytest.raises(ValueError):
        v21.effective_config({'diagnostic': False}, 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL',
                             output_score_precision='legacy_bf16_scores', route_storage='aligned16')


def test_ulp_gap_and_delta_report_real_distances():
    capture = pytest.importorskip('scripts.v21b_geometry_capture')
    a = torch.tensor([1.0, -2.0, 0.0, float('-inf')])
    b = a.clone()
    b[0] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    b[1] = torch.nextafter(torch.nextafter(torch.tensor(-2.0), torch.tensor(-3.0)), torch.tensor(-3.0))
    assert capture._float_ulp_gap(a, b) == 2
    delta = capture._tensor_delta(a, b)
    assert not delta['bitwise_equal'] and delta['differing_elements'] == 2
    assert delta['nonfinite_mismatch'] == 0 and delta['max_ulp'] == 2 and delta['max_abs'] > 0
    assert capture._tensor_delta(a, a.clone())['bitwise_equal']
