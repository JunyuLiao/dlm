"""CPU geometry checks for the explicitly historical LOCAL support diagnostic."""
import pytest
import torch

from scripts.v18_legacy_geometry_report import count_state


def state(window):
    return dict(q=torch.empty(1, 2, 3, 4), k=torch.empty(1, 1, 70, 4), mask=None,
                is_causal=False, sliding_window=window, source_id='T60/example',
                layer=0, layer_kind='local' if window else 'global',
                frontier_arm='T60', derived_mask=False)


def test_global_geometry_same_as_native_and_local_omits_old_keys():
    global_row = count_state(state(None))
    local_row = count_state(state(4))
    assert global_row['native_legal_pairs'] == 420
    assert global_row['legacy_repro_legal_pairs'] == 420
    assert local_row['native_legal_pairs'] == 420
    # Historical noncausal window removes older keys but retains later keys.
    assert local_row['legacy_repro_legal_pairs'] == 30
    assert local_row['native_legal_pairs_omitted_by_legacy'] == 390


def test_derived_and_wrong_arm_rejected():
    with pytest.raises(ValueError):
        count_state(dict(state(4), derived_mask=True))
    with pytest.raises(ValueError):
        count_state(dict(state(4), frontier_arm='T50'))
