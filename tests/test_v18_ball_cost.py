"""CPU gates for the bounded parent/metadata timing diagnostic."""
import numpy as np
import pytest
import torch

from scripts.v18_ball_cost import split_tiles, timing_summary
from scripts.v18_ball_metadata import cpu_reference


def test_static_prefix_excludes_canvas_boundary_tile():
    assert split_tiles(130, 65) == (1, 3)
    assert split_tiles(64, 64) == (0, 1)
    with pytest.raises(ValueError):
        split_tiles(64, 65)


def test_per_cta_share_is_normalized_per_record_not_summed_as_wall():
    timings = torch.tensor([[[[[10, 10, 20, 10, 40, 10],
                                 [10, 10, 0, 0, 80, 0]]]]], dtype=torch.uint64)
    result = timing_summary(timings)
    assert result['active_ctas'] == 2
    assert result['fields']['pz']['nonzero_ctas'] == 1
    assert np.isclose(result['pz_over_serial_router_stage_per_cta_median'], 1 / 6)


def test_cpu_reference_partial_tile_center_and_radius():
    z = torch.zeros((1, 1, 65, 32), dtype=torch.float32)
    z[..., 64, 0] = 7
    ref = cpu_reference(z)
    assert ref.shape == (1, 1, 2, 33)
    assert ref[0, 0, 1, 0] == 7
    assert ref[0, 0, 1, 32] == 0
