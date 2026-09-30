"""v27 SparseD port: average-pooled top-k tile selection and its protocol contract (CPU)."""
import json

import pytest
import torch

from experiments.numerical_qk_reuse.v20_controls import SPARSED_KEEPS, SPARSED_SKIPS, sparsed_bitmap
from scripts.v21_freeze_panel import freeze_v27
from scripts.v21_run import validate_protocol
from tests.test_v27_panel import ARMS, _setup


def _scores(prefix_tiles=10, canvas=256, hot=(2, 7, 4)):
    """One head, 256 canvas queries; prefix tile j gets a higher score the earlier it appears in ``hot``."""
    nk = prefix_tiles * 64 + canvas
    scores = torch.zeros((1, 1, canvas, nk))
    for rank, j in enumerate(hot):
        scores[..., j * 64:(j + 1) * 64] = 5.0 - rank
    return scores, prefix_tiles * 64


def test_keeps_the_top_average_pooled_prefix_tiles_and_every_canvas_tile():
    scores, prefix = _scores()
    skipped, eligible = sparsed_bitmap(scores, prefix, 0.3)          # ceil(0.3 * 10) = 3 kept prefix tiles
    kept = (~skipped[0, 0]).tolist()
    for qb in range(2):
        assert [j for j in range(10) if kept[qb][j]] == [2, 4, 7]
        assert all(kept[qb][10:])                                     # canvas tiles are never skipped
    assert eligible.all()


def test_keep_ratio_sets_the_count():
    scores, prefix = _scores(prefix_tiles=20)
    for keep, n in ((0.1, 2), (0.2, 4), (0.3, 6)):
        skipped, _ = sparsed_bitmap(scores, prefix, keep)
        assert int((~skipped[0, 0, 0, :20]).sum()) == n


def test_rejects_unnamed_settings_and_bad_geometry():
    scores, prefix = _scores()
    with pytest.raises(ValueError):
        sparsed_bitmap(scores, prefix, 1.0)
    with pytest.raises(ValueError):
        sparsed_bitmap(scores, scores.shape[-1], 0.3)
    assert SPARSED_KEEPS == (0.1, 0.2, 0.3) and SPARSED_SKIPS == (1, 10)


def test_sparsed_arm_freezes_as_a_labelled_port(tmp_path):
    arms = dict(ARMS, SparseD_k30_s10=dict(kind='sparsed_fa4', keep=0.3, skip_steps=10),
                SparseD_k10_s1=dict(kind='sparsed_fa4', keep=0.1, skip_steps=1))
    sp, op, bp, pool = _setup(tmp_path, arms)
    protocol = freeze_v27(sp, op, bp, pool, tmp_path / 'out')
    contract = protocol['arm_contracts']['SparseD_k30_s10']
    assert contract['kind'] == 'v27_sparsed' and contract['keep'] == 0.3 and contract['skip_steps'] == 10
    assert contract['scope'] == 'GLOBAL_ONLY_NATIVE_LOCAL' and contract['consumer'] == 'fa4'
    validate_protocol(json.loads((tmp_path / 'out' / 'protocol.json').read_bytes()))


def test_unnamed_sparsed_setting_is_refused(tmp_path):
    arms = dict(ARMS, SparseD_bad=dict(kind='sparsed_fa4', keep=0.25, skip_steps=10))
    sp, op, bp, pool = _setup(tmp_path, arms)
    with pytest.raises(ValueError):
        freeze_v27(sp, op, bp, pool, tmp_path / 'out')
