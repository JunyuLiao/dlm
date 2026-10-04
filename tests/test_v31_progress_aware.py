"""v31 progress-aware re-selection on the MAGE port (MAGE_RESELECT, MAGE_ROWW): re-selection at the listed canvas
calls, the carry using the latest selection, row weights in the worst-row statistic (CPU; FA4 entry points stubbed)."""
import math

import torch

from experiments.numerical_qk_reuse import v27_fa4
from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter

TYPES = ['sliding_attention'] * 5 + ['full_attention']
H, HK, G = 16, 2, 8


def _adapter(**kw):
    a = VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', mage_granularity='qblock_max', mage_k=64, **kw)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    return a


def test_weights_steer_the_block_statistic():
    torch.manual_seed(0)
    n, pt, kt, qb = 256, 40, 44, 2
    head_lse = torch.randn(H, n, pt) * 0.1
    head_lse[3, 5, 10] = 3.0                      # row 5 (block 0) of head 3: tile 10 holds the largest share
    head_lse[3, 9, 20] = 2.5                      # row 9 (block 0) of head 3: tile 20, a smaller share
    mass = torch.softmax(torch.cat([head_lse, torch.randn(H, n, kt - pt)], -1), -1)
    a = _adapter(mage_select_step=1, mage_reselect=[3], mage_row_weight='cgate')
    plain = a._mage_units(mass, head_lse, H, n, qb, kt, pt, 1, G)
    assert bool(plain[0, 3, 0, 10]) and not bool(plain[0, 3, 0, 20])
    w = torch.ones(n)
    w[5] = 0.0                                    # row 5 was accepted: it no longer decides
    a._mage_units_w = w
    weighted = a._mage_units(mass, head_lse, H, n, qb, kt, pt, 1, G)
    assert bool(weighted[0, 3, 0, 20]) and not bool(weighted[0, 3, 0, 10])
    assert bool((weighted[0, :, :, :pt].sum(-1) == 1).all())          # the per-unit budget is unchanged
    w = torch.ones(n)
    w[:128] = 0.0                                 # every row of block 0 accepted: block 0 falls back to unweighted
    a._mage_units_w = w
    fallback = a._mage_units(mass, head_lse, H, n, qb, kt, pt, 1, G)
    assert torch.equal(fallback[0, :, 0], plain[0, :, 0])


def test_conf_weights_follow_juyu_prior():
    a = _adapter(mage_select_step=1, mage_reselect=[4], mage_row_weight='conf')
    scaled = torch.full((1, 4, 6), -20.0)
    scaled[0, 0, 0] = 20.0                        # confident row: p_top ~ 1 -> weight 1
    scaled[0, 1] = 0.0                            # uniform row: p_top = 1/6 -> 1 + 3 sqrt(5/6)
    w = a._row_weight_from_logits(scaled, 2, entropy_bound=1.0)
    assert abs(float(w[0]) - 1.0) < 1e-4
    assert abs(float(w[1]) - (1 + 3 * math.sqrt(5 / 6))) < 1e-4


def test_reselect_replaces_the_held_lists_and_feeds_the_carry():
    log, sel = [], []
    v27_fa4.block_sparse_tensors = lambda kept, q_block=128: ('lists', kept.clone())
    v27_fa4.dense = lambda q, k, v, s: (log.append('dense'), 'dense')[1]
    v27_fa4.sparse_lists = lambda q, k, v, lists, s: (log.append(('sparse', lists[1])), 'sparse')[1]

    def fake_select(self, q, k, v, scale, prefix, n):
        kept = torch.zeros((1, H, 2, (prefix + n) // 64), dtype=torch.bool)
        kept[..., prefix // 64:] = True
        kept[..., len(sel) % (prefix // 64)] = True                # each selection differs
        sel.append((kept, self._mage_units_w))
        return 'select', kept
    orig = VllmMethodAdapter._mage_select_fa4
    VllmMethodAdapter._mage_select_fa4 = fake_select
    try:
        a = _adapter(mage_select_step=1, mage_carry_first=True, mage_reselect=[3], mage_row_weight='cgate')
        q, b, prefix, n = torch.zeros(1, H, 256, 4), dict(k=None, v=None), 64 * 40, 256
        a.canvas_id = 1
        a._mage_row_w = torch.ones(n)
        out = [a._mage(5, q, b, 1.0, prefix, n) for _ in range(6)]
        assert out == ['dense', 'select', 'sparse', 'select', 'sparse', 'sparse']
        assert a.calls['mage_selections'] == 1 and a.calls['mage_reselections'] == 1
        assert sel[0][1] is None and sel[1][1] is not None          # weights only in the re-selection
        assert torch.equal(log[-1][1], sel[1][0])                   # held calls use the re-selection
        a.canvas_id = 2                                             # next canvas: call 0 carries the LATEST selection
        assert a._mage(5, q, b, 1.0, prefix + n, n) == 'sparse'
        want = torch.ones((1, H, 2, (prefix + 2 * n) // 64), dtype=torch.bool)
        want[..., :prefix // 64] = sel[1][0][..., :prefix // 64]
        assert torch.equal(log[-1][1], want)
        assert [a._mage(5, q, b, 1.0, prefix + n, n) for _ in range(3)] == ['select', 'sparse', 'select']
    finally:
        VllmMethodAdapter._mage_select_fa4 = orig


def test_invalid_configurations_are_refused():
    bad = [dict(mage_select_step=1, mage_reselect=[1]), dict(mage_select_step=1, mage_row_weight='cgate'),
           dict(mage_select_step=1, mage_reselect=[3], mage_row_weight='bogus')]
    for kw in bad:
        try:
            _adapter(**kw)
            raise AssertionError(f'accepted {kw}')
        except ValueError:
            pass
    try:
        VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', mage_granularity='kvhead', mage_select_step=1,
                          mage_reselect=[3], mage_row_weight='cgate')
        raise AssertionError('row weights accepted on kvhead')
    except ValueError:
        pass


if __name__ == '__main__':
    import sys
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    sys.exit(0)
