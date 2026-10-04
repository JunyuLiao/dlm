"""v31 progress clock, round 4 (CPU): Junyu Liao's C gate tracked inside the clock matches his reference
implementation (value_direction_hopper/query_adaptive.State with enable_cgate), several thresholds give one
re-selection per crossing, the 'settle' signal, and the budget that follows progress."""
import torch

from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
from experiments.value_direction_hopper.query_adaptive import State

TYPES = ['sliding_attention'] * 5 + ['full_attention']


def _adapter(**kw):
    a = VllmMethodAdapter(TYPES, arm='mage', mage_select='fa4', mage_granularity='qblock_max', mage_k=4096,
                          mage_select_step=1, **kw)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0, mage_selections=0, mage_reused_calls=0)
    return a


def _steps(n=24, v=7, steps=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(steps):
        logits = torch.randn(1, n, v, generator=g) * (1 + 2 * i)                      # sharper as denoising proceeds
        acc = torch.rand(1, n, generator=g) < 0.2 + 0.15 * i
        out.append((logits, acc))
    return out


def _stats(logits, n):
    x = logits[0, :n].float()
    return x.argmax(-1), (x.amax(-1) - torch.logsumexp(x, -1)).exp()


def test_jcgate_matches_junyu_reference():
    n = 24
    ref = State('T', None, m_ref=1.0, fast_t=True, diagnostics=False)
    ref.enable_cgate()
    a = _adapter(mage_reselect_trigger=[1.0], mage_trigger_signal='settle', mage_row_weight='jcgate')
    a.canvas_id, a._canvas_step = 3, 0
    canvas = torch.zeros(1, n)
    ref.begin(48, canvas)
    for i, (logits, acc) in enumerate(_steps(n)):
        ref.observe_logits(logits, acc, 48 - i)
        ref.begin(47 - i, canvas)                                  # the weights the next call would use
        a._canvas_step = i + 1
        arg, p_top = _stats(logits, n)
        a._progress_observe(arg, p_top, acc[0], logits, n, 1.0)
        jc = a._jc
        settled = (1 - torch.exp(-jc['r'] / 2.5)) * (1 - jc['q']) * (1 - (1 - p_top).clamp_min(0).sqrt())
        mine = (1 + 3.0 * (1 - settled)).clamp(1, 4)
        assert torch.allclose(mine, ref.used_weights[0], atol=1e-6), (i, mine, ref.used_weights[0])
    assert a.calls['trigger_checks'] == 5 and a.calls.get('triggers', 0) == 0      # threshold 1.0 never reached


def test_thresholds_trigger_once_each_with_weights_and_budget():
    n = 8
    a = _adapter(mage_reselect_trigger=[0.25, 0.5, 0.75], mage_row_weight='cgate', mage_reselect_kmin=1024)
    a.canvas_id = 1
    logits = torch.zeros(1, n, 5)
    fr = [0.125, 0.375, 0.375, 0.875, 1.0]                         # crosses 0.25 at step 2, 0.5 + 0.75 at step 4
    arg, p_top = _stats(logits, n)
    sched = []
    for step, f in enumerate(fr, 1):
        a._canvas_step = step
        acc = torch.arange(n) < round(f * n)
        before = a._trig_at
        a._progress_observe(arg, p_top, acc, logits, n, 1.0)
        if a._trig_at != before:
            sched.append((a._trig_at, a._trig_k, a._mage_row_w.clone()))
    assert [s[0] for s in sched] == [(1, 2), (1, 4)]
    assert sched[0][1] == int(4096 * (1 - 0.375)) // 64 * 64 == 2560
    assert sched[1][1] == 1024                                     # max(kmin, 4096 * (1 - 0.875) = 512)
    assert torch.equal(sched[1][2], (~(torch.arange(n) < 7)).float())
    assert a.calls['triggers'] == 2 and a.calls['trigger_checks'] == 4            # no read after the last threshold
    a.canvas_id, a._canvas_step = 2, 1                             # a new canvas resets the clock
    a._progress_observe(arg, p_top, torch.ones(n, dtype=torch.bool), logits, n, 1.0)
    assert a._trig_at == (2, 1) and a._trig_count == 3


def test_single_threshold_keeps_round3_behaviour():
    n = 8
    a = _adapter(mage_reselect_trigger=0.5, mage_row_weight='cgate')
    assert a.mage_reselect_trigger == (0.5,)
    a.canvas_id = 1
    logits = torch.zeros(1, n, 5)
    arg, p_top = _stats(logits, n)
    for step, k in enumerate([2, 3, 4, 6, 8], 1):
        a._canvas_step = step
        a._progress_observe(arg, p_top, torch.arange(n) < k, logits, n, 1.0)
    assert a._trig_at == (1, 3) and a._trig_k is None and a.calls['triggers'] == 1 and a.calls['trigger_checks'] == 3


def test_invalid_round4_configurations():
    bad = [dict(mage_reselect_trigger=[0.5, 1.5]), dict(mage_trigger_signal='settle'),
           dict(mage_reselect_trigger=0.5, mage_trigger_signal='bogus'), dict(mage_reselect_kmin=1024),
           dict(mage_reselect_trigger=0.5, mage_reselect_kmin=32), dict(mage_row_weight='jcgate', mage_reselect=[3])]
    for kw in bad:
        try:
            _adapter(**kw)
            raise AssertionError(f'accepted {kw}')
        except ValueError:
            pass


def test_clock_trace_records_both_signals_per_canvas():
    n = 8
    a = _adapter(mage_reselect_trigger=0.5, mage_clock_trace=True)
    logits = torch.zeros(1, n, 5)
    arg, p_top = _stats(logits, n)
    for canvas, ks in ((1, [2, 4, 6]), (2, [8])):
        a.canvas_id = canvas
        for step, k in enumerate(ks, 1):
            a._canvas_step = step
            a._progress_observe(arg, p_top, torch.arange(n) < k, logits, n, 1.0)
    tr = a._clock_receipt()['clock_trace']
    assert [[x[0] for x in c] for c in tr] == [[0.25, 0.5, 0.75], [1.0]]
    assert all(0.0 <= x[1] <= 1.0 for c in tr for x in c)
    assert a.calls['triggers'] == 2 and a.calls['trigger_checks'] == 3            # tracing kept the clock reading
    assert _adapter(mage_reselect_trigger=0.5)._clock_receipt() == {}


def test_sink_and_recent_tiles_are_kept_within_the_budget():
    torch.manual_seed(0)
    H, n, pt, kt, qb, G = 16, 256, 40, 44, 2, 8
    head_lse = torch.randn(H, n, pt)
    head_lse[:, :, 10] += 9.0                                      # one clearly dominant scored tile everywhere
    mass = torch.softmax(torch.cat([head_lse, torch.randn(H, n, kt - pt)], -1), -1)
    a = _adapter(mage_sink=1, mage_recent=128)
    kept = a._mage_units(mass, head_lse, H, n, qb, kt, pt, 4, G)
    pre = kept[0, :, :, :pt]
    assert bool((pre.sum(-1) == 4).all())                          # exactly k tiles per unit
    assert bool(pre[..., 0].all() and pre[..., pt - 1].all() and pre[..., pt - 2].all() and pre[..., 10].all())
    assert bool(kept[0, :, :, pt:].all())
    for bad in (dict(mage_recent=100), dict(mage_sink=64), dict(mage_sink=-1)):
        try:
            _adapter(**bad)
            raise AssertionError(f'accepted {bad}')
        except ValueError:
            pass


def test_relative_progress_ignores_rows_accepted_at_the_first_step():
    n = 10
    logits = torch.zeros(1, n, 5)
    arg, p_top = _stats(logits, n)
    fired = {}
    for rel in (False, True):
        a = _adapter(mage_reselect_trigger=0.5, mage_trigger_relative=rel)
        a.canvas_id = 1
        for step, k in enumerate([6, 7, 8, 10], 1):                # absolute 0.6 at step 1; relative 0.25, 0.5, 1.0
            a._canvas_step = step
            a._progress_observe(arg, p_top, torch.arange(n) < k, logits, n, 1.0)
        fired[rel] = a._trig_at
    assert fired == {False: (1, 1), True: (1, 3)}
    try:
        _adapter(mage_trigger_relative=True)
        raise AssertionError('relative progress accepted without a trigger')
    except ValueError:
        pass


def test_split_tensors_deal_each_unit_into_balanced_parts():
    torch.manual_seed(1)
    kept = torch.rand(1, 3, 2, 40) < 0.3
    kept[..., 36:] = True
    counts, idx = VllmMethodAdapter._split_tensors(kept, 2)
    assert counts.shape == (2, 3, 2) and idx.shape == (2, 3, 2, 40)
    for h in range(3):
        for b in range(2):
            want = torch.nonzero(kept[0, h, b]).flatten().tolist()
            got = idx[0, h, b, :counts[0, h, b]].tolist() + idx[1, h, b, :counts[1, h, b]].tolist()
            assert got == want                                         # disjoint, complete, ascending
            assert abs(int(counts[0, h, b]) - int(counts[1, h, b])) <= 1


def test_pool_routes_reselections_and_is_stored_with_the_selection():
    from experiments.numerical_qk_reuse import v27_fa4
    v27_fa4.block_sparse_tensors = lambda kept, q_block=128: ('lists', kept.clone())
    v27_fa4.dense = lambda q, k, v, s: 'dense'
    v27_fa4.sparse_lists = lambda q, k, v, lists, s: 'sparse'
    calls = []

    def fake_select(self, q, k, v, scale, prefix, n):
        kept = torch.zeros((1, 16, 2, (prefix + n) // 64), dtype=torch.bool)
        kept[..., prefix // 64:] = True
        if self.mage_pool is not None:
            self._last_pool = kept.clone()
        calls.append('dense_observe')
        return 'select', kept

    def fake_pool(self, q, k, v, scale, prefix, n, pool):
        calls.append(('pool', int(pool.sum())))
        return 'pool_select', pool.clone()
    o1, o2 = VllmMethodAdapter._mage_select_fa4, VllmMethodAdapter._mage_select_pool
    VllmMethodAdapter._mage_select_fa4, VllmMethodAdapter._mage_select_pool = fake_select, fake_pool
    try:
        a = _adapter(mage_carry_first=True, mage_reselect_trigger=0.5, mage_pool=4)
        q, b, prefix, n = torch.zeros(1, 16, 256, 4), dict(k=None, v=None), 64 * 40, 256
        a.canvas_id = 1
        assert [a._mage(5, q, b, 1.0, prefix, n) for _ in range(2)] == ['dense', 'select']
        assert a.mage_state[5]['pool'] is not None
        a._trig_canvas, a._trig_at = 1, (1, 2)
        assert a._mage(5, q, b, 1.0, prefix, n) == 'pool_select'
        assert calls == ['dense_observe', ('pool', 16 * 2 * 4)]
    finally:
        VllmMethodAdapter._mage_select_fa4, VllmMethodAdapter._mage_select_pool = o1, o2
    for bad in (dict(mage_pool=4), dict(mage_reselect_trigger=0.5, mage_pool=1)):
        try:
            _adapter(**bad)
            raise AssertionError(f'accepted {bad}')
        except ValueError:
            pass


def test_coverage_budget_is_one_balanced_k_per_canvas():
    H, n, pt, qb = 16, 256, 400, 2
    a = _adapter(mage_kcover=0.9, mage_kq=0.75, mage_kmax=16384)
    peaked = torch.full((H, n, pt), -30.0)
    peaked[..., 5] = 0.0                                           # all mass on one tile: k stays at the floor
    assert a._coverage_tiles(peaked, n, qb, pt, 64) == 64
    flat = torch.zeros(H, n, pt)                                   # uniform: 90% needs 360 tiles, capped at 256
    assert a._coverage_tiles(flat, n, qb, pt, 64) == 256
    mixed = flat.clone()
    mixed[:12] = peaked[:12]                                       # 12 of 16 heads peaked -> the 0.75 quantile is peaked
    k = a._coverage_tiles(mixed, n, qb, pt, 64)
    assert k == 64 and a.calls['kcover_selections'] == 3
    for bad in (dict(mage_kcover=1.0), dict(mage_kcover=0.9, mage_kmax=2048)):
        try:
            _adapter(**bad)
            raise AssertionError(f'accepted {bad}')
        except ValueError:
            pass


if __name__ == '__main__':
    import sys
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    sys.exit(0)
