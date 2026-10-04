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


if __name__ == '__main__':
    import sys
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
    sys.exit(0)
