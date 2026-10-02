"""v27 C gate (group member's query sensitivity, ported from the design note): exact weight dynamics and config guards."""
import math

import pytest
import torch

BASE = {'diagnostic': False, 'policy': {'local': {'log_threshold': -3.0}, 'global': {'log_threshold': -2.0}}}


def _logits(probs_top, vocab=8):
    """[1, Q, V] logits whose processed top-1 probability at position i is probs_top[i] (top token = 0)."""
    rows = []
    for p in probs_top:
        rest = (1 - p) / (vocab - 1)
        rows.append([math.log(p)] + [math.log(rest)] * (vocab - 1))
    return torch.tensor([rows], dtype=torch.float32)


def _expected(q, r, u, beta=3.0, tau=2.5):
    g = 1 - math.exp(-r / tau)
    return min(max(1 + beta * (1 - g * (1 - q) * (1 - u)), 1.0), 1 + beta)


def test_cgate_weights_follow_the_design_note():
    from experiments.value_direction_hopper.query_adaptive import State
    st = State('T', None, m_ref=1., beta=3., gamma=.5, diagnostics=False, fast_t=True)
    st.enable_cgate(tau=2.5, gamma_q=.65)
    canvas = torch.zeros(1, 3, dtype=torch.long)
    st.begin(48, canvas)                                     # first call of a canvas: fully protected
    assert torch.equal(st.used_weights, torch.full((1, 3), 4.0))
    p1 = [0.99, 0.30, 0.80]
    acc1 = torch.tensor([[True, False, True]])
    st.observe_logits(_logits(p1), acc1, 48)
    st.begin(47, canvas)
    want = []
    for i in range(3):
        a = float(acc1[0, i])
        q = .65 * 1 + .35 * (1 - a)
        r = (0 + 1) * a                                      # no flip information at the first call
        u = math.sqrt(max(1 - p1[i], 0))
        want.append(_expected(q, r, u))
    torch.testing.assert_close(st.used_weights, torch.tensor([want]), rtol=1e-5, atol=1e-5)
    # second call: position 0 flips its top-1 token, position 2 stays and is accepted again
    p2 = [0.95, 0.40, 0.90]
    l2 = _logits(p2)
    l2[0, 0] = l2[0, 0].roll(1)                              # top token of position 0 moves -> flip
    acc2 = torch.tensor([[True, True, True]])
    q_prev = [.65 + .35 * (1 - float(acc1[0, i])) for i in range(3)]
    r_prev = [float(acc1[0, i]) for i in range(3)]
    st.observe_logits(l2, acc2, 47)
    st.begin(46, canvas)
    want = []
    for i in range(3):
        q = .65 * q_prev[i] + .35 * 0.0
        r = 0.0 if i == 0 else (r_prev[i] + 1) * 1.0
        u = math.sqrt(max(1 - p2[i], 0))
        want.append(_expected(q, r, u))
    torch.testing.assert_close(st.used_weights, torch.tensor([want]), rtol=1e-5, atol=1e-5)
    assert float(st.used_weights[0, 0]) == pytest.approx(4.0)   # flipped: r = 0 -> g = 0 -> fully protected
    st.begin(48, canvas)                                     # a new canvas resets the C-gate history
    assert torch.equal(st.used_weights, torch.full((1, 3), 4.0))


def test_cgate_requires_the_fast_t_path():
    from experiments.value_direction_hopper.query_adaptive import State
    with pytest.raises(ValueError):
        State('C', None, m_ref=1., diagnostics=False).enable_cgate()
    with pytest.raises(ValueError):
        State('T', None, m_ref=1., diagnostics=False, fast_t=False).enable_cgate()


def test_cgate_config_guards():
    from experiments.numerical_qk_reuse import v21
    arm, scope = 'M3_R3_A8_current_output', 'GLOBAL_ONLY_NATIVE_LOCAL'
    line = dict(output_score_precision='fp32_scores_bf16_pv', bootstrap_policy='native_bootstrap2_observe1',
                output_layout='model_major', consumer64=2, memory_caps='long', fa4_consumer=True, score_period=64,
                fused_observe=True, async_route=True, risk_state='dense_prefix', decision_interval=6,
                threshold_shift='minus_ln2', carry_first=True)
    cfg = v21.effective_config(dict(BASE), arm, scope, sensitivity='cgate', **line)
    v21.validate_effective(cfg, cfg['condition'])
    plain = v21.effective_config(dict(BASE), arm, scope, **line)
    assert cfg['sensitivity'] == 'cgate' and 'sensitivity' not in plain and cfg['fingerprint'] != plain['fingerprint']
    with pytest.raises(ValueError, match='C gate'):
        v21.effective_config(dict(BASE), arm, scope, sensitivity='other', **line)
