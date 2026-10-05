"""v14 CVM-T qualification: exact T fast path, declared planner rule, v5 kernel parity."""
import math
import os

import pytest
import torch

V5 = os.environ.get('V14_V5_BUILD')
SUPPORT = os.environ.get('V11_SUPPORT_BUILD')
JUNYU = dict(library='/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/value_direction_db080045f7a5fbce.so',
             torch_library='/home/exouser/dyh/numerical_qk_reuse_native_20260924/build_cp1/torch_4c65c048754f9fb7/'
                           'value_direction_torch_4c65c048754f9fb7.so')
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


# ------------------------------------------------------------------ T fast path
def run_state(fast, schedule, seed=0, vocab=37, q=13):
    from experiments.value_direction_hopper.query_adaptive import State
    g = torch.Generator().manual_seed(seed)
    st = State('T', None, m_ref=14.2, beta=3., gamma=.5, diagnostics=False, fast_t=fast)
    weights = []
    for cur_step in schedule:
        st.begin(cur_step, torch.zeros(1, q, dtype=torch.long))
        weights.append(None if st.used_weights is None else st.used_weights.clone())
        logits = torch.randn(1, q, vocab, generator=g)
        logits[0, :4, :2] = 5.                                   # exact ties: argmax picks the first index
        if cur_step % 3 == 0:
            logits[0, 5] = logits[0, 5].flip(-1)                  # forced flips
        st.observe_logits(logits, torch.ones(1, q, dtype=torch.bool), cur_step)
    return weights, st


@pytest.mark.parametrize('schedule', [
    list(range(48, 40, -1)),                                        # one canvas
    list(range(48, 44, -1)) + list(range(48, 38, -1)),              # partial canvas, then reset
    [48, 47] + [48, 47, 46, 45] + [48],                             # repeated short canvases (two-step bootstrap)
])
def test_fast_t_weight_sequence_is_identical_to_frozen(schedule):
    slow, s1 = run_state(False, schedule)
    fast, s2 = run_state(True, schedule)
    assert len(slow) == len(fast)
    for a, b in zip(slow, fast):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.equal(a, b) and a.dtype == b.dtype and a.is_contiguous() and b.is_contiguous()
    assert torch.equal(s1.temporal, s2.temporal) and torch.equal(s1.previous_top, s2.previous_top)
    assert s2.margin is None and s2.t_history                       # no margin work, explicit sentinel


def test_fast_t_first_two_iterations_have_no_weight_or_unit_weight():
    weights, _ = run_state(True, [48, 47, 46])
    assert weights[0] is None and torch.equal(weights[1], torch.ones_like(weights[1]))


def test_fast_t_refuses_non_exact_configurations():
    from experiments.value_direction_hopper.query_adaptive import State
    with pytest.raises(ValueError):
        State('MT', None, m_ref=1., diagnostics=False, fast_t=True)
    with pytest.raises(ValueError):
        State('T', None, m_ref=1., diagnostics=True, fast_t=True)


# ------------------------------------------------------------------ planner rule
def one_tile(log_rho_value, s):
    from experiments.value_direction_hopper.cvm import plan_reference
    log_rho = torch.full((1, 1, 1, 128), -math.inf)
    log_rho[0, 0, 0, 7] = math.log(log_rho_value)                  # one sensitive row
    sens = torch.ones(1, 128)
    sens[0, 7] = s
    return log_rho, sens


def test_declared_rule_example_drop_restore_monotone():
    from experiments.value_direction_hopper.cvm import plan_reference
    tau = math.log(.2)
    skipped = torch.ones(1, 1, 1, 1, dtype=torch.bool)             # anchor dropped the tile
    for s, expect in ((1., True), (2.5, False), (1.75, False)):     # drop, restore, stays restored
        log_rho, sens = one_tile(.1, s)
        skipped = plan_reference(log_rho, sens, skipped, 1, tau)
        assert bool(skipped[0, 0, 0, 0]) is expect
    log_rho, sens = one_tile(.1, 2.)                                 # exactly at threshold: retained
    assert not plan_reference(log_rho, sens, torch.ones(1, 1, 1, 1, dtype=torch.bool), 1, tau)[0, 0, 0, 0]


def test_row_specific_sensitivity_is_not_averaged_and_protect_limits_scope():
    from experiments.value_direction_hopper.cvm import plan_reference
    tau = math.log(.2)
    log_rho = torch.full((1, 1, 2, 128), math.log(.1))
    sens = torch.ones(1, 128)
    sens[0, 3] = 2.5                                                 # one hot row -> restores its tile
    skipped = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    out = plan_reference(log_rho, sens, skipped, 1, tau)             # tile 1 is outside the prunable prefix
    assert not out[0, 0, 0, 0] and out[0, 0, 0, 1]
    assert plan_reference(log_rho, sens.mean(-1, keepdim=True).expand(1, 128), skipped, 1, tau)[0, 0, 0, 0]


@cuda
def test_gpu_planner_matches_reference():
    from experiments.value_direction_hopper.cvm import plan, plan_reference
    g = torch.Generator().manual_seed(3)
    for q, kt, protect in ((256, 40, 37), (200, 9, 9), (256, 133, 128)):
        log_rho = torch.randn(1, 16, kt, q, generator=g) - 1.2
        log_rho[0, 3, :, 5] = math.inf
        log_rho[0, 4, :, :64] = -math.inf
        sens = 1 + 3 * torch.rand(1, q, generator=g)
        skipped = torch.rand(1, 16, (q + 127) // 128, kt, generator=g) > .4
        ref = plan_reference(log_rho, sens, skipped, protect, math.log(.3))
        got = plan(log_rho.cuda().contiguous(), sens.cuda().contiguous(), skipped.cuda().contiguous(), protect, math.log(.3))
        assert torch.equal(got.cpu(), ref)


# ------------------------------------------------------------------ v5 kernel qualification
def fresh_inputs(prefix, seed, d=512, hk=2, h=16, nq=256):
    from experiments.value_direction_hopper.masks import geometry
    g = torch.Generator(device='cuda').manual_seed(seed)
    nk = prefix + nq
    q = (torch.randn(1, h, nq, d, generator=g, device='cuda') * .3).to(torch.bfloat16).contiguous()
    k = (torch.randn(1, hk, nk, d, generator=g, device='cuda') * .3).to(torch.bfloat16).contiguous()
    v = torch.randn(1, hk, nk, d, generator=g, device='cuda').to(torch.bfloat16).contiguous()
    z = torch.randn(1, hk, nk, 32, generator=g, device='cuda').contiguous()
    ref = (torch.rand(1, hk, generator=g, device='cuda') + .5).contiguous()
    sens = (1 + 3 * torch.rand(1, nq, generator=g, device='cuda')).contiguous()
    packed, _ = geometry(1, nq, nk, device='cuda')
    return q, k, v, z, ref, sens, packed


def v5_call(q, k, v, z, ref, sens, packed, *, export=False, protect=-1, threshold=-1.0):
    from experiments.value_direction_hopper.cvm import V5Kernel
    kern = V5Kernel()
    kern.export, kern.protect_tile = export, protect
    out = kern(q, k, v, z, ref, mask=packed, scale=q.shape[-1] ** -.5, log_threshold=threshold, mode='value',
               precision='tf32x3_register', tma=True, sensitivity=sens)
    return out, kern.last_export


needs_v5 = pytest.mark.skipif(not (torch.cuda.is_available() and V5 and SUPPORT), reason='CUDA + V14_V5_BUILD + V11_SUPPORT_BUILD')


@needs_v5
@pytest.mark.parametrize('prefix,threshold', [(129, -1.0), (1645, -1.0), (1645, -3.1366905212402343), (3000, 0.0)])
def test_v5_without_protect_or_export_is_bit_identical_to_junyu_v4(prefix, threshold):
    from experiments.value_direction_hopper.cuda import Kernel
    from experiments.value_direction_hopper.cvm import load_v5
    load_v5(V5)
    v4 = Kernel(JUNYU['library'], torch_library=JUNYU['torch_library'])
    x = fresh_inputs(prefix, prefix)
    old = v4(*x[:5], mask=x[6], scale=512 ** -.5, log_threshold=threshold, mode='value', precision='tf32x3_register',
             tma=True, sensitivity=x[5])
    new, _ = v5_call(*x, threshold=threshold)
    newx, exported = v5_call(*x, threshold=threshold, export=True)
    torch.cuda.synchronize()
    for a, b, c in ((old.output, new.output, newx.output), (old.skipped, new.skipped, newx.skipped),
                    (old.eligible, new.eligible, newx.eligible), (old.log_normalizer, new.log_normalizer, newx.log_normalizer)):
        assert torch.equal(a.view(torch.uint8) if a.dtype != torch.bool else a, b.view(torch.uint8) if b.dtype != torch.bool else b)
        assert torch.equal(b.view(torch.uint8) if b.dtype != torch.bool else b, c.view(torch.uint8) if c.dtype != torch.bool else c)
    assert exported.shape == (1, 16, (prefix + 256 + 63) // 64, 256)


@needs_v5
@pytest.mark.parametrize('prefix', [1645, 3000])
def test_protect_is_coherent_across_roles_and_export_reconstructs_decisions(prefix):
    from experiments.value_direction_hopper import support
    from experiments.value_direction_hopper.cvm import load_v5
    load_v5(V5)
    support.load(SUPPORT)
    x = fresh_inputs(prefix, prefix + 1)
    protect = prefix // 64
    out, exported = v5_call(*x, export=True, protect=protect, threshold=0.0)
    torch.cuda.synchronize()
    assert not out.skipped[..., protect:].any() and out.skipped[..., :protect].any()
    # the PV role consumed exactly the protected decision: the support consumer on the same map is bit-identical
    cons = support.attention(x[0], x[1], x[2], out.skipped.contiguous(), out.eligible.contiguous(), scale=512 ** -.5,
                             mask=x[6].tensor)[0]
    torch.cuda.synchronize()
    assert torch.equal(cons.view(torch.int16), out.output.view(torch.int16))
    # exported unweighted log risk + log s reconstructs the kernel's own keep decision (prefix tiles)
    tau = 0.0
    risk = exported[..., :protect, :] + torch.log(x[5])[:, None, None, :]           # [B,H,J,Q]
    keep_rec = risk.reshape(1, 16, protect, 2, 128).amax(-1).permute(0, 1, 3, 2) >= tau
    keep_kernel = (~out.skipped & out.eligible)[..., :protect]
    mismatch = keep_rec != keep_kernel
    near = (risk.reshape(1, 16, protect, 2, 128).amax(-1).permute(0, 1, 3, 2) - tau).abs() < 1e-4
    assert not (mismatch & ~near).any(), int((mismatch & ~near).sum())
