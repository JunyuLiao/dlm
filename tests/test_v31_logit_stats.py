"""v31 fused logit statistics: one-pass row stats match the torch ops the hook used before (GPU only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Triton kernel needs CUDA')


def _logits(seed, rows=256, vocab=262144):
    g = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn(1, rows, vocab, device='cuda', generator=g) * 3
    x[0, 3, 1000] = x[0, 3, 7] = x[0, 3].max() + 5          # tie: the first index must win
    x[0, 5] = 0.                                             # a uniform (padded) row: maximal entropy
    x[0, 9, 123] = 80.                                       # a near one-hot row
    return x


def test_row_stats_match_torch():
    from experiments.numerical_qk_reuse.v31_logit_stats import row_stats
    x = _logits(0)
    s = row_stats(x)
    assert torch.equal(s.argmax, x.argmax(-1))
    assert int(s.argmax[0, 3]) == 7
    assert torch.equal(s.max, x.amax(-1))
    torch.testing.assert_close(s.lse, torch.logsumexp(x, -1), rtol=1e-6, atol=1e-5)
    logp = x.log_softmax(-1)
    torch.testing.assert_close(s.entropy, -(logp.exp() * logp).sum(-1), rtol=1e-5, atol=1e-4)


def test_accepted_rule_matches_legacy_mask():
    from experiments.numerical_qk_reuse.v31_logit_stats import accepted_from_entropy, row_stats
    from experiments.numerical_qk_reuse.vllm_adapter import accepted_mask
    x = _logits(1)
    for bound in (0.5, 5.0, 50.0):
        legacy = accepted_mask(x, bound)
        fused = accepted_from_entropy(row_stats(x).entropy, bound)
        assert (legacy != fused).sum() <= 1                   # only an exact tie at the bound may differ


def test_state_update_matches_observe_logits():
    from experiments.numerical_qk_reuse.v31_logit_stats import observe_logits_from_stats, row_stats
    from experiments.value_direction_hopper.query_adaptive import State
    ref = State('T', None, m_ref=1., diagnostics=False, fast_t=True)
    new = State('T', None, m_ref=1., diagnostics=False, fast_t=True)
    for st in (ref, new):
        st.enable_cgate()
    canvas = torch.zeros(1, 256, dtype=torch.long, device='cuda')
    g = torch.Generator(device='cuda').manual_seed(3)
    for step in range(4):
        x = _logits(10 + step)
        accepted = torch.rand(1, 256, device='cuda', generator=g) < 0.4
        for st in (ref, new):
            st.begin(48 - step, canvas)
        torch.testing.assert_close(ref.used_weights, new.used_weights, rtol=1e-5, atol=1e-5)
        ref.observe_logits(x, accepted, 48 - step)
        observe_logits_from_stats(new, row_stats(x), accepted, 48 - step)
        assert torch.equal(ref.previous_top, new.previous_top)
        assert torch.equal(ref.temporal, new.temporal)
        assert torch.equal(ref.cg_q, new.cg_q) and torch.equal(ref.cg_r, new.cg_r)
        torch.testing.assert_close(ref.cg_u, new.cg_u, rtol=1e-3, atol=1e-3)   # sqrt(1 - p_top) amplifies FP32 rounding near p_top = 1
