"""v31: the vLLM adapter's C-gate acceptance mask equals the official sampler's entropy-bound rule.

The reference below is a verbatim transcription of phase 3-4 of vllm 0.30.0
model_executor/models/diffusion_gemma.py::_compiled_sample_step (token entropy of the temperature-scaled logits,
sorted, accepted while cumulative - running max <= entropy_bound)."""
import pytest
import torch


def _official(scaled, entropy_bound):
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    token_entropy = -(probs * log_probs).sum(dim=-1)
    sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
    cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
    cummax_ent = torch.cummax(sorted_ent, dim=-1).values
    sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
    eb_mask = torch.zeros_like(sorted_mask)
    eb_mask.scatter_(1, sorted_idx, sorted_mask)
    return eb_mask


@pytest.mark.parametrize('bound', [0.0, 0.5, 2.0, 50.0])
def test_accepted_mask_matches_official_rule(bound):
    from experiments.numerical_qk_reuse.vllm_adapter import accepted_mask
    g = torch.Generator().manual_seed(3)
    scaled = torch.randn(1, 256, 512, generator=g) * torch.linspace(0.2, 6, 256)[None, :, None]
    got = accepted_mask(scaled, bound)
    want = _official(scaled.float(), bound)
    assert got.dtype == torch.bool and got.shape == (1, 256)
    assert torch.equal(got, want)
    if bound == 0.0:
        assert int(got.sum()) == 1            # only the lowest-entropy position passes a zero bound
