import pytest
import torch

from experiments.numerical_qk_reuse.v31_cached_refresh import CachedRefreshState, topk_fixed


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/H100 unavailable")


def test_cuda_hybrid_matches_cpu_oracle_and_preserves_fixed_budget():
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(7)
    raw = torch.randn((2, 16, 128), device=device, generator=gen)
    selected = topk_fixed(raw.amax(-1), 4)
    state = CachedRefreshState.from_dense(raw, selected)
    fresh = raw + 0.01 * torch.randn(raw.shape, device=device, generator=gen)
    result = state.refresh(fresh, policy="bounded_swaps", frontier_m=16, swap_budget=2)
    assert torch.equal(result.selected.sum(-1), torch.full((2,), 4, device=device))
    assert result.swaps <= 4


def test_cuda_selector_does_not_mutate_held_attention_output():
    device = torch.device("cuda")
    held_output = torch.randn((2, 128, 16), device=device)
    before = held_output.clone()
    raw = torch.randn((2, 8, 128), device=device)
    selected = topk_fixed(raw.amax(-1), 3)
    state = CachedRefreshState.from_dense(raw, selected, dense_output=held_output)
    state.refresh(raw + 0.01, policy="global_topk", frontier_m=8)
    assert torch.equal(held_output, before)
