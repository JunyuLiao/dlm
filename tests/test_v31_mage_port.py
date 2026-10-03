"""v31 MAGE port (prior-art baseline): selection follows MAGE eq. 5 at 64-key tile granularity."""
import torch


def _adapter(k_tokens):
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    types = ['sliding_attention'] * 5 + ['full_attention']
    return VllmMethodAdapter(types, arm='mage', mage_k=k_tokens)


def test_mage_selection_shapes_budget_and_canvas():
    torch.manual_seed(0)
    H, HK, D, n, prefix = 16, 2, 64, 256, 64 * 40 + 17          # prefix not tile-aligned
    a = _adapter(64 * 5)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    q = torch.randn(1, H, n, D)
    k = torch.randn(1, HK, prefix + n, D)
    kept = a._mage_select(q, k, D ** -.5, prefix, n, chunk=640)
    kt = -(-(prefix + n) // 64)
    assert kept.shape == (1, H, 2, kt) and kept.dtype == torch.bool
    first = prefix // 64
    assert bool(kept[..., first:].all())                          # canvas tiles always kept
    per_head = kept[0, :, 0, :first].sum(-1)
    assert torch.equal(per_head, torch.full((H,), 5))             # budget: mage_k / 64 prefix tiles per head
    for g in range(HK):                                           # one list per KV head, shared by its 8 q heads
        blk = kept[0, g * 8:(g + 1) * 8]
        assert bool((blk == blk[0:1]).all())
    assert torch.equal(kept[0, :, 0], kept[0, :, 1])              # same list for every query block


def test_mage_selects_highest_mass_tiles():
    torch.manual_seed(1)
    H, HK, D, n, prefix = 16, 2, 32, 128, 64 * 20
    a = _adapter(64 * 2)
    a.calls.update(mage_kept_prefix_tiles=0, mage_prefix_tiles=0)
    q = torch.randn(1, H, n, D) * 0.01
    k = torch.randn(1, HK, prefix + n, D) * 0.01
    # make tiles 3 and 11 of KV head 0, and tile 7 of KV head 1, dominate every query of the group
    k[0, 0, 3 * 64:4 * 64] += q[0, :8].mean((0, 1)) * 400
    k[0, 0, 11 * 64:12 * 64] += q[0, :8].mean((0, 1)) * 300
    k[0, 1, 7 * 64:8 * 64] += q[0, 8:].mean((0, 1)) * 400
    kept = a._mage_select(q, k, 1.0, prefix, n, chunk=4096)
    assert set(torch.nonzero(kept[0, 0, 0, :20]).flatten().tolist()) == {3, 11}
    assert 7 in torch.nonzero(kept[0, 8, 0, :20]).flatten().tolist()
