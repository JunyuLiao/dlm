"""v31 realized-sparsity receipt: the adapter counts kept wholly-prefix tiles per sparse call."""
from types import SimpleNamespace

import torch


def test_kept_prefix_fraction_counts_prefix_tiles_per_call():
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    if v27_fa4._BST is None:                                          # lazily bound by v27_fa4.load() in real calls
        v27_fa4._BST = lambda **kw: SimpleNamespace(**kw)
    a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='allkept')
    a._kept_prefix, a._prefix_total = None, 0
    H, QB, KT, prefix = 2, 1, 10, 64 * 6 + 5                          # 6 wholly-prefix tiles, 4 canvas tiles
    a.paged = dict(prefix=prefix)
    # head 0 keeps prefix tiles 0, 3 and canvas 6..9; head 1 keeps prefix 1, 2, 5 and canvas 6..9
    order = torch.tensor([[[[0, 3, 6, 7, 8, 9, 1, 2, 4, 5]], [[1, 2, 5, 6, 7, 8, 9, 0, 3, 4]]]], dtype=torch.int32)
    cnt = torch.tensor([[[6], [7]]], dtype=torch.int32)
    lists = SimpleNamespace(full_block_idx=order, full_block_cnt=cnt, block_size=(128, 64))
    a._split(lists)
    a._kept_account(lists)
    a._kept_account(lists)                                            # a held call reuses the same map
    r = a._kept_receipt()
    assert r['kept_prefix_tiles'] == 2 * 5 and r['sparse_prefix_tiles'] == 2 * H * QB * 6
    assert abs(r['kept_prefix_fraction'] - 5 / 12) < 1e-4                # receipt rounds to 5 decimals
