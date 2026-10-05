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


def test_dense_routed_calls_are_excluded_from_sparse_kept_and_counted_in_global_work():
    """The method's bootstrap dense call runs FA4 with an all-kept list (v27_fa4.dense): the legacy fraction counts it,
    sparse_kept_prefix_fraction must not, and global_prefix_work_fraction counts it as fully dense."""
    from experiments.numerical_qk_reuse import v27_fa4
    from experiments.numerical_qk_reuse.vllm_adapter import VllmMethodAdapter
    if v27_fa4._BST is None:
        v27_fa4._BST = lambda **kw: SimpleNamespace(**kw)
    a = VllmMethodAdapter(['sliding_attention'] * 5 + ['full_attention'], arm='allkept')
    a._kept_prefix, a._prefix_total = None, 0
    a._sparse_kept, a._sparse_total, a._global_tiles = None, 0, 0
    H, QB, KT, prefix = 2, 1, 10, 64 * 6 + 5
    a.paged = dict(prefix=prefix)
    sparse = SimpleNamespace(full_block_idx=torch.tensor([[[[0, 3, 6, 7, 8, 9, 1, 2, 4, 5]],
                                                           [[1, 2, 5, 6, 7, 8, 9, 0, 3, 4]]]], dtype=torch.int32),
                             full_block_cnt=torch.tensor([[[6], [7]]], dtype=torch.int32), block_size=(128, 64))
    dense = SimpleNamespace(full_block_idx=torch.arange(KT, dtype=torch.int32).expand(1, H, QB, KT).contiguous(),
                            full_block_cnt=torch.full((1, H, QB), KT, dtype=torch.int32), block_size=(128, 64))
    v27_fa4._ALLKEPT[('test', H, QB, KT)] = dense
    try:
        for lists in (dense, sparse, sparse):                         # one bootstrap dense call, two sparse calls
            a._split(lists)
            a._kept_account(lists)
            a._global_tiles += H * QB * 6
        a._global_tiles += H * QB * 6                                 # one dense call outside the lists (observation)
        r = a._kept_receipt()
    finally:
        v27_fa4._ALLKEPT.pop(('test', H, QB, KT), None)
    assert abs(r['kept_prefix_fraction'] - (12 + 10) / 36) < 1e-4         # legacy: includes the dense call
    assert abs(r['sparse_kept_prefix_fraction'] - 10 / 24) < 1e-4
    assert abs(r['global_prefix_work_fraction'] - (12 + 10 + 12) / 48) < 1e-4
