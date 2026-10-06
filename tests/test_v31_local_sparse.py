import torch

from experiments.numerical_qk_reuse.v31_local_sparse import alias64, geometry, select


def test_local_geometry_uses_window_and_canvas_tiles():
    eligible, mandatory, kt = geometry(4096, 256)
    assert kt == 68
    assert all(set(m).issubset(set(e)) for e, m in zip(eligible, mandatory))
    assert len(eligible[0]) < kt
    assert all(any(t >= 64 for t in m) for m in mandatory)


def test_local_selector_keeps_budget_plus_canvas_and_is_finite():
    prefix, n = 4096, 256
    _, _, kt = geometry(prefix, n)
    z = torch.randn(16, 2, kt, 128)
    kept, count, total = select(z, prefix, n, 512)
    assert kept.shape == (1, 16, 2, kt)
    assert 0 < count <= total
    assert torch.isfinite(kept.float()).all()
    assert (kept[..., 64:].sum(-1) > 0).all()


def test_alias64_preserves_interleaved_cache_and_table_without_copy():
    cache = torch.arange(5 * 128 * 2 * 8).reshape(5, 128, 2, 8)
    k, v = cache.split(4, -1)
    table = torch.tensor([3, 1, 4], dtype=torch.int32)
    ak, av, at = alias64(k, v, table, 300)
    assert ak.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    assert av.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    assert at.tolist() == [[6, 7, 2, 3, 8]]
    assert torch.equal(ak[at[0].long()].reshape(-1, 2, 4)[:300], k[table.long()].reshape(-1, 2, 4)[:300])
    assert torch.equal(av[at[0].long()].reshape(-1, 2, 4)[:300], v[table.long()].reshape(-1, 2, 4)[:300])
