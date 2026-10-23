import importlib.util
from pathlib import Path
import torch
import pytest
p = Path(__file__).resolve().parents[1]/'code/replay.py'
spec = importlib.util.spec_from_file_location('native_replay_test', p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_logical_cache_keeps_native_token_head_order_and_ignores_padding():
    cache = torch.arange(4*2*3*4).reshape(4, 2, 3, 4)
    before = cache.clone()
    logical = m.logical_cache(cache, torch.tensor([2, 0]), 5)
    expected = torch.cat([cache[2].transpose(0, 1), cache[0].transpose(0, 1)], 0)[:5]
    assert torch.equal(logical, expected)
    assert torch.equal(cache, before)
    with pytest.raises(ValueError):
        m.logical_cache(cache, torch.tensor([4]), 1)


def test_canonical_discards_only_physical_page_identity():
    records = [dict(prepare={'a': 1}, layers={'layer': dict(
        page_table_sha256='x', logical_kv_sha256='y', cache_stride=[1])})]
    out = m.canonical(records)
    assert 'page_table_sha256' not in out[0]['layers']['layer']
    assert out[0]['layers']['layer']['logical_kv_sha256'] == 'y'
    assert records[0]['layers']['layer']['page_table_sha256'] == 'x'
