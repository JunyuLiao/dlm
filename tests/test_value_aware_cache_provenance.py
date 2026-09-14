from copy import deepcopy
import json

import pytest

from experiments.diffusion_gemma_value_aware.cache_provenance_audit import (
    projected_source, validate_reuse,
)
from experiments.diffusion_gemma_value_aware.protocol import sha


def fixture():
    source = dict(screen=True, records=[dict(probe='execution', eligible=4),
        dict(probe='diagnostic', eligible=4)], distributions=[1],
        completion_tokens=[1, 2, 3], thresholds=None,
        generation_metadata={'thinking': False})
    raw = json.dumps(source).encode()
    return source, raw, dict(path='source.json', sha256=sha(raw))


def test_dense_observer_projection_preserves_source_and_generation():
    source, raw, origin = fixture()
    before = deepcopy(source)
    cached = projected_source(source, origin, dense=True)
    validate_reuse(source, cached, origin, raw, dense=True)
    assert source == before
    assert cached['screen'] is False and cached['distributions'] == []
    assert cached['records'] == [source['records'][0]]
    assert cached['completion_tokens'] == source['completion_tokens']


def test_sparse_reuse_allows_only_added_origin():
    source, raw, origin = fixture()
    source['screen'] = False
    raw = json.dumps(source).encode()
    origin['sha256'] = sha(raw)
    cached = projected_source(source, origin, dense=False)
    validate_reuse(source, cached, origin, raw, dense=False)
    assert cached['records'] == source['records']


@pytest.mark.parametrize('key,value', [
    ('completion_tokens', [1, 9, 3]),
    ('thresholds', {'local': 1}),
    ('generation_metadata', {'thinking': True}),
    ('records', [{'probe': 'execution', 'eligible': 5}]),
])
def test_undocumented_cache_changes_rejected(key, value):
    source, raw, origin = fixture()
    cached = projected_source(source, origin, dense=True)
    cached[key] = value
    with pytest.raises(ValueError, match='differs'):
        validate_reuse(source, cached, origin, raw, dense=True)


def test_changed_source_hash_rejected():
    source, raw, origin = fixture()
    cached = projected_source(source, origin, dense=True)
    with pytest.raises(ValueError, match='hash'):
        validate_reuse(source, cached, origin, raw + b' ', dense=True)
