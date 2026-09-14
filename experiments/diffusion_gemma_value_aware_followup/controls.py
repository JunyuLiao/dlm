"""Matched no-value control: original value rule with magnitude ratio exactly1.

This preserves the retained, pre-block online state and scalar threshold used
by value-aware routing. It is distinct from the legacy seen-maximum BLASST
extension and its inverse-length threshold. Actual V still forms the output;
only routing ignores its magnitude. No frozen original operator is edited.
"""
import json
from pathlib import Path
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware import run, routing
from experiments.diffusion_gemma_value_aware.operators import ValueCache
from experiments.diffusion_gemma_value_aware.protocol import sha

CONFIG = dict(method='value', pooling='rms', mode='no_value_control')


def source_sha():
    return sha(Path(__file__).read_bytes())


def unity_metadata(meta):
    # A new mapping, never mutate the cache or the true value metadata used by
    # another probe. ref/ref is exactly1 even when actual V is all zero.
    return dict(meta, rms=meta['ref'][..., None].expand_as(meta['rms']))


class UnityCache(ValueCache):
    def get(self, layer, v, valid, prefix):
        return unity_metadata(super().get(layer, v, valid, prefix))


class ControlAttention(routing.Attention):
    def __init__(self, config=None, thresholds=None, screen=False):
        if screen or any((config or {}).get(k) != v for k, v in CONFIG.items()):
            raise ValueError('explicit no-value control configuration required')
        super().__init__(config, thresholds, screen)
        self.cache = UnityCache()


def cache_control(adapter, row, path, fp, config=None, thresholds=None, root=None):
    digest = source_sha()
    if path.exists() and json.loads(path.read_text()).get('control_sha256') != digest:
        raise ValueError('no-value control source changed')
    original = run.generate
    def generate(*args, **kwargs):
        with patch.object(run, 'Attention', ControlAttention):
            result, arrays = original(*args, **kwargs)
        result['control_sha256'] = digest
        return result, arrays
    kwargs = dict(config=config, thresholds=thresholds)
    if root is not None:
        kwargs['root'] = root
    with patch.object(run, 'generate', generate):
        return run.cache(adapter, row, path, fp, **kwargs)
