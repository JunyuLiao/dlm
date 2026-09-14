"""Dispatch the matched no-value control without altering the frozen engine."""
import json
from unittest.mock import patch

from experiments.diffusion_gemma_value_aware import run as original
from experiments.diffusion_gemma_value_aware.execution import provenance as base_provenance
from .controls import CONFIG, cache_control, source_sha
from .engine import cached as base_cached, check_result
from .protocol import score


def is_control(config):
    return config.get('mode') == CONFIG['mode']


def provenance(config):
    return dict(control_sha256=source_sha()) if is_control(config) else base_provenance(config)


def cached(adapter, root, row, stage, name, config, thresholds, execution):
    if not is_control(config):
        return base_cached(adapter, root, row, stage, name, config, thresholds, execution=execution)
    path = original.shard_path(root, stage, name, row['id'])
    if not path.exists() and adapter is None:
        raise FileNotFoundError(path)
    with patch.object(original, 'score', score):
        result = cache_control(adapter, row, path, execution['fingerprint'], config, thresholds, root)
    if result.get('control_sha256') != source_sha():
        raise ValueError('no-value control source mismatch')
    check_result(result, row, execution['fingerprint'], config, thresholds)
    return result
