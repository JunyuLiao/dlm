"""Reuse native generation, baseline implementations and immutable shard checks."""
import json
import gzip
import os
from pathlib import Path
import time
from unittest.mock import patch
from contextlib import nullcontext

import torch

from experiments.diffusion_gemma_value_aware_gpu import runner as backend
from experiments.diffusion_gemma_value_aware_gpu.routing import Attention as BaselineAttention
from experiments.diffusion_gemma_value_aware_gpu import routing as baseline_routing
from experiments.diffusion_gemma_value_aware.routing import Attention as LegacyReference
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from .protocol import PARENT, sha, score
from .config import PROJECTED
from .routing import Attention


def write_output(path, out):
    """Compress verbose routing buckets without changing their numeric values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record_path = path.with_suffix('.records.json.gz')
    payload = gzip.compress(json.dumps(out['records'], separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    temp = record_path.with_suffix('.tmp')
    temp.write_bytes(payload)
    temp.replace(record_path)
    saved = dict(out)
    saved.pop('records')
    saved['records_source'] = dict(path=str(record_path), sha256=sha(payload))
    _write(path, saved)


def load_output(path):
    out = json.loads(path.read_text())
    if 'records_source' in out:
        source = out['records_source']
        raw = Path(source['path']).read_bytes()
        if sha(raw) != source['sha256']:
            raise ValueError('Compressed raw routing buckets changed')
        out['records'] = json.loads(gzip.decompress(raw))
    return out


def generate(adapter, row, name, config=None, thresholds=None, validate=False, trusted=False, observer=None):
    config = config or {}
    routers = []
    def factory(c, t, validate=False):
        if name in PROJECTED or name.startswith('seedcheck/'):
            router = Attention(c, t, validate, trusted, observer)
        elif name == 'unweighted_centered':
            router = LegacyReference(c, t)
        else:
            router = BaselineAttention(c, t, validate=validate)
        routers.append(router)
        return router
    original_prepare = baseline_routing._prepare_attention_scores
    def capture(module, q, k, v, mask, **kwargs):
        out = original_prepare(module, q, k, v, mask, **kwargs)
        _, ev, scores, valid = out
        kind = baseline_routing._attention_type(module, kwargs.get('sliding_window'))
        observer.capture(int(module.layer_idx), int(module._blasst_2d_runtime.current_denoising_iteration),
            kind, max(0, k.shape[-2]-q.shape[-2]), scores, valid, ev, v.shape[1])
        return out
    observe = patch.object(baseline_routing, '_prepare_attention_scores', capture) if observer is not None and name == 'dense' else nullcontext()
    with patch.object(backend, 'Attention', factory), patch.object(backend, 'score', score), observe:
        out = backend.generate(adapter, row, config, thresholds, validate=validate)
    router, = routers
    if isinstance(router, Attention):
        out['work_accounting'] = router.work_accounting()
        out['projection_matrices'] = router.cache.projections.manifest
        out['backend'] = 'trusted_projected_reference' if trusted else 'jl_fp32_custom_gpu_native_output'
    else:
        out['backend'] = 'unchanged_legacy_unweighted_reference' if name == 'unweighted_centered' else out['backend']
    out['candidate'] = name
    if observer is not None and hasattr(observer, 'sources'):
        out['shared_state_sources'] = observer.sources
    return out


def cached(adapter, root, row, stage, label, name, config, thresholds, contract, **kwargs):
    path = shard_path(root, stage, label, row['id'])
    if path.exists():
        out = load_output(path)
        check_result(out, row, contract['fingerprint'], config, thresholds)
        if out['candidate'] != name:
            raise ValueError('Candidate cache identity mismatch')
        if out.get('imported_source'):
            source = out['imported_source']
            check_sources({source['path']: source['sha256']})
        return out
    if adapter is None:
        raise FileNotFoundError(path)
    _write(root/'progress.json', dict(pid=os.getpid(), stage=stage, condition=label, id=row['id'], started=time.time()))
    out = generate(adapter, row, name, config, thresholds, **kwargs)
    out['fingerprint'] = contract['fingerprint']
    check_result(out, row, contract['fingerprint'], config, thresholds)
    write_output(path, out)
    _append(root/'completed.jsonl', dict(stage=stage, condition=label, id=row['id'], path=str(path), finished=time.time()))
    print(time.strftime('%FT%TZ', time.gmtime()), stage, label, row['id'], 'complete', flush=True)
    return out


def import_aime(root, row, label, name, config, thresholds, contract):
    """Only actually identical AIME prompt/settings/operator/thresholds may reuse."""
    if row['benchmark'] != 'aime26' or name in PROJECTED or name == 'unweighted_centered':
        return None
    stage = 'dense' if name == 'dense' else 'final'
    path = shard_path(root, stage, label, row['id'])
    if path.exists():
        return cached(None, root, row, stage, label, name, config, thresholds, contract)
    prior = json.loads((PARENT/'execution_contract.json').read_text())
    source = shard_path(PARENT, stage, label, row['id'])
    if not source.exists():
        return None
    out = json.loads(source.read_text())
    try:
        check_result(out, row, prior['fingerprint'], config, thresholds)
    except ValueError:
        return None
    check_sources(prior['sources'])
    out.update(fingerprint=contract['fingerprint'], candidate=name,
        imported_source=dict(path=str(source), sha256=sha(source.read_bytes()), original_fingerprint=prior['fingerprint']),
        diagnostic_convention='Historical FP32 block-contribution reduction; shared-state JL diagnostics recompute all methods uniformly')
    check_result(out, row, contract['fingerprint'], config, thresholds)
    write_output(path, out)
    _append(root/'completed.jsonl', dict(stage=stage, condition=label, id=row['id'], path=str(path),
        finished=time.time(), imported_source=out['imported_source']))
    return out
