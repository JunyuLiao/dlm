"""Shared calibration-state proposals and projection-seed diagnostics.

These are proposal distributions, not substitutes for sparse-trajectory
verification or downstream accuracy. Each sampled physical tile is weighted
once. Final generations never participate in this module.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from experiments.diffusion_gemma_value_aware.protocol import sha, frozen_write
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
from experiments.diffusion_gemma_value_aware.operators import (
    Config as LegacyConfig, value_summaries, screen_risks)
from experiments.diffusion_gemma_value_aware_gpu.kernels import route as legacy_route
from .config import Config, PROJECTED, BASELINES, PRIMARY_SEED, SENSITIVITY_SEEDS
from .projections import Projections
from . import kernels


def sketches_for(snapshot, config, bank):
    ident = snapshot['identity']
    value = snapshot['values'].float()
    valid = snapshot['kv_valid']
    ref = (value.norm(dim=-1).square().masked_fill(~valid, 0.).sum(-1)/valid.sum(-1).clamp_min(1)).sqrt().clamp_min(1e-12)
    if config.method == 'mass_exact':
        sketches = torch.zeros((*value.shape[:-1], 1), dtype=torch.float32, device=value.device)
    elif config.family == 'identity':
        sketches = value
    else:
        def project(seed):
            matrices = bank.get(ident['layer'], ident['native_head_count'], value.shape[-1],
                config.family, config.rank, seed, value.device)
            return value@matrices[ident['native_kv_head']:ident['native_kv_head']+1]
        sketches = project(config.projection_seed)
        if config.method == 'cancellation_guard':
            sketches = torch.cat((sketches, project(config.guard_seed)), -1)
    width = config.rank if config.method == 'cancellation_guard' else sketches.shape[-1]
    return sketches, sketches[..., :width].norm(dim=-1), ref


def legacy_state(state):
    z = state['logz']
    total = torch.logsumexp(z, -1, keepdim=True)
    mass = torch.exp(z-torch.where(torch.isfinite(total), total, 0.)).masked_fill(~state['active'], 0.)
    return dict(state, mass=mass, block_mean=state['mu'], contrib=mass[..., None]*state['mu'])


def source_index(root, calibration_outputs):
    sources = []
    for out in calibration_outputs:
        sources.extend(out['shared_state_sources'])
    if not sources or any(s['split'] != 'calibration' for s in sources):
        raise ValueError('Shared calibration sources must be nonempty and calibration-only')
    frozen_write(root/'shared_state_index.json', sources)
    return sources


def shared_screen(root, sources, contract):
    bank = Projections()
    extra = {f'{name}__seed{seed}': dict(c, projection_seed=seed)
             for name, c in PROJECTED.items() if name.startswith('jl_') for seed in SENSITIVITY_SEEDS}
    configs = {**PROJECTED, **extra}
    completed = []
    for item in sources:
        path = Path(item['path'])
        if item['split'] != 'calibration' or sha(path.read_bytes()) != item['sha256']:
            raise ValueError('Invalid shared calibration state provenance')
        dest = root/'shared_screen'/(path.stem+'.npz')
        proof = dest.with_suffix('.json')
        identity = dict(fingerprint=contract['fingerprint'], source=item,
                        screen_source_sha256=sha(Path(__file__).read_bytes()))
        if proof.exists():
            cached = json.loads(proof.read_text())
            if cached['identity'] != identity or sha(dest.read_bytes()) != cached['arrays_sha256']:
                raise ValueError('Shared screen cache changed')
            completed.append(cached)
            continue
        data = torch.load(path, weights_only=True, map_location='cuda')
        scores, valid, value = data['scores'], data['valid'], data['values']
        full = kernels.block_statistics(scores, valid, value.float())
        legacy = legacy_state(full)
        meta = value_summaries(value, data['kv_valid'])
        arrays = {}
        eligible = full['eligible'][0, 0]
        for name, kwargs in configs.items():
            config = Config(**kwargs)
            sk, norm, ref = sketches_for(data, config, bank)
            st = full if config.family == 'identity' else kernels.block_statistics(scores, valid, sk, norm)
            _, _, risks = kernels.route(st, ref, config, [-float('inf')], True)
            arrays[name] = risks[0, 0, 0][eligible].float().cpu().numpy()
            del st
        for name, kwargs in BASELINES.items():
            config = LegacyConfig(**kwargs)
            if name == 'unweighted_centered':
                risks, _ = screen_risks(legacy, meta, config)
                arrays[name] = risks[0, 0][eligible].float().cpu().numpy()
            else:
                # Existing screen_risks is the authority for dense-prefix
                # proposals (not full candidate routing with a guessed budget).
                risks, _ = screen_risks(legacy, meta, config)
                arrays[name] = risks[0, 0][eligible].float().cpu().numpy()
        dest.parent.mkdir(parents=True, exist_ok=True)
        # File handle avoids NumPy silently appending an extra .npz extension.
        temp = dest.with_suffix('.tmp')
        with temp.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(dest)
        result = dict(identity=identity, arrays_sha256=sha(dest.read_bytes()), path=str(dest),
            count=int(eligible.sum()), layer=item['layer'], head=item['head'], step=item['step'],
            benchmark=item['id'].split('/')[0], attention_type=item['attention_type'],
            matrices=bank.manifest, rule='All previously processed blocks retained; empirical proposals only')
        _write(proof, result); completed.append(result)
        print('shared screen', item['id'], item['layer'], item['step'], flush=True)
    frozen_write(root/'shared_screen_index.json', completed)
    return completed


def distributions(root, name, benchmark):
    index = json.loads((root/'shared_screen_index.json').read_text())
    pieces = {kind: [] for kind in ('local', 'global')}
    sources = {}
    for row in index:
        if row['benchmark'] != benchmark:
            continue
        source = row['identity']['source']
        if source['split'] != 'calibration':
            raise ValueError('Final/development data cannot fit threshold CDFs')
        path = Path(row['path'])
        if sha(path.read_bytes()) != row['arrays_sha256']:
            raise ValueError('Calibration CDF changed')
        with np.load(path) as data:
            pieces[row['attention_type']].append(data[name].astype(np.float64))
        sources[str(path)] = row['arrays_sha256']
    values = {kind:np.sort(np.concatenate(parts)) for kind, parts in pieces.items()}
    if any(np.isnan(v).any() or not np.isfinite(v).any() for v in values.values()):
        raise ValueError('Invalid calibration distribution')
    return values, sources
