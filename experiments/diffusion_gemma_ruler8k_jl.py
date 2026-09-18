"""RULER8K/13-task comparison. Versioned adapter; frozen kernels are unchanged.

BLASST is cap-one by default. Only a joint local=global=1 calibration
ceiling below the requested whole-model target unlocks aggressive thresholds.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import fcntl
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np
import torch
from dllm.models import create_adapter
from dllm.evaluation.ruler import official
from experiments import diffusion_gemma_jl_lowrank_multibench_v2 as parent
from experiments.diffusion_gemma_jl_output_aware import config as configuration
from experiments.diffusion_gemma_jl_output_aware.diagnostics import SnapshotObserver
from experiments.diffusion_gemma_value_aware_followup import evidence
from experiments.diffusion_gemma_value_aware.calibration import quantile_threshold
from experiments.diffusion_gemma_aime26.calibration import fit as blasst_fit
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import (
    _write, _append, _fingerprint, _install_dense, _set_context, _request)

ROOT = Path('results/diffusion_gemma_ruler8k_jl130_v13')
MODULE = 'experiments.diffusion_gemma_ruler8k_jl'
POOL = Path('results/blasst/diffusion_gemma/paper_8k_n650/manifest')
RULER = Path('reference/RULER')
BENCHMARK = 'ruler8k'
TARGETS = (.5, .75)
KINDS = ('local', 'global')
PROJECTED = dict(full_centered=dict(family='identity'), jl_gaussian_r2=dict(family='gaussian', rank=2))
BASELINES = dict(blasst=dict(method='blasst'), mass=dict(method='mass'))
CONFIGS = {**BASELINES, **PROJECTED}
CONDITIONS = ['dense'] + [f'{name}_s{int(t*100)}' for t in TARGETS for name in CONFIGS]
EXPECTED, MAX_POINTS, TOLERANCE = 1170, 16, .02
read, sha, frozen_write = parent.read, parent.sha, parent.frozen_write
fb, runner, screen, shared_analysis = parent.fb, parent.runner, parent.screen, parent.shared_analysis
shard_path = parent.shard_path


@lru_cache(None)
def scorers():
    return official.load_scorers(RULER)


def score(row, prediction):
    return float(scorers()[row['task_base']]([official.postprocess_prediction(prediction)], [row['outputs']])) / 100.


@contextmanager
def scope():
    # Process-local dispatch only; this worker is single threaded.
    with patch.object(configuration, 'DIMENSIONS', (2, 8, 16, 32)), \
         patch.object(runner, 'PROJECTED', PROJECTED), patch.object(runner, 'score', score), \
         patch.object(evidence, 'score', score):
        yield


def select(pool, seed=42):
    result = {k: [] for k in ('calibration', 'final', 'development')}
    for task in official.PAPER_TASKS:
        rows = [r for r in pool if r['task'] == task]
        if len(rows) < 13: raise ValueError('Insufficient cached task examples')
        rows.sort(key=lambda r: sha(f'{seed}|{task}|{r["sample_id"]}|{r["prompt_sha256"]}'))
        for split, subset in (('calibration', rows[:2]), ('final', rows[2:12]), ('development', rows[12:13])):
            result[split].extend(dict(r, split=split) for r in subset)
    return result


def audit_setup(setup):
    seen = {k: set() for k in ('id', 'source_id', 'prompt_hash')}
    for split, n in (('calibration', 2), ('final', 10), ('development', 1)):
        rows = setup[split]
        if Counter(r['task'] for r in rows) != dict.fromkeys(official.PAPER_TASKS, n):
            raise ValueError('Incorrect task quotas')
        for key in seen:
            keys = [r[key] for r in rows]
            if len(set(keys)) != len(keys) or set(keys) & seen[key]: raise ValueError('Split overlap or duplicates')
            seen[key].update(keys)
        for row in rows:
            if row['split'] != split or row['calibration'] != (split == 'calibration') or row['benchmark'] != BENCHMARK:
                raise ValueError('Split metadata changed')
            if sha(row['prompt']) != row['prompt_hash'] or row['seed'] != 42 or len(row['prompt_tokens']) != row['prompt_token_count']:
                raise ValueError('Prompt/seed/token metadata changed')
    evidence.check_sources(setup['dataset_sources'])


def prepare(root=ROOT):
    path = root/'setup.json'
    if path.exists():
        setup = read(path); audit_setup(setup); return setup
    verified = official.verify_checkout(RULER)
    meta = read(POOL/'manifest.json')
    if meta['context_length'] != 8192 or meta['length_mode'] != 'total' or meta['seed'] != 42:
        raise ValueError('Cached RULER generation settings changed')
    pool = [json.loads(line) for line in (POOL/'samples.jsonl').read_text().splitlines()]
    if Counter(r['task'] for r in pool) != dict.fromkeys(official.PAPER_TASKS, 50): raise ValueError('Expected complete650 pool')
    if any(sha(r['prompt']) != r['prompt_sha256'] for r in pool): raise ValueError('Cached prompt changed')
    previous = read(parent.ROOT/'setup.json')
    tokenizer = create_adapter('diffusion_gemma', previous['model'], device='cpu', precision='float32', revision=previous['revision']).load_tokenizer()
    splits = {}
    for split, rows in select(pool).items():
        splits[split] = []
        for r in rows:
            tokens = tokenizer.encode_prompt(r['prompt'], {'thinking': False})
            if len(tokens) != r['actual_prompt_length']: raise ValueError('Pinned tokenization differs from source pool')
            splits[split].append(dict(id=f'{BENCHMARK}/{r["sample_id"]}', source_id=r['sample_id'],
                benchmark=BENCHMARK, task=r['task'], task_base=r['task_base'], outputs=r['outputs'],
                official_index=r['official_index'], generator_seed=r['generator_seed'], split=split,
                calibration=split == 'calibration', prompt=r['prompt'], prompt_hash=r['prompt_sha256'],
                prompt_tokens=tokens, prompt_token_count=len(tokens), generation_budget=r['tokens_to_generate'], seed=42))
    sources = {str(p): sha(p.read_bytes()) for p in (POOL/'manifest.json', POOL/'samples.jsonl')}
    sources.update({str(RULER/p): sha((RULER/p).read_bytes()) for p in official.SOURCE_FILES})
    setup = dict(schema='ruler8k_jl130_v13', **splits,
        **{k: deepcopy(previous[k]) for k in ('model', 'revision', 'precision', 'tile_size', 'regions', 'decoding')},
        input_budget=8192, context_length_semantics='Official RULER total generator budget, before chat-template overhead; preserve cached prompts without truncation',
        tasks=list(official.PAPER_TASKS), targets=list(TARGETS), configs=CONFIGS, conditions=CONDITIONS,
        split_seed=42, projection_seed=1729, expected=EXPECTED, ruler_checkout=verified, dataset_sources=sources,
        selection='Per-task SHA256(seed|task|source_id|prompt_hash) sort: first2 calibration, next10 final, next1 development',
        exposure='Previously generated/partly evaluated650-example pool. New calibration/final split is disjoint; not fresh held-out confirmation.',
        calibration_policy='26 calibration questions with full official task budgets. Existing dense-state CDF and inverse-length BLASST fit; sparse-trajectory verification, up to16 joint points; no final scores or sparsity used to select thresholds.',
        blasst_gate='Probe lambda_local=lambda_global=1 on all26 calibration questions. Above-one thresholds permitted only if count-weighted overall ceiling is strictly below target. Otherwise cap both at1; allocate per-type goals under their measured ceilings when necessary.',
        tolerance=TOLERANCE, monitoring_interval_seconds=1800, hardware_speedup_claim=False)
    audit_setup(setup); frozen_write(path, setup)
    for split in splits: frozen_write(root/f'{split}_manifest.json', setup[split])
    frozen_write(root/'dataset_audit.json', dict(passed=True, disjoint=True, counts={s: len(v) for s, v in splits.items()},
        task_counts={s: dict(Counter(r['task'] for r in v)) for s, v in splits.items()}, source_hashes=sources))
    return setup


def execution(root=ROOT):
    from experiments import diffusion_gemma_ruler8k_jl_report as reporting
    import transformers, triton
    setup = prepare(root)
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=17, failures=0, errors=0, skipped=0): raise ValueError('Incomplete CPU/CUDA test gate')
    inherited = read(parent.ROOT/'execution_contract.json')
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__)
    if runtime != inherited['runtime']: raise ValueError('Pinned numerical runtime changed')
    files = [Path(__file__), Path(reporting.__file__), Path('tests/test_ruler8k_jl.py'), root/'tests.xml', root/'setup.json']
    sources = evidence.merge_sources(inherited['sources'], setup['dataset_sources'], {str(p): sha(p.read_bytes()) for p in files})
    evidence.check_sources(sources)
    contract = dict(schema='ruler8k_execution_v13', sources=sources, runtime=runtime, expected=EXPECTED,
        projection_seed=1729, projection_dtype='float32', tf32=False, unchanged_numerical_kernels=True,
        calibration_only=True, max_points=MAX_POINTS, blasst_gate=setup['blasst_gate'], hardware_speedup_claim=False)
    contract['fingerprint'] = _fingerprint(contract); frozen_write(root/'execution_contract.json', contract); return contract


class LambdaAttention(runner.BaselineAttention):
    """Observe the exact valid length already computed by the unchanged router."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.lambda_records = []

    def __call__(self, module, q, k, v, mask, **kwargs):
        original = runner.baseline_routing._prepare_attention_scores
        def observe(*args, **kw):
            out = original(*args, **kw)
            if self.config.method == 'blasst' and self.thresholds:
                valid = torch.broadcast_to(out[3], out[2].shape)
                length = int(valid.any((1, 2)).sum(-1).item())
                kind = runner.baseline_routing._attention_type(module, kw.get('sliding_window'))
                p = self.thresholds[kind]
                logt = p['log_scale'] - math.log(length) if 'log_scale' in p else p['log_threshold']
                if p.get('cap_one'): logt = min(0., logt)
                if p.get('unattainable') and p.get('cap_one'): logt = 0.
                self.lambda_records.append(dict(layer=int(module.layer_idx), step=int(module._blasst_2d_runtime.current_denoising_iteration),
                    attention_type=kind, valid_kv_length=length, log_lambda=logt, lambda_value=math.exp(logt)))
            return out
        with patch.object(runner.baseline_routing, '_prepare_attention_scores', observe):
            return super().__call__(module, q, k, v, mask, **kwargs)


@contextmanager
def lambda_observation():
    original_generate = runner.generate
    def generate(*args, **kwargs):
        instances = []
        class Observed(LambdaAttention):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw); instances.append(self)
        with patch.object(runner, 'BaselineAttention', Observed): out = original_generate(*args, **kwargs)
        if instances and instances[0].lambda_records: out['effective_lambda'] = instances[0].lambda_records
        return out
    with patch.object(runner, 'generate', generate): yield


def smoke(adapter, root, setup, contract):
    folder = root/'validation'/contract['fingerprint']; cases = []; sources = {}
    examples = [setup['development'][0], max(setup['development'], key=lambda r:len(r['prompt_tokens']))]
    for original in examples:
        row = dict(original, generation_budget=16)
        path = folder/(sha(row['id'])+'.native.json')
        if path.exists(): native = read(path)
        else:
            binding = _install_dense(adapter)
            try:
                _set_context(binding, row); generated = adapter.generate(_request(row))
            finally: binding.close()
            native = dict(completion_tokens=generated.completion_tokens, generation_metadata=generated.metadata)
            frozen_write(path, native)
        sources[str(path)] = sha(path.read_bytes())
        for name, cfg in {'dense': {}, **CONFIGS}.items():
            for unpruned in ((True,) if name == 'dense' else (True, False)):
                threshold = -1000. if unpruned else (0. if name == 'blasst' else -2.)
                policy = None if name == 'dense' else {k: dict(log_threshold=threshold, **(dict(unpruned=True, tau=0.) if unpruned and name in PROJECTED else {})) for k in KINDS}
                tag = name + ('/unpruned' if unpruned else '/pruned')
                out = runner.cached(adapter, folder, row, 'smoke', tag, name, cfg, policy, contract, validate=True)
                path = shard_path(folder, 'smoke', tag, row['id']); sources[str(path)] = sha(path.read_bytes())
                sources[out['records_source']['path']] = out['records_source']['sha256']
                if unpruned and (out['completion_tokens'] != native['completion_tokens'] or any(r['skipped'] for r in out['records'])):
                    raise AssertionError('Native dense/unpruned parity failed')
                if any(out['generation_metadata'].get(k) != native['generation_metadata'].get(k) for k in evidence.DECODING_FIELDS):
                    raise AssertionError('Unrelated decoding changed')
                if not unpruned and name in PROJECTED:
                    ref = runner.cached(adapter, folder, row, 'reference', tag, name, cfg, policy, contract, trusted=True)
                    path = shard_path(folder, 'reference', tag, row['id']); sources[str(path)] = sha(path.read_bytes())
                    sources[ref['records_source']['path']] = ref['records_source']['sha256']
                    if out['completion_tokens'] != ref['completion_tokens']: raise AssertionError('GPU/trusted token mismatch')
                    if any(sum(r[f] for r in out['records']) != sum(r[f] for r in ref['records']) for f in ('eligible','skipped')):
                        raise AssertionError('GPU/trusted physical counts mismatch')
                if any(not sum(r[f'{region}_eligible'] for r in out['records']) for region in ('prefix','canvas')):
                    raise AssertionError('Missing prefix/canvas eligibility')
                cases.append(dict(id=row['id'], name=name, unpruned=unpruned, passed=True, checked_calls=len(out['kernel_validation'])))
    proof = dict(passed=True, fingerprint=contract['fingerprint'], cases=cases, sources=sources)
    frozen_write(root/'smoke.json', proof); return proof


def goals_for(target, ceiling):
    """Only joint overall ceiling failure unlocks >1. No final data inputs."""
    aggressive = ceiling['achieved']['overall'] < target
    if aggressive or all(ceiling['achieved'][k] >= target for k in KINDS):
        return aggressive, dict.fromkeys(KINDS, target)
    counts = ceiling['eligible']; total = sum(counts.values())
    lo, hi = target, 1.
    for _ in range(64):
        mid = (lo+hi)/2
        weighted = sum(counts[k]*min(mid, ceiling['achieved'][k]) for k in KINDS)/total
        if weighted < target: lo = mid
        else: hi = mid
    return False, {k: min(hi, ceiling['achieved'][k]) for k in KINDS}


def within(achieved, target, goals):
    return abs(achieved['overall']-target) <= TOLERANCE+1e-12 and all(abs(achieved[k]-goals[k]) <= TOLERANCE+1e-12 for k in KINDS)


def measure_point(adapter, root, setup, contract, name, policy):
    label = f'{name}/{_fingerprint(policy)[:16]}'; sources = {}; outputs = []
    for row in setup['calibration']:
        out = runner.cached(adapter, root, row, 'calibration', label, name, CONFIGS[name], policy, contract)
        outputs.append(out); path = shard_path(root, 'calibration', label, row['id'])
        sources[row['id']] = dict(path=str(path), sha256=sha(path.read_bytes()))
    _, achieved = fb.measured(outputs)
    eligible = {k: sum(r['eligible'] for o in outputs for r in o['records'] if r['attention_type'] == k) for k in KINDS}
    return dict(policy=policy, achieved=achieved, eligible=eligible, sources=sources)


def audit_point(point, setup, contract, name):
    if set(point['sources']) != {r['id'] for r in setup['calibration']}: raise ValueError('Incomplete/contaminated calibration point')
    outputs = []
    for row in setup['calibration']:
        src = point['sources'][row['id']]; evidence.check_sources({src['path']: src['sha256']})
        out = runner.load_output(Path(src['path']))
        runner.check_result(out, row, contract['fingerprint'], CONFIGS[name], point['policy']); outputs.append(out)
    _, actual = fb.measured(outputs)
    if actual != point['achieved']: raise ValueError('Calibration physical counts differ')
    counts = {k: sum(r['eligible'] for o in outputs for r in o['records'] if r['attention_type'] == k) for k in KINDS}
    if counts != point['eligible']: raise ValueError('Calibration weights differ')


def ceiling_test(adapter, root, setup, contract):
    path = root/'blasst_cap_one.json'
    if path.exists(): result = read(path)
    else:
        policy = {k: dict(log_threshold=0., cap_one=True) for k in KINDS}
        result = measure_point(adapter, root, setup, contract, 'blasst', policy)
        frozen_write(path, result)
    if result['policy'] != {k: dict(log_threshold=0., cap_one=True) for k in KINDS}: raise ValueError('Ceiling probe changed')
    audit_point(result, setup, contract, 'blasst'); return result


def proposal_values(root, name):
    if name != 'blasst': return screen.distributions(root, name, BENCHMARK)
    parts = {k: [] for k in KINDS}; sources = {}
    for item in read(root/'shared_screen_index.json'):
        src = item['identity']['source']; evidence.check_sources({src['path']: src['sha256'], item['path']: item['arrays_sha256']})
        state = torch.load(src['path'], weights_only=True, map_location='cpu')
        length = int(state['kv_valid'].sum())
        with np.load(item['path']) as arrays: parts[item['attention_type']].append(arrays[name].astype(np.float64)+math.log(length))
        sources.update({src['path']: src['sha256'], item['path']: item['arrays_sha256']})
    return {k: np.sort(np.concatenate(v)) for k, v in parts.items()}, sources


def next_point(trace, target, goals, values, blasst=False, capped=False):
    policy = {}
    for k in KINDS:
        # Existing independent per-type search, applied to an explicitly frozen
        # attainable allocation when cap-one cannot fit both types to target.
        if blasst:
            candidate = fb.scalar_next(trace, goals[k], original=False)
            entry = candidate[k]; entry['cap_one'] = capped
        else:
            candidate = fb.proposal(trace, goals[k], values); entry = candidate[k]
        policy[k] = entry
    return policy


def audit_policy(root, p, setup, contract):
    name, target = p['name'], p['target']
    if name not in CONFIGS or target not in TARGETS or p['config'] != CONFIGS[name] or p['fingerprint'] != contract['fingerprint'] or p['heldout_used']:
        raise ValueError('Policy provenance mismatch')
    evidence.check_sources(p['distribution_sources'])
    goals = dict.fromkeys(KINDS, target)
    if name == 'blasst':
        ceiling = read(root/'blasst_cap_one.json'); audit_point(ceiling, setup, contract, name)
        aggressive, goals = goals_for(target, ceiling)
        if p['aggressive_allowed'] != aggressive or any(x['cap_one'] != (not aggressive) for x in p['policy'].values()):
            raise ValueError('Illegal above-one threshold authorization')
        if any(entry.get('cap_one') != (not aggressive) for point in p['trace'] for entry in point['policy'].values()):
            raise ValueError('Calibration trace violates above-one authorization')
    if p['goals'] != goals: raise ValueError('Per-type calibration goals changed')
    for point in p['trace']: audit_point(point, setup, contract, name)
    point = p['trace'][p['selected_round']]
    if p['policy'] != point['policy'] or p['measured'] != point['achieved'] or not within(p['measured'], target, goals):
        raise ValueError('Unverified/off-target policy')


def calibrate(adapter, root, setup, contract, name, target, ceiling):
    label = f'{name}_s{int(target*100)}'; dest = root/'policies'/BENCHMARK/f'{label}.json'
    if dest.exists():
        result = read(dest); audit_policy(root, result, setup, contract); return result
    values, sources = proposal_values(root, name)
    values = {k: np.maximum(v, -1e30) for k, v in values.items()}
    aggressive, goals = goals_for(target, ceiling) if name == 'blasst' else (False, dict.fromkeys(KINDS, target))
    if name == 'blasst':
        sources[str(root/'blasst_cap_one.json')] = sha((root/'blasst_cap_one.json').read_bytes())
        fits = {k: blasst_fit([values[k]], targets=(goals[k],)) for k in KINDS}
        initial = {k: dict(log_scale=fits[k]['targets'][str(goals[k])]['log_scale'], cap_one=not aggressive) for k in KINDS}
        frozen_write(root/'calibration_fits'/f'{label}.json', fits)
    else: initial = {k: dict(log_threshold=quantile_threshold([values[k]], target)['log_threshold']) for k in KINDS}
    path = root/'calibration_traces'/f'{label}.json'; trace = read(path) if path.exists() else []
    for point in trace: audit_point(point, setup, contract, name)
    for iteration in range(len(trace), MAX_POINTS):
        if any(within(x['achieved'], target, goals) for x in trace): break
        policy = initial if not trace else next_point(trace, target, goals, values, name == 'blasst', not aggressive)
        if any(x['policy'] == policy for x in trace): break
        fb.phase(root, 'calibration', condition=label, point=iteration+1, goals=goals, aggressive_allowed=aggressive)
        point = measure_point(adapter, root, setup, contract, name, policy); trace.append(point); _write(path, trace)
        print('calibration', label, iteration+1, point['achieved'], flush=True)
    candidates = [(i, p) for i, p in enumerate(trace) if within(p['achieved'], target, goals)]
    if not candidates: raise RuntimeError(f'{label}: no verified policy within2pp after{len(trace)} points; preserved trace')
    i, chosen = min(candidates, key=lambda pair: max(abs(pair[1]['achieved'][k]-(target if k == 'overall' else goals[k])) for k in ('overall', *KINDS)))
    result = dict(fingerprint=contract['fingerprint'], benchmark=BENCHMARK, name=name, config=CONFIGS[name], target=target,
        policy=chosen['policy'], measured=chosen['achieved'], trace=trace, selected_round=i, goals=goals,
        aggressive_allowed=aggressive, heldout_used=False, distribution_sources=sources,
        calibration_ids=[r['id'] for r in setup['calibration']])
    audit_policy(root, result, setup, contract); frozen_write(dest, result); return result


def freeze_conditions(root, setup, contract):
    smoke_proof = read(root/'smoke.json'); evidence.check_sources(smoke_proof['sources'])
    if not smoke_proof['passed'] or smoke_proof['fingerprint'] != contract['fingerprint']: raise ValueError('Missing smoke gate')
    conditions = {}
    for label in CONDITIONS:
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        sources = {str(root/'smoke.json'): sha((root/'smoke.json').read_bytes())}
        policy, target = None, 0.
        if name != 'dense':
            path = root/'policies'/BENCHMARK/f'{label}.json'
            if not path.exists(): continue
            p = read(path); audit_policy(root, p, setup, contract); policy, target = p['policy'], p['target']
            sources[str(path)] = sha(path.read_bytes())
        c = dict(name=name, target=target, config={} if name == 'dense' else CONFIGS[name], thresholds={BENCHMARK: policy},
            fingerprint=contract['fingerprint'], sources=sources)
        frozen_write(root/'final_configs'/f'{label}.json', c); conditions[label] = c
    return conditions


def work(root):
    setup, contract = prepare(root), execution(root)
    torch.backends.cuda.matmul.allow_tf32 = False; fb.phase(root, 'load_model')
    adapter = create_adapter('diffusion_gemma', setup['model'], device='cuda', precision='bfloat16', revision=setup['revision']).load()
    with scope(), lambda_observation():
        fb.phase(root, 'actual_model_smoke'); smoke(adapter, root, setup, contract)
        fb.phase(root, 'dense_calibration_states'); dense = []
        for i, row in enumerate(setup['calibration']):
            # All26 calibration cases/layers at step0; existing observer also
            # samples steps4/12/24 on first2, bounded one head/query tile per call.
            observer = SnapshotObserver(root, row, i)
            dense.append(runner.cached(adapter, root, row, 'calibration_dense', 'dense', 'dense', {}, None, contract, observer=observer))
        states = screen.source_index(root, dense)
        fb.phase(root, 'dense_final')
        for row in setup['final']:
            try: runner.cached(adapter, root, row, 'dense', 'dense', 'dense', {}, None, contract)
            except Exception: fb.failure(root, 'dense', id=row['id'])
        fb.phase(root, 'calibration_proposals')
        with patch.object(screen, 'PROJECTED', PROJECTED), patch.object(screen, 'BASELINES', BASELINES), patch.object(screen, 'SENSITIVITY_SEEDS', ()):
            screen.shared_screen(root, states, contract)
        fb.phase(root, 'blasst_cap_one_ceiling'); ceiling = ceiling_test(adapter, root, setup, contract)
        for target in TARGETS:
            for name in CONFIGS:
                try: calibrate(adapter, root, setup, contract, name, target, ceiling)
                except Exception: fb.failure(root, 'calibration', method=name, target=target)
        conditions = freeze_conditions(root, setup, contract)
        fb.phase(root, 'shared_state_diagnostics')
        try:
            with patch.object(shared_analysis, 'PROJECTED', PROJECTED), patch.object(shared_analysis, 'BASELINES', BASELINES), patch.object(shared_analysis, 'TARGETS', TARGETS):
                shared_analysis.analyze(root, states, contract)
        except Exception: fb.failure(root, 'shared_state_diagnostics')
        for label, c in conditions.items():
            if label == 'dense': continue
            fb.phase(root, 'final', condition=label, expected=130)
            for row in sorted(setup['final'], key=lambda r: (len(r['prompt_tokens']), r['id'])):
                try: runner.cached(adapter, root, row, 'final', label, c['name'], c['config'], c['thresholds'][BENCHMARK], contract)
                except Exception: fb.failure(root, 'final', condition=label, id=row['id'])
    del adapter; torch.cuda.empty_cache(); fb.phase(root, 'report'); audit = report(root)
    if audit['complete']: fb.phase(root, 'independent_regeneration'); verify(root)
    _write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'], expected=EXPECTED, finished=time.time()))
    fb.phase(root, 'finished', complete=audit['complete'])


def report(root=ROOT):
    from experiments.diffusion_gemma_ruler8k_jl_report import regenerate
    with scope(): return regenerate(root)


def verify(root=ROOT):
    before = read(root/'audit.json')
    if not before['complete'] or before['completed'] != EXPECTED: raise ValueError('Incomplete final sweep')
    evidence.check_sources({str(root/p): h for p, h in before['artifacts'].items()})
    if report(root) != before: raise ValueError('Raw-only regeneration differs')
    result = dict(passed=True, completed=EXPECTED, inference_performed=False, audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json', result); return result


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); fb.gpu_idle()
        with (root/'run.log').open('a', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'work', '--output', str(root)], stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=child.pid, supervisor_pid=os.getpid(), started=time.time()))
            while True:
                try: code = child.wait(timeout=1800); break
                except subprocess.TimeoutExpired:
                    _append(root/'monitor.jsonl', dict(time=time.time(), pid=child.pid,
                        **{k: read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase','progress')}))
            _write(root/'supervisor_terminal.json', dict(exit_code=code, pid=child.pid, finished=time.time()))
            if code: raise SystemExit(code)


def launch(root=ROOT):
    execution(root); fb.gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise', '--output', str(root)], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, expected=EXPECTED, monitoring_seconds=1800)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output', type=Path, default=ROOT); args = parser.parse_args()
    result = globals()[args.command](args.output)
    if args.command in ('prepare','report','verify'): print(json.dumps(dict(command=args.command, complete=result.get('complete'), passed=result.get('passed'))))
