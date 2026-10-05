"""RULER4K Gaussian-rank sweep.

This is a fresh, resumable 4K cohort using the audited RULER8K routing
pipeline.  Kernels, operators, calibration and scoring are inherited from the
RULER8K implementation; only the pinned 4K manifest and versioned output root
are new.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch

from experiments import diffusion_gemma_ruler8k_jl as base
from experiments import diffusion_gemma_ruler8k_jl_v14 as previous
from experiments import diffusion_gemma_ruler8k_jl_report as reporting
from experiments.diffusion_gemma_jl_output_aware import config as configuration
from experiments.diffusion_gemma_value_aware_followup import evidence

ROOT = Path('results/diffusion_gemma_ruler4k_gaussian_rank_sweep_v16')
MODULE = 'experiments.diffusion_gemma_ruler4k_gaussian_sweep'
POOL = Path('results/blasst/diffusion_gemma/paper_4k_n650/manifest')
RULER = Path('reference/RULER')
BENCHMARK = 'ruler4k'
TARGETS = (.75, .5)
KINDS = ('local', 'global')
BASELINES = dict(blasst=dict(method='blasst'), mass=dict(method='mass'))
PROJECTED = {
    'full_centered': dict(family='identity'),
    **{f'jl_gaussian_r{r}': dict(family='gaussian', rank=r)
       for r in (1, 2, 4, 8, 16, 32)},
}
CONFIGS = {**BASELINES, **PROJECTED}
CONDITIONS = ['dense'] + [f'{n}_s{int(t*100)}' for t in TARGETS for n in CONFIGS]
EXPECTED = 130 * len(CONDITIONS)
MAX_POINTS, TOLERANCE = 16, .02
TEST_COUNT = 41


def _sha(data):
    return base.sha(data)


def _task_base(task):
    if task.startswith('niah_'): return 'niah'
    if task == 'vt': return 'variable_tracking'
    if task == 'cwe': return 'common_words_extraction'
    if task == 'fwe': return 'freq_words_extraction'
    if task.startswith('qa_'): return 'qa'
    raise ValueError(task)


def _pool_rows():
    rows = [json.loads(x) for x in (POOL/'samples.jsonl').read_text().splitlines()]
    if len(rows) != 650 or Counter(r['task'] for r in rows) != dict.fromkeys(base.official.PAPER_TASKS, 50):
        raise ValueError('Expected 650 balanced 4K pool')
    for i, r in enumerate(rows):
        r['_pool_index'] = i
        if 'input' not in r or not r.get('outputs'):
            raise ValueError('Malformed official 4K raw row')
    return rows


def _select(rows, seed=42):
    result = {k: [] for k in ('calibration', 'final', 'development')}
    for task in base.official.PAPER_TASKS:
        subset = [r for r in rows if r['task'] == task]
        subset.sort(key=lambda r: _sha(f'{seed}|{task}|{r["_pool_index"]}|{_sha(r["input"])}'))
        for split, values in (('calibration', subset[:2]), ('final', subset[2:12]), ('development', subset[12:13])):
            result[split].extend((dict(r, split=split) for r in values))
    return result


def audit_setup(setup):
    seen = {k: set() for k in ('id', 'source_id', 'prompt_hash')}
    for split, n in (('calibration', 2), ('final', 10), ('development', 1)):
        rows = setup[split]
        if Counter(r['task'] for r in rows) != dict.fromkeys(base.official.PAPER_TASKS, n):
            raise ValueError(f'Incorrect {split} quotas')
        for key in seen:
            vals = [r[key] for r in rows]
            if len(vals) != len(set(vals)) or set(vals) & seen[key]:
                raise ValueError('Split overlap or duplicate identity')
            seen[key].update(vals)
        for r in rows:
            if (r['split'] != split or r['benchmark'] != BENCHMARK or
                    r['calibration'] != (split == 'calibration') or
                    _sha(r['prompt']) != r['prompt_hash'] or
                    len(r['prompt_tokens']) != r['prompt_token_count'] or r['seed'] != 42):
                raise ValueError('Manifest metadata changed')
    evidence.check_sources(setup['dataset_sources'])


def prepare(root=ROOT):
    path = root/'setup.json'
    if path.exists():
        setup = base.read(path); audit_setup(setup); return setup
    previous = base.read(Path('results/diffusion_gemma_ruler8k_gaussian_rank_sweep_v15/setup.json'))
    meta = base.read(POOL/'manifest.json')
    if meta.get('context_length') != 4096 or meta.get('length_mode') != 'total' or meta.get('seed') != 42:
        raise ValueError('4K pool generation settings changed')
    rows = _pool_rows()
    tokenizer = base.create_adapter('diffusion_gemma', previous['model'], device='cpu',
                                     precision='float32', revision=previous['revision']).load_tokenizer()
    splits = {k: [] for k in ('calibration', 'final', 'development')}
    for split, selected in _select(rows).items():
        for r in selected:
            prompt = r['input']
            tokens = tokenizer.encode_prompt(prompt, {'thinking': False})
            source_id = f'ruler_4096_{r["task"]}_p{int(r["_pool_index"]):04d}'
            row = dict(id=f'{BENCHMARK}/{source_id}', source_id=source_id, benchmark=BENCHMARK,
                       task=r['task'], task_base=_task_base(r['task']), outputs=r['outputs'],
                       official_index=int(r['index']), generator_seed=int(r['index']), split=split,
                       calibration=split == 'calibration', prompt=prompt, prompt_hash=_sha(prompt),
                       prompt_tokens=tokens, prompt_token_count=len(tokens),
                       generation_budget=int({'vt': 30, 'cwe': 120, 'fwe': 50,
                                              'qa_1': 32, 'qa_2': 32}.get(r['task'], 128)),
                       seed=42)
            splits[split].append(row)
    src = {str(p): _sha(p.read_bytes()) for p in (POOL/'manifest.json', POOL/'samples.jsonl', POOL/'generation_commands.json')}
    src.update({str(RULER/p): _sha((RULER/p).read_bytes()) for p in base.official.SOURCE_FILES})
    for p in (RULER/'scripts/data/synthetic/json/PaulGrahamEssays.json',
              RULER/'scripts/data/synthetic/json/squad.json', RULER/'scripts/data/synthetic/json/hotpotqa.json',
              RULER/'scripts/data/synthetic/json/english_words.json'):
        src[str(p)] = _sha(p.read_bytes())
    setup = dict(schema='ruler4k_gaussian_rank_sweep_v16', **splits,
        **{k: deepcopy(previous[k]) for k in ('model', 'revision', 'precision', 'tile_size', 'regions', 'decoding')},
        input_budget=4096, context_length_semantics='Official RULER total generator budget before chat-template overhead; 4K prompts are preserved verbatim',
        tasks=list(base.official.PAPER_TASKS), targets=list(TARGETS), configs=CONFIGS, conditions=CONDITIONS,
        split_seed=42, projection_seed=1729, diagnostic_projection_seeds=[1729, 2718, 31415],
        expected=EXPECTED, ruler_checkout=base.official.verify_checkout(RULER), dataset_sources=src,
        added_gaussian_ranks=[1, 2, 4, 8, 16, 32],
        selection='Per-task SHA256(seed|task|raw_index|raw_prompt_hash) sort: first2 calibration, next10 final, next1 development',
        exposure='Freshly generated 4K official pool, deterministic split; no final scores or sparsity used to select thresholds.',
        calibration_policy='26 calibration questions with official task budgets. Existing empirical-CDF BLASST fit and scalar refinement, max16 joint points, within2pp overall/global/local. Lambda cap-one gate is preserved; above-one is allowed only when the joint lambda=1 ceiling misses the whole-model target.',
        blasst_gate='Joint lambda_local=lambda_global=1 ceiling controls above-one authorization.',
        tolerance=TOLERANCE, monitoring_interval_seconds=1800, hardware_speedup_claim=False,
        benchmark=BENCHMARK, generation_projection_seed=1729)
    audit_setup(setup); base.frozen_write(path, setup)
    for split in splits: base.frozen_write(root/f'{split}_manifest.json', splits[split])
    base.frozen_write(root/'dataset_audit.json', dict(passed=True, disjoint=True,
        counts={s: len(v) for s, v in splits.items()},
        task_counts={s: dict(Counter(r['task'] for r in v)) for s, v in splits.items()}, source_hashes=src))
    return setup


@contextmanager
def scope():
    with patch.object(configuration, 'DIMENSIONS', (1, 2, 4, 8, 16, 32)), \
         patch.object(base.runner, 'PROJECTED', PROJECTED), \
         patch.object(base.runner, 'score', base.score), \
         patch.object(evidence, 'score', base.score):
        yield


def execution(root=ROOT):
    import transformers, triton
    setup = prepare(root)
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'Expected complete {TEST_COUNT}-test gate, got {counts}')
    inherited = base.read(Path('results/diffusion_gemma_ruler8k_gaussian_rank_sweep_v15/execution_contract.json'))
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__)
    if runtime != inherited['runtime']:
        raise ValueError('Pinned numerical runtime changed')
    files = [Path(__file__), Path(reporting.__file__), Path('tests/test_ruler8k_gaussian_sweep.py'), root/'tests.xml', root/'setup.json']
    sources = evidence.merge_sources(inherited['sources'], setup['dataset_sources'], {str(p): _sha(p.read_bytes()) for p in files})
    evidence.check_sources(sources)
    contract = dict(schema='ruler4k_execution_v16', sources=sources, runtime=runtime, expected=EXPECTED,
        projection_seed=1729, projection_dtype='float32', tf32=False, unchanged_numerical_kernels=True,
        calibration_only=True, max_points=MAX_POINTS, tolerance=TOLERANCE,
        benchmark=BENCHMARK, hardware_speedup_claim=False)
    contract['fingerprint'] = base._fingerprint(contract); base.frozen_write(root/'execution_contract.json', contract)
    return contract


@contextmanager
def active():
    attrs = dict(ROOT=ROOT, MODULE=MODULE, BENCHMARK=BENCHMARK, TARGETS=TARGETS, KINDS=KINDS,
                 CONFIGS=CONFIGS, PROJECTED=PROJECTED, BASELINES=BASELINES,
                 CONDITIONS=CONDITIONS, EXPECTED=EXPECTED, MAX_POINTS=MAX_POINTS,
                 TOLERANCE=TOLERANCE, prepare=prepare, execution=execution, scope=scope,
                 report=report, verify=verify)
    with ExitStack() as stack:
        for key, value in attrs.items(): stack.enter_context(patch.object(base, key, value))
        # The v14 compatibility layer rehydrates cached runner metadata after
        # gzip writes; this is required by the provenance-aware smoke/audit
        # path and does not alter routing or numerical behavior.
        stack.enter_context(previous.complete_metadata())
        yield


def work(root=ROOT):
    with active():
        setup, contract = prepare(root), execution(root)
        torch.backends.cuda.matmul.allow_tf32 = False; base.fb.phase(root, 'load_model')
        adapter = base.create_adapter('diffusion_gemma', setup['model'], device='cuda', precision='bfloat16', revision=setup['revision']).load()
        with scope(), base.lambda_observation():
            base.fb.phase(root, 'actual_model_smoke'); base.smoke(adapter, root, setup, contract)
            base.fb.phase(root, 'dense_calibration_states'); dense = []
            for i, row in enumerate(setup['calibration']):
                observer = base.SnapshotObserver(root, row, i)
                dense.append(base.runner.cached(adapter, root, row, 'calibration_dense', 'dense', 'dense', {}, None, contract, observer=observer))
            states = base.screen.source_index(root, dense)
            base.fb.phase(root, 'dense_final')
            for row in setup['final']:
                try: base.runner.cached(adapter, root, row, 'dense', 'dense', 'dense', {}, None, contract)
                except Exception: base.fb.failure(root, 'dense', id=row['id'])
            base.fb.phase(root, 'calibration_proposals')
            with patch.object(base.screen, 'PROJECTED', PROJECTED), patch.object(base.screen, 'BASELINES', BASELINES), patch.object(base.screen, 'SENSITIVITY_SEEDS', ()):
                base.screen.shared_screen(root, states, contract)
            base.fb.phase(root, 'blasst_cap_one_ceiling'); ceiling = base.ceiling_test(adapter, root, setup, contract)
            for target in TARGETS:
                for name in CONFIGS:
                    try: base.calibrate(adapter, root, setup, contract, name, target, ceiling)
                    except Exception: base.fb.failure(root, 'calibration', method=name, target=target)
            conditions = base.freeze_conditions(root, setup, contract)
            base.fb.phase(root, 'shared_state_diagnostics')
            try:
                with patch.object(base.shared_analysis, 'PROJECTED', PROJECTED), patch.object(base.shared_analysis, 'BASELINES', BASELINES), patch.object(base.shared_analysis, 'TARGETS', TARGETS):
                    base.shared_analysis.analyze(root, states, contract)
            except Exception: base.fb.failure(root, 'shared_state_diagnostics')
            for label, c in conditions.items():
                if label == 'dense': continue
                base.fb.phase(root, 'final', condition=label, expected=130)
                for row in sorted(setup['final'], key=lambda r: (len(r['prompt_tokens']), r['id'])):
                    try: base.runner.cached(adapter, root, row, 'final', label, c['name'], c['config'], c['thresholds'][BENCHMARK], contract)
                    except Exception: base.fb.failure(root, 'final', condition=label, id=row['id'])
        del adapter; torch.cuda.empty_cache(); base.fb.phase(root, 'report'); audit = report(root)
        if audit['complete']: base.fb.phase(root, 'independent_regeneration'); verify(root)
        base._write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'], expected=EXPECTED, finished=time.time()))
        base.fb.phase(root, 'finished', complete=audit['complete'])


def report(root=ROOT):
    with active():
        result = reporting.regenerate(root)
        # The frozen generic reporter is reused verbatim; adjust only prose
        # labels after its audit, then refresh the artifact hash in audit.json.
        p = root/'report.md'; text = p.read_text().replace('RULER8K', 'RULER4K').replace('8K denotes', '4K denotes').replace('complete1170', 'complete2470').replace('1170 hash-checked', '2470 hash-checked')
        p.write_text(text)
        result['artifacts']['report.md'] = _sha(p.read_bytes()); base._write(root/'audit.json', result)
        return result


def verify(root=ROOT):
    before = base.read(root/'audit.json')
    if not before['complete'] or before['completed'] != EXPECTED: raise ValueError('Incomplete final sweep')
    base.evidence.check_sources({str(root/p): h for p, h in before['artifacts'].items()})
    if report(root) != before: raise ValueError('Raw-only regeneration differs')
    result = dict(passed=True, completed=EXPECTED, inference_performed=False, audit_sha256=_sha((root/'audit.json').read_bytes()))
    base._write(root/'regeneration_verification.json', result); return result


def launch(root=ROOT):
    with active(): return base.launch(root)


def supervise(root=ROOT):
    with active(): return base.supervise(root)


if __name__ == '__main__':
    import importlib
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare','launch','supervise','work','report','verify')); parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args(); result = getattr(importlib.import_module(MODULE), args.command)(args.output)
    if args.command in ('prepare','report','verify'): print(json.dumps(dict(command=args.command, complete=result.get('complete'), passed=result.get('passed'))))
