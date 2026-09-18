"""Version15: extend the frozen RULER8K experiment, without numerical changes.

Five additional ranks at both targets. All prior outputs/policies are imported
with immutable provenance; no prior experiment file is modified.
"""
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from experiments import diffusion_gemma_ruler8k_jl_v14 as previous
from experiments import diffusion_gemma_ruler8k_jl_report as reporting

base = previous.base
ROOT = Path('results/diffusion_gemma_ruler8k_gaussian_rank_sweep_v15')
MODULE = 'experiments.diffusion_gemma_ruler8k_gaussian_sweep'
RANKS = (1, 4, 8, 16, 32)
DIAGNOSTIC_RANKS = (1, 2, 4, 8, 16, 32)
SEEDS = (1729, 2718, 31415)
OLD_CONFIGS, OLD_PROJECTED = deepcopy(base.CONFIGS), deepcopy(base.PROJECTED)
OLD_CONDITIONS = tuple(base.CONDITIONS)
NEW = {f'jl_gaussian_r{r}': dict(family='gaussian', rank=r) for r in RANKS}
CONFIGS, PROJECTED = {**OLD_CONFIGS, **NEW}, {**OLD_PROJECTED, **NEW}
CONDITIONS = list(OLD_CONDITIONS) + [f'{n}_s{int(t*100)}' for t in (.75, .5) for n in NEW]
EXPECTED = 130 * len(CONDITIONS)
BASE_AUDIT = base.audit_policy
TEST_COUNT = 41


@contextmanager
def scope():
    with patch.object(base.configuration, 'DIMENSIONS', DIAGNOSTIC_RANKS), \
         patch.object(base.runner, 'PROJECTED', PROJECTED), \
         patch.object(base.runner, 'score', base.score), patch.object(base.evidence, 'score', base.score):
        yield


def prepare(root=ROOT):
    old = base.read(previous.ROOT/'setup.json')
    setup = dict(deepcopy(old), schema='ruler8k_gaussian_rank_sweep_v15',
        configs=CONFIGS, conditions=CONDITIONS, expected=EXPECTED,
        added_gaussian_ranks=list(RANKS), diagnostic_ranks=list(DIAGNOSTIC_RANKS),
        diagnostic_projection_seeds=list(SEEDS), generation_projection_seed=1729,
        predecessor=str(previous.ROOT), expected_new_final=1300, expected_reused_final=1170,
        diagnostic_selection='One lexicographically first calibration ID per task, all30 layers at step0; same full-dimensional retained support for projection distortion; all six ranks and three prespecified seeds.',
        generation_order='All five new ranks at75%, then all five at50%; no score-based exclusions.',
        budget_caveat='Preserve official output budgets, including VT30. Dense VT hit this cap on all10 examples; apparent sparse improvements may reflect answer formatting. Excluding VT is post-hoc sensitivity only.')
    base.audit_setup(setup)
    base.frozen_write(root/'setup.json', setup)
    for split in ('calibration', 'final', 'development'):
        base.frozen_write(root/f'{split}_manifest.json', setup[split])
    base.frozen_write(root/'dataset_audit.json', dict(passed=True, identical_to_previous=True,
        source_path=str(previous.ROOT/'setup.json'), source_sha256=base.sha((previous.ROOT/'setup.json').read_bytes()),
        counts={s: len(setup[s]) for s in ('calibration','final','development')}, disjoint=True))
    return setup


def execution(root=ROOT):
    import transformers, triton
    from experiments import diffusion_gemma_ruler8k_projection_diagnostics as diagnostics
    from experiments import diffusion_gemma_ruler8k_gaussian_report as report_module
    prepare(root)
    prior = base.read(previous.ROOT/'execution_contract.json')
    base.evidence.check_sources(prior['sources'])
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, triton=triton.__version__)
    if runtime != prior['runtime']: raise ValueError('Frozen numerical runtime changed')
    suites = ET.parse(root/'tests.xml').getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'Require complete{TEST_COUNT}-test CPU/CUDA gate; got{counts}')
    proof_path, audit_path = previous.ROOT/'regeneration_verification.json', previous.ROOT/'audit.json'
    proof, audit = base.read(proof_path), base.read(audit_path)
    if not proof['passed'] or proof['audit_sha256'] != base.sha(audit_path.read_bytes()) or not audit['complete'] or audit['completed'] != 1170:
        raise ValueError('Previous complete/independently regenerated baseline required')
    files = [Path(__file__), Path(diagnostics.__file__), Path(report_module.__file__),
        Path('tests/test_ruler8k_gaussian_sweep.py'), root/'tests.xml', root/'setup.json',
        previous.ROOT/'execution_contract.json', proof_path, audit_path]
    sources = base.evidence.merge_sources(prior['sources'], {str(p): base.sha(p.read_bytes()) for p in files})
    contract = dict(schema='ruler8k_gaussian_rank_sweep_v15', sources=sources, runtime=runtime,
        expected=EXPECTED, prior_fingerprint=prior['fingerprint'], projection_seed=1729,
        diagnostic_projection_seeds=list(SEEDS), added_ranks=list(RANKS),
        projection_dtype='float32', tf32=False, unchanged_numerical_kernels=True,
        calibration_only=True, max_points=base.MAX_POINTS, tolerance=base.TOLERANCE,
        hardware_speedup_claim=False)
    contract['fingerprint'] = base._fingerprint(contract)
    base.frozen_write(root/'execution_contract.json', contract)
    return contract


def selected_sources(sources):
    # Input identities are frozen; task is resolved against the old manifest,
    # never inferred by a fragile substring of a sample ID.
    setup = base.read(previous.ROOT/'setup.json')
    tasks = {r['id']: r['task'] for r in setup['calibration']}
    selected = {}
    for src in sorted(sources, key=lambda s:(s['id'],s['layer'],s['step'])):
        if src['split'] != 'calibration' or src['id'] not in tasks:
            raise ValueError('Only calibration snapshots may enter diagnostics')
        if src['step'] == 0:
            selected.setdefault((base.BENCHMARK, tasks[src['id']], src['layer'], 0), src)
    return selected


def audit_policy(root, p, setup, contract):
    if 'imported_policy' not in p:
        return BASE_AUDIT(root, p, setup, contract)
    source = p['imported_policy']; path = Path(source['path'])
    if path != previous.ROOT/'policies'/base.BENCHMARK/f'{p["name"]}_s{int(p["target"]*100)}.json':
        raise ValueError('Unrecognized imported policy')
    base.evidence.check_sources({str(path): source['sha256']})
    old = base.read(path)
    expected = dict(old, fingerprint=contract['fingerprint'], imported_policy=source)
    if p != expected or p['name'] not in OLD_CONFIGS:
        raise ValueError('Imported baseline threshold/configuration changed')
    old_setup, old_contract = base.read(previous.ROOT/'setup.json'), base.read(previous.ROOT/'execution_contract.json')
    if any(setup[s] != old_setup[s] for s in ('calibration','final','development')):
        raise ValueError('Imported policy evaluated on changed examples')
    return BASE_AUDIT(previous.ROOT, old, old_setup, old_contract)


@contextmanager
def active():
    attrs = dict(prepare=prepare, execution=execution, scope=scope, MODULE=MODULE,
        CONFIGS=CONFIGS, PROJECTED=PROJECTED, CONDITIONS=CONDITIONS, EXPECTED=EXPECTED,
        audit_policy=audit_policy)
    with ExitStack() as stack:
        for key, value in attrs.items(): stack.enter_context(patch.object(base, key, value))
        stack.enter_context(previous.complete_metadata())
        stack.enter_context(scope())
        stack.enter_context(patch.object(base.shared_analysis, 'selected_sources', selected_sources))
        yield


def reuse(root, setup, contract):
    prior = base.read(previous.ROOT/'execution_contract.json')
    audit = base.read(previous.ROOT/'audit.json')
    base.evidence.check_sources({str(previous.ROOT/p): h for p,h in audit['artifacts'].items()})
    for label in OLD_CONDITIONS:
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        policy = None
        if name != 'dense':
            path = previous.ROOT/'policies'/base.BENCHMARK/f'{label}.json'
            old = base.read(path)
            p = dict(old, fingerprint=contract['fingerprint'], imported_policy=dict(path=str(path), sha256=base.sha(path.read_bytes())))
            audit_policy(root, p, setup, contract)
            base.frozen_write(root/'policies'/base.BENCHMARK/f'{label}.json', p)
            policy = p['policy']
        stage = 'dense' if name == 'dense' else 'final'
        for row in setup['final']:
            base.fb.alias(root, row, stage, label, name, {} if name == 'dense' else OLD_CONFIGS[name], policy, contract,
                base.shard_path(previous.ROOT, stage, label, row['id']), prior['fingerprint'])
        print('reused', label, 130, flush=True)
    sources = base.read(previous.ROOT/'shared_state_index.json')
    if len(sources) != 780 or len(selected_sources(sources)) != 390: raise ValueError('Unexpected shared-state coverage')
    base.evidence.check_sources({s['path']:s['sha256'] for s in sources})
    base.frozen_write(root/'shared_state_index.json', sources)
    oldfolder = previous.ROOT/'validation'/prior['fingerprint']
    folder = root/'validation'/contract['fingerprint']
    for original in [setup['development'][0], max(setup['development'], key=lambda r:len(r['prompt_tokens']))]:
        row = dict(original, generation_budget=16)
        native = oldfolder/(base.sha(row['id'])+'.native.json')
        oldproof = base.read(previous.ROOT/'smoke.json')
        base.evidence.check_sources({str(native):oldproof['sources'][str(native)]})
        base.frozen_write(folder/native.name, base.read(native))
        for name, cfg in {'dense':{}, **OLD_CONFIGS}.items():
            for unpruned in ((True,) if name == 'dense' else (True, False)):
                threshold = -1000. if unpruned else (0. if name == 'blasst' else -2.)
                policy = None if name == 'dense' else {k: dict(log_threshold=threshold,
                    **(dict(unpruned=True, tau=0.) if unpruned and name in OLD_PROJECTED else {})) for k in base.KINDS}
                tag = name + ('/unpruned' if unpruned else '/pruned')
                stages = ('smoke','reference') if name in OLD_PROJECTED and not unpruned else ('smoke',)
                for stage in stages:
                    source = base.shard_path(oldfolder, stage, tag, row['id'])
                    base.evidence.check_sources({str(source):oldproof['sources'][str(source)]})
                    base.fb.alias(folder, row, stage, tag, name, cfg, policy, contract, source, prior['fingerprint'])
    base.frozen_write(root/'reuse.json', dict(passed=True, final_outputs=1170, inference_performed=False,
        source_audit=str(previous.ROOT/'audit.json'), source_audit_sha256=base.sha((previous.ROOT/'audit.json').read_bytes()),
        shared_states=len(sources), selected_diagnostic_states=390, fingerprint=contract['fingerprint']))


def work(root=ROOT):
    from experiments import diffusion_gemma_ruler8k_projection_diagnostics as diagnostics
    with active():
        setup, contract = prepare(root), execution(root)
        base.fb.phase(root, 'reuse_completed_v14'); reuse(root, setup, contract)
        torch.backends.cuda.matmul.allow_tf32 = False
        base.fb.phase(root, 'load_model')
        adapter = base.create_adapter('diffusion_gemma', setup['model'], device='cuda', precision='bfloat16', revision=setup['revision']).load()
        base.fb.phase(root, 'new_rank_actual_model_smoke'); base.smoke(adapter, root, setup, contract)
        states = base.read(root/'shared_state_index.json')
        base.fb.phase(root, 'new_rank_calibration_proposals')
        with patch.object(base.screen, 'PROJECTED', NEW), patch.object(base.screen, 'BASELINES', {}), patch.object(base.screen, 'SENSITIVITY_SEEDS', ()):
            base.screen.shared_screen(root, states, contract)
        for target in (.75, .5):
            for name in NEW:
                try: base.calibrate(adapter, root, setup, contract, name, target, None)
                except Exception: base.fb.failure(root, 'calibration', method=name, target=target)
        conditions = base.freeze_conditions(root, setup, contract)
        base.fb.phase(root, 'balanced_shared_state_diagnostics')
        try:
            with patch.object(base.shared_analysis, 'PROJECTED', PROJECTED), patch.object(base.shared_analysis, 'BASELINES', base.BASELINES), patch.object(base.shared_analysis, 'TARGETS', base.TARGETS):
                base.shared_analysis.analyze(root, states, contract)
        except Exception: base.fb.failure(root, 'shared_state_diagnostics')
        base.fb.phase(root, 'common_full_support_projection_diagnostics')
        try: diagnostics.collect(root, states, contract)
        except Exception: base.fb.failure(root, 'common_support_diagnostics')
        for label in CONDITIONS:
            if label in OLD_CONDITIONS or label not in conditions: continue
            c = conditions[label]
            base.fb.phase(root, 'final', condition=label, expected=130)
            for row in sorted(setup['final'], key=lambda r:(len(r['prompt_tokens']),r['id'])):
                try: base.runner.cached(adapter, root, row, 'final', label, c['name'], c['config'], c['thresholds'][base.BENCHMARK], contract)
                except Exception: base.fb.failure(root, 'final', condition=label, id=row['id'])
        del adapter; torch.cuda.empty_cache()
        base.fb.phase(root, 'report'); audit = report(root)
        if audit['complete']:
            base.fb.phase(root, 'independent_regeneration'); verify(root)
        base._write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'], expected=EXPECTED, finished=time.time()))
        base.fb.phase(root, 'finished', complete=audit['complete'])


def report(root=ROOT):
    from experiments import diffusion_gemma_ruler8k_gaussian_report as module
    with active(): return module.regenerate(root)


def verify(root=ROOT):
    before = base.read(root/'audit.json')
    if not before['complete'] or before['completed'] != EXPECTED: raise ValueError('Incomplete final sweep')
    base.evidence.check_sources({str(root/p):h for p,h in before['artifacts'].items()})
    if report(root) != before: raise ValueError('Raw-only regeneration differs')
    result = dict(passed=True, completed=EXPECTED, inference_performed=False, audit_sha256=base.sha((root/'audit.json').read_bytes()))
    base._write(root/'regeneration_verification.json', result); return result


def launch(root=ROOT):
    with active(): return base.launch(root)


def supervise(root=ROOT):
    with active(): return base.supervise(root)


if __name__ == '__main__':
    import importlib
    p = argparse.ArgumentParser(); p.add_argument('command', choices=('prepare','launch','work','supervise','report','verify'))
    p.add_argument('--output', type=Path, default=ROOT); args = p.parse_args()
    # Dispatch through the canonical module before entering any patch context.
    # Diagnostic/report imports must never capture already-patched base globals.
    result = getattr(importlib.import_module(MODULE), args.command)(args.output)
    if args.command in ('prepare','report','verify'): print(json.dumps(dict(command=args.command, complete=result.get('complete'), passed=result.get('passed'))))
