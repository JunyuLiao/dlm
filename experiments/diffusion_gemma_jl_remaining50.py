"""Frozen-policy transfer to the other50 cached LongBench v2 questions.

Only cohort selection, cache provenance and orchestration are new. Reuse the
existing numerical kernels, generation engine, NeMo scorer and raw reporter.
"""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from dllm.models import create_adapter
from experiments import diffusion_gemma_jl_directional50 as prior
from experiments.diffusion_gemma_jl_focused import protocol as parent, report as backend
from experiments.diffusion_gemma_jl_output_aware import runner
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint

ROOT = Path('results/diffusion_gemma_jl_longbench_remaining50_v7')
PREVIOUS, LB = prior.ROOT, parent.LB
MODULE = 'experiments.diffusion_gemma_jl_remaining50'
NEW = {'full_centered': {'family': 'identity'}, 'jl_gaussian_r32': {'family': 'gaussian', 'rank': 32}}
CONFIGS = {**parent.BASELINES, **NEW}
CONDITIONS = ['dense'] + [name+'_s50' for name in CONFIGS]
BASELINE_LABELS = ['dense'] + [name+'_s50' for name in parent.BASELINES]
EXPECTED, TEST_COUNT = 300, 6
read, sha, frozen_write = parent.read, parent.sha, parent.frozen_write
check_sources = prior.fb.check_sources


def complement(source, previous):
    pool = [r for r in source if r['benchmark'] == 'longbench_v2']
    old = [r for r in previous if r['benchmark'] == 'longbench_v2']
    if len(pool) != 100 or len(old) != 50:
        raise ValueError('Require the source100 and preceding50 LongBench questions')
    for rows in (pool, old):
        for key in ('id', 'prompt_hash'):
            if len({r[key] for r in rows}) != len(rows):
                raise ValueError('Duplicate sample ID or prompt hash')
    by_id = {r['id']: r for r in pool}
    for row in old:
        if row['id'] not in by_id or any(row[k] != by_id[row['id']][k] for k in
                ('prompt_hash', 'prompt_tokens', 'seed', 'generation_budget')):
            raise ValueError('Previous cohort is not an inference-compatible subset of100')
    selected = sorted((deepcopy(r) for r in pool if r['id'] not in {x['id'] for x in old}), key=lambda r: r['id'])
    if len(selected) != 50 or {r['prompt_hash'] for r in selected} & {r['prompt_hash'] for r in old}:
        raise ValueError('Require exactly50 disjoint complementary questions')
    return selected


def require_complete(root, expected, artifacts=True):
    a, v = read(root/'audit.json'), read(root/'regeneration_verification.json')
    if not (a['complete'] and a['completed'] == a['expected'] == expected and not a['missing']
            and not a['violations'] and v['passed'] and v['completed'] == expected
            and v['inference_performed'] is False and v['audit_sha256'] == sha((root/'audit.json').read_bytes())):
        raise ValueError('Require independently verified complete predecessor results')
    if artifacts:
        check_sources({str(root/p): h for p, h in a['artifacts'].items()})
    return {str(root/p): sha((root/p).read_bytes()) for p in ('audit.json', 'regeneration_verification.json')}


def prepare(root=ROOT):
    old, cached = read(PREVIOUS/'setup.json'), read(LB/'setup.json')
    rows = complement(cached['final'], old['final'])
    for key in ('model', 'revision', 'precision', 'tile_size', 'regions'):
        if old[key] != cached[key]:
            raise ValueError(f'Incompatible cached inference setting: {key}')
    tuning = []
    for source in (old, cached, read(prior.fb.ROOT/'setup.json')):
        tuning += source['calibration'] + source['development']
    for key in ('id', 'prompt_hash'):
        if {r[key] for r in rows} & {r[key] for r in tuning}:
            raise ValueError('Complement overlaps historical calibration/development')
    if any(r['generation_budget'] != 4096 or r['seed'] != 42 or sha(r['prompt']) != r['prompt_hash'] for r in rows):
        raise ValueError('Unrequested budget, seed or prompt mutation')
    paths = [PREVIOUS/'setup.json', PREVIOUS/'execution_contract.json', LB/'setup.json', LB/'execution_contract.json']
    setup = dict(schema='jl_remaining50_v7', **{k: deepcopy(old[k]) for k in
        ('model', 'revision', 'precision', 'tile_size', 'regions', 'decoding', 'input_budget')},
        final=rows, calibration=[], development=[], configs=CONFIGS, conditions=CONDITIONS, targets=[.5],
        selection=dict(rule='Exact complement of prior50 in cached100, sorted by ID; no score or length selection',
            domains=dict(sorted(Counter(r['task'] for r in rows).items())),
            subtasks=dict(sorted(Counter(r['sub_domain'] for r in rows).items())),
            source=str(LB/'setup.json'), excluded_source=str(PREVIOUS/'setup.json')),
        projection_seed=1729, projection_family='gaussian', projection_dimension=32,
        policy_source=str(PREVIOUS), baseline_source=str(LB),
        calibration_policy='Import exact prior full-budget calibrated50% local/global thresholds; no new fitting or final-score selection. Achieved transfer sparsity is measured, not guaranteed.',
        final_generation_slots=100, reused_baseline_slots=200, monitoring_interval_seconds=900,
        validation_policy='Reuse audited actual-model native/unpruned and trusted/pruned parity for the identical numerical implementations; no redundant dense replay.',
        exposure='New to these two directional final evaluations, but previously examined with cached baselines. Methods were chosen after the previous50; not fresh fully held-out confirmation.',
        source_hashes={str(p): sha(p.read_bytes()) for p in paths})
    frozen_write(root/'setup.json', setup)
    frozen_write(root/'final_manifest.json', rows)
    frozen_write(root/'selection.json', dict(setup['selection'], ids=[r['id'] for r in rows]))
    frozen_write(root/'dataset_audit.json', dict(passed=True, samples=50, expected=EXPECTED,
        disjoint_previous50=True, calibration_development_overlap=[], source_rows_verbatim=True,
        domains=setup['selection']['domains'], truncated=sum(r['truncated'] for r in rows),
        new_generation_slots=100, reusable_outputs=200, fresh_heldout_claim=False))
    return setup


def execution(root=ROOT):
    import triton, transformers
    prepare(root)
    old = read(PREVIOUS/'execution_contract.json')
    runtime = dict(torch=torch.__version__, triton=triton.__version__, transformers=transformers.__version__)
    if runtime != old['runtime']:
        raise ValueError('Cached runtime changed; renewed parity evidence required')
    xml = root/'tests.xml'
    suites = ET.parse(xml).getroot().findall('testsuite')
    counts = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if counts != dict(tests=TEST_COUNT, failures=0, errors=0, skipped=0):
        raise ValueError(f'Complementary-cohort test gate failed: {counts}')
    paths = (Path(__file__), Path('tests/test_jl_remaining50.py'), xml, root/'setup.json',
             PREVIOUS/'execution_contract.json', LB/'execution_contract.json')
    sources = {**old['sources'], **read(LB/'execution_contract.json')['sources'],
               **{str(p): sha(p.read_bytes()) for p in paths}}
    check_sources(sources)
    data = dict(schema='jl_remaining50_execution_v7', parent_fingerprint=old['fingerprint'], sources=sources,
        runtime=runtime, setup_sha256=sha((root/'setup.json').read_bytes()), numerical_algorithms_unchanged=True,
        projection_seed=1729, projection_dtype='float32', matmul_allow_tf32=False,
        targets=[.5], final_generation_slots=100, no_recalibration=True, hardware_speedup_claim=False)
    data['fingerprint'] = _fingerprint(data); frozen_write(root/'execution_contract.json', data)
    return data


def audit_policy(root, p, setup, contract):
    if p['fingerprint'] != contract['fingerprint'] or p['heldout_used'] or p['target'] != .5:
        raise ValueError('Policy identity, target or fitting scope changed')
    expected = PREVIOUS/'policies/longbench_v2'/f"{p['name']}_s50.json"
    if p['imported_policy'] != str(expected):
        raise ValueError('Unexpected threshold source')
    check_sources(p['sources']); old = read(expected)
    if any(p[k] != old[k] for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')):
        raise ValueError('Previously calibrated threshold or operator changed')
    if p['config'] != CONFIGS[p['name']] or p['benchmark'] != 'longbench_v2':
        raise ValueError('Unexpected requested method/benchmark')


def smoke_audit(root, setup, contract):
    data = read(root/'inherited_validation.json')
    if not data['passed'] or data['numerical_algorithms_unchanged'] is not True:
        raise ValueError('Missing compatible actual-model parity evidence')
    check_sources(data['sources'])
    return {**data['sources'], str(root/'inherited_validation.json'): sha((root/'inherited_validation.json').read_bytes())}


def reuse(root, setup, contract):
    proofs = {**require_complete(PREVIOUS, 640), **require_complete(LB, 900)}
    old_setup, old_contract = read(PREVIOUS/'setup.json'), read(PREVIOUS/'execution_contract.json')
    # Re-audit the actual raw GPU smoke, not just its passed flag. This includes
    # full-centered's inherited proof and Gaussian32's native/pruned checks.
    validation = prior.smoke_audit(PREVIOUS, old_setup, old_contract)
    frozen_write(root/'inherited_validation.json', dict(passed=True, sources=validation,
        numerical_algorithms_unchanged=True, new_smoke_generation_calls=0,
        note='Existing identical kernels, runtime,128x64 masks,GQA,sketch caching and native/pruned generation parity.'))
    for name, cfg in CONFIGS.items():
        src = PREVIOUS/'policies/longbench_v2'/f'{name}_s50.json'; old = read(src)
        prior.audit_policy(PREVIOUS, old, old_setup, old_contract)
        p = {k: deepcopy(old[k]) for k in ('name', 'benchmark', 'config', 'target', 'policy', 'measured')}
        p.update(fingerprint=contract['fingerprint'], heldout_used=False, imported_policy=str(src),
            sources={str(src): sha(src.read_bytes()), **proofs}, calibration_ids=old.get('calibration_ids'),
            cap_one_unattainable=old.get('cap_one_unattainable'),
            rule='Unchanged completed-study threshold, no new calibration; evaluate physical sparsity transfer.')
        audit_policy(root, p, setup, contract); frozen_write(root/'policies/longbench_v2'/f'{name}_s50.json', p)
    for label in CONDITIONS:
        name = 'dense' if label == 'dense' else label.rsplit('_s', 1)[0]
        src = PREVIOUS/'final_configs'/f'{label}.json'; old = read(src); check_sources(old['sources'])
        if old['config'] != ({} if name == 'dense' else CONFIGS[name]):
            raise ValueError('Final operator differs from preceding study')
        sources = {**contract['sources'], **proofs, **smoke_audit(root, setup, contract), str(src): sha(src.read_bytes())}
        if name != 'dense':
            policy_path = root/'policies/longbench_v2'/f'{label}.json'
            sources[str(policy_path)] = sha(policy_path.read_bytes())
            if read(policy_path)['policy'] != old['thresholds']['longbench_v2']:
                raise ValueError('Frozen policy/config mismatch')
        c = dict(fingerprint=contract['fingerprint'], name=name, target=0. if name == 'dense' else .5,
            config=old['config'], thresholds={'longbench_v2': old['thresholds']['longbench_v2']},
            sources=sources, expected_per_benchmark={'longbench_v2': 50})
        frozen_write(root/'final_configs'/f'{label}.json', c)
    imports = []
    source_fp = read(LB/'execution_contract.json')['fingerprint']
    for label in BASELINE_LABELS:
        c, old = read(root/'final_configs'/f'{label}.json'), read(LB/'final_configs'/f'{label}.json')
        if c['config'] != old['config'] or c['thresholds']['longbench_v2'] != old['thresholds'].get('longbench_v2'):
            raise ValueError('The100-question baseline differs from the requested frozen threshold')
        check_sources(old['sources'])
        stage = 'dense' if label == 'dense' else 'final'
        for row in setup['final']:
            source = shard_path(LB, stage, label, row['id'])
            prior.fb.alias(root, row, stage, label, c['name'], c['config'], c['thresholds']['longbench_v2'],
                contract, source, source_fp)
            imports.append(dict(id=row['id'], condition=label, source=str(source), sha256=sha(source.read_bytes())))
        print('reused baseline', label, 50, flush=True)
    if len(imports) != 200:
        raise ValueError('Require all200 baseline cache entries before new inference')
    frozen_write(root/'baseline_imports.json', dict(passed=True, count=200, imports=imports, sources=proofs))
    reuse_diagnostics(root, contract)


def reuse_diagnostics(root, contract):
    states = [s for s in read(PREVIOUS/'shared_state_index.json') if s['id'].startswith('longbench_v2/')]
    check_sources({s['path']: s['sha256'] for s in states})
    frozen_write(root/'shared_state_index.json', states)
    index = []
    for item in read(PREVIOUS/'shared_diagnostics_index.json'):
        src = Path(item['path']); data = read(src); ident = data['identity']
        if not ident['source']['id'].startswith('longbench_v2/') or ident['name'] not in CONFIGS or ident['target'] != .5:
            continue
        check_sources({str(src): item['sha256'], ident['policy_path']: ident['policy_sha256'], **ident['diagnostic_sources']})
        dest_policy = root/'policies/longbench_v2'/f"{ident['name']}_s50.json"
        if read(ident['policy_path'])['policy'] != read(dest_policy)['policy']:
            raise ValueError('Historical shared diagnostic threshold changed')
        ident.update(fingerprint=contract['fingerprint'], policy_path=str(dest_policy), policy_sha256=sha(dest_policy.read_bytes()))
        dest = root/'shared_diagnostics'/src.name; frozen_write(dest, data)
        index.append(dict(path=str(dest), sha256=sha(dest.read_bytes())))
    if len(index) != len(backend.selected_sources(states))*len(CONFIGS):
        raise ValueError('Inherited shared diagnostic coverage incomplete')
    frozen_write(root/'shared_diagnostics_index.json', index)


def accounting(root):
    counts = {}
    for label in CONDITIONS:
        stage = 'dense' if label == 'dense' else 'final'
        out = [read(p) for p in (root/stage/label/'shards').glob('*.json')]
        counts[label] = dict(completed=len(out), reused=sum(bool(x.get('imported_source')) for x in out))
    data = dict(conditions=counts, new_generation_slots=100, expected_reused_baselines=200,
        completed_new_generations=sum(v['completed']-v['reused'] for v in counts.values()),
        completed_reused=sum(v['reused'] for v in counts.values()), new_calibration_or_smoke_calls=0)
    _write(root/'inference_accounting.json', data); return data


def cohort_comparison(root):
    previous = read(PREVIOUS/'per_sample.json'); current = read(root/'per_sample.json')
    groups = defaultdict(list)
    for cohort, rows in (('previous50', previous), ('remaining50', current)):
        for row in rows:
            if row['benchmark'] == 'longbench_v2' and row['condition'] in CONDITIONS:
                groups[cohort, row['condition']].append(row)
                groups['pooled100', row['condition']].append(row)
    result = []
    for (cohort, label), rows in sorted(groups.items()):
        expected = 100 if cohort == 'pooled100' else 50
        if len(rows) != expected or len({r['id'] for r in rows}) != expected:
            continue
        result.append(dict(cohort=cohort, condition=label, **backend.summarize(rows)))
    _write(root/'cohort_comparison.json', result); backend.csv_write(root/'cohort_comparison.csv', result)
    return result


def write_report(root, setup, rows, tasks, compared, diag, audit, policies):
    audit.update(expected=EXPECTED, complete=audit['completed'] == EXPECTED and not audit['missing'] and not audit['violations'])
    display = [dict(condition=r['condition'], threshold=backend.threshold_text(r), score=f"{r['correct']:g}/{r['count']}",
        accuracy=100*r['accuracy'], delta_pp=100*r['delta'], ci95_pp=[100*v for v in r['paired_ci95']],
        whole=100*r['overall_physical_sparsity'], global_s=100*r['global_physical_sparsity'], local_s=100*r['local_physical_sparsity'],
        mass=100*r['overall_mass'], agreement=100*r['token_agreement'], local_operator_error=r['overall_relative_error'],
        unparsed=r['unparsed_answers'], length_limited=r['length_terminated']) for r in rows if r['split'] == 'full']
    backend.csv_write(root/'main_results.csv', display)
    text = '# Frozen50%-target routing on the remaining50 LongBench v2 questions\n\n'
    text += f"Status: {'COMPLETE' if audit['complete'] else 'INCOMPLETE'}; {audit['completed']}/{EXPECTED} audited outputs.100 new directional generations;200 exact cached dense/BLASST/mass results.\n\n"
    text += '## Setup and threshold provenance\n\nThe exact other50 IDs from the preceding100-question NeMo study are selected without examining scores; no overlap in IDs or prompt hashes with the previous50 or relevant calibration/development sets. All50 count in headline accuracy. These samples already have baseline exposure and methods were chosen after previous results; this is not fresh fully held-out confirmation.\n\n'
    text += backend.table([dict(domain=d, count=n) for d,n in setup['selection']['domains'].items()], ('domain','count'))+'\n\n'
    text += f"BF16 DiffusionGemma revision {setup['revision']}; unchanged cached NeMo prompts/MCQ scorer,4096-token generation budget,32K input cap, seed42, canvas256,max48 denoising steps,thinkingFalse. The native0.4–0.8 schedule is unchanged; temperature0 is a sentinel, not greedy. {sum(r['truncated'] for r in setup['final'])}/50 source prompts retain inherited truncation.\n\n"
    text += 'Full-dimensional centered and Gaussian32 centered use the exact preceding full-budget calibrated local/global scalar thresholds. No recalibration, new seed selection or threshold changes are allowed. Gaussian32 seed1729 and per-layer/native-KV-head matrix hashes remain fixed. Both use128-query×64-KV physical tiles, prefix+canvas eligibility, native structural masks/GQA, attention-weighted centered online updates, valid-KV RMS reference scaling, worst-valid-query gating, retained ties/first support, unchanged state on skip and original-V renormalization without compensation.\n\n'
    text += 'Dense and original/aggressive BLASST and mass-only outputs are imported from the completed100-question cache only after exact prompt/budget/seed/config/threshold and decoding checks. BLASST uses exact token-QK block maxima and physical skip only when every valid query votes skip. Original lambda is capped at1; aggressive lambda may exceed1 with the preceding-seen-maximum convention. Existing lambda=exp(log_scale)/valid_KV_length or scalar exp(log_threshold) rules and unattainable-boundary flags are unchanged. Mass-only remains the existing max-based candidate-mass bound, not mass_exact. thresholds.json preserves exact policy provenance.\n\n'
    text += '## Remaining50 results\n\n'+backend.findings(rows, compared)+'\n\n'
    text += backend.table(display, ('condition','threshold','score','accuracy','delta_pp','ci95_pp','whole','global_s','local_s','mass','agreement','local_operator_error','unparsed','length_limited'))+'\n\n'
    text += '## Domain results\n\n'+backend.table([dict(domain=r['task'],condition=r['condition'],score=f"{r['correct']:g}/{r['count']}",accuracy=100*r['accuracy']) for r in tasks],('domain','condition','score','accuracy'))+'\n\n'
    text += '## Previous50, remaining50 and pooled100\n\nPooling is supplementary and uses identical policies, not a new independently chosen cohort. Physical counts and token numerators/denominators are pooled before division. Previous50 Gaussian calibration members remain included in pooled100.\n\n'
    cohorts = cohort_comparison(root)
    text += backend.table([dict(cohort=r['cohort'],condition=r['condition'],score=f"{r['correct']:g}/{r['count']}",
        whole=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity']) for r in cohorts], ('cohort','condition','score','whole','global_s','local_s'))+'\n\n'
    text += '## Measurement and interpretation limits\n\nPhysical sparsity is SUM(skipped eligible tiles)/SUM(eligible tiles), never averaged sample percentages. Overall/global/local, prefix/canvas/boundary and per-layer/head/step counts are saved. Dense prefill is excluded. Target50% describes threshold calibration, not a promise of final48–52% sparsity; all transfer drift is reported without retuning. Original BLASST may be unattainable at the requested physical budget; compare actual sparsities.\n\n'
    text += 'Retained mass and full-dimensional local operator error use the corresponding dense attention on each sparse run\'s QKV, not a replay of the dense generation after divergence. Shared diagnostics are reused historical calibration states at unchanged policies, not newly sampled remaining50 states. The seed checks also remain historical calibration diagnostics, not new multi-seed generation evidence. Token agreement includes every generated position after divergence; missing/extra tokens disagree and EOS is included. Paired prompt bootstrap95% CIs are exploratory, fixed-policy/seed, and not multiplicity-corrected.\n\n'
    text += 'QK, block softmax and projected PV are still computed. Sketch caching and invalidation are unchanged; work_accounting records projection/storage/refresh overhead. Native retained PV remains a dense-shaped masked matmul. No hardware speedup or FlashAttention comparison is claimed.\n\n'
    text += '![Remaining50 tradeoffs](figures/longbench_v2_tradeoffs.png)\n\n![Target versus achieved sparsity](figures/longbench_v2_target_actual.png)\n\n'
    text += f"Execution failures are preserved in failures.jsonl; missing={len(audit['missing'])}, violations={len(audit['violations'])}. Actual-model GPU smoke evidence is raw-audited and inherited for identical algorithms/runtime. No redundant dense or calibration inference is run.\n\n"
    text += f"Regenerate without inference: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m {MODULE} report`; repeat independently with `verify`.\n"
    (root/'report.md').write_text(text)


BASE_PLOTS = backend.plots


def regenerate(root=ROOT):
    # Only registry, cohort size and report/provenance adapters differ.
    def scoped_plots(folder, rows):
        from matplotlib.figure import Figure
        original = Figure.suptitle
        def title(fig, text, *args, **kwargs):
            return original(fig, text.replace('50/75% targets', '50% target; remaining50'), *args, **kwargs)
        with patch.object(Figure, 'suptitle', title):
            return [p for p in BASE_PLOTS(folder, rows) if 'longbench_v2' in p]
    with patch.object(backend, 'prepare', prepare), patch.object(backend, 'execution', execution), \
         patch.object(backend, 'PROJECTED', NEW), patch.object(backend, 'TARGETS', (.5,)), \
         patch.object(backend.common, 'TARGETS', (.5,)), patch.object(backend, 'audit_policy', audit_policy), \
         patch.object(backend, 'smoke_audit', smoke_audit), patch.object(backend, 'write_report', write_report), \
         patch.object(backend, 'plots', scoped_plots):
        audit = backend.regenerate(root)
    accounting(root)
    audit['sources'][str(PREVIOUS/'per_sample.json')] = sha((PREVIOUS/'per_sample.json').read_bytes())
    for name in ('inference_accounting.json', 'cohort_comparison.json', 'cohort_comparison.csv'):
        audit['artifacts'][name] = sha((root/name).read_bytes())
    _write(root/'audit.json', audit); return audit


def verify(root=ROOT):
    before = read(root/'audit.json')
    if not before['complete'] or before['completed'] != EXPECTED:
        raise ValueError('Require all300 completed audited outputs')
    check_sources({str(root/p): h for p,h in before['artifacts'].items()})
    if regenerate(root) != before:
        raise ValueError('Independent raw-only regeneration changed')
    proof = dict(passed=True, completed=EXPECTED, inference_performed=False, audit_sha256=sha((root/'audit.json').read_bytes()))
    _write(root/'regeneration_verification.json', proof); return proof


def work(root):
    setup, contract = prepare(root), execution(root)
    prior.fb.phase(root, 'audit_and_reuse_cached_baselines'); reuse(root, setup, contract); accounting(root)
    torch.backends.cuda.matmul.allow_tf32 = False
    prior.fb.phase(root, 'load_model')
    adapter = create_adapter('diffusion_gemma', setup['model'], device='cuda', precision='bfloat16', revision=setup['revision']).load()
    with patch.object(runner, 'PROJECTED', NEW):
        for name, cfg in NEW.items():
            label = name+'_s50'; c = read(root/'final_configs'/f'{label}.json')
            prior.fb.phase(root, 'final', condition=label, expected=50)
            for row in sorted(setup['final'], key=lambda r: (len(r['prompt_tokens']), r['id'])):
                try:
                    runner.cached(adapter, root, row, 'final', label, name, cfg, c['thresholds']['longbench_v2'], contract)
                except Exception:
                    prior.fb.failure(root, 'final', condition=label, id=row['id'])
            accounting(root)
    del adapter; torch.cuda.empty_cache()
    prior.fb.phase(root, 'report'); audit = regenerate(root)
    if audit['complete']:
        prior.fb.phase(root, 'independent_report_verification'); verify(root)
    _write(root/'terminal.json', dict(complete=audit['complete'], completed=audit['completed'], expected=EXPECTED, finished=time.time()))
    prior.fb.phase(root, 'finished', complete=audit['complete'])


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); prior.fb.gpu_idle()
        with (root/'run.log').open('a', buffering=1) as log:
            child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'work', '--output', str(root)], stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=child.pid, supervisor_pid=os.getpid(), started=time.time()))
            while True:
                try:
                    code = child.wait(timeout=900); break
                except subprocess.TimeoutExpired:
                    state = {k: read(root/f'{k}.json') if (root/f'{k}.json').exists() else None for k in ('phase', 'progress')}
                    _append(root/'monitor.jsonl', dict(time=time.time(), pid=child.pid, **state))
            _write(root/'supervisor_terminal.json', dict(exit_code=code, pid=child.pid, finished=time.time()))
            if code:
                raise SystemExit(code)


def launch(root):
    require_complete(PREVIOUS, 640); require_complete(LB, 900); execution(root); prior.fb.gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        child = subprocess.Popen([sys.executable, '-u', '-m', MODULE, 'supervise', '--output', str(root)],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid, new_generations=100, reused_baselines=200, targets=[.5])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare','launch','supervise','work','report','verify'))
    parser.add_argument('--output', type=Path, default=ROOT); args = parser.parse_args()
    result = (regenerate if args.command == 'report' else globals()[args.command])(args.output)
    if args.command in ('prepare','report','verify'):
        print(json.dumps({'command':args.command,'complete':result.get('complete'),'passed':result.get('passed')}))
