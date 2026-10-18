"""Frozen/resumable HumanEval164 K256 study with work-only V2 calibration."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_value_native_study as native
from scripts import v31_official as official
from scripts import v27_humaneval as he
from scripts.v31_score_humaneval import run_all
from scripts.v31_humaneval_value_worker import CONTROL, V2, V3A, REFINEMENTS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'results/humaneval_value_k256_20261018/attempt001'
SPEC = ROOT / 'docs/HUMANEVAL_VALUE_K256_PLAN_20261018.md'
SOURCE = Path('/home/exouser/humaneval_query_sensitivity_v2/source_data/HumanEval.jsonl.gz')
MODES = ('thinking_on', 'thinking_off')
CANDIDATES = (V2, V3A, *REFINEMENTS)
REFERENCES = ('dense_full_fix51994', 'dense_piecewise', 'allkept_fa4')
LABELS = ('control_before', *CANDIDATES, 'control_after', *REFERENCES)
GRID = (0., .01, .03, .1, .3, 1.)
SETTINGS = dict(dataset='humaneval', count=164, shards=2, max_len=9216,
                block=64, chunk=4096, memory=.80, mage_k=256)
read, write, sha = base.read, base.write, base.sha


def tau_label(t):
    return 'v2_tau_' + format(t, '.12g').replace('.', 'p').replace('-', 'm')


def first_canvas(row):
    a = row['receipts']['adapter']
    phases = a['value_canvas_phase_tiles']
    if not phases:
        raise ValueError('missing first-canvas accounting')
    first = phases[min(phases, key=int)]
    eligible = sum(x['eligible_tiles'] for x in first.values())
    kept = sum(x['kept_tiles'] for x in first.values())
    if not eligible or not 0 <= kept <= eligible:
        raise ValueError('invalid first-canvas physical work')
    return eligible, kept


def work(rows, weights=None):
    weights = weights or {}
    e = k = 0.
    for r in rows:
        a, b = first_canvas(r)
        w = weights.get(str(r['index']), 1.)
        e += w*a
        k += w*b
    return dict(eligible=e, kept=k, kept_fraction=k/e, sparsity=1-k/e,
                kept_per_problem=k/sum(weights.get(str(r['index']), 1.) for r in rows))


def choose_threshold(reference, candidates):
    """No correctness, timing, or monotonicity assumption enters selection."""
    if not candidates:
        raise ValueError('no measured thresholds')
    return min(candidates, key=lambda t: (
        abs(candidates[t]['kept_fraction']-reference['kept_fraction']),
        abs(candidates[t]['kept_per_problem']-reference['kept_per_problem']), t))


def refinement_points(reference, candidates):
    ordered = sorted(candidates)
    target = reference['kept_fraction']
    crossings = [(a, b) for a, b in zip(ordered, ordered[1:])
                 if (candidates[a]['kept_fraction']-target)*(candidates[b]['kept_fraction']-target) < 0]
    if crossings:
        a, b = min(crossings, key=lambda ab: sum(abs(candidates[t]['kept_fraction']-target) for t in ab))
        return ([a+(b-a)/3, a+2*(b-a)/3] if not a else
                [a**(2/3)*b**(1/3), a**(1/3)*b**(2/3)])
    best = choose_threshold(reference, candidates)
    j = ordered.index(best)
    intervals = [(ordered[i], ordered[i+1]) for i in (j-1, j) if 0 <= i < len(ordered)-1]
    return [math.sqrt(a*b) if a else b/2 for a, b in intervals]


def match(reference, candidate):
    return dict(matched=(abs(candidate['sparsity']-reference['sparsity']) <= .0025
                         and abs(candidate['kept_fraction']/reference['kept_fraction']-1) <= .01),
                sparsity_delta_pp=100*(candidate['sparsity']-reference['sparsity']),
                kept_fraction_ratio=candidate['kept_fraction']/reference['kept_fraction'],
                absolute_kept_ratio=candidate['kept']/reference['kept'])


def calibration_indices(manifest):
    ordered = sorted(range(164), key=lambda i: (len(manifest[i]['prompt_tokens']), i))
    indices, weights = [], {}
    for group in np.array_split(ordered, 24):
        i = int(group[len(group)//2])
        indices.append(i)
        weights[str(i)] = len(group)
    return indices, weights


def prepare(study):
    if study.exists():
        raise RuntimeError('new attempt directory required')
    from transformers import AutoProcessor
    gate = study.parent/'preflight/attempt001/tests.xml'
    suites = ET.parse(gate).getroot().findall('testsuite')
    if (sum(int(s.attrib.get('tests', 0)) for s in suites) < 90
            or any(int(s.attrib.get(k, 0)) for s in suites for k in ('failures', 'errors', 'skipped'))):
        raise RuntimeError('complete H100 test gate required')
    study.mkdir(parents=True)
    execution = study / 'private/execution'
    files = subprocess.check_output(['git', 'ls-files'], cwd=ROOT, text=True).splitlines()
    for rel in files:
        p = ROOT / rel
        if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.yml', '.toml') and rel.split('/')[0] in ('src', 'scripts', 'experiments', 'tests'):
            dest = execution/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
    if not (execution/'scripts/v31_humaneval_value_study.py').exists():
        raise RuntimeError('commit study source before freezing')
    if sha(SOURCE) != 'b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef':
        raise ValueError('HumanEval source changed')
    tasks = [json.loads(x) for x in gzip.decompress(SOURCE.read_bytes()).decode().splitlines() if x.strip()]
    tasks.sort(key=lambda x: int(x['task_id'].split('/')[1]))
    if [x['task_id'] for x in tasks] != [f'HumanEval/{i}' for i in range(164)]:
        raise ValueError('full164 identity required')
    shutil.copyfile(SOURCE, study/'private/HumanEval.jsonl.gz')
    processor = AutoProcessor.from_pretrained(str(base.MODEL), local_files_only=True)
    manifest_info = {}
    for mode in MODES:
        thinking = mode == 'thinking_on'
        pool = study/f'private/pools/{mode}'
        manifest, gold = [], {}
        for i, task in enumerate(tasks):
            encoded = processor.apply_chat_template([{'role': 'user', 'content': he.PREFIX+task['prompt']}],
                add_generation_prompt=True, enable_thinking=thinking, tokenize=True, return_tensors='pt')
            if hasattr(encoded, 'keys'):
                encoded = encoded['input_ids']
            ids = encoded[0].tolist()
            manifest.append(dict(id=f'humaneval/{i}', source_id=task['task_id'], benchmark='humaneval',
                prompt=he.PREFIX+task['prompt'], prompt_tokens=ids, prompt_token_count=len(ids),
                thinking=thinking, generation_budget=8192))
            gold[f'humaneval/{i}'] = {k: task[k] for k in ('task_id', 'prompt', 'test', 'entry_point')}
        write(pool/'humaneval_generation_manifest.json', manifest, True)
        write(pool/'gold.json', gold, True)
        cells = [dict(dataset='humaneval', id=m['id'], index=i, prompt_tokens=m['prompt_tokens'], seed=42)
                 for i, m in enumerate(manifest)]
        schedule = study/f'private/schedules/{mode}'
        write(schedule/'cells.json', cells, True)
        order = sorted(range(164), key=lambda i: len(manifest[i]['prompt_tokens']))
        warm = [cells[order[-1]], cells[order[len(order)//2]]]
        write(schedule/'warmup.json', warm, True)
        write(schedule/'smoke.json', warm, True)
        ix, weights = calibration_indices(manifest)
        write(schedule/'calibration.json', [cells[i] for i in ix], True)
        write(schedule/'calibration_weights.json', weights, True)
        for i in range(2):
            write(schedule/f'shard{i+1:03d}.json', cells[i::2], True)
        manifest_info[mode] = dict(thinking=thinking, questions=164,
            prompt_mean=statistics.mean(len(m['prompt_tokens']) for m in manifest),
            prompt_min=min(len(m['prompt_tokens']) for m in manifest),
            prompt_max=max(len(m['prompt_tokens']) for m in manifest))
    inventory = ROOT/'results/v31_value_v2_equal_work_20261011/model_source_fingerprints.json'
    shutil.copyfile(inventory, study/'model_source_fingerprints.json')
    for f in read(inventory)['files']:
        p = base.MODEL/f['file']
        if p.stat().st_size != f['bytes'] or sha(p) != f['sha256']:
            raise RuntimeError('model inventory mismatch')
    shutil.copyfile(SPEC, study/'spec.md')
    shutil.copyfile(gate, study/'preflight_tests.xml')
    protocol = dict(schema='humaneval164_value_k256_v1',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        execution_fingerprints={str(p.relative_to(execution)): sha(p) for p in execution.rglob('*') if p.is_file()},
        input_fingerprints={str(p.relative_to(study)): sha(p) for root in (study/'private/pools', study/'private/schedules') for p in root.rglob('*') if p.is_file()},
        model_inventory_sha256=sha(study/'model_source_fingerprints.json'),
        spec_sha256=sha(study/'spec.md'), runtime=base.runtime_identity(),
        preflight_tests_sha256=sha(study/'preflight_tests.xml'),
        suites={m: SETTINGS for m in MODES}, manifests=manifest_info,
        native=native.native_contract(base.MODEL), seed=42, references=REFERENCES,
        candidates=CANDIDATES, calibration_grid=GRID, calibration_midpoints=2,
        full_cohort_work_refinements=2, passes=['audit', 'clean'],
        primary_control='control_before', control_after='technical bookend, reported separately',
        calibration='transductive attention work only; no answer scoring before threshold freeze',
        timing='clean request/prefill boundaries; separate audit; zero measured graph captures',
        provenance='Yuhan inherited V31; Gaussian32 and query sensitivity are prior user methods',
        status='frozen_before_generation')
    write(study/'protocol.json', protocol, True)
    write(study/'receipts/source_fingerprints.json', dict(source_commit=protocol['source_commit'],
        execution_fingerprints=protocol['execution_fingerprints']), True)
    write(study/'state.json', dict(stage='prepared', paused=False))
    return protocol


def verify(study):
    p = read(study/'protocol.json')
    base.verify(study, p)
    if sha(study/'spec.md') != p['spec_sha256'] or sha(study/'model_source_fingerprints.json') != p['model_inventory_sha256']:
        raise RuntimeError('protocol identity changed')
    base.SUITES.update(p['suites'])
    return p


def entry(arm, threshold=None, label=None):
    return dict(arm=arm, threshold=threshold, label=label or arm)


def schedule(cells, warm, family, pass_name, threshold, calibration=None):
    rows, entries, groups = [], [entry(CONTROL)], []
    if family == 'shared':
        warm_entries = [entry(m, threshold if m == V2 else None) for _ in warm for m in CANDIDATES]
        for c in warm:
            rows.extend([c]*len(CANDIDATES))
        entries.extend(warm_entries)
        warm_count = len(rows)
        for c in cells:
            shift = c['index'] % len(CANDIDATES)
            arms = list(CANDIDATES[shift:]+CANDIDATES[:shift])
            if pass_name == 'audit':
                arms.reverse()
            group = [entry(CONTROL, label='control_before')]+[entry(m, threshold if m == V2 else None) for m in arms]+[entry(CONTROL, label='control_after')]
            groups.append(dict(start=len(rows), end=len(rows)+len(group), labels=[g['label'] for g in group]))
            rows.extend([c]*len(group))
            entries.extend(group)
    elif family == 'calibration':
        for c in warm:
            for t in (0., 1.):
                rows.append(c); entries.append(entry(V2, t, tau_label(t)))
        warm_count = len(rows)
        for c in cells:
            group = [entry(CONTROL, label='control_before')]+[entry(V2, t, tau_label(t)) for t in calibration]
            groups.append(dict(start=len(rows), end=len(rows)+len(group), labels=[g['label'] for g in group]))
            rows.extend([c]*len(group)); entries.extend(group)
    elif family == V2:
        for c in warm:
            rows.append(c); entries.append(entry(V2, threshold))
        warm_count = len(rows)
        for c in cells:
            groups.append(dict(start=len(rows), end=len(rows)+1, labels=[V2]))
            rows.append(c); entries.append(entry(V2, threshold))
    else:
        rows.extend([c for c in warm for _ in CANDIDATES])
        warm_count = len(rows)
        for c in cells:
            groups.append(dict(start=len(rows), end=len(rows)+1, labels=[family]))
            rows.append(c)
    return rows, entries, groups, warm_count


def validate(row, cell, family, live, pass_name, oracle=False):
    errors = []
    a = ((row.get('receipts') or {}).get('adapter') or {})
    checks = [(row.get('dataset') == cell['dataset'] and row.get('index') == cell['index'] and row.get('panel_seed') == 42 and row.get('repeat') == 0, 'input/seed'),
        (row.get('cuda_graph_captures') == 0, 'measured CUDA capture'),
        (row.get('block_size') == 64 and row.get('chunk') == 4096 and row.get('max_model_len') == 9216, 'geometry'),
        (row.get('fix_51994') is True and row.get('fa4_local_fix') is True, 'dense/LOCAL fixes'),
        (row.get('value_clean_timing') == (pass_name == 'clean') and row.get('value_audit') == (pass_name == 'audit'), 'pass flags'),
        (row.get('denoise_forwards', 0) > 0 and row.get('canvases', 0) > 0 and row.get('output_tokens', 0) > 0, 'empty native output'),
        (row.get('budget') == 8192 and row.get('local_kv_budget') is None, 'output/LOCAL budget'),
        (row.get('cuda_matmul_allow_tf32') is False, 'FP32 precision'),
        (row.get('engine_steps') == row.get('prefill_steps', 0)+row.get('sampler_calls', 0), 'native call accounting')]
    if pass_name == 'clean':
        checks += [(row.get('timer_policy') == 'request_and_prefill_boundaries', 'clean boundaries'),
                   (abs(row['wall_s']-row['decode_s']-row['prefill_s']) < 2e-5, 'time sum')]
    if family in ('shared', 'calibration', V2):
        arm = live['arm']
        checks += [(a.get('humaneval_live_arm') == arm, 'live arm'),
                   (a.get('humaneval_threshold') == live.get('threshold'), 'live threshold'),
                   (a.get('humaneval_prefix_budget_tokens') == 256, 'prefix budget')]
        if arm != V2 and pass_name == 'audit':
            checks.append((a.get('humaneval_fixed_budget_errors') == 0, 'fixed query-tile budget'))
        if arm in REFINEMENTS:
            checks += [(a.get('value_refinement') == arm and a.get('value_refinement_layers') == [29], 'refinement scope'),
                       (a.get('value_refinement_backend') == 'device_exact_v1', 'device backend')]
            if pass_name == 'audit':
                checks.append((a.get('value_refinement_work', {}).get('budget_errors') == 0, 'refined budget'))
                if oracle:
                    checks.append((a.get('value_refinement_work', {}).get('humaneval_native_oracle_checks', 0) > 0, 'native oracle'))
    if pass_name == 'audit':
        checks += [(a.get('global_calls') == 5*row['denoise_forwards'], 'GLOBAL calls'),
                   (a.get('native_local_calls_audited') == 25*row['denoise_forwards'], 'LOCAL calls'),
                   (a.get('native_local_skipped_tiles') == 0, 'LOCAL skips'),
                   (a.get('global_eligible_tiles', 0) > 0 and a.get('native_local_eligible_rectangles', 0) > 0, 'physical denominator')]
        if family in ('shared', 'calibration', V2):
            first_canvas(row)
    for okay, name in checks:
        if not okay:
            errors.append(name)
    return errors


def verify_leaf(study, leaf):
    done = read(leaf/'attempt_complete.json')
    if done['protocol_sha256'] != sha(study/'protocol.json'):
        raise RuntimeError('leaf protocol changed')
    for name, digest in done['files_sha256'].items():
        if sha(leaf/name) != digest:
            raise RuntimeError('leaf artifact changed')
    return done


def run_job(study, stage, mode, pass_name, family, cells_file, threshold=None, calibration=None, oracle=False):
    p = verify(study)
    if (study/'PAUSED_USER_REQUEST.json').exists():
        raise RuntimeError('campaign pause marker prevents launch')
    parent = study/stage/mode/pass_name/family/cells_file.stem
    done = list(parent.glob('attempt*/attempt_complete.json'))
    if len(done) > 1:
        raise RuntimeError('ambiguous completed attempt')
    if done:
        verify_leaf(study, done[0].parent)
        return done[0].parent
    number = max([int(x.name[7:]) for x in parent.glob('attempt*')], default=0)+1
    leaf = parent/f'attempt{number:03d}'
    (leaf/'private').mkdir(parents=True)
    cells = read(cells_file)
    warm = read(study/f'private/schedules/{mode}/warmup.json')
    rows, entries, groups, warm_count = schedule(cells, warm, family, pass_name, threshold, calibration)
    write(leaf/'private/cells.json', rows, True)
    write(leaf/'private/arms.json', entries, True)
    write(leaf/'config.json', dict(protocol_sha256=sha(study/'protocol.json'), stage=stage, mode=mode,
        pass_name=pass_name, family=family, threshold=threshold, calibration=calibration,
        groups=groups, warmup_requests=warm_count, oracle=oracle), True)
    base.wait_idle()
    if base.runtime_identity() != p['runtime']:
        raise RuntimeError('runtime identity changed')
    execution = study/'private/execution'
    env = base.env_for(study, mode, CONTROL)
    shared = family in ('shared', 'calibration', V2)
    env.update(STUDY_ARM=family, PYTHONHASHSEED='0', REPEATS='1', TRACE=str(int(pass_name == 'audit')),
        FROZEN_SOURCE_SHA256=sha(study/'receipts/source_fingerprints.json'),
        VALUE_AUDIT=str(int(pass_name == 'audit')), VALUE_CLEAN_TIMING=str(int(pass_name == 'clean')),
        VALUE_BOUNDARY_TIMING=str(int(pass_name == 'clean')),
        HUMANEVAL_REQUEST_ARMS=str(leaf/'private/arms.json'),
        HUMANEVAL_BASE_WORKER=str(execution/'scripts/v31_vllm_paired_bench.py'), HUMANEVAL_ORACLE=str(int(oracle)))
    worker = 'v31_humaneval_value_worker.py' if shared else 'v31_vllm_paired_bench.py'
    arm = 'mage' if shared else 'allkept' if family == 'allkept_fa4' else 'dense'
    cg = 'default' if family == 'dense_full_fix51994' else 'PIECEWISE'
    cmd = [base.PYTHON, str(execution/'scripts'/worker), str(base.MODEL), str(study/f'private/pools/{mode}'),
           str(leaf/'private/cells.json'), str(leaf/'records.jsonl'), str(leaf/'private/generation.private.jsonl'), arm, cg]
    write(study/'state.json', dict(stage=stage, mode=mode, pass_name=pass_name, family=family,
        active_attempt=str(leaf.relative_to(study)), resumable=True))
    start = time.monotonic()
    print(dict(event='launch', stage=stage, mode=mode, pass_name=pass_name, family=family, requests=len(rows)), flush=True)
    with (leaf/'private/worker.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=execution, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(leaf/'private/process.json', dict(pid=proc.pid, command=cmd), True)
        last = start
        while proc.poll() is None:
            if time.monotonic()-last >= 1800:
                count = sum(1 for _ in (leaf/'records.jsonl').open()) if (leaf/'records.jsonl').exists() else 0
                print(dict(event='progress', stage=stage, mode=mode, pass_name=pass_name,
                           family=family, records=count, planned=len(rows)), flush=True)
                last = time.monotonic()
            time.sleep(5)
    public = base.jsonl(leaf/'records.jsonl') if (leaf/'records.jsonl').exists() else []
    private = base.jsonl(leaf/'private/generation.private.jsonl') if (leaf/'private/generation.private.jsonl').exists() else []
    errors = []
    if proc.returncode or len(public) != len(rows) or len(private) != len(rows):
        errors.append('worker/coverage failure')
    if not errors:
        for i, (r, c) in enumerate(zip(public, rows)):
            live = entries[i+1] if shared else None
            errors.extend(validate(r, c, family, live, pass_name, oracle))
            if shared and r['receipts']['adapter'].get('humaneval_request_index') != i+1:
                errors.append('live request index')
    receipt = dict(status='failed' if errors else 'complete', errors=sorted(set(errors)), returncode=proc.returncode,
        records=len(public), measured_records=len(rows)-warm_count, gpu_reservation_s=time.monotonic()-start,
        protocol_sha256=sha(study/'protocol.json'))
    write(leaf/'receipt.json', receipt, True)
    if errors:
        raise RuntimeError('preserved failed attempt '+str(leaf)+': '+str(receipt['errors']))
    exports = defaultdict(list)
    for g in groups:
        for i, label in zip(range(g['start'], g['end']), g['labels']):
            exports[label].append(i)
    files = ['records.jsonl', 'private/generation.private.jsonl', 'config.json', 'receipt.json', 'private/cells.json', 'private/arms.json']
    for label, indices in exports.items():
        for name, data in [(f'exports/{label}/records.jsonl', [dict(public[i], study_arm=label) for i in indices]),
                           (f'exports/{label}/private/run_{label}.private.jsonl', [private[i] for i in indices])]:
            dest = leaf/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('x') as f:
                for row in data:
                    f.write(json.dumps(row)+'\n')
            files.append(name)
    receipt['files_sha256'] = {name: sha(leaf/name) for name in files}
    write(leaf/'attempt_complete.json', receipt, True)
    print(dict(event='complete', stage=stage, mode=mode, pass_name=pass_name, family=family,
               records=len(public), gpu_reservation_s=receipt['gpu_reservation_s']), flush=True)
    return leaf


def exported(leaf, label):
    return base.jsonl(leaf/f'exports/{label}/records.jsonl')


def calibrate(study, mode):
    done = study/f'calibration/{mode}/threshold.json'
    if done.exists():
        return read(done)['threshold']
    cells = study/f'private/schedules/{mode}/calibration.json'
    weights = read(cells.parent/'calibration_weights.json')
    first = run_job(study, 'calibration_grid', mode, 'audit', 'calibration', cells, calibration=list(GRID))
    reference = work(exported(first, 'control_before'), weights)
    candidates = {t: work(exported(first, tau_label(t)), weights) for t in GRID}
    mids = refinement_points(reference, candidates)
    if mids:
        second = run_job(study, 'calibration_midpoints', mode, 'audit', 'calibration', cells, calibration=mids)
        for t in mids:
            candidates[t] = work(exported(second, tau_label(t)), weights)
    best = choose_threshold(reference, candidates)
    write(done, dict(status='work_calibrated', threshold=best, reference=reference,
        candidates={str(t): v for t, v in candidates.items()}, matching=match(reference, candidates[best]),
        question_count=24, weighted_population=164, accuracy_used=False,
        protocol_sha256=sha(study/'protocol.json')), True)
    print(dict(event='threshold_calibrated', mode=mode, threshold=best, matching=match(reference, candidates[best])), flush=True)
    return best


def extract(raw, entry_point, thinking):
    final, _ = official.load_final_response()
    visible = final(raw, thinking)
    return he.extract_code('<channel|>'+visible, entry_point)


def scorer_gate(study):
    dest = study/'qualification/scorer_gate.json'
    if dest.exists():
        if read(dest)['passed'] != 328:
            raise RuntimeError('scorer gate failed')
        return
    he.sandbox_preflight()
    tasks = [json.loads(x) for x in gzip.decompress((study/'private/HumanEval.jsonl.gz').read_bytes()).decode().splitlines() if x.strip()]
    jobs = []
    for thinking in (True, False):
        for t in tasks:
            text = '```python\n'+t['prompt']+t['canonical_solution']+'\n```<turn|>'
            raw = '<|channel>thought\n<channel|>'+text if thinking else text
            code, why = extract(raw, t['entry_point'], thinking)
            if why != 'parsed':
                raise RuntimeError('canonical extraction failed')
            jobs.append((t, code))
    results, timeouts = run_all(jobs, he)
    receipt = dict(passed=sum(x[0] for x in results), expected=328,
        timeouts_first_pass=timeouts, timeouts_after_rerun=sum(x[1] == 'timeout' for x in results),
        scorer_sha256=sha(Path(he.__file__)), modes=['thinking_on', 'thinking_off'],
        extractor='official final_response(raw, thinking), then existing first complete-function extractor',
        program=he.PROGRAM, sandbox=he.SANDBOX)
    write(dest, receipt, True)
    if receipt['passed'] != 328:
        raise RuntimeError('canonical scorer gate failed')
    print(dict(event='scorer_gate', passed=328), flush=True)


def full_audit(study, mode, threshold):
    folder = study/f'private/schedules/{mode}'
    leaves = [run_job(study, 'runs', mode, 'audit', 'shared', folder/f'shard{i:03d}.json', threshold) for i in (1, 2)]
    reference = work([r for leaf in leaves for r in exported(leaf, 'control_before')])
    candidates = {threshold: work([r for leaf in leaves for r in exported(leaf, V2)])}
    chosen_paths = {threshold: leaves}
    for iteration in range(2):
        best = choose_threshold(reference, candidates)
        if match(reference, candidates[best])['matched']:
            break
        grid_report = read(study/f'calibration/{mode}/threshold.json')
        all_points = {float(t): v for t, v in grid_report['candidates'].items()}
        all_points.update(candidates)
        point = candidates[best]['kept_fraction']
        others = sorted((t for t in all_points if t != best), key=lambda t: abs(t-best))
        opposite = [t for t in others if (all_points[t]['kept_fraction']-reference['kept_fraction'])*(point-reference['kept_fraction']) < 0]
        neighbor = opposite[0] if opposite else (best*2 if point > reference['kept_fraction'] else best/2)
        t = math.sqrt(best*neighbor) if best and neighbor else max(best, neighbor)/2
        if t in candidates:
            break
        extra = [run_job(study, f'work_refinement{iteration+1:03d}', mode, 'audit', V2, folder/f'shard{i:03d}.json', t) for i in (1, 2)]
        candidates[t] = work([r for leaf in extra for r in exported(leaf, V2)])
        chosen_paths[t] = extra
    best = choose_threshold(reference, candidates)
    result = dict(status='frozen_before_clean', threshold=best, reference=reference,
        candidates={str(t): v for t, v in candidates.items()}, matching=match(reference, candidates[best]),
        selected_v2_leaves=[str(p.relative_to(study)) for p in chosen_paths[best]],
        shared_leaves=[str(p.relative_to(study)) for p in leaves], accuracy_used=False,
        protocol_sha256=sha(study/'protocol.json'))
    dest = study/f'calibration/{mode}/full164_threshold.json'
    write(dest, result, True)
    print(dict(event='full164_work_match', mode=mode, threshold=best, **result['matching']), flush=True)
    return result


def selected_paths(study, mode, pass_name):
    result = defaultdict(list)
    for leaf in sorted((study/f'runs/{mode}/{pass_name}').glob('*/*/attempt*/attempt_complete.json')):
        verify_leaf(study, leaf.parent)
        for label in LABELS:
            path = leaf.parent/f'exports/{label}'
            if path.exists():
                result[label].append(path)
    if pass_name == 'audit':
        final = read(study/f'calibration/{mode}/full164_threshold.json')
        result[V2] = [study/p/f'exports/{V2}' for p in final['selected_v2_leaves']]
    return result


def score(study, mode, pass_name, paths):
    dest = study/f'private/scoring/{mode}/{pass_name}.json'
    if dest.exists():
        return read(dest)
    pool = study/f'private/pools/{mode}'
    gold = read(pool/'gold.json')
    pools = official.Pools(pool, ('thinking',))
    jobs, identities, reasons = [], [], Counter()
    fingerprints = {}
    for label in LABELS:
        records = []
        for path in paths[label]:
            f = path/f'private/run_{label}.private.jsonl'
            fingerprints[str(f.relative_to(study))] = sha(f)
            records.extend(base.jsonl(f))
        if len(records) != 164 or {r['index'] for r in records} != set(range(164)):
            raise RuntimeError('official scoring coverage mismatch: '+label)
        for r in records:
            _, binding, verified = official.bind(r, pools, False)
            if not verified or r['id'] != f"humaneval/{r['index']}" or r['panel_seed'] != 42:
                raise RuntimeError('official input binding failed')
            problem = gold[r['id']]
            code, why = extract(r['completion'], problem['entry_point'], mode == 'thinking_on')
            reasons[label+'|'+why] += 1
            jobs.append((problem, code))
            identities.append((label, str(r['index']), official.finish_class(r['finish_reason'])))
    results, timeouts = run_all(jobs, he)
    scores = defaultdict(dict)
    for (label, index, finish), (passed, outcome) in zip(identities, results):
        scores[label][index] = dict(passed=bool(passed), pass_and_finished=bool(passed) and finish == 'finished', finish=finish, outcome=outcome)
    result = dict(scores=dict(scores), extraction=dict(reasons), timeouts_first_pass=timeouts,
        input_sha256=fingerprints, protocol_sha256=sha(study/'protocol.json'))
    write(dest, result, True)
    return result


def summarize(study):
    verify(study)
    results = {}
    for mode in MODES:
        scores = {}
        paths = {}
        for pass_name in ('audit', 'clean'):
            paths[pass_name] = selected_paths(study, mode, pass_name)
            scores[pass_name] = score(study, mode, pass_name, paths[pass_name])
        summaries, arrays = {}, {}
        for label in LABELS:
            clean = {r['index']: r for p in paths['clean'][label] for r in base.jsonl(p/'records.jsonl')}
            audit = {r['index']: r for p in paths['audit'][label] for r in base.jsonl(p/'records.jsonl')}
            if set(clean) != set(range(164)) or set(audit) != set(clean):
                raise RuntimeError('full164 record coverage')
            values = np.array([[scores['clean']['scores'][label][str(i)]['passed'], clean[i]['denoise_forwards'], clean[i]['canvases'],
                clean[i]['output_tokens'], clean[i]['wall_s'], clean[i]['decode_s'], clean[i]['prefill_s']] for i in range(164)], dtype=float)
            arrays[label] = values
            a = [audit[i]['receipts']['adapter'] for i in range(164)]
            required = ('global_eligible_tiles', 'global_kept_tiles', 'native_local_eligible_rectangles', 'native_local_skipped_tiles')
            totals = {k: sum(x[k] for x in a) for k in required}
            e, k, local = totals['global_eligible_tiles'], totals['global_kept_tiles'], totals['native_local_eligible_rectangles']
            traces = [r['receipts'].get('trace') for r in audit.values()]
            steps = [int(n) for t in traces if t for n in t['canvas_steps']]
            item = dict(questions=164, correct=int(values[:, 0].sum()), accuracy=values[:, 0].mean(),
                audit_correct=sum(x['passed'] for x in scores['audit']['scores'][label].values()),
                pass_and_finished=sum(x['pass_and_finished'] for x in scores['clean']['scores'][label].values()),
                N_mean=values[:, 1].mean(), N_median=float(np.median(values[:, 1])), N_p90=float(np.quantile(values[:, 1], .9)),
                N_p95=float(np.quantile(values[:, 1], .95)), N_max=int(values[:, 1].max()), C_mean=values[:, 2].mean(),
                N_per_C=values[:, 1].sum()/values[:, 2].sum(), T_mean=values[:, 3].mean(),
                W_mean_s=values[:, 4].mean(), W_median_s=float(np.median(values[:, 4])), W_p90_s=float(np.quantile(values[:, 4], .9)),
                S_mean_s=values[:, 5].mean(), P_mean_s=values[:, 6].mean(), amortized_S_per_N_ms=1000*values[:, 5].sum()/values[:, 1].sum(),
                generation_caps=sum(x['finish'] == 'capped' for x in scores['clean']['scores'][label].values()),
                audited_canvas_steps_available=len(steps), audited_cap_canvases=sum(x >= 48 for x in steps),
                global_sparsity=1-k/e, local_sparsity=totals['native_local_skipped_tiles']/local,
                overall_sparsity=(e-k)/(e+local), physical_totals=totals,
                clean_audit_hash_differences=sum(clean[i]['output_hash'] != audit[i]['output_hash'] for i in range(164)),
                clean_audit_N_differences=sum(clean[i]['denoise_forwards'] != audit[i]['denoise_forwards'] for i in range(164)),
                clean_audit_score_differences=sum(scores['clean']['scores'][label][str(i)]['passed'] != scores['audit']['scores'][label][str(i)]['passed'] for i in range(164)))
            if label not in REFERENCES:
                item['first_canvas'] = work(list(audit.values()))
            extra = Counter()
            timing = Counter()
            for x in a:
                extra.update(x.get('value_refinement_work', {}))
                if x.get('value_audit_timings'):
                    timing.update(x['value_audit_timings'])
                if 'value_anchor_module_cuda_elapsed_ms' in x:
                    timing['anchor_module_cuda_elapsed_ms'] += x['value_anchor_module_cuda_elapsed_ms']
            item['refinement_work'] = dict(extra)
            item['audit_selector_times_ms'] = dict(timing)
            summaries[label] = item
        rng = np.random.default_rng(1729)
        indices = rng.integers(0, 164, size=(20000, 164))
        contrasts = {}
        for label in LABELS:
            if label == 'control_before':
                continue
            ref, val = arrays['control_before'], arrays[label]
            delta = (val[:, 0]-ref[:, 0])*100
            draws = delta[indices].mean(1)
            speed = ref[indices, 4].mean(1)/val[indices, 4].mean(1)
            nd = (val[:, 1]-ref[:, 1])[indices].mean(1)
            contrasts[label] = dict(accuracy_delta_pp=float(delta.mean()), accuracy_delta_ci95=np.quantile(draws, [.025, .975]).tolist(),
                W_speedup=float(ref[:, 4].mean()/val[:, 4].mean()), W_speedup_ci95=np.quantile(speed, [.025, .975]).tolist(),
                N_delta=float((val[:, 1]-ref[:, 1]).mean()), N_delta_ci95=np.quantile(nd, [.025, .975]).tolist(),
                candidate_only_correct=int(((val[:, 0] == 1) & (ref[:, 0] == 0)).sum()),
                control_only_correct=int(((val[:, 0] == 0) & (ref[:, 0] == 1)).sum()))
        results[mode] = dict(methods=summaries, paired_vs_v31=contrasts,
            threshold=read(study/f'calibration/{mode}/full164_threshold.json'))
    receipts = [read(p) for p in study.rglob('receipt.json') if '/private/' not in str(p)]
    report = dict(status='complete', protocol_sha256=sha(study/'protocol.json'), modes=results,
        gpu_reservation_s=sum(r['gpu_reservation_s'] for r in receipts),
        limitations=['single seed42; work calibration overlaps evaluated prompts',
            'V2/V3a Gaussian32 all GLOBAL; swaps/residual modify GLOBAL layer29 only',
            'effective physical sparsity from separate audit; clean/audit drift explicitly reported',
            'Q128/KV64 geometry rectangles, not a count of CUDA CTAs',
            'S/N is amortized across native trajectory, not a controlled per-forward price'])
    write(study/'summary.json', report, True)
    lines = ['# HumanEval164 K256 seed42', '', 'Clean native accuracy and request latency; separate audit sparsity. Full inputs, generated text, gold and token IDs remain private.', '']
    for mode, result in results.items():
        t = result['threshold']
        lines += [f'## {mode}', '', f"V2 threshold: {t['threshold']:.12g}; full164 first-canvas matching: {t['matching']}.", '',
            '| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in LABELS:
            m = result['methods'][label]
            lines.append(f"| {label} | {m['correct']}/164 | {m['N_mean']:.3f} | {m['C_mean']:.3f} | {m['N_per_C']:.3f} | {m['T_mean']:.2f} | {m['S_mean_s']:.5f} | {m['W_mean_s']:.5f} | {100*m['global_sparsity']:.3f}% | {100*m['overall_sparsity']:.3f}% |")
        lines += ['', 'Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.', '']
    lines += ['## Limits', '', *['- '+s for s in report['limitations']], '', f"Charged GPU reservation: {report['gpu_reservation_s']:.3f} seconds. Older LongBench campaigns remain paused."]
    (study/'summary.md').write_text('\n'.join(lines)+'\n')
    bound = ['protocol.json', 'spec.md', 'summary.json', 'summary.md', 'qualification/scorer_gate.json']
    write(study/'final_complete.json', dict(status='complete', protocol_sha256=sha(study/'protocol.json'),
        files_sha256={n: sha(study/n) for n in bound}, source_commit=read(study/'protocol.json')['source_commit'],
        expected_unique_questions=164, thinking_modes=MODES, passes=['audit', 'clean'],
        gpu_reservation_s=report['gpu_reservation_s']), True)
    write(study/'state.json', dict(stage='complete', active_attempt=None, gpu_reservation_s=report['gpu_reservation_s']))
    print(dict(event='campaign_complete', gpu_reservation_s=report['gpu_reservation_s']), flush=True)


def run(study):
    verify(study)
    scorer_gate(study)
    for mode in MODES:
        threshold = calibrate(study, mode)
        folder = study/f'private/schedules/{mode}'
        for pass_name in ('audit', 'clean'):
            run_job(study, 'qualification', mode, pass_name, 'shared', folder/'smoke.json', threshold,
                    oracle=pass_name == 'audit')
            for family in REFERENCES:
                run_job(study, 'qualification', mode, pass_name, family, folder/'smoke.json')
        full_audit(study, mode, threshold)
    for shard in (1, 2):
        for mi, mode in enumerate(MODES):
            folder = study/f'private/schedules/{mode}'
            threshold = read(study/f'calibration/{mode}/full164_threshold.json')['threshold']
            families = ['shared', *REFERENCES]
            shift = (shard+mi) % len(families)
            families = families[shift:]+families[:shift]
            for family in families:
                passes = ('clean', 'audit') if (shard+mi) % 2 else ('audit', 'clean')
                for pass_name in passes:
                    run_job(study, 'runs', mode, pass_name, family, folder/f'shard{shard:03d}.json', threshold)
    summarize(study)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report'])
    parser.add_argument('--study', type=Path, default=DEFAULT)
    args = parser.parse_args()
    study = args.study.resolve()
    if args.action == 'prepare':
        prepare(study)
    elif args.action == 'run':
        run(study)
    else:
        summarize(study)


if __name__ == '__main__':
    main()
