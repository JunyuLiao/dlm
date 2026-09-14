"""User-authorized small tuning screen and cache-reusing selected final sweep.

This orchestration changes neither frozen scientific sources nor the original
47-condition contract. The reduced bundle has its own contract/report, with a
validated link to the original final cache. No held-out scores select methods.
"""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

DEFAULT_ROOT = Path('results/diffusion_gemma_value_aware_128x64')
SCREEN_METHODS = ('blasst_original', 'sol_plain_gaussian', 'mass_exact', 'mass_value')
CANDIDATES = ('mass_exact', 'mass_value')
TARGETS = (25, 50, 75, 90)
SELECTION_RULE = (
    'Only complete calibration16 results at targets50/75; maximize the mean '
    'dense-relative score, equally weighting benchmarks and targets (LongBench '
    'uses an equal-task macro). Ties: higher measured physical sparsity, then '
    'retained mass, then pre-existing primary mass_exact. This is a tuning '
    'heuristic, not a claim of superiority at matched actual sparsity. Final '
    'comparisons require measured-sparsity matching and held-out evaluation.'
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    _write(path, data)


def guard_legacy_run(root, stage):
    """An interrupted old finalizer must not resurrect the superseded sweep."""
    if stage in ('run-final', 'finalize') and (root/'efficient_iteration'/'iteration_plan.json').exists():
        raise RuntimeError(
            'User superseded the 47-condition sweep with efficient_iteration. '
            'Do not restart the old finalizer/run-final. See ITERATION_SCOPE_CHANGE.md '
            'and efficient_iteration/job.json; use the efficient_iteration launcher only.'
        )


def screen_conditions(contract):
    names = ['dense'] + [f'{method}_s{target}' for method in SCREEN_METHODS for target in (50, 75)]
    return {name: contract['conditions'][name] for name in names}


def final_condition_names(winner):
    if winner is not None and winner not in CANDIDATES:
        raise ValueError('unknown screening finalist')
    methods = ['blasst_original', 'sol_plain_gaussian'] + ([winner] if winner else [])
    return ['dense'] + [f'{method}_s{target}' for method in methods for target in TARGETS]


def screen_rows(setup):
    rows = setup['calibration']
    heldout = [r for r in setup['final'] if not r['calibration']]
    assert len(rows) == 16
    assert sum(r['benchmark'] == 'aime26' for r in rows) == 6
    tasks = defaultdict(int)
    for row in rows:
        if row['benchmark'] == 'longbench':
            tasks[row['task']] += 1
    assert len(tasks) == 5 and set(tasks.values()) == {2}
    for key in ('id', 'prompt_hash'):
        assert len({r[key] for r in rows}) == len(rows)
        assert not {r[key] for r in rows} & {r[key] for r in heldout}
    return rows


def link_input(path, source):
    """Never replace a file or redirect an existing cache link."""
    source = source.resolve(strict=True)
    if path.is_symlink():
        if path.resolve(strict=True) != source:
            raise RuntimeError(f'incompatible input link: {path}')
    elif path.exists():
        raise RuntimeError(f'refusing to replace existing input: {path}')
    else:
        path.symlink_to(source, target_is_directory=source.is_dir())


def prepare(root):
    from .protocol import fingerprint, frozen_write, prepare as original_prepare, sha
    setup = original_prepare(root)
    contract = read(root/'final_contract.json')
    fp = fingerprint(root)
    assert contract['fingerprint'] == fp
    rows = screen_rows(setup)
    dest = root/'efficient_iteration'
    dest.mkdir(exist_ok=True)
    for name in ('setup.json', 'smoke.json', 'refinement_smoke.json', 'ranking_guard_smoke.json', 'final'):
        link_input(dest/name, root/name)
    # Preserve optional explanatory references without linking generated reports.
    for name in ('literature_verification.md', 'README.md'):
        if (root/name).exists():
            link_input(dest/name, root/name)
    assert fingerprint(dest) == fp
    frozen_write(dest/'screen_manifest.json', rows)
    conditions = screen_conditions(contract)
    plan = dict(schema='efficient_iteration_v1', fingerprint=fp,
        original_contract=str(root/'final_contract.json'),
        original_contract_sha256=sha((root/'final_contract.json').read_bytes()),
        original_expected=3760, screening_conditions=conditions, screening_samples=16,
        screening_expected=144, screen_ids=[r['id'] for r in rows],
        final_fixed_methods=['blasst_original', 'sol_plain_gaussian'],
        candidate_methods=list(CANDIDATES), final_targets=list(TARGETS),
        final_expected=1040, final_examples=80, selection_rule=SELECTION_RULE,
        heldout_used_for_selection=False, calibration_used_for_screening=True,
        diagnostics='unchanged full diagnostics on reduced scope; no algorithm/source fingerprint changes',
        extra_seed_sweep=False, monitor_seconds=900)
    frozen_write(dest/'iteration_plan.json', plan)
    snapshot = dest/'superseded_job.json'
    if not snapshot.exists():
        write(snapshot, dict(job=read(root/'job.json'), progress=read(root/'progress.json'),
            captured_at=time.time(), reason='explicit user request to reduce sweep'))
    return dest, setup, contract, plan


def matching_sources(root, row, name, condition):
    """Never read held-out outputs: caller is restricted to calibration16."""
    from .run import shard_path
    if name == 'dense':
        path = shard_path(root, 'screen', 'dense', row['id'])
        return [path] if path.exists() else []
    sources = []
    # Only the six explicitly shared AIME tuning examples may reuse final-path
    # caches. No heldout24 or LongBench final result is opened for selection.
    if row['benchmark'] == 'aime26' and row['calibration']:
        path = shard_path(root, 'final', name, row['id'])
        if path.exists():
            sources.append(path)
    policy_path = root/'verified_policies'/row['benchmark']/f'{name}.json'
    if not policy_path.exists():
        return sources
    policy = read(policy_path)
    expected = condition.get('thresholds', {}).get(row['benchmark'])
    if policy['config'] != condition['config']:
        return sources
    for point in policy['trace']:
        if point['policy'] != expected or row['id'] not in point['source_ids']:
            continue
        path = shard_path(root, 'calibration', point['source_condition'], row['id'])
        if path.exists():
            sources.append(path)
    return list(dict.fromkeys(sources))


def reuse_screen(root, dest, setup, plan):
    from .evaluate import reuse_dense
    from .protocol import sha
    from .report import inspect_output
    from .run import shard_path
    reused = []; missing = []; rejected = []
    for name, condition in plan['screening_conditions'].items():
        for row in screen_rows(setup):
            path = shard_path(dest, 'screening', name, row['id'])
            if path.exists():
                data, _, errors = inspect_output(read(path), row, plan['fingerprint'], condition)
                if errors:
                    raise RuntimeError(f'incompatible screening cache {path}: {errors}')
                origin = data.get('reused_screening_source', data.get('reused_dense_source'))
                if origin:
                    reused.append(dict(id=row['id'], condition=name, source=origin['path']))
                continue
            for source in matching_sources(root, row, name, condition):
                if name == 'dense':
                    reuse_dense(source, path, row, plan['fingerprint'])
                else:
                    data, _, errors = inspect_output(read(source), row, plan['fingerprint'], condition)
                    if errors or data['screen']:
                        rejected.append(dict(source=str(source), errors=errors or ['screen observer']))
                        continue
                    data['reused_screening_source'] = dict(path=str(source), sha256=sha(source.read_bytes()))
                    write(path, data)
                reused.append(dict(id=row['id'], condition=name, source=str(source)))
                break
            else:
                missing.append(dict(id=row['id'], condition=name))
    write(dest/'cache_reuse.json', dict(reused=reused, pending=missing, expected=144,
        rejected_sources=rejected, heldout_used=False, reason='identical sample/config/threshold/fingerprint'))
    return len(missing)


def select_candidate(summaries):
    """Prespecified tuning ranking; never consume the final-run summaries."""
    index = {(r['benchmark'], r['condition']): r for r in summaries}
    ranking = []
    for method in CANDIDATES:
        points = [index.get((benchmark, f'{method}_s{target}'))
            for benchmark in ('aime26', 'longbench') for target in (50, 75)]
        if any(p is None or p['count'] != (6 if p['benchmark'] == 'aime26' else 10) for p in points):
            continue
        ranking.append(dict(method=method,
            mean_dense_delta=sum(p['accuracy']-p['dense_accuracy'] for p in points)/4,
            mean_physical_sparsity=sum(p['overall_physical_sparsity'] for p in points)/4,
            mean_retained_mass=sum(p['overall_denominator_mass'] for p in points)/4))
    ranking.sort(key=lambda r: (-round(r['mean_dense_delta'], 12),
        -round(r['mean_physical_sparsity'], 12), -round(r['mean_retained_mass'], 12),
        CANDIDATES.index(r['method'])))
    return (ranking[0]['method'] if ranking else None), ranking


def summarize_screen(dest, setup, plan):
    from .protocol import score, sha
    from .report import inspect_output, summary, csv_write
    from .report_metrics import aggregate
    from .run import shard_path
    from experiments.diffusion_gemma_solattn_blasst_multibench.controlled_report import token_counts
    raw = []; sources = {}; missing = []; violations = []
    for row in screen_rows(setup):
        dense_path = shard_path(dest, 'screening', 'dense', row['id'])
        dense = read(dense_path) if dense_path.exists() else None
        for name, condition in plan['screening_conditions'].items():
            path = shard_path(dest, 'screening', name, row['id'])
            if not path.exists() or dense is None:
                missing.append(dict(id=row['id'], condition=name)); continue
            data, records, errors = inspect_output(read(path), row, plan['fingerprint'], condition)
            if errors:
                violations.append(dict(id=row['id'], condition=name, errors=errors)); continue
            sources[str(path)] = sha(path.read_bytes())
            matching, compared = token_counts(dense['completion_tokens'], data['completion_tokens'])
            raw.append(dict(id=row['id'], benchmark=row['benchmark'], task=row['task'], condition=name,
                accuracy=score(row, data['prediction']), dense_accuracy=score(row, dense['prediction']),
                matching=matching, compared=compared, exact_match=data['completion_tokens']==dense['completion_tokens'],
                termination_reason=data['termination_reason'], output_length=len(data['completion_tokens']),
                aggregates={kind: aggregate([r for r in records if kind=='overall' or r['attention_type']==kind])
                    for kind in ('overall', 'global', 'local')}))
    groups = defaultdict(list); tasks = defaultdict(list)
    for row in raw:
        groups[row['benchmark'], row['condition']].append(row)
        tasks[row['benchmark'], row['task'], row['condition']].append(row)
    summaries = [dict(benchmark=b, condition=n, **summary(g)) for (b, n), g in groups.items()]
    audit = dict(complete=len(raw)==144 and not missing and not violations, expected=144, completed=len(raw),
        missing=missing, violations=violations, sources=sources, heldout_used=False)
    write(dest/'screening_audit.json', audit)
    write(dest/'screening_per_sample.json', raw)
    write(dest/'screening_summary.json', summaries)
    csv_write(dest/'screening_summary.csv', summaries)
    csv_write(dest/'screening_per_task.csv', [dict(benchmark=b, task=t, condition=n, **summary(g))
        for (b, t, n), g in tasks.items()])
    return summaries, audit


def freeze_reduced(root, dest, setup, contract, plan):
    from .protocol import frozen_write, sha
    summaries, audit = summarize_screen(dest, setup, plan)
    winner, ranking = select_candidate(summaries)
    decision = dict(selected_methods=[winner] if winner else [], heldout_used=False,
        selection_rule=SELECTION_RULE, ranking=ranking, screening_complete=audit['complete'],
        screening_audit_sha256=sha((dest/'screening_audit.json').read_bytes()),
        rationale='Small calibration-only screen; final point scores cannot establish a matched-sparsity win by themselves',
        failure='No complete candidate: continue independent reference conditions only' if winner is None else None)
    frozen_write(dest/'candidate_decision.json', decision)
    names = final_condition_names(winner)
    reduced = dict(contract, conditions={name: contract['conditions'][name] for name in names},
        secondary_scope='Only plain Sol Gaussian all four targets; other diagnostic runs archived',
        user_authorized_scope_reduction=True, original_condition_count=47,
        original_contract_sha256=plan['original_contract_sha256'],
        screening_selection=decision, heldout_used_for_selection=False,
        sources={**contract['sources'], str(root/'final_contract.json'):plan['original_contract_sha256'],
            str(Path(__file__).resolve()):sha(Path(__file__).read_bytes()),
            **audit['sources'], **{str(dest/p):sha((dest/p).read_bytes()) for p in
                ('iteration_plan.json', 'screen_manifest.json', 'screening_audit.json', 'candidate_decision.json')}})
    frozen_write(dest/'final_contract.json', reduced)
    write(dest/'selection_status.json', dict(finalist=winner, conditions=len(names), expected=80*len(names)))
    return reduced


def execute(root):
    from dllm.models import create_adapter
    from .protocol import MODEL, REVISION
    from .evaluate import run_group
    from .execution import require_refinement_smoke
    from .run import shard_path, smoke
    dest, setup, contract, plan = prepare(root)
    reuse_screen(root, dest, setup, plan)
    require_refinement_smoke(dest, plan['fingerprint'], [c['config'] for c in plan['screening_conditions'].values()])
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    smoke(adapter, setup, dest)
    write(dest/'phase.json', dict(stage='screening', expected=144, started=time.time()))
    for name, condition in plan['screening_conditions'].items():
        for benchmark in ('aime26', 'longbench'):
            rows = [r for r in screen_rows(setup) if r['benchmark']==benchmark
                and not shard_path(dest, 'screening', name, r['id']).exists()]
            run_group(adapter, dest, rows, 'screening', name, condition['config'],
                condition.get('thresholds', {}).get(benchmark), plan['fingerprint'])
    reduced = freeze_reduced(root, dest, setup, contract, plan)
    write(dest/'phase.json', dict(stage='selected-final', expected=80*len(reduced['conditions']), started=time.time()))
    # Baseline first, then cheap Sol configurations, then the one selected mass
    # method. Completed BLASST and dense shards are validated/reused by cache().
    order = ['dense'] + [n for n in reduced['conditions'] if n.startswith('blasst_original')]
    order += [n for n in reduced['conditions'] if n.startswith('sol_plain_gaussian')]
    order += [n for n in reduced['conditions'] if n not in order]
    for name in order:
        condition = reduced['conditions'][name]
        for benchmark in ('aime26', 'longbench'):
            rows = [r for r in setup['final'] if r['benchmark']==benchmark]
            run_group(adapter, dest, rows, 'final', name, condition['config'],
                condition.get('thresholds', {}).get(benchmark), plan['fingerprint'])
    write(dest/'execution_terminal.json', dict(finished=time.time(), conditions=len(order)))


def progress(dest):
    screen = len(list((dest/'screening').glob('*/shards/*.json')))
    contract_path = dest/'final_contract.json'
    result = dict(screening_completed=screen, screening_expected=144)
    if contract_path.exists():
        names = read(contract_path)['conditions']
        counts = {n:len(list((dest/'final'/n/'shards').glob('*.json'))) for n in names}
        result.update(final_completed=sum(counts.values()), final_expected=80*len(names), final_counts=counts)
    if (dest/'phase.json').exists():
        result['phase'] = read(dest/'phase.json')
    if (dest/'progress.json').exists():
        result['progress'] = read(dest/'progress.json')
    return result


def report(dest):
    from .report import report as existing_report
    existing_report(dest)
    audit = read(dest/'audit.json')
    if audit['complete']:
        p = dest/'report.md'
        text = p.read_text()
        introduction = ('> User-authorized reduced scope: calibration16 screening at50/75; '
            'dense + original BLASST + plain Sol Gaussian + one selected mass method at all four targets. '
            'The original47-condition sweep was superseded, not completed. Historical mechanisms '
            'described below are background, not additional required final configurations. '
            'See `iteration_plan.json`, `screening_summary.csv` and `candidate_decision.json`. '
            'Full AIME30 includes the six tuning problems; heldout24 is the confirmatory AIME view.\n\n')
        reproduction = ('\nReduced-bundle regeneration (CPU only): '
            '`PYTHONPATH=src CUDA_VISIBLE_DEVICES="" python -m experiments.diffusion_gemma_value_aware.efficient_iteration report`\n')
        p.write_text(introduction+text+reproduction)
    return audit['complete']


def launch(root):
    dest = root/'efficient_iteration'
    if not (dest/'iteration_plan.json').exists():
        raise RuntimeError('run prepare and its tests before launching')
    with (dest/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=20)
        if gpu.returncode or gpu.stdout.strip():
            raise RuntimeError(f'GPU not verified idle; refusing competing worker: {gpu.stdout} {gpu.stderr}')
        from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _append
        with (dest/'run.log').open('a', buffering=1) as log:
            worker = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, 'execute', '--root', str(root)],
                stdout=log, stderr=subprocess.STDOUT)
            job = dict(pid=worker.pid, supervisor_pid=os.getpid(), stage='efficient-iteration', started=time.time(),
                output=str(dest), monitoring_interval_seconds=900, supersedes_expected=3760)
            write(dest/'job.json', job); write(root/'job.json', job)
            write(root/'progress.json', dict(pid=worker.pid, stage='efficient-iteration initialization',
                path=str(dest/'run.log'), started=time.time()))
            while True:
                try:
                    code = worker.wait(timeout=900); break
                except subprocess.TimeoutExpired:
                    state = dict(time=time.time(), pid=worker.pid, alive=worker.poll() is None,
                        stage='efficient-iteration', disk_free_bytes=shutil.disk_usage(dest).free, **progress(dest))
                    _append(dest/'monitor.jsonl', state); _append(root/'monitor.jsonl', state)
                    if 'progress' in state:
                        write(root/'progress.json', state['progress'])
                    print(json.dumps(state), flush=True)
        write(dest/'worker_terminal.json', dict(exit_code=code, finished=time.time(), **progress(dest)))
        cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2')
        result = subprocess.run([sys.executable, '-u', '-m', __spec__.name, 'report', '--root', str(root)], env=cpu_env)
        audit = read(dest/'audit.json') if (dest/'audit.json').exists() else {}
        screening = read(dest/'screening_audit.json') if (dest/'screening_audit.json').exists() else {}
        selection = read(dest/'candidate_decision.json') if (dest/'candidate_decision.json').exists() else {}
        status = dict(complete=code==0 and result.returncode==0 and audit.get('complete') is True
                and screening.get('complete') is True and bool(selection.get('selected_methods')),
            screening_complete=screening.get('complete', False), selected_methods=selection.get('selected_methods', []),
            worker_exit_code=code, report_exit_code=result.returncode, finished=time.time(), **progress(dest))
        write(dest/'terminal.json', status)
        print(json.dumps(status), flush=True)
        return 0 if status['complete'] else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'execute', 'launch', 'report', 'status'])
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.command == 'prepare':
        dest, setup, _, plan = prepare(args.root)
        pending = reuse_screen(args.root, dest, setup, plan)
        print(json.dumps(dict(output=str(dest), screen_expected=144, pending_screen_inferences=pending)))
    elif args.command == 'execute':
        execute(args.root)
    elif args.command == 'launch':
        sys.exit(launch(args.root))
    elif args.command == 'report':
        sys.exit(0 if report(args.root/'efficient_iteration') else 1)
    else:
        print(json.dumps(progress(args.root/'efficient_iteration'), indent=2))


if __name__ == '__main__':
    main()
