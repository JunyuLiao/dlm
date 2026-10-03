"""Frozen calibration, two-seed screen, and step-aware finalist evaluation.

Uses a single local/global threshold pair per arm at every denoising call.
All choices are development choices on the already exposed AIME26 manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np

from . import aime_temporal_sweep as base
from . import aime_query_sensitivity_uniform as uniform
from .aime_query_sensitivity_uniform_report import summarize, question_disagreements
from .experiment import atomic, fingerprint, sha
from .query_adaptive import State, observe


DEFAULT_ROOT = Path('/home/exouser/aime_query_sensitivity_gated_v2')
PRIOR = Path('/home/exouser/aime_query_sensitivity_uniform_causal_v1')
CONTROLS = base.DLMDIR / 'results/query_adaptive_aime_temporal_v14'
METHODS = ('M_gate', 'M_gate_norm', 'C_gate', 'T_gate')
TAUS = (1.5, 2.5, 3.5)
SCREEN_IDS = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28)
SCREEN_SEEDS = (42, 43)
FINAL_SEEDS = (42, 43, 44)


def arms():
    return [dict(name=f'{method}_tau{str(tau).replace(".", "p")}',
                 method=method, tau=tau) for method in METHODS for tau in TAUS]


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'frozen protocol/configuration changed: {path}')
    else:
        atomic(path, value)
    return value


def prepare(root):
    final, calibration = base._load_rows()
    sources = [Path(__file__), Path(base.__file__), Path(uniform.__file__),
               Path(__file__).with_name('query_adaptive.py'),
               Path(__file__).with_name('query_sensitivity_uniform.py'),
               Path(__file__).with_name('integration.py')]
    protocol = dict(schema='aime26_gated_v2', arms=arms(), target=50,
        calibration_ids=[r['id'] for r in calibration],
        screen_ids=[f'aime26/{i}' for i in SCREEN_IDS],
        screen_seeds=list(SCREEN_SEEDS), final_seeds=list(FINAL_SEEDS),
        trajectory_gamma=.65, beta=3., flip_gamma=.5,
        normalized_margin='clip((log1p(m_ref)-log1p(margin))/sigma,0,1)',
        normalization='sigma=max(IQR(log1p(raw margins))/1.349,0.05) on dense calibration trajectories; frozen',
        selection=('Among attained arms with screen correct count >= dense minus one, '
                   'select shortest mean canvas steps, then highest accuracy as the second finalist. '
                   'Tie-break on cap rate and steps. If none meet the accuracy guard, '
                   'take shortest and most accurate as exploratory finalists.'),
        desired_steps='within 10% of matched dense canvas steps; report actual gap even if missed',
        timing='30 prompts seed42; dense, archived Gaussian32 s50, and two finalists; diagnostics/counters off; per-condition warmup excluded',
        exposure='Calibration and screen overlap the already exposed final AIME26; no held-out claim',
        source_hashes={str(p.resolve()): sha(p) for p in sources})
    root.mkdir(parents=True, exist_ok=True)
    _freeze(root/'protocol.json', protocol)
    _freeze(root/'final_manifest.json', final)
    _freeze(root/'calibration_manifest.json', calibration)
    return protocol, final, calibration


def model_config():
    cfg = json.loads((PRIOR/'configuration.json').read_text())
    return {key: cfg[key] for key in ('model', 'revision', 'library',
        'torch_library', 'm_ref', 'beta', 'gamma', 'rank', 'projection_seed',
        'physical_tile', 'canvas', 'max_steps')}


class MarginCollector(State):
    def observe_logits(self, logits, accepted, cur_step):
        super().observe_logits(logits, accepted, cur_step)
        # Include zero margins; do not infer distributions from quantiles.
        self.margin_samples.extend(self.margin.flatten().detach().cpu().tolist())


def normalization(adapter, root, cfg, calibration):
    path = root/'normalization.json'
    if path.exists():
        return json.loads(path.read_text())
    values = []
    for row in calibration:
        dest = root/'normalization_samples'/f"{row['source_id']}.json"
        if dest.exists():
            data = json.loads(dest.read_text())
        else:
            state = MarginCollector('native_dense', None, m_ref=cfg['m_ref'],
                                    diagnostics=True, seed=int(row['seed']))
            with observe(adapter.model, state):
                output = adapter.generate(base._request(row))
            data = dict(id=row['id'], seed=row['seed'], margins=state.margin_samples,
                        score=base.aime_score(row, output.text),
                        metadata=output.metadata)
            atomic(dest, data)
        values.extend(data['margins'])
    q = np.quantile(np.log1p(values), [.25, .5, .75])
    result = dict(sigma=max(float((q[2]-q[0])/1.349), .05),
                  log_margin_quantiles=q.tolist(), samples=len(values),
                  calibration_ids=[r['id'] for r in calibration],
                  m_ref=cfg['m_ref'], frozen=True,
                  formula='clip((log1p(m_ref)-log1p(margin))/sigma,0,1)')
    atomic(path, result)
    print(json.dumps(dict(event='normalization_complete', **result)), flush=True)
    return result


def arm_config(root, protocol, arm, scale, final, calibration):
    reference = {'M_gate':'M_prior', 'M_gate_norm':'M_prior',
                 'C_gate':'C_prior', 'T_gate':'T_prior'}[arm['method']]
    initial = json.loads((PRIOR/'thresholds'/f'{reference}_s50.json').read_text())['policy']
    cfg = dict(model_config(), schema='aime26_gated_arm_v2', arm=arm,
        methods=[arm['method']], targets=[50], seeds=list(FINAL_SEEDS),
        trajectory_gamma=.5, smooth_trajectory_gamma=.8,
        gate_trajectory_gamma=.65, gate_tau=arm['tau'], margin_scale=scale,
        initial_policy=initial, protocol_fingerprint=fingerprint(protocol),
        source_hashes=protocol['source_hashes'])
    cfg['fingerprint'] = fingerprint(cfg)
    directory = root/'arms'/arm['name']
    _freeze(directory/'configuration.json', cfg)
    _freeze(directory/'final_manifest.json', final)
    _freeze(directory/'calibration_manifest.json', calibration)
    return directory, cfg


def condition(method, seed):
    return f'dense_seed{seed}' if method == 'dense' else f'{method}_s50_seed{seed}'


def archived(method, ids, seeds):
    output = []
    for seed in seeds:
        output.extend(json.loads(p.read_text()) for p in
            (CONTROLS/'final'/condition(method, seed)).glob('*.json'))
    return [r for r in output if r['id'] in ids]


def metrics(rows, seeds):
    value = summarize(rows)
    accuracies = [statistics.mean(r['score'] for r in rows if r['seed'] == seed)
                  for seed in seeds]
    value.update(accuracy_std=statistics.stdev(accuracies) if len(seeds)>1 else 0.,
        cap_rate=value['cap_canvases']/value['total_canvases'],
        mean_question_steps=statistics.mean(r['steps'] for r in rows),
        mean_question_seconds_instrumented=statistics.mean(r['seconds'] for r in rows),
        disagreement_questions=sum(r['seed_disagreement'] for r in
            question_disagreements('metrics', rows, seeds)))
    value.update({f'accuracy_seed{seed}': acc for seed, acc in zip(seeds, accuracies)})
    return value


def select_finalists(results, dense):
    eligible = [r for r in results if r['correct'] >= dense['correct']-1]
    pool = eligible or results
    fast = min(pool, key=lambda r: (r['mean_canvas_steps'], r['cap_rate'], -r['accuracy']))
    remaining = [r for r in pool if r['arm'] != fast['arm']]
    if not remaining:
        remaining = [r for r in results if r['arm'] != fast['arm']]
    second = max(remaining, key=lambda r: (r['accuracy'], -r['mean_canvas_steps'], -r['cap_rate']))
    return dict(arms=[fast['arm'], second['arm']], accuracy_guard_passed=bool(eligible),
                dense_screen=dense, results=results,
                rationale='shortest trajectory within accuracy guard, plus accuracy leader')


def load_stage(directory, stage, method, seeds, ids):
    rows = []
    for seed in seeds:
        subset = [json.loads(p.read_text()) for p in
                  (directory/stage/condition(method, seed)).glob('*.json')]
        if {r['id'] for r in subset} != ids or len(subset) != len(ids):
            raise ValueError(f'incomplete {directory.name} {stage} seed{seed}')
        rows.extend(subset)
    return rows


def validate_rows(rows, policy, beta=3.):
    assert policy['call1'] == policy['call2'] == policy['late']
    for row in rows:
        assert row['policy'] == policy
        first = [s for s in row['step_records'] if s['iteration'] == 1]
        assert len(first) == row['total_canvases']
        assert all(abs(s['sensitivity_mean']-(1+beta)) < 1e-6 for s in first)
        assert all(s['threshold_phase'] == 'uniform' for s in row['step_records'])


def run(root):
    protocol, final, calibration = prepare(root)
    if not base.torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    cfg = model_config()
    adapter = base._adapter(cfg)
    projections = base.Projections()
    scale = normalization(adapter, root, cfg, calibration)['sigma']
    configs, policies = {}, {}
    for arm in arms():
        directory, arm_cfg = arm_config(root, protocol, arm, scale, final, calibration)
        configs[arm['name']] = (directory, arm_cfg)
        threshold = uniform._calibrate_one(adapter, directory, arm_cfg, calibration,
                                           projections, arm['method'], 50)
        if threshold['status'] == 'attained':
            policies[arm['name']] = threshold['policy']
        print(json.dumps(dict(event='arm_calibrated', arm=arm['name'],
                              status=threshold['status'])), flush=True)
    atomic(root/'calibration_complete.json', dict(attained=list(policies),
             total=len(arms()), finished=time.time()))
    screen = [r for r in final if r['id'] in set(protocol['screen_ids'])]
    results = []
    for arm in arms():
        if arm['name'] not in policies:
            continue
        directory, arm_cfg = configs[arm['name']]
        rows = []
        for seed in SCREEN_SEEDS:
            for source in screen:
                row = dict(source, seed=seed)
                rows.append(base._cached(adapter, directory, 'screen',
                    condition(arm['method'], seed), row, arm['method'],
                    policies[arm['name']], arm_cfg, projections))
        validate_rows(rows, policies[arm['name']])
        results.append(dict(arm=arm['name'], method=arm['method'], tau=arm['tau'],
                            **metrics(rows, SCREEN_SEEDS)))
    _csv(root/'screen_summary.csv', results)
    dense = metrics(archived('dense', set(protocol['screen_ids']), SCREEN_SEEDS), SCREEN_SEEDS)
    selection = select_finalists(results, dense)
    _freeze(root/'selection.json', selection)
    print(json.dumps(dict(event='selected', arms=selection['arms'],
                          accuracy_guard_passed=selection['accuracy_guard_passed'])), flush=True)
    for arm in arms():
        if arm['name'] not in selection['arms']:
            continue
        directory, arm_cfg = configs[arm['name']]
        policy = policies[arm['name']]
        for seed in FINAL_SEEDS:
            cond = condition(arm['method'], seed)
            for source in final:
                row = dict(source, seed=seed)
                prior = base._shard(directory, 'screen', cond, row)
                dest = base._shard(directory, 'final', cond, row)
                if prior.exists() and not dest.exists():
                    out = json.loads(prior.read_text())
                    expected = fingerprint([arm_cfg['fingerprint'], 'screen', cond,
                        arm['method'], policy, row['id'], row['prompt_hash'], seed])
                    if out['identity'] != expected:
                        raise ValueError(f'invalid screen identity {prior}')
                    out['reused_from'] = str(prior)
                    out['identity'] = fingerprint([arm_cfg['fingerprint'], 'final', cond,
                        arm['method'], policy, row['id'], row['prompt_hash'], seed])
                    atomic(dest, out)
                base._cached(adapter, directory, 'final', cond, row,
                             arm['method'], policy, arm_cfg, projections)
        validate_rows(load_stage(directory, 'final', arm['method'], FINAL_SEEDS,
                                {r['id'] for r in final}), policy)
        atomic(directory/'final_complete.json', dict(passed=True, seeds=list(FINAL_SEEDS),
                 methods=[arm['method']], targets=[50], finished=time.time()))
    atomic(root/'final_complete.json', dict(passed=True, selected=selection['arms'],
                                            finished=time.time()))
    # Measure actual request latency with routing counters and diagnostic
    # entropy/quantile collection disabled. Warm up each condition separately.
    timing_conditions = [('dense', 'dense', cfg, None),
        ('gaussian32', 'gaussian32', cfg, json.loads(
            (CONTROLS/'thresholds/gaussian32_s50.json').read_text())['policy'])]
    for name in selection['arms']:
        _, arm_cfg = configs[name]
        timing_conditions.append((name, arm_cfg['arm']['method'], arm_cfg, policies[name]))
    for name, method, timing_cfg, policy in timing_conditions:
        dest_dir = root/'timing'/name
        if len(list(dest_dir.glob('*.json'))) == len(final):
            continue
        base._run_one(adapter, dict(final[0], seed=42), method, policy,
                      timing_cfg, projections, diagnostics=False)
        for source in final:
            dest = dest_dir/f"{source['source_id']}.json"
            if dest.exists():
                continue
            out = base._run_one(adapter, dict(source, seed=42), method, policy,
                                timing_cfg, projections, diagnostics=False)
            out['timing_condition'] = name
            atomic(dest, out)
        print(json.dumps(dict(event='timing_complete', condition=name)), flush=True)
    atomic(root/'timing_complete.json', dict(passed=True, finished=time.time()))
    report(root)


def report(root):
    protocol = json.loads((root/'protocol.json').read_text())
    final = json.loads((root/'final_manifest.json').read_text())
    ids = {r['id'] for r in final}
    selection = json.loads((root/'selection.json').read_text())
    pooled, per_seed, disagreement, calibrations = [], [], [], []
    for name in ['dense', 'gaussian32', 'temporal'] + selection['arms']:
        if name in ('dense', 'gaussian32', 'temporal'):
            rows = archived(name, ids, FINAL_SEEDS)
        else:
            directory = root/'arms'/name
            cfg = json.loads((directory/'configuration.json').read_text())
            method = cfg['arm']['method']
            rows = load_stage(directory, 'final', method, FINAL_SEEDS, ids)
            data = json.loads((directory/'thresholds'/f'{method}_s50.json').read_text())
            validate_rows(rows, data['policy'])
        pooled.append(dict(arm=name, **metrics(rows, FINAL_SEEDS)))
        for seed in FINAL_SEEDS:
            per_seed.append(dict(arm=name, seed=seed,
                                 **summarize([r for r in rows if r['seed']==seed])))
        disagreement.extend(question_disagreements(name, rows, FINAL_SEEDS))
    for arm in arms():
        path = root/'arms'/arm['name']/'thresholds'/f"{arm['method']}_s50.json"
        if path.exists():
            data = json.loads(path.read_text())
            pair = data['policy']['late']
            calibrations.append(dict(arm=arm['name'], status=data['status'],
                local_log_threshold=pair['local']['log_threshold'],
                global_log_threshold=pair['global']['log_threshold'],
                max_error=data['max_error']))
    _csv(root/'pooled_summary.csv', pooled)
    _csv(root/'per_seed_summary.csv', per_seed)
    _csv(root/'per_question_disagreement.csv', disagreement)
    _csv(root/'calibration_summary.csv', calibrations)
    timing = []
    for directory in sorted((root/'timing').iterdir()):
        rows = [json.loads(p.read_text()) for p in directory.glob('*.json')]
        if len(rows) != 30:
            raise ValueError(f'incomplete timing: {directory}')
        token_counts = [len(r['completion_tokens']) if isinstance(
            r.get('completion_tokens'), list) else int(r.get('completion_tokens', 0))
                        for r in rows]
        timing.append(dict(arm=directory.name, n=len(rows),
            mean_seconds=statistics.mean(r['seconds'] for r in rows),
            median_seconds=statistics.median(r['seconds'] for r in rows),
            p90_seconds=float(np.quantile([r['seconds'] for r in rows], .9)),
            mean_question_steps=statistics.mean(r['steps'] for r in rows),
            accuracy=statistics.mean(r['score'] for r in rows),
            completion_tokens=sum(token_counts),
            tokens_per_second=sum(token_counts)/sum(r['seconds'] for r in rows)))
    _csv(root/'timing_summary.csv', timing)
    lines = ['# AIME26 gated sensitivity: accuracy and generation cost', '',
        'All gates use the previous completed acceptance/flip history. Each arm has one frozen local/global threshold pair across all calls. '
        'Margin normalization is fitted once on six dense calibration trajectories. Calibration IDs: 2, 8, 14, 20, 23, 30.', '',
        f"Screen IDs: {list(SCREEN_IDS)}, seeds 42/43; 12 arms. Selected: {selection['arms']}. "
        "Screen and calibration prompts overlap the already exposed final 30 prompts: development evidence only.", '',
        'Dense accuracy is a reference, not an upper bound. The selection favors short trajectories while guarding accuracy; '
        'the desired step count is within 10% of dense. SD is sample SD across seeds.', '',
        '| Arm | Accuracy seeds 42 / 43 / 44 | Pooled ± SD | O/G/L sparsity | Mean / median / P90 calls per canvas | Caps / canvases | Disagreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in pooled:
        acc = ' / '.join(f"{100*r[f'accuracy_seed{s}']:.2f}" for s in FINAL_SEEDS)
        sparsity = ' / '.join(f"{100*r[f'{k}_sparsity']:.2f}" for k in ('overall','global','local'))
        calls = ' / '.join(f"{r[f'{k}_canvas_steps']:.2f}" for k in ('mean','median','p90'))
        lines.append(f"| {r['arm']} | {acc}% | {100*r['accuracy']:.2f}% ± {100*r['accuracy_std']:.2f} pp | {sparsity}% | {calls} | {r['cap_canvases']} / {r['total_canvases']} | {r['disagreement_questions']}/30 |")
    lines += ['', '## Phase sparsity (O/G/L)', '',
              '| Arm | Call 1 | Call 2 | Later |', '|---|---:|---:|---:|']
    for r in pooled:
        parts = [' / '.join(f"{100*r[f'{p}_{k}_sparsity']:.2f}" for k in ('whole','global','local'))
                 for p in ('call1','call2','late')]
        lines.append(f"| {r['arm']} | " + '% | '.join(parts) + '% |')
    lines += ['', '## End-to-end request timing', '',
        '30 prompts, seed 42, diagnostics and routing counters disabled; per-condition warmup excluded. '
        'Includes prefill and generation, excludes model loading. Output lengths can differ. These are fresh timings; '
        'archived instrumented latencies are not used as a speed baseline.', '',
        '| Arm | Mean / median / P90 seconds | Mean calls/question | Accuracy | Output tokens/s |',
        '|---|---:|---:|---:|---:|']
    for r in timing:
        lines.append(f"| {r['arm']} | {r['mean_seconds']:.3f} / {r['median_seconds']:.3f} / {r['p90_seconds']:.3f} | {r['mean_question_steps']:.2f} | {100*r['accuracy']:.2f}% | {r['tokens_per_second']:.2f} |")
    lines += ['', '## Calibration', '', '| Arm | Status | Local log τ | Global log τ | Max error pp |',
              '|---|---|---:|---:|---:|']
    for r in calibrations:
        lines.append(f"| {r['arm']} | {r['status']} | {r['local_log_threshold']:.6f} | {r['global_log_threshold']:.6f} | {100*r['max_error']:.2f} |")
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','run','report'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.root)
    elif args.stage == 'run': run(args.root)
    else: report(args.root)


if __name__ == '__main__':
    main()
