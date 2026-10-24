"""Matched official accuracy, native trajectory, work and clean timing readout."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / 'analysis001'
    out.mkdir(exist_ok=False)
    protocol = json.loads((root / 'e2e_protocol.json').read_text())
    scores = json.loads((root / 'scoring001/official_scores.json').read_text())
    outcomes = {(x['arm'], x['id'], x['seed']): x for x in scores['outcomes']}
    records = defaultdict(dict)
    summaries = defaultdict(dict)
    inputs = {'scoring001/official_scores.json': sha(root / 'scoring001/official_scores.json')}
    for suite in ['short', 'long']:
        for arm in protocol['arms']:
            bindings=json.loads((root/'result_directory_bindings.json').read_text())
            leaf = root / bindings['result_paths'].get(suite,{}).get(arm,'runs/panel/'+suite+'/'+arm)
            marker = leaf / 'final_complete.json'
            if not marker.exists():
                continue
            complete = json.loads(marker.read_text())
            if not complete['passed'] or complete['cells'] != len(protocol['panel'][suite]):
                raise RuntimeError('incomplete arm')
            for p in leaf.glob('*.json'):
                inputs[str(p.relative_to(root))] = sha(p)
            for p in sorted(leaf.glob('clean[0-9][0-9][0-9].json')):
                clean = json.loads(p.read_text())
                audit = json.loads((leaf / p.name.replace('clean', 'audit')).read_text())
                if any(clean[k] != audit[k] for k in ['output_sha256', 'N', 'C', 'T']):
                    raise RuntimeError('audit/clean mismatch')
                score = outcomes[arm, clean['id'], clean['seed']]
                native_canvas_steps=[]; pending=0
                for event in audit['native_trace']:
                    if event['commit']:
                        native_canvas_steps.append(pending);pending=0
                    else: pending+=1
                assert len(native_canvas_steps)==clean['C'] and sum(native_canvas_steps)+pending==clean['N']
                clean['actual_native_canvas_steps']=native_canvas_steps
                clean['native_uncommitted_tail_calls']=pending
                records[arm][clean['id'], clean['seed']] = dict(clean=clean, audit=audit, score=score)
    for arm, cells in records.items():
        by_task = defaultdict(list)
        for data in cells.values():
            by_task[data['clean']['benchmark']].append(data)
        for task, selected in by_task.items():
            N = sum(x['clean']['N'] for x in selected)
            C = sum(x['clean']['C'] for x in selected)
            work = [x['audit']['work'] for x in selected]
            global_eligible = sum(x['global_eligible_tiles'] for x in work)
            global_kept = sum(x['global_kept_tiles'] for x in work)
            local = sum(x['native_local_eligible_rectangles'] for x in work)
            selector = [x['audit']['selector'] for x in selected if x['audit']['selector']]
            inherited = [x['audit']['work']['adapter_receipt'] for x in selected if 'adapter_receipt' in x['audit']['work']]
            def phase_count(new_key, inherited_key):
                if selector: return sum(x[new_key] for x in selector)
                if inherited: return sum(x[inherited_key] for x in inherited) if all(inherited_key in x for x in inherited) else None
                return None
            native_trace = [c for x in selected for c in x['audit']['native_trace'] if not c['commit']]
            per_item = defaultdict(list)
            for x in selected:
                per_item[x['clean']['id']].append(x['score']['correct'])
            summaries[task][arm] = dict(cells=len(selected), items=len(per_item),
                correct=sum(x['score']['correct'] for x in selected), accuracy=statistics.mean(statistics.mean(v) for v in per_item.values()),
                capped=sum(x['clean']['finish_reason'] == 'length' for x in selected),
                no_final_response=sum(x['score']['no_final_response'] for x in selected),
                N=N, C=C, N_per_C=N/C, T=sum(x['clean']['T'] for x in selected),
                mean_N=statistics.mean(x['clean']['N'] for x in selected),
                mean_S=statistics.mean(x['clean']['S'] for x in selected),
                mean_W=statistics.mean(x['clean']['W'] for x in selected),
                S_per_N=sum(x['clean']['S'] for x in selected)/N,
                clean_timing_valid_cells=sum(x['clean']['timing_valid'] for x in selected),
                global_eligible=global_eligible, global_kept=global_kept,
                global_skipped=global_eligible-global_kept, global_sparsity=1-global_kept/global_eligible,
                local_eligible=local, local_kept=local, local_skipped=sum(x['native_local_skipped_tiles'] for x in work),
                overall_sparsity=(global_eligible-global_kept)/(global_eligible+local),
                phase_count_unit='GLOBAL attention invocations across5layers' if inherited else 'layer29 attention invocations' if selector else None,
                selector_refresh_calls=phase_count('selector_refresh_calls','optimized_refresh_calls'),
                held_mask_calls=phase_count('held_mask_calls','resident_sparse_held_calls'),
                carried_mask_calls=phase_count('carried_mask_calls','resident_sparse_carried_calls'),
                certificate_failures=sum(x['certificate_failures'] for x in selector) if selector else None,
                fallback_units=sum(x['fallback_units'] for x in selector) if selector else None,
                native_cap_calls=sum(c['cap_reached'] for c in native_trace),
                native_confidence_limited_calls=sum(not c['confidence'] for c in native_trace),
                actual_native_canvas_steps=[dict(id=x['clean']['id'], seed=x['clean']['seed'],
                    committed_canvas_calls=x['clean']['actual_native_canvas_steps'],
                    uncommitted_tail_calls=x['clean']['native_uncommitted_tail_calls']) for x in selected],
                native_uncommitted_tail_calls=sum(x['clean']['native_uncommitted_tail_calls'] for x in selected),
                committed_denoise_calls=N-sum(x['clean']['native_uncommitted_tail_calls'] for x in selected),
                committed_calls_per_canvas=(N-sum(x['clean']['native_uncommitted_tail_calls'] for x in selected))/C,
                emission_metadata_intervals=[dict(id=x['clean']['id'], seed=x['clean']['seed'],
                    **x['clean']['native_interval_steps'], caveat='CPU emission intervals may lag native commits; use actual native canvas steps for canvas claims') for x in selected])
    # Bootstrap paired item clusters, keeping the two seeds together.
    paired = defaultdict(dict)
    rng = np.random.default_rng(20261024)
    for task in summaries:
        ids = sorted({i for arm in records for (i, s), x in records[arm].items() if x['clean']['benchmark'] == task})
        draws = rng.integers(0, len(ids), size=(10000, len(ids)))
        for reference in ['dense_full_fix51994', 'dense_piecewise', 'allkept_layer29', 'v31_optimized', 'current_error_fixed4', 'sham_fixed4']:
            for arm in protocol['arms']:
                common = sorted(set(records[arm]) & set(records[reference]))
                common = [k for k in common if records[arm][k]['clean']['benchmark'] == task]
                if len(common) != len(ids)*2:
                    continue
                quality, motion = [], defaultdict(list)
                changes = dict(reference_correct_candidate_wrong=0, reference_wrong_candidate_correct=0)
                for k in common:
                    a, b = records[arm][k]['score']['correct'], records[reference][k]['score']['correct']
                    changes['reference_correct_candidate_wrong'] += b and not a
                    changes['reference_wrong_candidate_correct'] += a and not b
                for i in ids:
                    keys = [k for k in common if k[0] == i]
                    quality.append(statistics.mean(float(records[arm][k]['score']['correct'])-float(records[reference][k]['score']['correct']) for k in keys))
                    for field in ['N', 'W', 'S']:
                        motion[field].append(statistics.mean(math.log(records[arm][k]['clean'][field]/records[reference][k]['clean'][field]) for k in keys))
                delta = np.array(quality)
                row = dict(clusters=len(ids), cells=len(common), accuracy_difference_pp=float(delta.mean()*100),
                    accuracy_ci95_pp=np.percentile(delta[draws].mean(1)*100,[2.5,97.5]).tolist(), **changes)
                for field, values in motion.items():
                    v = np.array(values)
                    row[field] = dict(geometric_ratio=float(np.exp(v.mean())),
                        ci95=np.exp(np.percentile(v[draws].mean(1),[2.5,97.5])).tolist())
                row['timing_ratios_qualified'] = all(records[a][k]['clean']['timing_valid'] for a in [arm,reference] for k in common)
                paired[task][arm+'__vs__'+reference] = row
    result = dict(protocol_sha256=sha(root/'e2e_protocol.json'), summary=dict(summaries),
        paired=dict(paired), official_records=len(outcomes), method_promoted=False,
        limits=['Exposed descriptive cohort; screening, not formal noninferiority.',
                'V31 scope/budget differs from layer29 methods; report physical work.',
                'Different native trajectories change future support feasibility and physical work.',
                'No data-driven refinement inside the frozen attempt.'])
    for name, value in [('summary.json', result), ('input_hashes.json', inputs)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.environ.setdefault('MPLCONFIGDIR',str(root/'cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for axis, task in zip(axes, ['aime26','humaneval','longbench_v2']):
        entries=summaries.get(task,{})
        for arm, data in entries.items():
            axis.scatter(data['mean_N'],data['accuracy']*100,label=arm)
        axis.set_title(task);axis.set_xlabel('Mean complete-request denoising calls N');axis.set_ylabel('Official task accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=7)
    fig.tight_layout(rect=(0,.15,1,1));fig.savefig(out/'accuracy_trajectory.png',dpi=180);fig.savefig(out/'accuracy_trajectory.pdf')
    plt.close(fig)
    (out/'analysis_complete.json').write_text(json.dumps(dict(passed=True,
        summary_sha256=sha(out/'summary.json'),analysis_source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root/'e2e_protocol.json')),indent=2)+'\n')
    print(json.dumps({'tasks':list(summaries),'scored_records':len(outcomes)}))


if __name__ == '__main__':
    main()
