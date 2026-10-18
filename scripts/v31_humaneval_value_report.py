"""Post-generation, CPU-only reporting; keep audit and clean populations separate."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import numpy as np
from scripts import v31_humaneval_value_study as study


def values(root, mode, pass_name, label):
    paths = study.selected_paths(root, mode, pass_name)[label]
    rows = [r for p in paths for r in study.base.jsonl(p/'records.jsonl')]
    return {r['index']: r for r in rows}


def make_report(root, dest):
    if dest.exists():
        raise RuntimeError('use a new analysis attempt')
    dest.mkdir(parents=True)
    original = study.read(root/'summary.json')
    out = dict(status='complete', source_commit=study.read(root/'protocol.json')['source_commit'],
               protocol_sha256=study.sha(root/'protocol.json'), modes={},
               populations='clean accuracy/latency and audit accuracy/work are separate native populations',
               work_calibration_transfer_to_clean='not established; clean mask work was intentionally not instrumented',
               trajectory_drift_cause='not localized; differences associated with instrumentation and/or engine execution history',
               native_settings=study.read(root/'protocol.json')['native'],
               gpu_engine_reservation_s=original['gpu_reservation_s'])
    lines = ['# HumanEval164, seed42, K256: verified results', '',
        'All164 tasks, one measured generation per task and arm, thinking on/off, native output limit8192. '
        'K256 is four selectable complete prefix KV64 blocks per query-head/Q128 tile; canvas/boundary tiles stay dense. '
        'V2 uses a calibrated variable-budget threshold. LOCAL remains dense.', '',
        '**Interpretation:** clean accuracy and request time come from the same generation records. '
        'Audit accuracy and physical work are reported separately. Audit and clean output hashes differ substantially, '
        'including dense controls; the cause has not been localized. First-canvas threshold matching was established '
        'on the audit population and is not a verified clean-population work match.', '',
        'V2/V3a use existing Gaussian32 statistics on all five GLOBAL layers. Full-value swaps and direction-guarded '
        'residual use full V only at GLOBAL layer29, with V31 selection elsewhere. Residual explicitly modifies held '
        'attention output; the other value methods change support only. The inherited V31 lineage is Yuhan\'s.', '']
    for mode in study.MODES:
        info = dict(passes={}, bookends={}, overhead={}, first_canvas=original['modes'][mode]['threshold'])
        raw = {}
        for pass_name in ('clean', 'audit'):
            score = study.read(root/f'private/scoring/{mode}/{pass_name}.json')['scores']
            raw[pass_name] = {}
            info['passes'][pass_name] = {}
            for label in study.LABELS:
                by_index = values(root, mode, pass_name, label)
                raw[pass_name][label] = by_index
                rows = list(by_index.values())
                n = np.array([r['denoise_forwards'] for r in rows])
                c = np.array([r['canvases'] for r in rows])
                m = dict(questions=164, correct=sum(x['passed'] for x in score[label].values()),
                    accuracy=sum(x['passed'] for x in score[label].values())/164,
                    N_mean=float(n.mean()), N_median=float(np.median(n)), N_p90=float(np.quantile(n,.9)),
                    N_p95=float(np.quantile(n,.95)), N_max=int(n.max()), C_mean=float(c.mean()),
                    N_per_C=float(n.sum()/c.sum()), T_mean=statistics.mean(r['output_tokens'] for r in rows),
                    generation_caps=sum(x['finish']=='capped' for x in score[label].values()),
                    global_eligible_tiles=None, global_kept_tiles=None,
                    global_sparsity=None, local_sparsity=None, overall_sparsity=None,
                    S_mean_s=None, W_mean_s=None, amortized_S_per_N_ms=None)
                if pass_name == 'clean':
                    m.update(S_mean_s=statistics.mean(r['decode_s'] for r in rows),
                        W_mean_s=statistics.mean(r['wall_s'] for r in rows),
                        W_median_s=statistics.median(r['wall_s'] for r in rows),
                        W_p90_s=float(np.quantile([r['wall_s'] for r in rows],.9)),
                        amortized_S_per_N_ms=1000*sum(r['decode_s'] for r in rows)/int(n.sum()))
                else:
                    a = [r['receipts']['adapter'] for r in rows]
                    e = sum(x['global_eligible_tiles'] for x in a)
                    k = sum(x['global_kept_tiles'] for x in a)
                    local = sum(x['native_local_eligible_rectangles'] for x in a)
                    m.update(global_eligible_tiles=e, global_kept_tiles=k,
                        local_eligible_rectangles=local, global_sparsity=1-k/e,
                        local_sparsity=sum(x['native_local_skipped_tiles'] for x in a)/local,
                        overall_sparsity=(e-k)/(e+local))
                    trace = [r['receipts'].get('trace') for r in rows]
                    available = all(t and 'canvas_steps' in t for t in trace)
                    if available:
                        assert all(sum(r['receipts']['trace']['canvas_steps']) == r['denoise_forwards'] for r in rows)
                    m.update(native_interval_trace_available=available,
                        native_intervals_at_48_call_cap=(sum(v>=48 for t in trace for v in t['canvas_steps']) if available else None))
                info['passes'][pass_name][label] = m
            before, after = raw[pass_name]['control_before'], raw[pass_name]['control_after']
            info['bookends'][pass_name] = dict(hash_differences=sum(before[i]['output_hash'] != after[i]['output_hash'] for i in range(164)),
                N_differences=sum(before[i]['denoise_forwards'] != after[i]['denoise_forwards'] for i in range(164)),
                max_absolute_N_difference=max(abs(before[i]['denoise_forwards']-after[i]['denoise_forwards']) for i in range(164)),
                score_differences=sum(score['control_before'][str(i)]['passed'] != score['control_after'][str(i)]['passed'] for i in range(164)))
        clean = info['passes']['clean']
        v31 = clean['control_before']
        dense = clean['dense_full_fix51994']
        for label in study.LABELS:
            m = clean[label]
            m.update(W_speedup_vs_v31=v31['W_mean_s']/m['W_mean_s'], W_speedup_vs_dense_full=dense['W_mean_s']/m['W_mean_s'],
                W_change_vs_v31_pct=100*(m['W_mean_s']/v31['W_mean_s']-1),
                N_change_vs_v31_pct=100*(m['N_mean']/v31['N_mean']-1))
        for label in study.CANDIDATES:
            original_m = original['modes'][mode]['methods'][label]
            timing = original_m['audit_selector_times_ms']
            work = original_m['refinement_work']
            overhead = dict(audit_envelopes_ms_per_problem={k:v/164 for k,v in timing.items()},
                interpretation='instrumented CUDA module envelopes; include launch gaps and may include base observation; not clean marginal prices',
                work=work)
            if work.get('anchor_events'):
                overhead.update(anchors_per_problem=work['anchor_events']/164,
                    anchor_envelope_ms_per_event=timing['anchor_module_cuda_elapsed_ms']/work['anchor_events'])
            info['overhead'][label] = overhead
        info['clean_vs_audit'] = {label:dict(hash_differences=original['modes'][mode]['methods'][label]['clean_audit_hash_differences'],
            N_differences=original['modes'][mode]['methods'][label]['clean_audit_N_differences'],
            score_differences=original['modes'][mode]['methods'][label]['clean_audit_score_differences']) for label in study.LABELS}
        info['paired_clean_vs_v31'] = original['modes'][mode]['paired_vs_v31']
        out['modes'][mode] = info
        lines += [f'## {mode}: clean generation', '',
            '| Method | Pass@1 | N mean / median / P90 / max | C | N/C | T | S (s) | S/N (ms) | W (s) | Output caps |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in study.LABELS:
            m = clean[label]
            lines.append(f"| {label} | {m['correct']}/164 ({100*m['accuracy']:.2f}%) | {m['N_mean']:.3f} / {m['N_median']:.1f} / {m['N_p90']:.1f} / {m['N_max']} | {m['C_mean']:.3f} | {m['N_per_C']:.3f} | {m['T_mean']:.2f} | {m['S_mean_s']:.5f} | {m['amortized_S_per_N_ms']:.3f} | {m['W_mean_s']:.5f} | {m['generation_caps']} |")
        lines += ['', f'## {mode}: separate audit generation', '',
            '| Method | Pass@1 | N | C | N/C | T | GLOBAL sparsity | LOCAL sparsity | Overall sparsity | Output caps |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in study.LABELS:
            m = info['passes']['audit'][label]
            lines.append(f"| {label} | {m['correct']}/164 ({100*m['accuracy']:.2f}%) | {m['N_mean']:.3f} | {m['C_mean']:.3f} | {m['N_per_C']:.3f} | {m['T_mean']:.2f} | {100*m['global_sparsity']:.3f}% | {100*m['local_sparsity']:.1f}% | {100*m['overall_sparsity']:.3f}% | {m['generation_caps']} |")
        t = info['first_canvas']
        lines += ['', f"V2 threshold {t['threshold']:.12g}; audit first-canvas matching: {t['matching']}.", '',
            'Clean versus audit hash/score disagreements:', '', '| Method | Output hash differences | Correctness differences |', '|---|---:|---:|']
        for label, m in info['clean_vs_audit'].items():
            lines.append(f"| {label} | {m['hash_differences']}/164 | {m['score_differences']}/164 |")
        lines += ['', f"V31 technical bookends: {info['bookends']}.", '',
            'Paired20,000-draw question-bootstrap95% intervals (single seed, conditional on observed engines):', '',
            '| Method | Accuracy delta vs V31 (pp) [CI] | Request speedup vs V31 [CI] |', '|---|---:|---:|']
        for label in study.CANDIDATES:
            m = info['paired_clean_vs_v31'][label]
            lines.append(f"| {label} | {m['accuracy_delta_pp']:+.3f} {m['accuracy_delta_ci95']} | {m['W_speedup']:.4f} {m['W_speedup_ci95']} |")
        lines += ['', 'Audit selector/correction envelopes:', '', '| Method | Recorded ms/problem | Extra full-value anchors/problem |', '|---|---:|---:|']
        for label, m in info['overhead'].items():
            lines.append(f"| {label} | {m['audit_envelopes_ms_per_problem']} | {m.get('anchors_per_problem', 'not applicable')} |")
        lines.append('')
    lines += ['## Findings and boundaries', '',
        '- V3a is the best clean point candidate among the value methods: thinking on +1 correct and 1.7% lower W versus V31; thinking off -1 correct and 5.7% lower W. Both request-speedup intervals include1. No accuracy improvement is established.',
        '- All requested sparse methods are slower than dense FULL in the clean runs. PIECEWISE dense is faster still without thinking.',
        '- Full-value swaps spend about745ms/problem (thinking) and72ms/problem (non-thinking) in the audit anchor envelope. Directional residual spends about143ms/12ms. These are diagnostic envelopes, not clean subtraction estimates.',
        '- First-canvas calibration does not equalize the full trajectory. Audit V2 GLOBAL sparsity is29.98% versus V31 48.17% with thinking, and8.37% versus14.24% without thinking.',
        '- Audit/clean trajectory differences also occur in dense controls. Quality rankings differ across passes. Do not attach audit masks/work to clean outputs as if they were identical, infer a clean work match, or claim robust superiority from these single-seed point estimates.',
        '- Overall sparsity uses actual denoising GLOBAL full-support plus LOCAL window-intersecting Q128/KV64 rectangles. Prefill and commits remain dense and are outside this sparsity denominator. This is a common geometry accounting unit, not CUDA CTA count.',
        '- C follows the inherited bench definition ceil(output_tokens/256). N is sampler_calls minus C and is independently checked against GLOBAL-call scope and available denoising-interval trace sums. S/N is amortized over native generation, not a controlled forward price.',
        '- All planned records, source/input bindings and sampler geometry checks pass. There are58 completed engine jobs, zero failed jobs, zero measured CUDA captures, zero LOCAL skips, and zero fixed-K budget errors. Older LongBench campaigns remain paused.', '']
    study.write(dest/'summary.json', out, True)
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    study.write(dest/'final_complete.json', dict(status='complete_cpu_analysis',
        files_sha256={n:study.sha(dest/n) for n in ['summary.json','summary.md']},
        native_completion_sha256=study.sha(root/'final_complete.json'),
        analysis_source_sha256=study.sha(Path(__file__))), True)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--study', type=Path, default=study.DEFAULT)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    root = a.study.resolve()
    make_report(root, a.out or root/'analysis/attempt001')
