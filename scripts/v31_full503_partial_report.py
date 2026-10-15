"""Official, resumable CPU analysis of the user-paused first two shards.

This never launches generation or changes the frozen native protocol. Each
scoring attempt is new; only validated completed scores are reusable.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


LABELS = {
    'dense_full_fix51994': 'Dense FULL (fixed)',
    'dense_piecewise': 'Dense PIECEWISE',
    'allkept_fa4': 'All-kept FA4',
    'v31_progress_dense_control': 'V31 progress control',
    'v31_no_progress_dense_hold': 'No-progress hold',
    'v31_every_step_dense_select_sparse_output': 'Fresh every step',
    'raw_global_progress_dense_s2': 'Raw / progress / 2',
    'stable_global_progress_dense_s2': 'Stable / progress / 2',
    'raw_global_no_progress_dense_s2': 'Raw / no progress / 2',
    'stable_global_no_progress_dense_s2': 'Stable / no progress / 2',
    'stable_global_no_progress_dense_s1': 'Stable / no progress / 1',
    'stable_global_no_progress_dense_s4': 'Stable / no progress / 4',
    'stable_global_no_progress_dense_s8': 'Stable / no progress / 8',
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj, immutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if immutable and p.exists():
        if p.read_text() != body:
            raise RuntimeError('preserve prior analysis artifact: '+str(p))
        return
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(body)
    tmp.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    study, out = args.study.resolve(), args.output.resolve()
    execution = study/'private/execution'
    sys.path[:0] = [str(execution/'src'), str(execution)]
    import numpy as np
    from scripts import v31_official as official
    from scripts.v31_refresh_factor_analysis import cell_key, contrast
    from scripts.v31_full503_refresh_study import verify

    native = read(study/'specs/protocol.json')
    verify(study, native)
    planned = sorted(read(study/'private/cells_shard01.json')+
                     read(study/'private/cells_shard02.json'), key=lambda c: c['index'])
    expected = {cell_key(c) for c in planned}
    assert len(planned) == len(expected) == 252
    cells = out/'private/cells_shards01_02.json'
    write(cells, planned, True)
    identity = dict(schema='v31_first_two_shards_analysis_v1',
        native_protocol_sha256=sha(study/'specs/protocol.json'),
        generation_source_commit=native['source_commit'],
        analysis_sha256=sha(__file__),
        scorer_sha256=sha(execution/'scripts/v31_score_longbench_official.py'),
        scorer_rules_sha256=sha(execution/'scripts/v31_official.py'),
        contrast_sha256=sha(execution/'scripts/v31_refresh_factor_analysis.py'),
        cells_sha256=sha(cells), cells_per_arm=252, shards=[1, 2],
        planned_full503_complete=False, clean_timing='paused',
        bootstrap=dict(problem_clusters=True, replicates=10000, seed=51994,
                       intervals='descriptive, unadjusted for multiple comparisons'),
        scope='Previously examined deterministic modulo-index shards; not all 503 problems.')
    write(out/'specs/analysis_protocol.json', identity, True)
    queue = read(study/'queue_state.json')
    entries = [e for e in queue['completed'] if e['shard'] in (1, 2)]
    index_path = out/'index.json'
    index = read(index_path) if index_path.exists() else dict(scored={})
    for arm in native['arms']:
        selected = sorted([e for e in entries if e['arm'] == arm], key=lambda e: e['shard'])
        if len(selected) != 2:
            continue
        for entry in selected:
            leaf = study/entry['path']
            if any(sha(leaf/name) != h for name, h in entry['artifacts_sha256'].items()):
                raise RuntimeError('completed native shard changed')
            receipt = read(leaf/'receipt.json')
            if receipt['status'] != 'complete' or receipt['validation_errors']:
                raise RuntimeError('unclean native receipt')
        if arm in index['scored']:
            cached = index['scored'][arm]
            if cached['native_shards'] != selected:
                raise RuntimeError('scored native source differs')
            if any(sha(out/rel) != h for rel, h in cached['artifacts_sha256'].items()):
                raise RuntimeError('completed scoring artifact changed')
            continue
        for attempt in range(1, 1000):
            scoring = out/f'scoring/{arm}/attempt{attempt:03d}'
            if not scoring.exists():
                break
        private = scoring/f'private/run_{arm}.private.jsonl'
        private.parent.mkdir(parents=True)
        with private.open('w') as target:
            for entry in selected:
                target.write((study/entry['path']/f'private/run_{arm}.private.jsonl').read_text())
        prefix = scoring/'longbench'
        with (scoring/'scorer.log').open('w') as stream:
            subprocess.run([sys.executable, str(execution/'scripts/v31_score_longbench_official.py'),
                str(prefix), str(study/'private/manifests'), str(study/'private/manifests'),
                '--cells', str(cells), str(private)], cwd=execution, check=True,
                stdout=stream, stderr=subprocess.STDOUT)
        scores = read(prefix.with_suffix('.official.json'))
        assert set(scores) == {arm} and set(scores[arm]) == expected
        names = ['longbench.official.json', 'longbench.summary.json', 'longbench.binding.json',
                 'longbench.judge_and_finished.json']
        state = dict(native_shards=selected, scores=str(prefix.with_suffix('.official.json').relative_to(out)),
            summary=str(prefix.with_suffix('.summary.json').relative_to(out)),
            artifacts_sha256={str((scoring/name).relative_to(out)):sha(scoring/name) for name in names})
        write(scoring/'score_complete.json', dict(status='complete', cells=252, **state), True)
        index['scored'][arm] = state
        write(index_path, index)
        print(json.dumps(dict(event='official_partial_scoring_complete',arm=arm,cells=252)), flush=True)
    if len(index['scored']) != 13:
        print(json.dumps(dict(status='waiting_for_shard02',scored_arms=len(index['scored']))))
        return
    pause = read(study/'PAUSED_USER_REQUEST.json')
    assert pause['completed_shards'] == 26 and not list(study.glob('attempt*/*/shard03'))
    rows, scores, configs, gpu_seconds = {}, {}, {}, {}
    for arm in native['arms']:
        all_rows, cfgs = [], []
        selected = index['scored'][arm]['native_shards']
        for entry in selected:
            leaf = study/entry['path']
            all_rows.extend(json.loads(x) for x in (leaf/'records.jsonl').read_text().splitlines())
            cfgs.append(read(leaf/'config.json'))
        rows[arm] = {cell_key(r):r for r in all_rows}
        assert len(all_rows) == 252 and set(rows[arm]) == expected
        scores[arm] = read(out/index['scored'][arm]['scores'])[arm]
        configs[arm] = cfgs
        gpu_seconds[arm] = sum(read(study/e['path']/'receipt.json')['gpu_seconds'] for e in selected)
    reference = 'v31_progress_dense_control'
    results = []
    for arm, rr in rows.items():
        for key, r in rr.items():
            for field in official.COMPARE_FIELDS+('vllm','torch','gpu','seed_base','gpu_memory_utilization',
                    'fix_51994','fa4_local_fix','frozen_source_identity_sha256'):
                if r.get(field) != rows[reference][key].get(field):
                    raise RuntimeError('record fairness mismatch: '+field)
            a = r['receipts']['adapter']
            assert r['cuda_graph_captures'] == 0
            assert not any(a.get(k, 0) for k in ('native_local_skipped_tiles','order_errors',
                                               'stable_budget_errors','every_step_budget_errors'))
        for cfg in configs[arm]:
            for field in ('source_sha256','host_fingerprint','gpu_identity',
                          'model_source_inventory_sha256','manifests_sha256'):
                if cfg[field] != configs[reference][0][field]:
                    raise RuntimeError('configuration fairness mismatch: '+field)
        vals = list(rr.values())
        audits = [r['receipts']['adapter'] for r in vals]
        total = lambda k: sum(a.get(k, 0) or 0 for a in audits)
        n = np.array([r['denoise_forwards'] for r in vals])
        result = dict(arm=arm,label=LABELS[arm],cells=252,correct=sum(scores[arm].values()),
            accuracy_pct=100*np.mean(list(scores[arm].values())),mean_N=float(n.mean()),
            p95_N=float(np.quantile(n,.95)),mean_NC=float(np.mean([r['denoise_forwards']/max(1,r['canvases']) for r in vals])),
            mean_T=float(np.mean([r['output_tokens'] for r in vals])),
            N_increased_cells=sum(r['denoise_forwards']>rows[reference][key]['denoise_forwards'] for key,r in rr.items()),
            global_eligible=total('global_eligible_tiles'),global_kept=total('global_kept_tiles'),
            global_skipped=total('global_eligible_tiles')-total('global_kept_tiles'),
            global_sparsity=1-total('global_kept_tiles')/max(1,total('global_eligible_tiles')),
            LOCAL_calls=total('native_local_calls_audited'),LOCAL_eligible=total('native_local_eligible_rectangles'),
            LOCAL_skipped=total('native_local_skipped_tiles'),request_graph_captures=0,
            cheap_swaps=total('stable_swaps'),cheap_noop_units=total('stable_unit_noops'),
            cheap_updated_units=total('stable_unit_updates'),cheap_selector_audit_ms=total('stable_selector_audit_ms'),
            every_step_selector_audit_ms=total('every_step_selector_audit_ms'),
            every_step_dense_observations=total('every_step_dense_observations'),
            extra_observer_QK=total('every_step_observer_qk_rectangles'),extra_observer_PV=total('every_step_observer_pv_rectangles'),
            extra_tail_QK=total('every_step_tail_qk_rectangles'),
            cache_bytes_max=max(a.get('stable_state_bytes',0) or 0 for a in audits),
            audit_GPU_reservation_seconds=gpu_seconds[arm],by_shard={},
            paired_vs_V31=contrast(rr,rows[reference],scores[arm],scores[reference]))
        for shard in (1,2):
            keys=[k for k,r in rr.items() if r['index']%4 == shard-1]
            result['by_shard'][str(shard)]=dict(cells=len(keys),correct=sum(scores[arm][k] for k in keys),
                mean_N=float(np.mean([rr[k]['denoise_forwards'] for k in keys])))
        results.append(result)
    factor=[]
    def pair(candidate, ref, label):
        factor.append(dict(factor=label,candidate=candidate,reference=ref,
            **contrast(rows[candidate],rows[ref],scores[candidate],scores[ref])))
    for progress in ('progress','no_progress'):
        pair(f'stable_global_{progress}_dense_s2',f'raw_global_{progress}_dense_s2','stable_minus_raw_s2')
    for mode in ('raw','stable'):
        pair(f'{mode}_global_progress_dense_s2',f'{mode}_global_no_progress_dense_s2','progress_on_minus_off_s2')
    for interval in (2,4,8):
        pair(f'stable_global_no_progress_dense_s{interval}','stable_global_no_progress_dense_s1','interval_minus_1')
    pair('v31_progress_dense_control','v31_no_progress_dense_hold','progress_without_cheap')
    pair('v31_every_step_dense_select_sparse_output','v31_progress_dense_control','fresh_every_step_minus_V31')
    write(out/'native_results.json',results,True)
    write(out/'factor_contrasts.json',factor,True)
    write(out/'fairness_audit.json',dict(status='complete',arms=13,cells_per_arm=252,total_cells=3276,
        completed_native_hashes_valid=True,source_sampler_runtime_inputs_matched=True,
        LOCAL_dense=True,request_graph_captures=0,budget_errors=0,validation_errors=[]),True)
    lines=['# Paused after shard02: LongBench-v2 partial results','',
        'All 13 arms completed the same 252 seed-1 problems: 126 in each of shards 01 and 02. The full 503-problem experiment is paused; 251 problems per arm remain. These are partial, previously examined benchmark results, not a completed 503-problem audit. No shard03 worker launched.','',
        '| Arm | Shard01 /126 | Shard02 /126 | Combined /252 | Accuracy | Mean N | N vs V31 | N/C | Output tokens | GLOBAL skipped |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['label']} | {r['by_shard']['1']['correct']} | {r['by_shard']['2']['correct']} | {r['correct']} | {r['accuracy_pct']:.2f}% | {r['mean_N']:.2f} | {100*(r['paired_vs_V31']['N_ratio']-1):+.2f}% | {r['mean_NC']:.2f} | {r['mean_T']:.1f} | {100*r['global_sparsity']:.2f}% |")
    lines+=['','## Paired factor comparisons','',
        '| Candidate minus reference | Correct delta | Wins / losses | Accuracy delta pp [95% CI] | N ratio [95% CI] | N/C ratio | Output-length ratio |',
        '|---|---:|---:|---|---|---:|---:|']
    for r in factor:
        ac=r['accuracy_delta_ci95_pp'];nc=r['N_ratio_ci95']
        lines.append(f"| {LABELS[r['candidate']]} minus {LABELS[r['reference']]} | {r['correct_delta']:+d} | {r['wins']} / {r['losses']} | {r['accuracy_delta_pp']:+.2f} [{ac[0]:.2f}, {ac[1]:.2f}] | {r['N_ratio']:.3f} [{nc[0]:.3f}, {nc[1]:.3f}] | {r['NC_ratio']:.3f} | {r['T_ratio']:.3f} |")
    lines+=['','Intervals use 10,000 paired problem-cluster bootstrap replicates, seed 51994; they are descriptive and unadjusted for multiple comparisons. Nonsignificance does not establish noninferiority. The four interval-2 arms isolate the stable bundle and progress conditionally; intervals 1/2/4/8 are compared only for stable no-progress.','',
        '## Operator diagnostics and execution checks','',
        'The pre-generation CPU/CUDA/FA4 gates, complete independent factor replay, and new every-step real-Q/K/V qualification passed; see the parent study qualification receipts. The first-round cross-arm audit and this 3,276-cell source/sampler/runtime/hash audit pass. All evaluated requests have zero new CUDA captures and all LOCAL calls remain dense. Warmup graph captures are outside evaluated requests. Native GLOBAL physical counts include actual dense-refresh/carry phases; K=128 is the eligible-prefix budget, with existing protected tail/structural tiles preserved. Short prefixes keep all available tiles.','',
        'No-progress hold retains the existing per-canvas initialization/carry and disables mid-canvas progress refresh. Fresh every step selects from current full-support V31 mass without a previous-map bonus or cache, discards the dense observer output, then returns native sparse FA4 output on that same call; its extra observer QK/PV and tail-QK work are separate in native_results.json. No extra transformer/denoising forward is inserted.','',
        '## Selector overhead and clean timing','',
        'native_results.json reports selector CUDA-event audit totals, state memory, physical work, and instrumented GPU reservation seconds. These are audit measurements, not clean latency. Clean timing remains intentionally absent and paused. N, N/C and output length describe the native trajectory, not speedup. The full study has no final_complete.json. Resume requires an explicit user instruction.','']
    summary='\n'.join(lines)
    path=out/'summary.md'
    if path.exists() and path.read_text()!=summary:
        raise RuntimeError('preserve prior partial summary')
    path.write_text(summary)
    verify(study,native)
    write(out/'partial_complete.json',dict(status='complete',scope='shards01_02_only',
        arms=13,cells_per_arm=252,total_native_cells=3276,full503_complete=False,
        official_scoring_complete=True,study_paused=True,source_commit=native['source_commit'],
        analysis_protocol_sha256=sha(out/'specs/analysis_protocol.json'),summary_sha256=sha(path),
        fairness_audit_sha256=sha(out/'fairness_audit.json'),clean_timing='paused'),True)
    print(json.dumps(dict(status='partial_complete',cells_per_arm=252,total_cells=3276,study_paused=True)))


if __name__ == '__main__':
    main()
