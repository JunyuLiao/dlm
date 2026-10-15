"""Official full503 scoring and source-matched trajectory/work report."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.v31_full503_refresh_study import read,sha,write,verify
from scripts.v31_refresh_factor_analysis import cell_key,contrast
from scripts import v31_official as official


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);args=p.parse_args();study=args.study.resolve()
    if (study/'final_complete.json').exists():raise RuntimeError('preserve completed report')
    protocol=read(study/'specs/protocol.json');verify(study,protocol)
    execution=study/'private/execution';completion=read(study/'native_generation_complete.json')
    expected={cell_key(c) for c in read(study/'private/cells_full503.json')}
    reportdir=study/'reports';reportdir.mkdir(exist_ok=True)
    rows,scores,configs={},{},{}
    for arm in completion['arms']:
        collected=[];configs[arm]=[]
        scoring=study/'scoring'/arm;private=scoring/f'private/run_{arm}.private.jsonl';private.parent.mkdir(parents=True,exist_ok=True)
        with private.open('w') as out:
            for entry in [e for e in completion['shards'] if e['arm']==arm]:
                leaf=study/entry['path']
                for name,h in entry['artifacts_sha256'].items():
                    if sha(leaf/name)!=h:raise RuntimeError('completed shard changed')
                receipt=read(leaf/'receipt.json')
                if receipt['status']!='complete' or receipt['validation_errors']:raise RuntimeError('unclean native receipt')
                cfg=read(leaf/'config.json');configs[arm].append(cfg)
                collected.extend(json.loads(line) for line in (leaf/'records.jsonl').read_text().splitlines())
                out.write((leaf/'private'/f'run_{arm}.private.jsonl').read_text())
        rows[arm]={cell_key(r):r for r in collected}
        if len(collected)!=503 or set(rows[arm])!=expected:raise RuntimeError('native full503 coverage mismatch')
        subprocess.run([sys.executable,str(execution/'scripts/v31_score_longbench_official.py'),str(scoring/'longbench'),
            str(study/'private/manifests'),str(study/'private/manifests'),'--cells',str(study/'private/cells_full503.json'),str(private)],check=True,cwd=execution)
        scores[arm]=next(iter(read(scoring/'longbench.official.json').values()))
        if set(scores[arm])!=expected:raise RuntimeError('official full503 scorer coverage mismatch')
    reference='v31_progress_dense_control';fairness=[];results=[]
    for arm,rr in rows.items():
        base=rows[reference]
        for key,r in rr.items():
            for field in official.COMPARE_FIELDS+('vllm','torch','gpu','seed_base','gpu_memory_utilization','fix_51994','fa4_local_fix','frozen_source_identity_sha256'):
                if r.get(field)!=base[key].get(field):raise RuntimeError('native fairness mismatch: '+field)
            a=r['receipts']['adapter']
            if r['cuda_graph_captures'] or a.get('native_local_skipped_tiles',0) or a.get('order_errors',0) or a.get('stable_budget_errors',0) or a.get('every_step_budget_errors',0):raise RuntimeError('native output/routing violation')
        for cfg in configs[arm]:
            for field in ('source_sha256','host_fingerprint','gpu_identity','model_source_inventory_sha256','manifests_sha256'):
                if cfg[field]!=configs[reference][0][field]:raise RuntimeError('source/runtime config mismatch')
        vals=list(rr.values());audits=[r['receipts']['adapter'] for r in vals]
        total=lambda k:sum(a.get(k,0) or 0 for a in audits)
        n=np.array([r['denoise_forwards'] for r in vals]);nc=np.array([r['denoise_forwards']/max(1,r['canvases']) for r in vals]);tokens=np.array([r['output_tokens'] for r in vals])
        result=dict(arm=arm,cells=503,correct=sum(scores[arm].values()),accuracy=100*np.mean(list(scores[arm].values())),
            mean_N=float(n.mean()),p95_N=float(np.quantile(n,.95)),mean_NC=float(nc.mean()),mean_T=float(tokens.mean()),
            global_kept=total('global_kept_tiles'),global_eligible=total('global_eligible_tiles'),
            global_sparsity=1-total('global_kept_tiles')/max(1,total('global_eligible_tiles')),
            local_calls=total('native_local_calls_audited'),local_skipped=total('native_local_skipped_tiles'),
            graph_captures=sum(r['cuda_graph_captures'] for r in vals),
            always_dense_observations=total('every_step_dense_observations'),
            extra_dense_QK=total('every_step_observer_qk_rectangles'),extra_dense_PV=total('every_step_observer_pv_rectangles'),
            extra_tail_QK=total('every_step_tail_qk_rectangles'),
            every_step_selector_audit_ms=total('every_step_selector_audit_ms'),
            cheap_selector_audit_ms=total('stable_selector_audit_ms'),
            cheap_swaps=total('stable_swaps'),fresh_swaps=total('every_step_swaps'),
            stable_cache_bytes_max=max((a.get('stable_state_bytes',0) or 0 for a in audits),default=0),
            paired={ref:contrast(rr,rows[ref],scores[arm],scores[ref]) for ref in ('v31_progress_dense_control','v31_no_progress_dense_hold','dense_full_fix51994')})
        results.append(result);fairness.append(dict(arm=arm,cells=503,source_sampler_runtime_matched=True,LOCAL_dense=True,graph_captures=0))
    factor=[]
    def pair(candidate,reference,label):
        factor.append(dict(factor=label,candidate=candidate,reference=reference,**contrast(rows[candidate],rows[reference],scores[candidate],scores[reference])))
    for progress in ('progress','no_progress'):
        pair(f'stable_global_{progress}_dense_s2',f'raw_global_{progress}_dense_s2','stable_minus_raw_s2')
    for mode in ('raw','stable'):
        pair(f'{mode}_global_progress_dense_s2',f'{mode}_global_no_progress_dense_s2','progress_on_minus_off_s2')
    for period in (2,4,8):pair(f'stable_global_no_progress_dense_s{period}','stable_global_no_progress_dense_s1','interval_minus_1')
    pair('v31_progress_dense_control','v31_no_progress_dense_hold','progress_without_cheap')
    pair('v31_every_step_dense_select_sparse_output','v31_progress_dense_control','every_step_minus_V31')
    write(reportdir/'native_results.json',results);write(reportdir/'factor_contrasts.json',factor)
    write(reportdir/'fairness_audit.json',dict(status='complete',arms=fairness))
    lines=['# Full LongBench-v2 refresh comparison','',
        'All 13 arms completed and officially scored on 503 matched seed-1 problems. Previously examined benchmark; parameters remained frozen. N is denoising calls, not latency. Physical tiles below describe output-consumer work; every-step dense observation work is additional.','',
        '| Arm | Correct /503 | Mean N | N vs V31 | Mean N/C | Mean output tokens | GLOBAL sparsity |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:lines.append(f"| {r['arm']} | {r['correct']} | {r['mean_N']:.2f} | {100*(r['paired']['v31_progress_dense_control']['N_ratio']-1):+.2f}% | {r['mean_NC']:.2f} | {r['mean_T']:.1f} | {100*r['global_sparsity']:.2f}% |")
    lines+=['','## Paired factor comparisons','',
        '| Comparison | Correct delta | Wins / losses | Accuracy delta pp [95% CI] | N ratio [95% CI] |',
        '|---|---:|---:|---|---|']
    for r in factor:
        ci=r['accuracy_delta_ci95_pp'];ni=r['N_ratio_ci95']
        lines.append(f"| {r['candidate']} / {r['reference']} | {r['correct_delta']:+d} | {r['wins']} / {r['losses']} | {r['accuracy_delta_pp']:+.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {r['N_ratio']:.3f} [{ni[0]:.3f}, {ni[1]:.3f}] |")
    lines+=['','Paired problem bootstrap: 10,000 replicates, seed 51994. Intervals are descriptive and unadjusted for multiple contrasts; nonsignificance is not noninferiority. Always-refresh uses fresh current statistics and returns same-call sparse output, without a held-map bonus. No-progress hold retains per-canvas selection/carry and disables mid-canvas refresh. These are separate named baselines.','',
        '## Operator correctness and overhead','',
        'Focused CUDA/FA4 tests, real-Q/K/V CPU-map/native-output checks, repeated native controls and observation-only parity passed before target generation. See qualification receipts. Per-arm selector audit totals, extra full QK/PV work for every-step selection, physical GLOBAL/LOCAL counts, and cache memory are in native_results.json. All LOCAL calls stayed dense and actual request CUDA captures are zero. Audit times are not clean latency.','',
        '## Clean timing','',
        'Intentionally absent and paused. No speedup is inferred from N, output length or physical sparsity. All frozen sources/inputs, completed shard hashes, official score coverage and fairness audits agree.','']
    (study/'summary.md').write_text('\n'.join(lines))
    verify(study,protocol)
    write(study/'final_complete.json',dict(status='complete',native_cells=6539,arms=13,cells_per_arm=503,
        source_commit=protocol['source_commit'],protocol_sha256=sha(study/'specs/protocol.json'),official_scoring_complete=True,
        analysis_source_sha256=sha(__file__),scorer_source_sha256=sha(execution/'scripts/v31_score_longbench_official.py'),
        fairness_audit_sha256=sha(reportdir/'fairness_audit.json'),summary_sha256=sha(study/'summary.md'),clean_timing='paused'),True)


if __name__=='__main__':main()
