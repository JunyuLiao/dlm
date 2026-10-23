"""Verify and consolidate the completed user-directed optimized kernel panels."""
from pathlib import Path
import collections
import json
import math
import statistics

import numpy as np

from scripts import optimized_refresh_campaign as base
from scripts.v31_official import cell_key

ROOT=base.ROOT
OLD=ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
FOLLOW=ROOT/'results/optimized_refresh_followup_20261023/attempt003'
SOURCES={'humaneval':OLD,'aime26_k128':OLD,
         'aime26_k256':FOLLOW/'aime26_k256','math500':FOLLOW/'math500_lower',
         'ruler128k':FOLLOW/'ruler128k'}


def load_rows(study,suite,arm,pass_name):
    paths=sorted((study/f'runs/{suite}').glob(arm+f'*/{pass_name}/shard???/attempt???'))
    return [row for p in paths if (p/'attempt_complete.json').exists() for row in base.read_lines(p/'records.jsonl')]


def paired(scores,clean,cluster_of,ruler=False):
    rng=np.random.default_rng(20261023);result={}
    keys=sorted(next(iter(scores.values())));groups=collections.defaultdict(list)
    for key in keys:groups[cluster_of(key)].append(key)
    units=list(groups.values());draw=rng.integers(0,len(units),size=(10000,len(units)))
    for reference in ['v31_optimized','dense_full_fix51994','allkept_fa4']:
        result[reference]={}
        for arm in scores:
            delta=np.array([statistics.mean(float(scores[arm][k])-float(scores[reference][k]) for k in group) for group in units])
            scale=1 if ruler else 100
            row=dict(cells=len(keys),clusters=len(units),cluster_kind='RULER task' if ruler else 'problem',
                accuracy_difference_pp=float(delta.mean()*scale),accuracy_difference_ci95_pp=list(map(float,np.percentile(delta[draw].mean(1)*scale,[2.5,97.5]))))
            for field in ['wall_s','decode_s','denoise_forwards']:
                logs=np.array([statistics.mean(math.log(clean[arm][k][field]/clean[reference][k][field]) for k in group) for group in units])
                row[field]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=list(map(float,np.exp(np.percentile(logs[draw].mean(1),[2.5,97.5])))))
            result[reference][arm]=row
    return result


def build(dest):
    if not (FOLLOW/'final_complete.json').exists():raise RuntimeError('requested follow-up is incomplete')
    if (OLD/'runs/longbench_v2').exists():raise RuntimeError('LongBench launched after stop instruction')
    dest.mkdir(parents=True,exist_ok=False)
    all_summary={};all_paired={};receipts={};sources={};work={};latencies={};resources=[]
    for panel,study in SOURCES.items():
        suite='aime26' if panel.startswith('aime26') else panel
        protocol=base.verify(study);out=study/f'analysis/{suite}/attempt001'
        for rel,digest in protocol.get('reference_sha256',{}).items():
            if base.sha(ROOT/rel)!=digest:raise RuntimeError('referenced evidence drift')
        marker=base.read(out/'analysis_complete.json')
        if marker['protocol_sha256']!=base.sha(study/'protocol.json'):raise RuntimeError('analysis protocol mismatch')
        summary=base.read(out/'summary.json');scores=base.read(out/'official.json')
        expected=protocol['suites'][suite]['count'];audit={};clean={};checks={}
        for arm in base.ARMS:
            ar=load_rows(study,suite,arm,'audit');cr=load_rows(study,suite,arm,'clean')
            if len(ar)!=expected or len(cr)!=expected:raise RuntimeError('coverage mismatch')
            am={cell_key(x):x for x in ar};cm={cell_key(x):x for x in cr}
            if len(am)!=expected or len(cm)!=expected or am.keys()!=cm.keys() or am.keys()!=scores[arm].keys():raise RuntimeError('duplicate/unpaired coverage')
            if any((am[k]['output_hash'],am[k]['denoise_forwards'])!=(cm[k]['output_hash'],cm[k]['denoise_forwards']) for k in am):raise RuntimeError('trajectory mismatch')
            if any('cuda_graph_captures' not in x or x['cuda_graph_captures']!=0 or x.get('request_jit_event_count')!=0 or x.get('request_jit_events')!=[] for x in cr):raise RuntimeError('missing/nonzero clean capture or JIT counter')
            adapters=[x['receipts']['adapter'] for x in ar]
            if any(a.get('native_local_skipped_tiles')!=0 or a.get('global_calls')!=5*x['denoise_forwards'] or a.get('native_local_calls_audited')!=25*x['denoise_forwards'] for a,x in zip(adapters,ar)):raise RuntimeError('layer scope or forward accounting')
            if arm.startswith(('v2','v31')):
                if any(a.get('optimized_refresh_extra_model_forwards')!=0 or a.get('resident_sparse_extra_forwards')!=0 or a.get('optimized_refresh_budget_errors')!=0 or a.get('resident_sparse_held_kv_copy_calls')!=0 or a.get('resident_sparse_held_selector_calls')!=0 or a.get('resident_sparse_held_list_builds')!=0 or a.get('value_nonfinite_attention_calls')!=0 for a in adapters):raise RuntimeError('sparse kernel receipt violation')
            if arm.startswith('v2') and any(a['optimized_refresh_library_sha256']!=protocol['kernel_sha256'] for a in adapters):raise RuntimeError('V2 library identity mismatch')
            expected_budget=protocol['suites'][suite]['mage_k']
            if arm.startswith('v31') and any(x['mage_k']!=expected_budget for x in ar+cr):raise RuntimeError('V31 budget drift')
            audit[arm]=ar;clean[arm]=cm
            checks[arm]=dict(audit_cells=len(ar),clean_cells=len(cr),trajectory_mismatches=0,clean_capture_count=0,clean_jit_count=0,local_skip_count=0,
                nonfinite_attention_field_coverage=sum('value_nonfinite_attention_calls' in a for a in adapters),
                extra_forward_receipt_coverage=sum('optimized_refresh_extra_model_forwards' in a for a in adapters),
                global_calls=sum(a['global_calls'] for a in adapters),local_calls=sum(a['native_local_calls_audited'] for a in adapters),
                source_protocol_identities=sorted({x['frozen_source_identity_sha256'] for x in ar+cr}),
                observed_refresh_kernels=sorted({a['optimized_refresh_kernel'] for a in adapters if a.get('optimized_refresh_kernel')}),
                observed_v31_observer_sha256=sorted({a['optimized_refresh_observer_sha256'] for a in adapters if a.get('optimized_refresh_observer_sha256')}),
                observed_v2_thresholds=sorted({x['value_threshold'] for x in ar+cr if x.get('value_threshold') is not None}),
                observed_v31_budgets=sorted({x['mage_k'] for x in ar+cr}) if arm.startswith('v31') else None,
                output_hash_agreement=True)
            phase_timing={}
            for a in adapters:
                for group in ['optimized_refresh_phase_timing','resident_sparse_phase_timing']:
                    for phase,row in a.get(group,{}).items():
                        dst=phase_timing.setdefault(phase,dict(calls=0,gpu_ms=0.))
                        dst['calls']+=row['calls'];dst['gpu_ms']+=row['gpu_ms']
            for row in phase_timing.values():row['mean_us']=1000*row['gpu_ms']/row['calls']
            latencies.setdefault(panel,{})[arm]=dict(audit_attention_gpu_by_phase=phase_timing,
                clean_mean_request_s=summary[arm]['mean_W'],clean_generation_s=summary[arm]['S'],clean_request_s=summary[arm]['W'],
                N=summary[arm]['N'],C=summary[arm]['C'],N_per_C=summary[arm]['N_per_C'],T=summary[arm]['T'],
                amortized_generation_ms_per_call=1000*summary[arm]['S_per_N'])
            work.setdefault(panel,{})[arm]={k:summary[arm][k] for k in ['global_eligible_tiles','global_kept_tiles','global_sparsity','local_eligible_rectangles','local_skipped_tiles','overall_eligible_rectangles','overall_skipped_rectangles','overall_sparsity','first_canvas_eligible','first_canvas_kept','first_canvas_sparsity','phase_work']}
        task_of=None
        if suite=='ruler128k':
            task_of={x['index']:x['task'] for x in base.read(study/f'private/pools/{suite}/{protocol["suites"][suite]["dataset"]}_rowinfo.json')['rows']}
        cluster=(lambda k:task_of[int(k.split('|')[1])]) if task_of else (lambda k:'|'.join(k.split('|')[:2]))
        all_paired[panel]=paired(scores,clean,cluster,suite=='ruler128k');all_summary[panel]=summary;receipts[panel]=checks
        sources[panel]=dict(study=str(study.relative_to(ROOT)),source_commit=protocol['source_commit'],protocol_sha256=base.sha(study/'protocol.json'),
            execution_sha256=protocol['execution_sha256'],kernel_sha256=protocol['kernel_sha256'],model_inventory_sha256=protocol['model_inventory_sha256'],
            runtime=protocol['runtime'],analysis_summary_sha256=base.sha(out/'summary.json'),official_scores_sha256=base.sha(out/'official.json'),
            reused_reference_arms=protocol.get('reference_arms',{}),v31_budget=protocol['suites'][suite]['mage_k'])
    for r in [OLD,ROOT/'results/optimized_refresh_followup_20261022/attempt002',FOLLOW]:
        # Explicitly named immutable roots; deduplicate linked reference receipts.
        for p in r.rglob('receipt.json'):
            try:d=base.read(p)
            except (ValueError,OSError):continue
            if 'worker_reservation_seconds' in d:resources.append((str(p.resolve()),d['worker_reservation_seconds']))
    resources=dict(resources)
    base.write(dest/'summary.json',all_summary);base.write(dest/'paired_uncertainty.json',all_paired)
    base.write(dest/'verification.json',receipts);base.write(dest/'source_identity.json',sources)
    base.write(dest/'physical_work.json',work);base.write(dest/'latency_and_trajectories.json',latencies)
    base.write(dest/'resource_accounting.json',dict(worker_reservation_seconds=sum(resources.values()),workers=len(resources),scope='completed-worker reservations in named evaluation roots; includes loading and explicit warm requests; earlier optimization/calibration failures are preserved separately',active_gpu_seconds=False))
    lines=['# Optimized V2/V31 multi-benchmark results','',
        'Completed requested comparison on one H100. HumanEval and AIME K128 results are preserved; MATH V2 was recalibrated and rerun at0.2; AIME V31K256 is additional; RULER uses the existing104 balanced sampled problems. LongBench-v2 was stopped before generation as requested. The earlier four-workload campaign remains partial.','',
        '| Panel | V2 | V31 | Corrected FULL dense | PIECEWISE dense | All-kept FA4 |','|---|---:|---:|---:|---:|---:|']
    for panel,d in all_summary.items():
        values=[f"{100*d[a]['accuracy']:.2f}% ({d[a]['correct']}/{d[a]['cells']})" if panel!='ruler128k' else f"{d[a]['official_score']:.3f} ({d[a]['correct']}/104 full-credit)" for a in base.ARMS]
        lines.append('| '+panel+' | '+' | '.join(values)+' |')
    lines+=['','RULER reports official fractional-credit score out of100; the other panels use official task accuracy. AIME covers30 questions × seeds42/43/44, MATH all500 seed42, HumanEval all164 seed42. RULER is13 tasks ×8 sampled prompts at seed42 with the original official prefilled/thinking-off format.','',
        '| Panel | V2 mean W(s) | V31 mean W(s) | FULL mean W(s) | V2 / FULL | V31 / FULL | V2 / V31 |','|---|---:|---:|---:|---:|---:|---:|']
    for panel,d in all_summary.items():
        v=d['v2_online_optimized']['mean_W'];r=d['v31_optimized']['mean_W'];b=d['dense_full_fix51994']['mean_W']
        lines.append(f'| {panel} | {v:.4f} | {r:.4f} | {b:.4f} | {v/b:.3f} | {r/b:.3f} | {v/r:.3f} |')
    lines+=['','Ratios above1 are slower. These are arithmetic mean clean request times; paired geometric ratios and95% clustered intervals are in paired_uncertainty.json. S/N is an amortized generation cost, not a measured single-forward price. Every measured request has a separate untimed warm request; warm-up cost remains in reservation accounting.','',
        '| Panel | V2 threshold | V2 GLOBAL sparsity | V2 held | V31 GLOBAL sparsity | V31 held |','|---|---:|---:|---:|---:|---:|']
    thresholds={'humaneval':.5600716206066294,'aime26_k128':.059460355750136064,'aime26_k256':.059460355750136064,'math500':.2,'ruler128k':.021}
    for panel,d in all_summary.items():
        v=d['v2_online_optimized'];r=d['v31_optimized']
        lines.append(f"| {panel} | {thresholds[panel]:.12g} | {100*v['global_sparsity']:.2f}% | {100*v['phase_work']['held']['sparsity']:.2f}% | {100*r['global_sparsity']:.2f}% | {100*r['phase_work']['held']['sparsity']:.2f}% |")
    lines+=['','Physical work uses audited eligible/kept Q128×KV64 GLOBAL tiles. LOCAL remains dense and has a separate denominator. Complete initial/refresh/held/carried, first-canvas and whole-request counts are in physical_work.json. Native denoising trajectories, output length, generation/request time and phase GPU latency are in latency_and_trajectories.json. Refresh/held event times are audit-only; clean timing excludes instrumentation.','',
        '| Panel | Method | N | N/C | T | S/N(ms) | S(s) | W(s) | Refresh(us) | Held(us) |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for panel,arms in latencies.items():
        for arm,x in arms.items():
            timing=x['audit_attention_gpu_by_phase'];refresh=timing.get('refresh',{}).get('mean_us');held=timing.get('held',{}).get('mean_us')
            refresh_text=f'{refresh:.2f}' if refresh is not None else 'not instrumented';held_text=f'{held:.2f}' if held is not None else 'not instrumented'
            lines.append(f"| {panel} | {arm} | {x['N']} | {x['N_per_C']:.3f} | {x['T']} | {x['amortized_generation_ms_per_call']:.3f} | {x['clean_generation_s']:.3f} | {x['clean_request_s']:.3f} | {refresh_text} | {held_text} |")
    mismatch={}
    for panel,d in all_summary.items():
        v=d['v2_online_optimized'];r=d['v31_optimized']
        phase={}
        for name in sorted(set(v['phase_work'])|set(r['phase_work'])):
            a=v['phase_work'].get(name,{});b=r['phase_work'].get(name,{})
            phase[name]=dict(v2_kept=a.get('kept_tiles',0),v31_kept=b.get('kept_tiles',0),kept_ratio=a.get('kept_tiles',0)/b['kept_tiles'] if b.get('kept_tiles') else None)
        mismatch[panel]=dict(first_canvas_absolute_kept_ratio=v['first_canvas_kept']/r['first_canvas_kept'],
            whole_request_absolute_kept_ratio=v['global_kept_tiles']/r['global_kept_tiles'],
            global_kept_fraction_delta_pp=100*(r['global_sparsity']-v['global_sparsity']),
            phases=phase,total_flops_matched=False)
    base.write(dest/'work_mismatch.json',mismatch)
    lines+=['','The phase GPU figures are per GLOBAL attention call and include the integrated refresh dispatch, projection/packing and selection work inside its event boundary. They are not clean end-to-end or isolated PV measurements. Native dense per-attention event timing was not instrumented; no native single-step kernel speedup is inferred from these rows.','',
        '| Panel | V2 − V31 accuracy points | Paired95% interval | V2 − FULL accuracy points | Paired95% interval |','|---|---:|---:|---:|---:|']
    for panel,p in all_paired.items():
        x=p['v31_optimized']['v2_online_optimized'];y=p['dense_full_fix51994']['v2_online_optimized']
        ci=lambda row:', '.join(f'{v:+.2f}' for v in row['accuracy_difference_ci95_pp'])
        lines.append(f"| {panel} | {x['accuracy_difference_pp']:+.2f} | [{ci(x)}] | {y['accuracy_difference_pp']:+.2f} | [{ci(y)}] |")
    lines+=['','Intervals cluster AIME seeds by question and RULER by task;10,000 paired bootstrap draws. Nonsignificance does not establish noninferiority. Calibration is computation-only and transductive: MATH uses32 examined work strata, AIME four examined prompts, RULER one prompt per task; HumanEval reuses its saved calibrated threshold. No new held-out claim.','',
        'The old MATH V2tau3.2 result is supplementary:392/500,77.58% GLOBAL and79.43% held sparsity. Its four short calibration requests were unrepresentative. The rerun at0.2 reaches the requested40–50% on the complete500 panel.','',
        'RULER matches approximate held kept fraction, not total FLOPs. The preserved first-canvas absolute-work matching attempt failed: V31 refresh is dense, V2 refresh is sparse, and trajectories differ. Reported absolute/phase work must accompany sparsity; work_mismatch.json records these ratios for every panel. Both controls and negative calibration evidence remain available.','',
        'Verification: complete paired coverage, audit/clean output-hash and denoising-call agreement, explicit zero clean capture/JIT counters, zero audited LOCAL skips, sparse-selector budget checks and no extra sparse model forwards all pass. Kernel/source/runtime/model identity and receipt field coverage are in verification.json and source_identity.json. Native dense has no sparse adapter receipt; absent fields are not reported as measured zero.','',
        'V2 remains genuinely online with current QK/softmax/Gaussian32 projected PV followed by conditional full PV. QK/routing repeats across four output groups in the selected kernel. Best preserved8K complete synthetic refresh427.90us remains slower than plain dense363.58us and native paged dense287.47us. No refresh-parity claim follows from sparsity. Yuhan authored the inherited cross-step reuse baseline; Junyu Gaussian32/value-direction routing and the optimized execution variants remain separately attributed.','',
        'All original partial attempts, raw generations, failed designs and calibration evidence remain immutable. The prior full campaign has no completion marker; this report covers only the explicitly completed requested panels.']
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    base.write(dest/'final_complete.json',dict(status='complete',utc=base.utc(),panels=list(SOURCES),summary_sha256=base.sha(dest/'summary.json'),verification_sha256=base.sha(dest/'verification.json'),source_identity_sha256=base.sha(dest/'source_identity.json'),report_script_sha256=base.sha(Path(__file__)),longbench_generation_started=False))
    print(dest,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();build(args.out.resolve())
