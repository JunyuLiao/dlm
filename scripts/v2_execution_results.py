"""Verify and publish the completed execution follow-up without partial-panel claims."""
import argparse
import collections
import json
import math
from pathlib import Path
import statistics
import numpy as np
from scripts import optimized_refresh_campaign as base
from scripts.v31_official import cell_key

def load(study,suite,arm,kind):
    return [r for p in sorted((study/f'runs/{suite}').glob(arm+f'*/{kind}/shard???/attempt???'))
            if (p/'attempt_complete.json').exists() for r in base.read_lines(p/'records.jsonl')]

def budget_comparison(study,suite,reference):
    """Paired budget contrast; historical timing remains explicitly labeled."""
    current=base.read(study/f'analysis/{suite}/attempt001/official.json')['v31_optimized']
    prior=base.read(reference/f'analysis/{suite}/attempt001/official.json')['v31_optimized']
    if current.keys()!=prior.keys():raise RuntimeError('budget comparison coverage mismatch')
    groups=collections.defaultdict(list)
    for key in current:groups['|'.join(key.split('|')[:2])].append(key)
    units=list(groups.values());rng=np.random.default_rng(20261023);draw=rng.integers(0,len(units),size=(10000,len(units)))
    delta=np.array([statistics.mean(float(current[k])-float(prior[k]) for k in group) for group in units])
    before={cell_key(x):x for x in load(reference,suite,'v31_optimized','clean')}
    after={cell_key(x):x for x in load(study,suite,'v31_optimized','clean')}
    if before.keys()!=current.keys() or after.keys()!=current.keys():raise RuntimeError('budget clean coverage mismatch')
    result=dict(cells=len(current),problem_clusters=len(units),current_correct=sum(current.values()),reference_correct=sum(prior.values()),
        accuracy_difference_pp=float(100*delta.mean()),accuracy_difference_ci95_pp=list(map(float,np.percentile(100*delta[draw].mean(1),[2.5,97.5]))),
        reference=str(reference.relative_to(base.ROOT)),reference_protocol_sha256=base.sha(reference/'protocol.json'),historical_timing=True)
    for field in ['wall_s','decode_s','denoise_forwards']:
        logs=np.array([statistics.mean(math.log(after[k][field]/before[k][field]) for k in group) for group in units])
        result[field]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=list(map(float,np.exp(np.percentile(logs[draw].mean(1),[2.5,97.5])))))
    return result

def publication_scope(root):
    manifest=base.read(root/'protocol.json')
    changes=sorted(root.glob('USER_SCOPE_CHANGE_*.json'))
    if not changes:
        if not (root/'final_complete.json').exists():raise RuntimeError('campaign incomplete')
        return manifest,manifest['order'],None
    if len(changes)!=1:raise RuntimeError('ambiguous user scope changes')
    path=changes[0];scope=base.read(path)
    if scope['original_protocol_sha256']!=base.sha(root/'protocol.json') or scope['original_order']!=manifest['order']:
        raise RuntimeError('scope change protocol mismatch')
    active=scope['active_order'];cancelled=scope['cancelled_panels']
    if len(set(active+cancelled))!=len(active+cancelled) or set(active+cancelled)!=set(manifest['order']):
        raise RuntimeError('invalid active/cancelled scope')
    for panel in cancelled:
        study=root/panel
        if not (study/'PAUSED_USER_REQUEST.json').exists() or not (study/'NOT_RUN_USER_CANCELLED.json').exists():
            raise RuntimeError('cancelled panel launch guard absent')
        if (study/'runs').exists() or (study/'qualification').exists():raise RuntimeError('cancelled panel unexpectedly started')
    if not all((root/panel/'final_complete.json').exists() for panel in active):raise RuntimeError('active campaign incomplete')
    return manifest,active,dict(path=path.name,sha256=base.sha(path),cancelled_panels=cancelled,original_campaign_complete=False)

def build(root,dest):
    if not (root/'protocol.json').exists():raise RuntimeError('campaign incomplete')
    manifest,active,scope=publication_scope(root);summaries={};sources={};verified={};latencies={};paired={}
    for panel in active:
        study=root/panel;p=base.verify(study);suite=next(iter(p['suites']));out=study/f'analysis/{suite}/attempt001'
        if not (study/'final_complete.json').exists():raise RuntimeError('panel incomplete')
        if base.sha(study/'protocol.json')!=manifest['panels'][panel]:raise RuntimeError('panel protocol mismatch')
        if base.read(out/'analysis_complete.json')['protocol_sha256']!=base.sha(study/'protocol.json'):raise RuntimeError('analysis identity mismatch')
        for rel,digest in p['reference_sha256'].items():
            if base.sha(base.ROOT/rel)!=digest:raise RuntimeError('reference drift')
        summary=base.read(out/'summary.json');scores=base.read(out/'official.json');clean={};checks={};phases={}
        expected=p['suites'][suite]['count'];cal=base.read(study/f'calibration_held/{suite}/selected_threshold.json')
        for arm in p['arms']:
            audit=load(study,suite,arm,'audit');cr=load(study,suite,arm,'clean');am={cell_key(x):x for x in audit};cm={cell_key(x):x for x in cr}
            if len(audit)!=expected or len(cr)!=expected or len(am)!=expected or len(cm)!=expected or am.keys()!=cm.keys() or cm.keys()!=scores[arm].keys():raise RuntimeError('unpaired coverage')
            for key in am:
                if (am[key]['output_hash'],am[key]['denoise_forwards'])!=(cm[key]['output_hash'],cm[key]['denoise_forwards']):raise RuntimeError('audit/clean mismatch')
            if any(x.get('cuda_graph_captures')!=0 or x.get('request_jit_event_count')!=0 or x.get('request_jit_events')!=[] for x in cr):raise RuntimeError('timed capture/JIT')
            receipts=[x['receipts']['adapter'] for x in audit]
            for x,a in zip(audit,receipts):
                if a.get('native_local_skipped_tiles')!=0 or a.get('global_calls')!=5*x['denoise_forwards'] or a.get('native_local_calls_audited')!=25*x['denoise_forwards']:raise RuntimeError('scope/forward count')
                if arm.startswith(('v2','v31')):
                    if any(a.get(k)!=0 for k in ['optimized_refresh_extra_model_forwards','resident_sparse_extra_forwards','optimized_refresh_budget_errors','resident_sparse_held_kv_copy_calls','resident_sparse_held_selector_calls','resident_sparse_held_list_builds','value_nonfinite_attention_calls']):raise RuntimeError('sparse invariant')
                if arm in p['new_arms'] and arm.startswith('v2'):
                    if a['optimized_refresh_library_sha256']!=p['kernel_sha256'] or not a.get('optimized_refresh_prefix_projection_reuse'):raise RuntimeError('new V2 identity')
                if arm in p['new_arms'] and arm.startswith('v31') and x['mage_k']!=p['suites'][suite]['mage_k']:raise RuntimeError('budget mismatch')
            timing={}
            for a in receipts:
                for group in ['optimized_refresh_phase_timing','resident_sparse_phase_timing']:
                    for phase,row in a.get(group,{}).items():
                        entry=timing.setdefault(phase,dict(calls=0,gpu_ms=0.));entry['calls']+=row['calls'];entry['gpu_ms']+=row['gpu_ms']
            for row in timing.values():row['mean_us']=1000*row['gpu_ms']/row['calls']
            phases[arm]=timing;clean[arm]=cm
            projection_fields={'prefix_projection_full_calls':'optimized_refresh_projection_full_calls',
                'prefix_projection_canvas_calls':'optimized_refresh_projection_canvas_calls',
                'reused_projection_rows':'optimized_refresh_reused_projection_rows'}
            projection={name:sum(a[key] for a in receipts) if all(a.get(key) is not None for a in receipts) else None for name,key in projection_fields.items()}
            checks[arm]=dict(audit_count=expected,clean_count=expected,audit_clean_trajectory_agreement=True,clean_captures=0,clean_jit_events=0,local_skips=0,
                source_identity_sha256=sorted({x['frozen_source_identity_sha256'] for x in audit+cr}),
                projection_receipt_coverage={name:sum(a.get(key) is not None for a in receipts) for name,key in projection_fields.items()},**projection)
        # Recompute paired intervals for all available comparators. Historical
        # controls are labeled and never used as source-matched speed evidence.
        if suite=='ruler128k':
            info=base.read(study/f'private/pools/{suite}/{p["suites"][suite]["dataset"]}_rowinfo.json')['rows'];tasks={x['index']:x['task'] for x in info}
            cluster=lambda k:tasks[int(k.split('|')[1])]
        else:cluster=lambda k:'|'.join(k.split('|')[:2])
        keys=sorted(next(iter(scores.values())));groups=collections.defaultdict(list)
        for key in keys:groups[cluster(key)].append(key)
        units=list(groups.values());rng=np.random.default_rng(20261023);draw=rng.integers(0,len(units),size=(10000,len(units)));comparisons={}
        for reference in scores:
            comparisons[reference]={}
            for arm in scores:
                delta=np.array([statistics.mean(float(scores[arm][k])-float(scores[reference][k]) for k in group) for group in units]);scale=1 if suite=='ruler128k' else 100
                result=dict(cells=expected,clusters=len(units),accuracy_difference_pp=float(delta.mean()*scale),accuracy_difference_ci95_pp=list(map(float,np.percentile(delta[draw].mean(1)*scale,[2.5,97.5]))),historical_control=reference in p['reference_arms'] or arm in p['reference_arms'])
                for field in ['wall_s','decode_s','denoise_forwards']:
                    logs=np.array([statistics.mean(math.log(clean[arm][k][field]/clean[reference][k][field]) for k in group) for group in units]);result[field]=dict(geometric_ratio=float(np.exp(logs.mean())),ci95=list(map(float,np.exp(np.percentile(logs[draw].mean(1),[2.5,97.5])))))
                comparisons[reference][arm]=result
        summaries[panel]=dict(suite=suite,v31_budget=p['suites'][suite]['mage_k'],calibration=cal,arms=summary,new_arms=p['new_arms'],reference_arms=p['reference_arms'])
        paired[panel]=comparisons;latencies[panel]=phases;verified[panel]=checks
        sources[panel]=dict(study=str(study.relative_to(base.ROOT)),protocol_sha256=base.sha(study/'protocol.json'),source_commit=p['source_commit'],execution_sha256=p['execution_sha256'],kernel_sha256=p['kernel_sha256'],runtime=p['runtime'],model_inventory_sha256=p['model_inventory_sha256'],summary_sha256=base.sha(out/'summary.json'),scores_sha256=base.sha(out/'official.json'))
    budget_contrasts={}
    old=base.ROOT/'results/optimized_refresh_benchmarks_20261022/attempt009'
    if 'aime26_k1920' in active:
        budget_contrasts['aime26_k1920_vs_k128']=budget_comparison(root/'aime26_k1920','aime26',old)
        prior=base.ROOT/'results/optimized_refresh_followup_20261023/attempt003/aime26_k256'
        budget_contrasts['aime26_k1920_vs_k256']=budget_comparison(root/'aime26_k1920','aime26',prior)
    if 'math500_k384' in active:budget_contrasts['math500_k384_vs_k128']=budget_comparison(root/'math500_k384','math500',old)
    dest.mkdir(parents=True,exist_ok=False)
    base.write(dest/'budget_contrasts.json',budget_contrasts)
    for name,value in [('summary',summaries),('paired_uncertainty',paired),('verification',verified),('source_identity',sources),('attention_latency',latencies)]:base.write(dest/(name+'.json'),value)
    lines=['# V2 execution optimization and benchmark follow-up','', 'Completed user-requested panels. Clean generation/request timing is separate from instrumented attention phase timing. Historical controls retain their original source and date; their timing is supplementary, not a matched execution-speed test. Native sampler, dense LOCAL, prefill and commits are preserved.','', '| Panel | Method | Official accuracy | N | N/C | Mean S (s) | Mean W (s) | GLOBAL all-step sparsity | Held sparsity |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for panel,data in summaries.items():
        for arm,x in data['arms'].items():
            held=x.get('phase_work',{}).get('held',{}).get('sparsity');held_text=f'{100*held:.2f}%' if held is not None else 'n/a'
            label=arm+(' (historical)' if arm in data['reference_arms'] else '')
            lines.append(f"| {panel} | {label} | {100*x['accuracy']:.2f}% | {x['N']} | {x['N_per_C']:.3f} | {x['S']/x['cells']:.4f} | {x['mean_W']:.4f} | {100*x['global_sparsity']:.2f}% | {held_text} |")
    scope_note='LongBench-v2 was cancelled by the user before qualification or generation; prepared artifacts and original four-panel protocol remain preserved. This report completes only the three-panel authorized scope.' if scope else 'LongBench-v2 uses fixedV2tau0.021 andV31K8192 without dense panels.'
    lines+=['', 'RULER uses its official fractional-credit score out of 100. V2 alone is recalibrated to 40-50% all-step GLOBAL physical sparsity on the examined 13-task calibration cohort. Full 104-problem work is measured independently in the table. '+scope_note+' Prior short-benchmark V2 generations remain intact. The frozen per-suite Markdown helper uses legacy held-target/four-question wording; the calibration receipt and this report specify the actual all-step target and 13 examined questions.','', '| Panel | V2 threshold | Calibration all-step sparsity | Full-panel all-step sparsity | Full-panel held sparsity |','|---|---:|---:|---:|---:|']
    for panel,data in summaries.items():
        cal=data['calibration'];x=data['arms']['v2_online_optimized'];cs=cal.get('global_sparsity')
        ctext=f'{100*cs:.2f}%' if cs is not None else 'preserved threshold'
        held=x['phase_work']['held']['sparsity']
        lines.append(f"| {panel} | {cal['threshold']:.12g} | {ctext} | {100*x['global_sparsity']:.2f}% | {100*held:.2f}% |")
    lines+=['', '| Budget comparison | Accuracy difference (pp) | Paired 95% interval (pp) |','|---|---:|---:|']
    for label,x in budget_contrasts.items():
        lo,hi=x['accuracy_difference_ci95_pp']
        lines.append(f"| {label} | {x['accuracy_difference_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines+=['', 'Intervals use 10,000 paired bootstrap draws, clustering AIME seeds by problem and RULER by task. Nonsignificance is not noninferiority. summary.json includes N,C,N/C,T,S/N,S,W, GLOBAL/LOCAL/overall physical counts, first-canvas and initial/refresh/held/carried work. attention_latency.json reports audit refresh/held timings; paired_uncertainty.json reports item-clustered accuracy and clean time/call ratios. Ratios above 1 indicate more time/calls. Sparsity and fewer calls alone are not speed evidence.','', 'The output-preserving execution change reuses only unchanged prefix projections within a canvas and changes CTA placement. Shared-routing and direct-paged candidates were slower and remain preserved negatives. Held execution uses the shared resident consumer without selection, metadata rebuild or KV copy; workload/mask skew is reported rather than presented as a method-specific kernel difference.','', 'Preserved HumanEval and prior AIME/MATH/RULER results: [completed earlier report](../../optimized_refresh_results_20261023/attempt003/summary.md).']
    (dest/'summary.md').write_text('\n'.join(lines)+'\n')
    (dest/'README.md').write_text('# Completed authorized follow-up\n\nSee [summary.md](summary.md) for official scores, clean timing, physical work and uncertainty. source_identity.json pins every panel and historical reference; verification.json records receipt checks. LongBench-v2 remains cancelled and unstarted. The frozen original four-panel campaign is preserved as partial/scope-superseded.\n')
    reservations=[]
    for panel in active:
        for path in (root/panel).rglob('receipt.json'):
            row=base.read(path)
            if 'worker_reservation_seconds' in row:reservations.append(dict(path=str(path.relative_to(base.ROOT)),seconds=row['worker_reservation_seconds']))
    base.write(dest/'resource_accounting.json',dict(worker_reservation_seconds=sum(x['seconds'] for x in reservations),workers=reservations,meaning='process elapsed reservations including engine loading, compilation and warm-up; not active GPU time'))
    base.write(dest/'report_identity.json',dict(source_sha256=base.sha(Path(__file__)),campaign_protocol_sha256=base.sha(root/'protocol.json'),scope_change=scope))
    base.write(dest/'final_complete.json',dict(status='complete_authorized_scope' if scope else 'complete',utc=base.utc(),campaign_protocol_sha256=base.sha(root/'protocol.json'),summary_sha256=base.sha(dest/'summary.json'),panels=list(summaries),scope_change=scope))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.study.resolve(),a.out.resolve())
