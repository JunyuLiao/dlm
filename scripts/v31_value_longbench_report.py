"""Matched, sanitized reporting from complete clean and audit target stages."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from v31_value_longbench_pipeline import ARMS
from v31_value_summarize import aggregate, load, paired


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell_key(row):
    return row['dataset'],row['index'],row['panel_seed'],row['repeat']


def stage(study, path, purpose):
    base=Path(path)
    marker=json.loads((base/'stage_complete.json').read_text())
    if marker['status']!='complete' or marker['arms']!=len(ARMS) or marker['cells_per_arm']!=503:
        raise ValueError('complete 503-cell all-arm stage required')
    result={}
    for arm in ARMS:
        index=json.loads((base/arm/'completed_shards.json').read_text())
        configs,records=[],[]
        for relative in index['attempts']:
            config,rows=load(study/relative)
            if config['arm']!=arm or config['purpose']!=purpose:
                raise ValueError('arm or pass identity mismatch')
            if configs and any(config[field]!=configs[0][field] for field in
                ('source_sha256','runtime_settings','model_revision','threshold','manifests_sha256')):
                raise ValueError('shard source or protocol mismatch')
            configs.append(config)
            records.extend(rows)
        if len(records)!=503 or len({cell_key(r) for r in records})!=503:
            raise ValueError('missing or duplicated target cells')
        scoring=base/arm/'scoring/official.official.json'
        scores=json.loads(scoring.read_text())
        if set(scores)!={arm} or len(scores[arm])!=503:
            raise ValueError('official scoring coverage mismatch')
        accuracy={}
        for key,value in scores[arm].items():
            ds,idx,seed,repeat=key.split('|')
            accuracy[(ds,int(idx),int(seed),int(repeat))]=bool(value)
        if set(accuracy)!={cell_key(r) for r in records}:
            raise ValueError('scorer and generation cell mismatch')
        result[arm]=dict(config=configs[0],records=records,accuracy=accuracy,
            summary=json.loads((base/arm/'scoring/official.summary.json').read_text()),
            scoring_sha256=digest(scoring),source_attempts=index['attempts'])
    return result


def accuracy_pair(reference,other,indices):
    keys=sorted(k for k in reference if k[1] in indices)
    if not keys or set(reference)!=set(other):
        raise ValueError('empty or unmatched scored cohort')
    values=np.array([[reference[k],other[k]] for k in keys],dtype=np.float64)
    groups=sorted({k[:2] for k in keys})
    clusters=np.array([[sum(values[i,0] for i,k in enumerate(keys) if k[:2]==g),
                        sum(values[i,1] for i,k in enumerate(keys) if k[:2]==g),
                        sum(k[:2]==g for k in keys)] for g in groups])
    rng=np.random.default_rng(1729)
    draws=clusters[rng.integers(len(groups),size=(10000,len(groups)))].sum(1)
    delta=(draws[:,1]-draws[:,0])/draws[:,2]
    only_reference=int(((values[:,0]==1)&(values[:,1]==0)).sum())
    only_other=int(((values[:,0]==0)&(values[:,1]==1)).sum())
    discordant=only_reference+only_other
    p=min(1.,2*sum(math.comb(discordant,i) for i in range(min(only_reference,only_other)+1))/2**discordant) if discordant else 1.
    return dict(cells=len(keys),question_clusters=len(groups),
        reference_correct=int(values[:,0].sum()),correct=int(values[:,1].sum()),
        reference_accuracy=float(values[:,0].mean()),accuracy=float(values[:,1].mean()),
        accuracy_delta=float((values[:,1]-values[:,0]).mean()),
        accuracy_delta_cluster_ci95=np.quantile(delta,[.025,.975]).tolist(),
        reference_only_correct=only_reference,other_only_correct=only_other,
        mcnemar_exact_p=p,interpretation='descriptive comparison; nonsignificance is not noninferiority')


def phases(records):
    rows={}
    for record in records:
        audit=record['receipts']['adapter']
        for name,value in audit['value_phase_tiles'].items():
            row=rows.setdefault(name,dict(global_calls=0,global_eligible_tiles=0,global_kept_tiles=0,
                local_calls=0,local_eligible_rectangles=0,local_skipped_tiles=0))
            row['global_calls']+=value['calls']
            row['global_eligible_tiles']+=value['eligible_tiles']
            row['global_kept_tiles']+=value['kept_tiles']
        for name,value in audit['native_local_phase_rectangles'].items():
            row=rows.setdefault(name,dict(global_calls=0,global_eligible_tiles=0,global_kept_tiles=0,
                local_calls=0,local_eligible_rectangles=0,local_skipped_tiles=0))
            row['local_calls']+=value['calls']
            row['local_eligible_rectangles']+=value['eligible_rectangles']
            row['local_skipped_tiles']+=value['skipped_tiles']
    for row in rows.values():
        row['global_skipped_tiles']=row['global_eligible_tiles']-row['global_kept_tiles']
        row['global_sparsity']=row['global_skipped_tiles']/row['global_eligible_tiles'] if row['global_eligible_tiles'] else None
        row['local_sparsity']=0.
        row['overall_eligible_rectangles']=row['global_eligible_tiles']+row['local_eligible_rectangles']
        row['overall_sparsity']=row['global_skipped_tiles']/row['overall_eligible_rectangles'] if row['overall_eligible_rectangles'] else None
    return rows


def main():
    p=argparse.ArgumentParser()
    for name in ('audit','clean','thresholds','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=root/'results/v31_value_selectors_20261008'
    destination=Path(args.output)
    if destination.exists() or not destination.resolve().is_relative_to(study):
        raise ValueError('new canonical report directory required')
    clean=stage(study,args.clean,'clean')
    audit=stage(study,args.audit,'audit')
    protocol=json.loads((study/'protocol_original_s1_20261008.json').read_text())
    split=json.loads((root/'results/v31_20261003/panels/lbt_split.json').read_text())
    # The inherited split's schema is validated by its frozen source hash.
    if digest(root/'results/v31_20261003/panels/lbt_split.json')!=protocol['original_split_sha256']:
        raise ValueError('original split identity changed')
    if 'explore_indices' not in split or split['n']!=503:
        raise ValueError('unexpected inherited split schema')
    exploration=set(split['explore_indices'])
    holdout=set(range(503))-exploration
    cohorts=dict(full503=set(range(503)),original_exploration160=exploration,
        original_holdout343=holdout,
        original_holdout_without_pilot332=holdout-set(protocol['preliminary_diagnostic_deviation']['exposed_holdout_indices']))
    control=clean['current_v31_control']['config']
    calibration=json.loads(Path(args.thresholds).read_text())
    if calibration['status']!='complete':
        raise ValueError('completed frozen calibration required')
    results={}
    for arm in ARMS:
        item,measured=clean[arm],audit[arm]
        config=item['config']
        for candidate in (config,measured['config']):
            if any(candidate[k]!=control[k] for k in ('source_sha256','model_revision','manifests_sha256',
                'model_source_inventory_sha256','host_fingerprint','gpu_identity')):
                raise ValueError('source, model or pool fairness violation')
            common_runtime=lambda c:{k:v for k,v in c['runtime_settings'].items()
                if k not in ('VALUE_AUDIT','VALUE_CLEAN_TIMING')}
            if common_runtime(candidate)!=common_runtime(control):
                raise ValueError('runtime or inherited reuse setting mismatch')
            expected=calibration['thresholds'].get(arm)
            if candidate['threshold']!=expected:
                raise ValueError('target threshold differs from frozen development threshold')
        if any(not r['value_clean_timing'] or r['value_audit'] or (r.get('receipts') or {}).get('timing') for r in item['records']):
            raise ValueError('instrumented timing record')
        clean_by={cell_key(r):r for r in item['records']}
        audit_by={cell_key(r):r for r in measured['records']}
        parity={field:sum(clean_by[k][field]!=audit_by[k][field] for k in clean_by)
            for field in ('output_hash','output_tokens','denoise_forwards','canvases','finish_reason')}
        counts=aggregate(measured['config'],measured['records'])
        counts['phase_global_local_overall']=phases(measured['records'])
        counts['density_source']='separate target audit; trajectory agreement reported explicitly'
        results[arm]=dict(clean=aggregate(config,item['records']),audit=counts,
            clean_audit_mismatched_cells=parity,official_summary=item['summary'],
            clean_gpu_seconds=sum(json.loads((study/x/'receipt.json').read_text())['gpu_seconds'] for x in item['source_attempts']),
            audit_gpu_seconds=sum(json.loads((study/x/'receipt.json').read_text())['gpu_seconds'] for x in measured['source_attempts']),
            comparisons={})
        for reference in ARMS[:4]:
            results[arm]['comparisons'][reference]=dict(
                timing=paired(clean[reference]['records'],item['records']),
                accuracy={name:accuracy_pair(clean[reference]['accuracy'],item['accuracy'],indices)
                          for name,indices in cohorts.items()})
    reference_sparsity=results['current_v31_control']['audit']['global_sparsity']
    for arm in ARMS:
        results[arm]['audit']['global_sparsity_delta_from_target_control']=results[arm]['audit']['global_sparsity']-reference_sparsity
    result=dict(status='complete LongBench panel; other target suites tracked separately',
        cells_per_arm=503,seeds=[1],source_commit=control['source_commit'],
        source_sha256=control['source_sha256'],protocol_sha256=digest(study/'protocol_original_s1_20261008.json'),
        thresholds_sha256=digest(args.thresholds),model_source_inventory_sha256=digest(study/'model_source_fingerprints.json'),
        metric='official LongBench-v2 0shot_think, thinking on, total cap16384',
        uncertainty='10000 paired question-cluster bootstrap draws, seed1729; no noninferiority claim',
        exposure='historically examined pool; original32 development at seeds1,2;11 disclosed pilot holdout exposures',
        timing='synchronized per engine step, one identical first-cell warm-up per shard; rotated arm order; no new timed CUDA graphs',
        denominator='actual decode calls on Q128 x KV64 rectangle geometry, native LOCAL window intersections; not CUDA CTA count',
        native_dense_attention_timing='not available inside native FULL graph; native dense geometry audit does not change consumer',
        arms=results)
    destination.mkdir(parents=True)
    (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['Complete independent LongBench-v2 panel: 503 official seed-1 cells per arm.\n',
        'Clean timing and instrumented audits are separate. Sparsity comes from the audit pass; their output and trajectory agreement is in summary.json. The inherited reuse system is Yuhan\'s work. These Gaussian32 selectors are this study\'s integration, distinct from Junyu\'s prior Gaussian32 and C_gate families.\n',
        '| Arm | Correct /503 | W mean s | S mean s | N | C | T | N/C | S/N ms | GLOBAL skip | Overall skip | W speedup vs FULL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm,item in results.items():
        c,a=item['clean'],item['audit']
        comparison=item['comparisons']['dense_full_fix51994']
        acc=comparison['accuracy']['full503']
        lines.append(f"| {arm} | {acc['correct']} | {c['W_mean_s']:.3f} | {c['S_mean_s']:.3f} | {c['N']} | {c['C']} | {c['T']} | {c['N_per_C']:.3f} | {c['S_per_N_ms']:.3f} | {100*a['global_sparsity']:.3f}% | {100*a['overall_rectangle_sparsity']:.3f}% | {comparison['timing']['W_speedup']:.3f} |")
    lines+=['\nLOCAL sparsity is zero. Overall sparsity uses actual GLOBAL support plus native LOCAL window-intersecting rectangles; it excludes prefill and commits and does not include the additional selector QK work. That extra work, phases, selection/discovery time, peak memory, and candidate counts are reported separately in summary.json. S/N is an amortized per-step cost.\n',
        'Paired accuracy differences and cluster intervals are reported against FULL, PIECEWISE, all-kept and the current control for full503, original exploration160, original holdout343, and the332-item sensitivity subset. This is a historically examined pool. Nonsignificance does not establish noninferiority. Offline full-dimensional diagnostics and kernel qualification are separate evidence.\n']
    (destination/'summary.md').write_text('\n'.join(lines))
    (destination/'panel_complete.json').write_text(json.dumps(dict(status='complete',cells_per_arm=503,arms=len(ARMS),
        summary_sha256=digest(destination/'summary.json'),source_commit=control['source_commit']),indent=2)+'\n')


if __name__=='__main__':
    main()
