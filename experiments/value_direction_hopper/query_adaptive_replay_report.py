"""Summarize correlated, task-balanced same-state replay diagnostics."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .experiment import atomic


def report(root):
    root=Path(root);source=root/'diagnostic_replays/shards/matched_dense'
    samples=[json.loads(path.read_text()) for path in sorted(source.glob('*.json'))]
    if not samples:return None
    snapshots=[(sample,snap) for sample in samples for snap in sample['snapshots']]
    methods=sorted({name for _,snap in snapshots for name in snap['methods'] if name!='dense'})
    result={}
    for method in methods:
        rows=[(sample,snap,r) for sample,snap in snapshots for r in snap['position_records'] if r['method']==method]
        entries=[snap['methods'][method] for _,snap in snapshots if method in snap['methods']]
        if not entries:continue
        granularity=[g for entry in entries for g in entry['granularity']]
        skipped=sum(e['skipped_tiles'] for e in entries);eligible=sum(e['eligible_tiles'] for e in entries)
        inside=[r for _,_,r in rows if r['inside_returned_answer']]
        outside=[r for _,_,r in rows if not r['inside_returned_answer']]
        sensitivity=np.array([r['sensitivity'] for _,_,r in rows]);median=float(np.median(sensitivity))
        low=[r for _,_,r in rows if r['sensitivity']<median]
        high=[r for _,_,r in rows if r['sensitivity']>=median]
        result[method]=dict(snapshots=len(entries),positions=len(rows),physical_sparsity=skipped/max(1,eligible),
            top1_disagreement=float(np.mean([r['top1_disagree'] for _,_,r in rows])),
            acceptance_disagreement=float(np.mean([r['accepted']!=r['dense_accepted'] for _,_,r in rows])),
            confidence_loss=float(np.mean([r['confidence_loss'] for _,_,r in rows])),
            returned_answer_disagreement=float(np.mean([r['top1_disagree'] for r in inside])) if inside else None,
            outside_answer_disagreement=float(np.mean([r['top1_disagree'] for r in outside])) if outside else None,
            high_sensitivity_disagreement=float(np.mean([r['top1_disagree'] for r in high])) if high else None,
            low_sensitivity_disagreement=float(np.mean([r['top1_disagree'] for r in low])) if low else None,
            mean_sampled_attention_error=float(np.mean([e['sampled_attention_relative_error'] for e in entries])),
            mean_fixed_temperature_confidence=float(np.mean([e['fixed_temperature_confidence'] for e in entries])),
            mean_actual_processed_entropy=float(np.mean([e['mean_entropy'] for e in entries])),
            mixed_query_tiles=sum(g['mixed_query_tiles'] for g in granularity),
            granularity_eligible=sum(g['eligible'] for g in granularity),
            reference_mask_disagreement=sum(g['reference_disagreement'] for g in granularity),
            newly_retained_vs_unweighted=sum(e.get('physical_tiles_newly_retained_vs_unweighted',0) for e in entries),
            newly_skipped_vs_unweighted=sum(e.get('physical_tiles_newly_skipped_vs_unweighted',0) for e in entries))
    audit=dict(samples=len(samples),snapshots=len(snapshots),methods=result,
        caveat='Correlated replay positions on repeatedly examined prompts; original decoder inputs kept identical across replays; matched-kernel dense is reference, not native SDPA.')
    atomic(root/'diagnostic_replays/analysis.json',audit)
    lines=['# Same-state query-sensitivity diagnostics','',
        f'{len(samples)} task-balanced RULER4K prompts, {len(snapshots)} step-2/3 snapshots. '
        'Each replay uses identical canvas, self-conditioning and prefix cache. '
        'The matched-kernel dense forward is returned to the native decoder, so alternate replays cannot create a different diagnostic trajectory.','',
        '| Method | Physical sparsity | Top-1 disagreement | Acceptance-set disagreement | Confidence loss | Sampled attention error | Mixed-query tiles | Reference/CUDA tile mismatch |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,item in result.items():
        lines.append(f"| {method} | {item['physical_sparsity']:.1%} | {item['top1_disagreement']:.1%} | "
            f"{item['acceptance_disagreement']:.1%} | {item['confidence_loss']:+.4f} | "
            f"{item['mean_sampled_attention_error']:.3f} | {item['mixed_query_tiles']} | "
            f"{item['reference_mask_disagreement']}/{item['granularity_eligible']} |")
    lines+=['','High-sensitivity versus low-sensitivity and returned-answer versus outside-answer damage rates are in `analysis.json`. '
        'Row-vote diagnostics use a small FP32 reference and must be qualified by their mask disagreement with the production kernel. '
        'These are replay associations, not evidence that selection caused downstream accuracy differences.','']
    (root/'diagnostic_replays/report.md').write_text('\n'.join(lines))
    return audit


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'query_adaptive_v3')
    args=parser.parse_args();report(args.root)
