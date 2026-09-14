"""Read-only reanalysis of original shards and calibration margins."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dllm.evaluation.ruler.io import read_jsonl, write_json
from dllm.evaluation.ruler.official import score_predictions
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.report import EXPECTED_CONDITIONS, _read_csv_rows
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.metrics import aggregate_routing_stats, positional_token_id_agreement, paired_bootstrap_ci
from experiments.diffusion_gemma_solattn_vs_blasst_ruler16k.calibration import candidate_lambdas, _loads_line
from .runner import BASE, ROOT, RULER


def history(root: Path = ROOT):
    dense = read_jsonl(BASE / "dense/predictions.jsonl")
    dense_map = {r['sample_id']: r for r in dense}
    baseline_config=json.loads((BASE/'dense/run_config.json').read_text())
    eager_map={r['sample_id']:r for r in read_jsonl(root/'dense_eager/predictions.jsonl')} if (root/'dense_eager/summary.json').exists() else {}
    results = []
    paths = [(name, BASE / name) for name in EXPECTED_CONDITIONS]
    paths += [(name, root / name) for name in ('dense_eager', 'lambda1_forward', 'lambda1_reverse') if (root / name / 'summary.json').exists()]
    for name, directory in paths:
        rows = read_jsonl(directory / 'predictions.jsonl')
        if len(rows) != 50 or len({r['sample_id'] for r in rows}) != 50:
            raise ValueError(f'incomplete/duplicate {name}')
        for row in rows:
            reference = dense_map[row['sample_id']]
            for key in ('prompt', 'inference_seed', 'outputs'):
                assert row[key] == reference[key], (name, key)
        primary = [r for r in rows if r['task'] != 'vt']
        scores, accuracy = score_predictions(primary, RULER)
        all_scores, _ = score_predictions(rows, RULER)
        delta = [score_predictions([r], RULER)[1] - score_predictions([dense_map[r['sample_id']]], RULER)[1] for r in primary]
        stats = []
        for row in primary:
            if row.get('routing_stats'):
                stats.extend(row['routing_stats'].get('per_call', [row['routing_stats']]))
        if not stats:
            stats = [r for r in _read_csv_rows(directory / 'attention_stats/per_step.csv') if r.get('benchmark') != 'vt']
        routing = aggregate_routing_stats(stats)
        if name.startswith('sol_'):
            for scope in ('overall','local','global'):
                routing[scope]['valid_qk_element_sparsity'] = None
        if name in ('dense','dense_eager'):
            for scope in ('overall','local','global'):
                routing[scope]['retained_dense_attention_mass'] = 1.
        config = json.loads((directory / 'run_config.json').read_text())
        for key in ('model_path','revision','resolved_model_revision','precision','temperature','block_size','steps','threshold','generation_extra','max_new_tokens','context_length','native_canvas_length'):
            assert config.get(key)==baseline_config.get(key),(name,key)
        if name.startswith('lambda1_'):
            assert config['blasst_lambda']==1.
            assert config['blasst_policy']=={'local_blasst_lambda':1.,'global_blasst_lambda':1.}
            assert config['q_tile_size']==config['kv_tile_size']==64
        result = dict(condition=name, primary_n=len(primary), accuracy=accuracy, per_task=scores, vt_accuracy=all_scores['vt'],
                      delta_vs_dense=float(np.mean(delta)), delta_ci95=list(paired_bootstrap_ci(delta)), routing=routing,
                      token_agreement=float(np.mean([positional_token_id_agreement(dense_map[r['sample_id']]['completion_tokens'],r['completion_tokens']) for r in primary])),
                      exact_match=float(np.mean([dense_map[r['sample_id']]['completion_tokens']==r['completion_tokens'] for r in primary])),
                      backend=config.get('attention_backend'), policy=config.get('blasst_policy'))
        result['blasst_mask_semantics']=config.get('blasst_mask_semantics', 'legacy_row_mask' if config.get('attention_backend')=='blasst-reference' else 'not_applicable')
        if eager_map:
            result['token_agreement_vs_eager']=float(np.mean([positional_token_id_agreement(eager_map[r['sample_id']]['completion_tokens'],r['completion_tokens']) for r in primary]))
            result['exact_match_vs_eager']=float(np.mean([eager_map[r['sample_id']]['completion_tokens']==r['completion_tokens'] for r in primary]))
        results.append(result)
    policy = json.loads((BASE/'calibration/blasst_policy.json').read_text())
    fit = {t:[{k:v for k,v in row.items() if k not in ('provenance',)} for row in policy[t]] for t in ('local','global')}
    paired = {}
    for left,right in [('sol_gaussian_s50','blasst_calibrated_s75'), ('lambda1_forward','lambda1_reverse'), ('dense_eager','lambda1_forward'), ('dense_eager','lambda1_reverse')]:
        directories = dict(paths)
        if left not in directories or right not in directories:
            continue
        a={r['sample_id']:r for r in read_jsonl(directories[left]/'predictions.jsonl') if r['task']!='vt'}
        b={r['sample_id']:r for r in read_jsonl(directories[right]/'predictions.jsonl') if r['task']!='vt'}
        delta=[score_predictions([b[k]],RULER)[1]-score_predictions([a[k]],RULER)[1] for k in a]
        paired[right+' minus '+left]=dict(delta=float(np.mean(delta)),ci95=list(paired_bootstrap_ci(delta)),n=len(delta))
    payload = dict(primary_tasks=['niah_multikey_1','niah_multivalue','niah_multiquery','fwe'], conditions=results, original_calibration=fit, paired_comparisons=paired)
    write_json(root/'historical_reanalysis.json', payload)
    for r in results:
        s=r['routing']['overall']
        print(r['condition'], 'accuracy',round(r['accuracy'],4),'tiles',round(s['full_tile_sparsity'],4), 'elements',s['valid_qk_element_sparsity'], 'mass',s['retained_dense_attention_mass'], 'agreement',round(r['token_agreement'],4),flush=True)
    return payload


def calibration_grid(root: Path = ROOT):
    """One streaming scan, strict physical threshold via sorted max margins."""
    policy = json.loads((BASE/'calibration/blasst_policy.json').read_text())
    grid = sorted(set(candidate_lambdas()) | {1., .999999} | {float(row['lambda']) for typ in ('local','global') for row in policy[typ]})
    logs = np.log(grid)
    totals = defaultdict(lambda: dict(eligible=0, traces=0, skipped=np.zeros(len(grid), dtype=np.int64), record_tiles=0, strict_record_tiles=0, tie_only_tiles=0, first_valid_tiles=0))
    path=BASE/'calibration/margin_traces.jsonl'
    with path.open('rb') as handle:
        for index,line in enumerate(handle):
            row = _loads_line(line)
            margins=np.asarray(row['margins'],dtype=np.float64)
            valid=np.asarray(row['valid_rows'],dtype=bool)
            eligible=valid.any(0)
            maxima=np.where(valid,margins,-np.inf).max(0)
            threshold=np.sort(maxima[eligible])
            skipped=np.searchsorted(threshold,logs,side='left')
            records=eligible & (maxima>=0)
            strict=eligible & (maxima>0)
            first=(valid & (np.cumsum(valid,axis=1)==1)).any(0)
            for scope in ('all_tasks', 'non_vt') if row.get('benchmark')!='vt' else ('all_tasks',):
                target=totals[scope+'|'+row['attention_type']]
                target['eligible']+=int(eligible.sum()); target['traces']+=1; target['skipped']+=skipped
                target['record_tiles']+=int(records.sum());target['strict_record_tiles']+=int(strict.sum())
                target['tie_only_tiles']+=int((records & ~strict).sum());target['first_valid_tiles']+=int(first.sum())
            if index%10000==0:
                print('calibration traces',index,flush=True)
    output={}
    for key,counts in totals.items():
        counts['grid']=[dict(lambda_value=value, sparsity=int(skip)/counts['eligible'],skipped_tiles=int(skip)) for value,skip in zip(grid,counts.pop('skipped'))]
        counts['nearest_grid_targets']={str(s):min(counts['grid'],key=lambda x:abs(x['sparsity']-s)) for s in (.25,.5,.75,.9)}
        output[key]=counts
    write_json(root/'calibration_grid_audit.json', output)
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    parser.add_argument('--calibration',action='store_true')
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    calibration_grid(args.output_dir) if args.calibration else history(args.output_dir)
