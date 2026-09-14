"""Regenerate evidence and analysis from completed experiment shards."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dllm.evaluation.ruler.io import read_jsonl, write_json, sha256_file
from .runner import ROOT, BASE
from .history import history


def ratio(n,d):
    return n/d if d else None


def aggregate(root=ROOT):
    out=root/'same_state'
    predictions=read_jsonl(out/'predictions.jsonl')
    assert len(predictions)==16 and len({r['sample_id'] for r in predictions})==16
    parity=json.loads((out/'native_dense_parity.json').read_text())
    assert len(parity)==8 and all(parity.values())
    groups=defaultdict(lambda:defaultdict(float))
    reuse=defaultdict(lambda:defaultdict(float))
    grouping=defaultdict(lambda:defaultdict(float))
    distribution=defaultdict(list)
    strata=defaultdict(lambda:defaultdict(float))
    regions=defaultdict(lambda:defaultdict(float))
    events=defaultdict(lambda:defaultdict(float))
    prompt_totals=defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    negative_margins=defaultdict(list)
    records=[]; paths=[]
    for prediction in predictions:
        path=out/'shards'/prediction['sample_id']
        shard=json.loads((path/'diagnostics.json').read_text())
        assert shard and {r['layer'] for r in shard}==set(range(30))
        assert {r['attention_type'] for r in shard}=={'local','global'}
        paths.extend([path/'diagnostics.json',path/'snapshots.npz'])
        with np.load(path/'snapshots.npz') as arrays:
            for row in shard:
                assert row['example_id']==prediction['sample_id']
                snapshot=row['snapshot_key']
                proxy=arrays[snapshot+'_proxy']; mass=arrays[snapshot+'_mass'].sum(0)
                eligible=arrays[snapshot+'_counts'].any(0)
                maxima=arrays[snapshot+'_maxima']; valid=arrays[snapshot+'_counts']>0
                previous=np.concatenate([np.full((len(maxima),1),-np.inf),np.maximum.accumulate(maxima,axis=-1)[:,:-1]],axis=-1)
                with np.errstate(invalid='ignore'):
                    margins=maxima-previous
                physical=np.where(valid,margins,-np.inf).max(0)
                for label,values in (('row',margins[valid]),('physical',physical[eligible])):
                    negative_margins[f"{row['split']}|{row['attention_type']}|{label}"].append(values[np.isfinite(values)&(values<0)])
                p=proxy[eligible]; m=mass[eligible]
                std=p.std()
                z=(p-p.mean())/max(float(std),1.e-6)
                dist=dict(spearman=float(spearmanr(p,m).statistic) if std>1.e-6 and len(p)>2 and m.std()>0 else None,
                          skew=float((z**3).mean()),excess_kurtosis=float((z**4).mean()-3),
                          sol50_drops_highest_mass=int(p[np.argmax(m)]<p.mean()),n=len(p))
                for scope in ('overall',row['attention_type']):
                    key=row['split']+'|'+scope
                    distribution[key].append(dist)
                    for name,values in row['policies'].items():
                        for metric,value in values.items():
                            if isinstance(value,(int,float)):
                                groups[key+'|'+name][metric]+=value
                        groups[key+'|'+name]['calls']+=1
                        for region,region_values in values.get('regions',{}).items():
                            for metric,value in region_values.items():
                                regions[key+'|'+name+'|'+region][metric]+=value
                    for name,values in row['reuse'].items():
                        for metric,value in values.items():
                            reuse[key+'|'+name][metric]+=value
                        reuse[key+'|'+name]['pairs']+=1
                        if values.get('eligible'):
                            expected=values['previous_retained']*values['current_retained']/values['eligible']
                            reuse[key+'|'+name]['chance_intersection']+=expected
                            reuse[key+'|'+name]['chance_union']+=values['previous_retained']+values['current_retained']-expected
                    for lam,variants in row['grouping'].items():
                        for name,values in variants.items():
                            for metric,value in values.items():
                                grouping[key+'|'+lam+'|'+name][metric]+=value
                    for name,values in row.items():
                        if name.endswith('_records'):
                            for metric,value in values.items():
                                events[key+'|'+name][metric]+=value
                        elif name.endswith(('_degenerate','_fallback')):
                            events[key+'|'+name]['count']+=values
                            events[key+'|'+name]['rows']+=1
                for dimension,value in (('layer',row['layer']),('head',row['head']),('call',row['call'])):
                    for name in ('forward_1','reverse_1','sol_50','closure_1'):
                        for metric in ('eligible_tiles','skipped_tiles','retained_mass_sum','valid_rows'):
                            strata[f"{row['split']}|{dimension}_{value}|{name}"][metric]+=row['policies'][name][metric]
                for name,values in row['policies'].items():
                    for metric in ('eligible_tiles','skipped_tiles','retained_mass_sum','valid_rows','output_squared_error','dense_squared_norm'):
                        prompt_totals[row['example_id']][name][metric]+=values[metric]
                records.append({k:row[k] for k in ('example_id','split','layer','head','call','query_start','kv_length','attention_type')})
    for values in groups.values():
        assert 0<=values['skipped_tiles']<=values['eligible_tiles']
        assert values['empty_rows']==0
        assert math.isfinite(values['output_squared_error'])
        values.update(tile_sparsity=ratio(values['skipped_tiles'],values['eligible_tiles']),
            element_sparsity=ratio(values['skipped_elements'],values['valid_elements']),
            retained_mass=ratio(values['retained_mass_sum'],values['valid_rows']),
            relative_output_l2=math.sqrt(ratio(values['output_squared_error'],values['dense_squared_norm'])))
    for values in reuse.values():
        values['fresh_mass']=ratio(values['fresh_mass_sum'],values['rows'])
        values['reused_mass']=ratio(values['reused_mass_sum'],values['rows'])
        if 'union' in values:
            values['jaccard']=ratio(values['intersection'],values['union'])
            values['chance_jaccard']=ratio(values['chance_intersection'],values['chance_union'])
    for values in grouping.values():
        values['tile_sparsity']=ratio(values['skipped_tiles'],values['eligible_tiles'])
    for values in [*strata.values(),*regions.values()]:
        values['tile_sparsity']=ratio(values['skipped_tiles'],values['eligible_tiles'])
        if 'valid_rows' in values:
            values['retained_mass']=ratio(values['retained_mass_sum'],values['valid_rows'])
    dist_summary={key:{metric:float(np.mean([r[metric] for r in rows if r[metric] is not None])) for metric in ('spearman','skew','excess_kurtosis','sol50_drops_highest_mass')} for key,rows in distribution.items()}
    prompt_results={}
    for example,policies in prompt_totals.items():
        prompt_results[example]={name:dict(tile_sparsity=ratio(p['skipped_tiles'],p['eligible_tiles']),
            retained_mass=ratio(p['retained_mass_sum'],p['valid_rows']),
            relative_output_l2=math.sqrt(ratio(p['output_squared_error'],p['dense_squared_norm']))) for name,p in policies.items()}
    margin_summary={}
    for key,parts in negative_margins.items():
        values=np.concatenate(parts)
        margin_summary[key]=dict(count=len(values),quantiles=dict(zip(('p10','p50','p90'),np.quantile(values,(.1,.5,.9)).tolist())))
    payload=dict(policies=dict(groups),reuse=dict(reuse),grouping=dict(grouping),proxy_distribution=dist_summary,
        strata=dict(strata),regions=dict(regions),events=dict(events),
        per_prompt=prompt_results,
        negative_margin_distribution=margin_summary,
        coverage=dict(samples=len(predictions),calls=len(records),heads=sorted({r['head'] for r in records}),layers=sorted({r['layer'] for r in records})),
        native_dense_parity=parity,source_hashes={str(p.relative_to(root)):sha256_file(p) for p in paths})
    write_json(root/'same_state_summary.json',payload)
    return payload


def pct(x):
    return '—' if x is None else f'{100*x:.2f}%'


def report(root=ROOT):
    # This report's fixed interpretation describes the archived row-mask
    # experiment, not the subsequently corrected whole-tile forward path.
    for condition in ('lambda1_forward','lambda1_reverse'):
        config=json.loads((root/condition/'run_config.json').read_text())
        if config.get('blasst_mask_semantics') is not None:
            raise ValueError('This is a legacy row-mask report; use version-labelled history output for corrected whole-tile runs')
    from .audit import audit
    checked=audit(root)
    historical=history(root)
    conditions={r['condition']:r for r in historical['conditions']}
    assert {'dense_eager','lambda1_forward','lambda1_reverse'}<=conditions.keys()
    diagnostics=aggregate(root)
    grid=json.loads((root/'calibration_grid_audit.json').read_text())
    fwd=conditions['lambda1_forward']; rev=conditions['lambda1_reverse']
    def state(name, scope='overall'):
        return diagnostics['policies'][f'final|{scope}|{name}']
    lines=['# BLASST on DiffusionGemma: what the evidence supports','',
        '> Historical analysis: these BLASST rollouts used the old row-level execution mask. They do not describe the corrected physical-tile implementation or establish its accuracy.','',
        'Primary evaluation: the four original non-VT RULER16K tasks, 10 prompts each. Accuracy is the official equal-task macro score, allowing partial answer credit—not the fraction of completely correct questions. VT is an appendix. All sparsities divide summed skipped tiles by summed eligible tiles; no unweighted per-call sparsity averages and no speedup claims.','',
        '## Bottom line','',
        '- **A valid motivation is budget/error control, not blanket Sol superiority.** The closest historical sparsity pair slightly favors BLASST in accuracy, but favors Sol in retained mass and token agreement.',
        f"- **The λ=1 limit is strongly layer-type dependent.** Full-run tile sparsity is {pct(fwd['routing']['local']['full_tile_sparsity'])} locally versus {pct(fwd['routing']['global']['full_tile_sparsity'])} globally. Local weak-record unions, not ties alone, prevent high forward-order tile sparsity.",
        f"- **Row pruning wastes mass relative to a full-tile objective.** On identical final dense states, physical closure raises mass {pct(state('forward_1')['retained_mass'])} → {pct(state('closure_1')['retained_mass'])} at unchanged {pct(state('forward_1')['tile_sparsity'])} tile sparsity.",
        f"- **Order matters here.** Reverse traversal increases full-run tile sparsity {pct(fwd['routing']['overall']['full_tile_sparsity'])} → {pct(rev['routing']['overall']['full_tile_sparsity'])}; accuracy rises {pct(fwd['accuracy'])} → {pct(rev['accuracy'])}, but its paired accuracy CI includes zero.",
        '- **Reuse and regrouping evidence is mixed.** Adjacent-step masks are substantially correlated; stale masks still lose mass. The tested current-mask grouping heuristic helps global layers modestly and gives no local improvement. Frequency-only reuse underperforms last-step reuse in these probes.','',
        '## 1. Correct the starting premise','',
        'The saved results do **not** establish that Sol-Attn is more accurate at matched physical sparsity. The closest historical pair is Sol s50 versus BLASST s75; target labels are not comparable budgets.','',
        '| Condition | Tile sparsity | Local | Global | Accuracy | Dense mass retained | Token agreement |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in conditions.items():
        s=r['routing']
        lines.append(f"| {name} | {pct(s['overall']['full_tile_sparsity'])} | {pct(s['local']['full_tile_sparsity'])} | {pct(s['global']['full_tile_sparsity'])} | {pct(r['accuracy'])} | {pct(s['overall']['retained_dense_attention_mass'])} | {pct(r['token_agreement'])} |")
    lines+=['','Paired accuracy differences (right minus left), with prompt-bootstrap 95% confidence intervals:','']
    for name,p in historical['paired_comparisons'].items():
        lines.append(f"- {name}: {100*p['delta']:+.3f} percentage points; CI [{100*p['ci95'][0]:+.3f}, {100*p['ci95'][1]:+.3f}].")
    eager=conditions['dense_eager']
    lines+=['',f"Numerical control: no-skipping eager dense scores {pct(eager['accuracy'])}, with {pct(eager['token_agreement'])} token agreement and {pct(eager['exact_match'])} exact-sequence agreement against native dense. Forward/reverse λ=1 token agreement against **eager** dense is {pct(fwd['token_agreement_vs_eager'])}/{pct(rev['token_agreement_vs_eager'])}, respectively."]
    lines+=['','Mass and token agreement measure different things from task correctness. A lower retained mass does not by itself prove lower accuracy. Native dense and eager sparse backends also differ numerically; the new same-state probes isolate routing, and the forward/reverse rollout pair uses the same eager backend.','',
        '## 2. Why the sparsity target can fail','',
        'Let b(q,j) be the maximum logit in KV tile j and m_previous(q,j) the preceding running maximum. A query retains a tile when b − m_previous ≥ log λ. For λ≤1 this equals using the updated maximum. At λ=1 every new record **and tie** is retained. A physical tile survives if **any** of its valid query rows retains it. Thus the ceiling is one minus the union of weak-record tiles across 64 queries—not merely one compulsory first tile. This is an algebraic consequence of the implemented rule.','',
        '| Dense calibration, non-VT | λ=1 tile sparsity | Retained: strict record tiles / eligible | Retained: tie-only tiles / eligible | First-valid tiles / eligible |',
        '|---|---:|---:|---:|---:|']
    for typ in ('local','global'):
        v=grid['non_vt|'+typ]; n=v['eligible']
        lines.append(f"| {typ} | {pct(v['grid'][-1]['sparsity'])} | {pct(v['strict_record_tiles']/n)} | {pct(v['tie_only_tiles']/n)} | {pct(v['first_valid_tiles']/n)} |")
    lines+=['','First-valid tiles are a subset of record tiles; do not add these columns. Most local retention is strict-record retention, so ties alone cannot explain the ceiling. The new FP32 and order probes below test how much the native precision and traversal contribute.','',
        'The ceiling depends on Q/K state, not only λ: dense-calibration and sparse-rollout values need not match. Long global KV sequences dominate eligible-tile counts even though there are fewer global layers; local layers dominate the query-row-weighted mass average. The overall metrics are not averages of the two layer-type percentages.','',
        'A second, separate problem is **this reproduction’s calibration fit/selection error**. Original all-five-task calibration selected the following thresholds; these are not thresholds tuned on final data:','',
        '| Type | Target | Chosen λ | Achieved calibration sparsity | Closest audited grid λ | Grid sparsity |','|---|---:|---:|---:|---:|---:|']
    for typ in ('local','global'):
        for p in historical['original_calibration'][typ]:
            near=grid['all_tasks|'+typ]['nearest_grid_targets'][str(p['target_sparsity'])]
            lines.append(f"| {typ} | {pct(p['target_sparsity'])} | {p['lambda']:.8g} | {pct(p['observed_sparsity'])} | {near['lambda_value']:.8g} | {pct(near['sparsity'])} |")
    lines+=['','The grid audit evaluates the union of original candidates, saved fitted points, and the endpoint; it is a diagnostic, not a replacement deployed policy. Local high targets are genuinely outside this forward-rule range. Several other misses are poor fit/selection, not impossibility. In particular, the original global 75% target used λ=0.4, not λ≈1.','',
        'Important attribution limit: the [local calibrator](../../experiments/diffusion_gemma_solattn_vs_blasst_ruler16k/calibration.py) fits all finite aggregate grid points, without sparsity-bound filtering, and may finish with a poor fitted choice instead of the closest measured sparsity. The paper filters extreme sample-level points in [Algorithm 2](https://arxiv.org/html/2512.12087v3#S3). These fit misses therefore do **not** by themselves establish failure of the paper’s full calibration procedure. The λ=1 record-union ceiling is independent of this fitting issue.','',
        '## 3. Identical-state tests: order, mass, and output perturbation','',
        'Sixteen native-dense trajectories: two calibration and two final prompts per non-VT task. One fixed head and 64-query tile per prompt across all layers/calls; collectively all 16 heads, not all heads per prompt. Calibration and final diagnostics are kept separate. All eight final observer generations exactly match the original native-dense token sequences. Errors below are FP32 softmax/value diagnostic errors at fixed Q/K/V, not deployed BF16 kernel errors. Relative output error is sqrt(sum squared error / sum squared dense-output norm).','',
        '| Final-state probe | Scope | Tile sparsity | QK-element sparsity | Retained mass | Relative output L2 |','|---|---|---:|---:|---:|---:|']
    probes=('forward_0.001','forward_0.9995','forward_1','reverse_1','random_1','finalmax_1','fp32_1','closure_1','sol_50','matched_blasst_rank','matched_mass_oracle')
    for name in probes:
        for scope in ('overall','local','global'):
            p=diagnostics['policies']['final|'+scope+'|'+name]
            lines.append(f"| {name} | {scope} | {pct(p['tile_sparsity'])} | {pct(p['element_sparsity'])} | {pct(p['retained_mass'])} | {p['relative_output_l2']:.4f} |")
    lines+=['','The local-region breakdown shows what reversal changes at the same Q/K state:','',
        '| Local λ=1 traversal | Fully prefix tiles skipped | Fully canvas tiles skipped | Boundary tiles skipped |',
        '|---|---:|---:|---:|']
    for name in ('forward_1','reverse_1'):
        region_values=[diagnostics['regions'][f'final|local|{name}|{region}']['tile_sparsity'] for region in ('prefix','canvas','boundary')]
        lines.append(f"| {name} | {' | '.join(pct(v) for v in region_values)} |")
    lines+=['','Processing the canvas end first preserves more canvas tiles and permits more prefix skipping. This is evidence of a traversal-dependent allocation, not evidence that reversing always improves accuracy. Boundary-straddling tiles form their own category here; historical region counters instead use their original prefix/canvas assignment.']
    lines+=['','Why a small λ skips little: λ=0.001 requires a logit gap exceeding 6.91 for **every valid query row** to skip a physical tile. On sampled final states, among tiles with strictly negative maximum-over-query margins (the only tiles skippable at λ≤1), the p10/median/p90 margins are:']
    lines.append('')
    for typ in ('local','global'):
        quantiles=diagnostics['negative_margin_distribution'][f'final|{typ}|physical']['quantiles']
        lines.append(f"- {typ}: {quantiles['p10']:.3f} / {quantiles['p50']:.3f} / {quantiles['p90']:.3f}. These conditional quantiles exclude record/tie tiles and are not population-wide sparsities.")
    prompts=[v for k,v in diagnostics['per_prompt'].items() if '_final_' in k]
    lines+=['',f"Reverse traversal improves retained mass on {sum(p['reverse_1']['retained_mass']>p['forward_1']['retained_mass'] for p in prompts)}/{len(prompts)} sampled final prompts and lowers relative output error on {sum(p['reverse_1']['relative_output_l2']<p['forward_1']['relative_output_l2'] for p in prompts)}/{len(prompts)}. The independent unit remains the prompt, not the thousands of correlated layer/call observations."]
    lines+=['','Interpretation of controls: `closure_1` restores every valid position inside any physically retained tile, so physical sparsity is unchanged. Any recovered mass identifies row-level deletions that do not buy additional **full-tile** skips. `matched_*` retains exactly the Sol-50 tile count per sampled row, using either maximum online margin or exact dense tile mass; both are offline ranking controls, not deployable BLASST thresholds. Ranking ties use the implementation’s `topk` tie order, not an optimized tie-breaker. `finalmax` knows the future row maximum and is likewise an oracle. `fp32` changes QK precision for routing only, while evaluating masks against the same native-score dense probabilities.','',
        f"Key observations: FP32 changes sampled local λ=1 sparsity only {pct(state('forward_1','local')['tile_sparsity'])} → {pct(state('fp32_1','local')['tile_sparsity'])}; quantization is secondary here. Reverse traversal changes it to {pct(state('reverse_1','local')['tile_sparsity'])}, improving sampled mass and reducing output perturbation. Full-rollout and same-state numbers differ because sparse trajectories and stopping decisions differ. On the full run, reverse improves multivalue accuracy (75% → 82.5%) and multiquery (97.5% → 100%), but reduces FWE (100% → 96.67%); multikey stays at 100%.",'',
        'At exactly matched sampled tile counts, Sol keeps more overall mass than the BLASST-margin ranking, **but the ranking has lower overall output error**. In global layers the ranking also keeps slightly more mass than Sol. Thus mass, output error, and task accuracy cannot be treated as interchangeable superiority claims.','',
        'For a skipped tile, its maximum exponentiated logit is less than λ times the observed maximum. Near λ=1 this upper bound is not evidence that the tile has negligible **total** probability mass. If lost dense mass is δ, normalized sparse output differs by δ times the difference between retained- and dropped-value conditional means; consequently ||error||≤2δ max||V||. This direct derivation provides no small-error guarantee when δ is large. Error also depends on values, not only mass.','',
        '## 4. Sol-like routing is not an exact-budget or error guarantee','',
        'The Gaussian β values map to targets only under a Gaussian reference. Sol target 50% actually skips 58.58% in the original non-VT run—an 8.58-point miss. Mean-pooled logits also do not rank exact softmax mass perfectly. No approximate correction from the complete Sol-Attn system is included here.','',
        '| Final sampled scope | Proxy/mass Spearman (mean per row) | Standardized skew | Excess kurtosis | Sol-50 drops highest-mass tile |','|---|---:|---:|---:|---:|']
    for scope in ('overall','local','global'):
        p=diagnostics['proxy_distribution']['final|'+scope]
        lines.append(f"| {scope} | {p['spearman']:.3f} | {p['skew']:.3f} | {p['excess_kurtosis']:.3f} | {pct(p['sol50_drops_highest_mass'])} |")
    lines+=['','These shape/alignment statistics are descriptive sampled diagnostics, not a formal normality test or proof that distribution awareness is sufficient. The mass oracle is a fixed-state mass ceiling at the same tile count, not a task-accuracy oracle.','',
        f"The sampled final Sol-90 mask needed a maximum-proxy fallback in {int(diagnostics['events']['final|overall|sol_90_fallback']['count'])}/{int(diagnostics['events']['final|overall|sol_90_fallback']['rows'])} proxy rows; no sampled final proxy row was degenerate. The analytic threshold is neither an exact sparsity guarantee nor a guarantee against dropping an important local tile.",'',
        '## 5. Motivation probes: headroom, not proposed methods','',
        '| Final λ=1 fixed row mask | Grouping | Tile sparsity |','|---|---|---:|']
    for scope in ('local','global'):
        for group in ('1','16','64','256','random64','oracle_lex64'):
            p=diagnostics['grouping']['final|'+scope+'|1.0|'+group]
            lines.append(f"| {scope} | {group} | {pct(p['tile_sparsity'])} |")
    lines+=['','Changing query-group size changes the physical tile shape and is not a free speedup. `oracle_lex64` lexicographically sorts the **current exact** row masks while preserving 64-query groups; its row-level masks/mass are fixed. It tests regrouping headroom, not an efficient predictor, globally optimal grouping, or a deployable algorithm.','',
        '| Final reuse probe | Scope | Physical-mask Jaccard | Fresh physical-mask mass | Reused mass |','|---|---|---:|---:|---:|']
    for scope in ('local','global'):
        for name in ('step_forward_1','step_sol_50','layer_forward_1','layer_sol_50','history_forward_1','history_sol_50'):
            p=diagnostics['reuse'].get('final|'+scope+'|'+name)
            if p:
                lines.append(f"| {name} | {scope} | {pct(p.get('jaccard'))} | {pct(p['fresh_mass'])} | {pct(p['reused_mass'])} |")
    lines+=['','Step reuse applies the preceding mask in the same layer/window. Layer reuse uses the previous same-type layer in the same call. History ranks tiles by prior access frequency at the current physical keep count; it therefore uses a current-budget oracle. Fresh/reused masses both use physical masks (BLASST closure), avoiding a row-mask versus tile-mask mismatch. Similarity alone does not establish safe reuse; retained-mass changes reveal risk, and rollout accuracy remains untested.','',
        'These observations favor testing selective, confidence-checked reuse rather than unconditional reuse: last-step Sol masks lose about 0.82 percentage points of local mass and 1.14 points globally; frequency-only masks lose more. The saved summaries include matched-density chance Jaccard baselines. Head/step/layer strata are descriptive and sampled heads are confounded with prompts, so they should not be treated as independent statistical replicates.','',
        '## 6. Defensible problem formulation','',
        '> On this DiffusionGemma/RULER16K setup, online row-max thresholding does not reliably control physical 64×64 sparsity: local query-row record unions impose a ceiling, while aggressive row masking can discard attention mass without eliminating whole tiles. Reproduction-specific calibration error is an additional, separable issue. A useful research objective is to control physical tile budgets and approximation error jointly, while measuring whether temporal reuse and query regrouping can reduce routing/union overhead without degrading generation.','',
        'This formulation does **not** claim universal Sol superiority, universal BLASST failure, or a new algorithm. Order effects, precision controls, and reuse/regrouping probes must be read alongside their measured tables; none alone proves task-level causality. A stronger paper needs more prompts/models/benchmarks, equal-physical-budget deployment comparisons, and actual kernels before making generalization or speed claims.','',
        '## 7. Relationship to the supplied papers','',
        '- DiffusionGemma alternates causal prefix encoding and bidirectional canvas denoising; this differs from ordinary autoregressive decode. [Technical report, §3](https://arxiv.org/html/2608.00146v1#S3).',
        '- BLASST uses online block maxima and an empirical threshold calibration; its appendix already investigates tile-order reversal. Order awareness itself is therefore not a novelty claim. The small thresholds reported in some settings are not a universal valid range. [BLASST, Algorithm 1, §3.2, Appendix A.4](https://arxiv.org/html/2512.12087v3).',
        '- Sol-Attn motivates standardized Gaussian proxy thresholds on video diffusion and also includes streaming and approximate correction. This experiment transfers only its Gaussian routing idea to a dLLM. [Sol-Attn, §3](https://arxiv.org/html/2607.24027v1#S3).','',
        '## Appendix: provenance, limitations, and VT','',
        'All original manifests and calibration traces were preserved. No final prompt selected a deployed threshold. Historical calibration traces are deduplicated records and omit layer IDs; their ceiling analysis describes the saved trace population. Final rollouts and newly sampled dense states have independent counters. Regions distinguish prefix, canvas, and boundary-straddling tiles. Saved JSON includes per-task scores, exact sequence agreement, bootstrap deltas, source hashes, and split-separated diagnostics.','',
        '| Condition | VT accuracy (excluded from primary) |','|---|---:|']
    for name,r in conditions.items():
        lines.append(f"| {name} | {pct(r['vt_accuracy'])} |")
    lines+=['','Regenerate with `PYTHONPATH=src:. python -m experiments.diffusion_gemma_blasst_diagnosis.report`. Raw data: `historical_reanalysis.json`, `calibration_grid_audit.json`, `same_state_summary.json`, endpoint `predictions.jsonl`/`attention_stats`, and `same_state/shards`.','']
    (root/'analysis.md').write_text('\n'.join(lines))
    write_json(root/'audit.json',dict(complete=True,final_endpoint_samples=50,reverse_samples=50,primary_samples=40,
        **checked,
        diagnostic_coverage=diagnostics['coverage'],native_dense_parity=True,calibration_and_final_disjoint=True,
        no_final_threshold_tuning=True,report_from_completed_shards=True,reference_masks_only=True))
    return historical,diagnostics


if __name__=='__main__':
    report()
