"""Canonical A1 identification/prediction readout and prespecified A2 gate."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    root=Path(__file__).resolve().parents[1];out=root/'analysis001';out.mkdir(exist_ok=False)
    inputs={}
    def read(p):
        inputs[str(p.relative_to(root))]=sha(p)
        return json.loads(p.read_text())
    protocol=read(root/'protocol.json');manifest=read(root/'manifest_identity.json')['identities']
    for job in ['smoke','full','a1']:assert read(root/'executions'/job/'final_complete.json')['passed']
    cells=[(r,s) for split in ['development','confirmation'] for r in manifest if r['split']==split for s in r['seeds']]
    coverage=[];forecast=[];controls=[];seen=set()
    for index,(identity,seed) in enumerate(cells):
        leaf=root/'executions/a1';q=read(leaf/f'cell{index:03d}_qualification.json');assert q['passed']
        arms={}
        for name in ['piecewise','allkept_unobserved','allkept_reference','allkept_observed']:
            v=read(leaf/f'cell{index:03d}_{name}'/'receipt.json');assert(v['id'],v['seed'])==(identity['id'],seed)
            key=(v['id'],seed,name);assert key not in seen;seen.add(key);arms[name]=v
        full=read(root/'executions/full'/f'cell{index:03d}_native_full/receipt.json');assert(full['id'],full['seed'])==(identity['id'],seed)
        controls.append(dict(id=identity['id'],seed=seed,benchmark=identity['benchmark'],split=identity['split'],
            arms={k:{f:v[f] for f in ['correct','N','C','T','N_per_C','W','S','S_per_N','timing_qualified','no_final_response','finish_reason','first_canvas_calls','output_sha256']} for k,v in (arms|{'dense_full':full}).items()}))
        for boundary in q['anchors']:
            anchor='late' if boundary==arms['allkept_reference']['first_canvas_calls']-2 else 'middle'
            for layer in [5,17,29]:
                folder=leaf/f'cell{index:03d}_allkept_observed/operators/layer{layer:02d}_after{boundary:02d}'
                bank=read(folder/'bank_choices.json');future_path=folder/'future_evaluation.json'
                future=read(future_path) if future_path.exists() else None
                for ui,unit in enumerate(bank['units']):
                    cover=dict(id=identity['id'],seed=seed,benchmark=identity['benchmark'],split=identity['split'],cohort=identity['cohort'],
                        layer=layer,boundary=boundary,anchor=anchor,head=unit['head'],qb=unit['qb'],prefix_blocks=unit['prefix_blocks'],
                        absolute_deletion=unit['drop_count'],candidate_count=len(unit['candidates']),feasible=unit['feasible'],
                        error_mass_pairs=len(unit['pairs']),identified_by_score={name:unit['feasible']>=4 and any(x['separated_by_score'][name] for x in unit['pairs']) for name in ['q_past','qkv_past']},
                        future_available=future is not None,choices=unit['choices'])
                    coverage.append(cover)
                    if future is None:continue
                    results=future['units'][ui]['results'];feasible=[x for x in results if x['feasible']]
                    if not feasible:continue
                    for score in ['q_past','qkv_past']:
                        chosen=unit['choices'][score];base=unit['choices']['current_error'];sham=unit['choices']['support_sham'];ds=unit['choices']['direction_sham']
                        row={k:cover[k] for k in ['id','seed','benchmark','split','cohort','layer','anchor','boundary','head','qb','absolute_deletion']}
                        row.update(score=score,identified=cover['identified_by_score'][score],forecast_loss={k:statistics.mean(x['forecast_loss'][k] for x in feasible) for k in feasible[0]['forecast_loss']},
                            residual_forecast_loss={k:statistics.mean(x['residual_forecast_loss'][k] for x in feasible) for k in feasible[0]['residual_forecast_loss']},
                            selected_actual_motion={k:results[i]['actual_motion'] if i is not None else None for k,i in [(score,chosen),('current_error',base),('support_sham',sham),('direction_sham',ds)]})
                        forecast.append(row)
    question_rows=[]
    for score in ['q_past','qkv_past']:
        for layer in [5,17,29]:
            for identity in manifest:
                for seed in identity['seeds']:
                    rows=[x for x in forecast if x['score']==score and x['layer']==layer and x['id']==identity['id'] and x['seed']==seed and x['anchor']=='late']
                    covers=[x for x in coverage if x['layer']==layer and x['id']==identity['id'] and x['seed']==seed and x['anchor']=='late']
                    identifying=len(covers)==4 and all(x['feasible']>=4 for x in covers) and any(x['identified_by_score'][score] for x in covers)
                    metrics={}
                    if rows:
                        for baseline in ['bias_carry_dense_forecast','direction_sham','q_past','persistence','future_motion_oracle']:
                            metrics['forecast_vs_'+baseline]=statistics.mean(x['forecast_loss'][score] for x in rows)/max(statistics.mean(x['forecast_loss'][baseline] for x in rows),1e-12)
                        metrics['residual_vs_carry']=statistics.mean(x['residual_forecast_loss'][score] for x in rows)/max(statistics.mean(x['residual_forecast_loss']['carry_current_bias'] for x in rows),1e-12)
                        for baseline in ['current_error','support_sham','direction_sham']:
                            available=[x for x in rows if x['selected_actual_motion'][score] is not None and x['selected_actual_motion'][baseline] is not None]
                            metrics['selected_motion_vs_'+baseline]=statistics.mean(x['selected_actual_motion'][score] for x in available)/max(statistics.mean(x['selected_actual_motion'][baseline] for x in available),1e-12) if available else None
                    prediction=identifying and metrics.get('forecast_vs_bias_carry_dense_forecast',999)<=.9 and metrics.get('forecast_vs_direction_sham',999)<=.9
                    if score=='qkv_past':prediction &= metrics.get('forecast_vs_q_past',999)<=.9
                    utility=all(metrics.get('selected_motion_vs_'+b) is not None and metrics['selected_motion_vs_'+b]<=.9 for b in ['current_error','support_sham'])
                    question_rows.append(dict(id=identity['id'],benchmark=identity['benchmark'],split=identity['split'],seed=seed,layer=layer,score=score,
                        identifying=identifying,prediction_pass=bool(prediction),ranking_utility_pass=utility,metrics=metrics))
    decisions=[]
    for score in ['q_past','qkv_past']:
        for layer in [5,17,29]:
            eligible=[];confirmed=[]
            for identity in manifest:
                rows=[x for x in question_rows if x['id']==identity['id'] and x['layer']==layer and x['score']==score]
                if identity['split']=='development' and len(rows)==2 and all(x['identifying'] for x in rows):eligible.append(identity['id'])
                if identity['split']=='confirmation' and len(rows)==2 and all(x['prediction_pass'] and x['ranking_utility_pass'] for x in rows):confirmed.append(identity['id'])
            selected=[x for x in question_rows if x['split']=='confirmation' and x['score']==score and x['layer']==layer]
            # Missing/non-identifying confirmation cells cannot be excluded.
            aggregate=bool(selected) and all(x['metrics'].get(k) is not None for x in selected for k in ['forecast_vs_bias_carry_dense_forecast','forecast_vs_direction_sham','selected_motion_vs_current_error','selected_motion_vs_support_sham'])
            if aggregate:
                aggregate=all(statistics.mean(x['metrics'][k] for x in selected)<=.9 for k in ['forecast_vs_bias_carry_dense_forecast','forecast_vs_direction_sham','selected_motion_vs_current_error','selected_motion_vs_support_sham'])
            decisions.append(dict(score=score,layer=layer,development_identifying_questions=eligible,confirmation_replicated_questions=confirmed,
                A2_qualified=len(eligible)>=2 and len(confirmed)>=2 and aggregate,engineering_passed=True))
    groups=[];rng=np.random.default_rng(20261025)
    for score in ['q_past','qkv_past']:
        for layer in [5,17,29]:
            for benchmark in ['aime26','humaneval','longbench_v2']:
                rows=[x for x in question_rows if x['score']==score and x['layer']==layer and x['benchmark']==benchmark]
                metrics={}
                for metric in ['forecast_vs_bias_carry_dense_forecast','forecast_vs_direction_sham','forecast_vs_q_past','residual_vs_carry','selected_motion_vs_current_error','selected_motion_vs_support_sham']:
                    byid=defaultdict(list)
                    for x in rows:
                        value=x['metrics'].get(metric)
                        if value is not None and value>0:byid[x['id']].append(np.log(value))
                    if byid:
                        values=np.array([np.mean(v) for v in byid.values()]);draws=rng.integers(0,len(values),(10000,len(values)))
                        metrics[metric]=dict(geometric_ratio=float(np.exp(values.mean())),ci95=np.exp(np.quantile(values[draws].mean(1),[.025,.975])).tolist(),independent_questions=len(values),available_cells=sum(len(v) for v in byid.values()))
                groups.append(dict(score=score,layer=layer,benchmark=benchmark,metrics=metrics))
    data=dict(passed=True,stage='A1_complete_A2_gated',coverage=coverage,forecast_by_unit=forecast,question_results=question_rows,
        aggregate_prediction=groups,score_location_decisions=decisions,controls=controls,A2_qualified=any(x['A2_qualified'] for x in decisions),
        source_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),timing_claim=False,
        interpretation='Prediction and selected operator motion are separate from native causal improvement and task quality. Only two ordinary current/future landmarks, first canvas, no online policy claim.')
    for name,value in [('summary.json',data),('input_hashes.json',inputs)]:
        (out/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    (out/'final_complete.json').write_text(json.dumps(dict(passed=True,summary_sha256=sha(out/'summary.json'),source_sha256=sha(__file__),A2_qualified=data['A2_qualified'],unique_item_seed_cells=24),indent=2)+'\n')
    print(json.dumps(dict(cells=24,coverage_units=len(coverage),A2_qualified=data['A2_qualified'],decisions=decisions)))


if __name__=='__main__':main()
