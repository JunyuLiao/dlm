"""Read-only correction of Stage A1 reporting, not a change to its gates."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[3]
OLD=REPO/'results/denoising_mechanism_stage_a_20261025/attempt002'


def main():
    out=Path(__file__).resolve().parent;inputs={}
    def read(p):
        inputs[str(p.relative_to(REPO))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    analysis=read(OLD/'analysis001/summary.json')
    read(OLD/'final_complete.json')
    identities={x['id']:x for x in read(OLD/'manifest_identity.json')['identities']}
    groups=defaultdict(list)
    for row in analysis['forecast_by_unit']:
        if row['score']=='qkv_past':
            groups[row['id'],row['seed'],row['layer'],row['anchor']].append(row)
    cells=[];crossings=[]
    for (identity,seed,layer,anchor),units in groups.items():
        for score in ['q_past','qkv_past']:
            values={}
            for baseline in ['persistence','bias_carry_dense_forecast','direction_sham']:
                arithmetic=statistics.mean(x['forecast_loss'][score] for x in units)/statistics.mean(x['forecast_loss'][baseline] for x in units)
                geometric=math.exp(statistics.mean(math.log(x['forecast_loss'][score]/x['forecast_loss'][baseline]) for x in units))
                values[baseline]=dict(executable_ratio_of_mean_unit_losses=arithmetic,README_geometric_unit_ratio=geometric)
                if anchor=='late' and baseline!='persistence' and (arithmetic<=.9)!=(geometric<=.9):
                    crossings.append(dict(id=identity,seed=seed,layer=layer,score=score,baseline=baseline,**values[baseline]))
            cells.append(dict(id=identity,seed=seed,layer=layer,anchor=anchor,score=score,benchmark=identities[identity]['benchmark'],
                available_units=len(units),expected_units=4,identified_units=sum(x['identified'] for x in units),ratios=values))
    aggregates=[]
    for task in ['aime26','humaneval','longbench_v2']:
        for layer in [5,17,29]:
            for anchor in ['middle','late']:
                for score in ['q_past','qkv_past']:
                    rows=[x for x in cells if (x['benchmark'],x['layer'],x['anchor'],x['score'])==(task,layer,anchor,score)]
                    values={}
                    for definition in ['executable_ratio_of_mean_unit_losses','README_geometric_unit_ratio']:
                        byid=defaultdict(list)
                        for row in rows:byid[row['id']].append(math.log(row['ratios']['persistence'][definition]))
                        values[definition]=math.exp(statistics.mean(statistics.mean(v) for v in byid.values()))
                    assert all(v>1 for v in values.values()),'persistence interpretation changed'
                    aggregates.append(dict(benchmark=task,layer=layer,anchor=anchor,score=score,persistence_ratios=values,
                        available_questions=len({x['id'] for x in rows}),expected_questions=4,available_seed_cells=len(rows),expected_seed_cells=8,
                        available_routing_units=sum(x['available_units'] for x in rows),expected_routing_units=32))
    assert len(crossings)==5
    pairs=read(OLD/'closeout001/identification.json')['selected_pair_details']
    selected={}
    for score in ['q_past','qkv_past']:
        selected[score]=dict(distinct=sum(x['selected_pairs'][score].get('distinct',False) for x in pairs),
            distinct_error_mass_matched=sum(x['selected_pairs'][score].get('distinct',False) and x['selected_pairs'][score].get('error_mass_matched',False) for x in pairs))
    assert selected['q_past']==dict(distinct=185,distinct_error_mass_matched=19)
    assert selected['qkv_past']==dict(distinct=179,distinct_error_mass_matched=18)
    native=read(OLD/'closeout001/first_canvas_native.json')['cells']
    common=[x for x in native if x['arm']=='allkept_layers5_17_29_canvas1']
    assert len(common)==24 and sum(x['confidence_limited_calls'] for x in common)==0
    resources=read(OLD/'closeout001/physical_and_resources.json')['totals']
    assert resources['native_sparse_applications']==0
    result=dict(passed=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),forecast_cells=cells,
        persistence_aggregates=aggregates,individual_threshold_crossings=crossings,selected_contrasts=selected,
        bank_wide_matched_pairs=24865,full_response_identifying_pairs=0,first_canvas_common_confidence_only_delays=0,
        missing_controls=['Q-only dense forecast J_A u_Q','Q-only direction sham with Q-only scale'],
        interpretation='Both past-motion forecasts lose to persistence at every task/layer/anchor aggregate. Joint-QKV controls cannot identify Q-only directional information. All frozen A2 no-go decisions remain.',
        accounting=dict(model_support_deletion=False,diagnostic_work_additional=True,audit_timing_is_clean=False,FULL_complete_marker=True,FULL_parent_wait_status_available=False,
            prior_raw_receipt_corrections=299,prior_published_aggregate_conclusions_changed=False),
        no_GPU_rerun=True,prior_Response_quality_and_total_work_behavior_not_invalidated=True)
    for name,v in [('summary.json',result),('input_hashes.json',inputs)]:
        with (out/name).open('x') as f:json.dump(v,f,indent=2,sort_keys=True)
    with (out/'final_complete.json').open('x') as f:json.dump(dict(passed=True,CPU_only=True,summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()),f,indent=2)
    print(json.dumps(dict(aggregates=len(aggregates),threshold_crossings=len(crossings),selected=selected)))


if __name__=='__main__':main()
