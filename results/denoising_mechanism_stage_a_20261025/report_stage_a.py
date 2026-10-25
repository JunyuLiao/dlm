"""Derived Stage A handoff, without changing frozen endpoints or gate decisions."""
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'attempt002'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'closeout001'
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(path):
        inputs[str(path.relative_to(BASE))] = sha(path)
        return json.loads(path.read_text())

    def save(name, value):
        (out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

    complete = read(ROOT / 'execution_complete.json')
    assert complete['passed'] and complete['A1_complete']
    analysis = read(ROOT / 'analysis001/summary.json')
    analysis_inputs = read(ROOT / 'analysis001/input_hashes.json')
    for path, value in analysis_inputs.items():
        assert sha(ROOT / path) == value, ('analysis input drift', path)
    assert not analysis['A2_qualified'], 'Qualified A2 needs a separately bound causal handoff'
    protocol = read(ROOT / 'protocol.json')
    manifest = read(ROOT / 'manifest_identity.json')['identities']
    identities = {(r['id'], seed): r for r in manifest for seed in r['seeds']}
    rng = np.random.default_rng(20261025)

    def cluster_interval(values):
        by_id = defaultdict(list)
        for key, value in values:
            by_id[key].append(value)
        means = np.array([statistics.mean(v) for v in by_id.values()])
        draws = rng.integers(0, len(means), (10000, len(means)))
        return dict(mean=float(means.mean()), ci95=np.quantile(means[draws].mean(1), [.025, .975]).tolist(),
                    independent_questions=len(means), seed_cells=sum(map(len, by_id.values())))

    receipts = []
    qualified = []
    resources = []
    for job in ['smoke', 'full', 'a1']:
        marker = read(ROOT / f'executions/{job}/final_complete.json')
        assert marker['passed'] and marker['protocol_sha256'] == sha(ROOT / 'protocol.json')
        resource = dict(marker, job=job)
        exit_path = ROOT / f'launches/{job}/exit_receipt.json'
        if exit_path.exists():
            resource['launcher'] = read(exit_path)
        else:
            resource['observed_exit'] = read(ROOT / 'controller_recovery001/full_observed_exit.json')
        resources.append(resource)
        for path in sorted((ROOT / f'executions/{job}').glob('*/receipt.json')):
            v = read(path)
            assert v['protocol_sha256'] == sha(ROOT / 'protocol.json')
            v['_job'] = job
            v['_relative_path'] = str(path.relative_to(ROOT))
            receipts.append(v)
        for path in sorted((ROOT / f'executions/{job}').glob('cell*_qualification.json')):
            q = read(path)
            assert q['passed'] and all(q['checks'].values())
            qualified.append(dict(job=job, **q))

    canonical = defaultdict(dict)
    for v in receipts:
        if not v['label'].startswith('cell'):
            continue
        if v['_job'] == 'full':
            arm = 'dense_full_fix51994'
        elif v['_job'] == 'a1' and v['label'].endswith('_piecewise'):
            arm = 'dense_piecewise'
        elif v['_job'] == 'a1' and v['label'].endswith('_allkept_reference'):
            arm = 'allkept_layers5_17_29_canvas1'
        else:
            continue
        key = (v['id'], v['seed'])
        assert key in identities and arm not in canonical[key]
        canonical[key][arm] = v
    assert len(canonical) == 24 and all(len(v) == 3 for v in canonical.values())

    quality = []
    transitions = []
    cell_rows = []
    first_canvas = []
    for key, arms in canonical.items():
        ident = identities[key]
        for arm, v in arms.items():
            first=[x for x in v['native_trace'] if not x['commit'] and x['canvas']==0]
            first_canvas.append(dict(id=key[0],seed=key[1],benchmark=ident['benchmark'],arm=arm,calls=len(first),
                confidence_limited_calls=sum(x['stable'] and x['history_ready'] and not x['confidence'] for x in first),
                stability_limited_calls=sum(x['confidence'] and not(x['stable'] and x['history_ready']) for x in first),
                both_limited_calls=sum(not x['confidence'] and not(x['stable'] and x['history_ready']) for x in first),
                terminal_flags={k:first[-1][k] for k in ['native_step','stable','history_ready','confidence','converged','cap']},
                entropy_reconstruction_flag_discrepancies=sum((x['entropy_mean']<.005)!=x['confidence'] for x in first),
                entropy_curve=[x['entropy_mean'] for x in first],entropy_slack_curve=[.005-x['entropy_mean'] for x in first],
                terminal_entropy_position_quantiles=np.quantile(first[-1]['entropy_by_position'],[0,.05,.5,.95,1]).tolist()))
            committed = defaultdict(int)
            for call in v['native_trace']:
                if not call['commit']:
                    committed[call['canvas']] += 1
            committed_calls = sum(n for c, n in committed.items() if c < v['C'])
            tail = v['N'] - committed_calls
            assert sum(committed.values()) == v['N'] and tail >= 0
            cell_rows.append(dict(id=key[0], seed=key[1], benchmark=ident['benchmark'], split=ident['split'], cohort=ident['cohort'], arm=arm,
                correct=v['correct'], N=v['N'], C=v['C'], T=v['T'], N_per_C=v['N_per_C'], first_canvas_calls=v['first_canvas_calls'],
                committed_canvas_calls=[committed[c] for c in range(v['C'])], uncommitted_tail_calls=tail,
                canvas_48_call_caps=sum(n == 48 for c, n in committed.items() if c < v['C']),
                output_cap=v['T'] >= (16384 if ident['benchmark']=='longbench_v2' else 8192),
                no_final_response=v['no_final_response'], finish_reason=v['finish_reason'], quality_detail=v['quality_detail'],
                W=v['W'], S=v['S'], S_per_N=v['S_per_N'], timing_qualified=v['timing_qualified'],
                output_sha256=v['output_sha256'], receipt=v['_relative_path']))
    for benchmark in ['aime26', 'humaneval', 'longbench_v2']:
        rows = [x for x in cell_rows if x['benchmark']==benchmark]
        for arm in sorted({x['arm'] for x in rows}):
            r = [x for x in rows if x['arm']==arm]
            quality.append(dict(benchmark=benchmark, arm=arm, cells=len(r), independent_questions=len({x['id'] for x in r}),
                correct=sum(x['correct'] for x in r), accuracy_interval=cluster_interval([(x['id'], int(x['correct'])) for x in r]),
                total_N=sum(x['N'] for x in r), total_C=sum(x['C'] for x in r), total_T=sum(x['T'] for x in r),
                mean_N_per_C=statistics.mean(x['N_per_C'] for x in r), mean_T=statistics.mean(x['T'] for x in r),
                output_capped_cells=sum(x['output_cap'] for x in r), canvas_caps=sum(x['canvas_48_call_caps'] for x in r),
                missing_final=sum(x['no_final_response'] for x in r), uncommitted_tail_calls=sum(x['uncommitted_tail_calls'] for x in r),
                mean_audit_W=statistics.mean(x['W'] for x in r), mean_audit_S=statistics.mean(x['S'] for x in r), timing_qualified=False))
        for candidate in ['dense_piecewise', 'allkept_layers5_17_29_canvas1']:
            pairs = [(arms['dense_full_fix51994'], arms[candidate]) for key, arms in canonical.items() if identities[key]['benchmark']==benchmark]
            transitions.append(dict(benchmark=benchmark, candidate=candidate, reference='dense_full_fix51994',
                reference_correct_candidate_wrong=[dict(id=a['id'],seed=a['seed']) for a,b in pairs if a['correct'] and not b['correct']],
                reference_wrong_candidate_correct=[dict(id=a['id'],seed=a['seed']) for a,b in pairs if not a['correct'] and b['correct']],
                quality_delta=cluster_interval([(a['id'], int(b['correct'])-int(a['correct'])) for a,b in pairs]),
                log_N_per_C_ratio=cluster_interval([(a['id'], math.log(b['N_per_C']/a['N_per_C'])) for a,b in pairs]),
                log_T_ratio=cluster_interval([(a['id'],math.log(b['T']/a['T'])) for a,b in pairs])))
    save('official_controls.json', dict(aggregates=quality, transitions=transitions, cells=cell_rows,
        interpretation='Controls only: A1 candidates were operator recomputations, never native sparse treatments. Four question clusters per task. Synchronized audit timing is not clean timing.'))
    save('first_canvas_native.json',dict(cells=first_canvas,
        note='Actual history-ready/stability/confidence/convergence flags are observed after the unchanged sampler. Entropy is separately reconstructed in float32 from the actual returned scaled logits; this is not a claim of exact compiled acceptance counts.'))

    coverage_groups = defaultdict(list)
    pair_details = []
    for cover in analysis['coverage']:
        key = tuple(cover[k] for k in ['benchmark','layer','anchor'])
        coverage_groups[key].append(cover)
    coverage = []
    bank_seen = set()
    pair_totals = defaultdict(lambda:dict(feasible_pairs=0,error_mass_matched_pairs=0,full_q_pairs=0,full_qkv_pairs=0,residual_q_pairs=0,residual_qkv_pairs=0))
    distinct = defaultdict(lambda:dict(units=0,q_vs_error=0,qkv_vs_error=0,q_vs_qkv=0,support_sham_vs_error=0))
    selection_oracle=[]
    for row in analysis['coverage']:
        key = (row['id'],row['seed'],row['layer'],row['boundary'])
        if key in bank_seen:
            continue
        bank_seen.add(key)
        index = next(i for i,c in enumerate(analysis['controls']) if (c['id'],c['seed'])==(row['id'],row['seed']))
        bank = read(ROOT / f'executions/a1/cell{index:03d}_allkept_observed/operators/layer{row["layer"]:02d}_after{row["boundary"]:02d}/bank_choices.json')
        future_path=ROOT / f'executions/a1/cell{index:03d}_allkept_observed/operators/layer{row["layer"]:02d}_after{row["boundary"]:02d}/future_evaluation.json'
        future=read(future_path) if future_path.exists() else None
        for unit in bank['units']:
            candidates=unit['candidates']
            f=[i for i,x in enumerate(candidates) if x['feasible']]
            group=(row['benchmark'],row['layer'],row['anchor'])
            t=pair_totals[group]
            t['feasible_pairs'] += len(f)*(len(f)-1)//2
            t['error_mass_matched_pairs'] += len(unit['pairs'])
            for score,short in [('q_past','q'),('qkv_past','qkv')]:
                t['full_'+short+'_pairs'] += sum(x['separated_by_score'][score] for x in unit['pairs'])
                t['residual_'+short+'_pairs'] += sum(abs(candidates[x['a']]['residual_scores'][score]-candidates[x['b']]['residual_scores'][score])/max(candidates[x['a']]['residual_scores'][score],candidates[x['b']]['residual_scores'][score],1e-6)>=.25 for x in unit['pairs'])
            ch=unit['choices']; d=distinct[group]; d['units']+=1
            if future is not None:
                future_unit=next(x for x in future['units'] if (x['head'],x['qb'])==(unit['head'],unit['qb']))
                results=future_unit['results'];feasible_future=[x for x in results if x['feasible']]
                if feasible_future and ch['current_error'] is not None:
                    best=min(x['actual_motion'] for x in feasible_future)
                    base=results[ch['current_error']]['actual_motion']
                    selection_oracle.append(dict(id=row['id'],seed=row['seed'],benchmark=row['benchmark'],layer=row['layer'],anchor=row['anchor'],boundary=row['boundary'],
                        head=unit['head'],qb=unit['qb'],absolute_deletion=unit['drop_count'],best_available_next_motion=best,current_error_choice_next_motion=base,
                        best_vs_current_error=best/max(base,1e-12),maximum_available_next_motion=max(x['actual_motion'] for x in feasible_future)))
            for label,a,b in [('q_vs_error','q_past','current_error'),('qkv_vs_error','qkv_past','current_error'),('q_vs_qkv','q_past','qkv_past'),('support_sham_vs_error','support_sham','current_error')]:
                d[label]+=ch[a] is not None and ch[b] is not None and ch[a]!=ch[b]
            pairs = {(x['a'], x['b']):x for x in unit['pairs']}
            chosen={}
            for score in ['q_past','qkv_past']:
                if ch[score] is None or ch['current_error'] is None:
                    chosen[score]=dict(available=False)
                else:
                    a,b=sorted([ch[score],ch['current_error']]); pair=pairs.get((a,b))
                    chosen[score]=dict(available=True,distinct=a!=b,error_mass_matched=pair is not None,
                        full_response_separated=bool(pair and pair['separated_by_score'][score]))
            pair_details.append(dict(id=row['id'],seed=row['seed'],benchmark=row['benchmark'],layer=row['layer'],anchor=row['anchor'],boundary=row['boundary'],
                head=unit['head'],qb=unit['qb'],prefix_blocks=unit['prefix_blocks'],absolute_deletion=unit['drop_count'],
                feasible=len(f),candidate_count=len(candidates),selected_pairs=chosen,
                abstention='fewer_than_four_feasible_alternatives' if len(f)<4 else None,
                exclusion_reasons={str(i):(['current_error_above_0.1'] if x['current_error']>.1 else [])+
                    (['whole_deleted_mass_above_0.2'] if x['deleted_mass']>.2 else []) for i,x in enumerate(candidates) if not x['feasible']}))
    for key, rows in coverage_groups.items():
        coverage.append(dict(zip(['benchmark','layer','anchor'],key)) | dict(units=len(rows),independent_questions=len({x['id'] for x in rows}),
            feasible_at_least_four=sum(x['feasible']>=4 for x in rows),future_available=sum(x['future_available'] for x in rows),
            identified_q=sum(x['identified_by_score']['q_past'] for x in rows),identified_qkv=sum(x['identified_by_score']['qkv_past'] for x in rows),
            pair_totals=pair_totals[key],distinct_choices=distinct[key]))
    save('identification.json',dict(coverage=coverage,selected_pair_details=pair_details,
        interpretation='All pairs and selected pairs are separate. Counts of candidates/pairs/units are coverage, not independent task samples. Middle and late have different native states and physical budgets are never pooled for inference.'))
    save('selection_oracle.json',dict(units=selection_oracle,
        note='Evaluation-only best available next-call motion in the frozen feasible bank. This bounds achievable operator-ranking utility; it is never used for support selection or causal promotion.'))

    # Preserve the frozen executable readout and disclose the README's different
    # within-question averaging description. Neither analysis changes selection.
    aggregation_sensitivity=[]
    for question in analysis['question_results']:
        rows=[x for x in analysis['forecast_by_unit'] if all(x[k]==question[k] for k in ['id','seed','layer','score']) and x['anchor']=='late']
        values={}
        for baseline in ['bias_carry_dense_forecast','direction_sham','q_past']:
            ratios=[max(x['forecast_loss'][question['score']],1e-12)/max(x['forecast_loss'][baseline],1e-12) for x in rows]
            if ratios:
                values['forecast_vs_'+baseline]=math.exp(statistics.mean(map(math.log,ratios)))
        for baseline in ['current_error','support_sham','direction_sham']:
            ratios=[max(x['selected_actual_motion'][question['score']],1e-12)/max(x['selected_actual_motion'][baseline],1e-12) for x in rows
                if x['selected_actual_motion'][question['score']] is not None and x['selected_actual_motion'][baseline] is not None]
            if ratios:
                values['selected_motion_vs_'+baseline]=math.exp(statistics.mean(map(math.log,ratios)))
        aggregation_sensitivity.append({k:question[k] for k in ['id','seed','benchmark','split','layer','score','identifying']} |
            dict(frozen_executable_ratio_of_unit_means=question['metrics'],README_geometric_mean_of_unit_ratios=values))
    save('aggregation_sensitivity.json',dict(rows=aggregation_sensitivity,
        note='The pre-generation executable averages unit losses before division. README describes mean paired log ratios within question. Both were frozen; this derived readout discloses that discrepancy, retains both calculations, and does not revise selection or promotion gates. An identification failure is independent of this averaging difference.'))
    floor_cases=[]
    for row in analysis['forecast_by_unit']:
        low=[k for k,v in row['forecast_loss'].items() if v<=1e-12]
        low_motion=[k for k,v in row['selected_actual_motion'].items() if v is not None and v<=1e-12]
        if low or low_motion:
            floor_cases.append({k:row[k] for k in ['id','seed','layer','anchor','head','qb','score']} |
                dict(forecast_losses_at_floor=low,selected_motion_at_floor=low_motion))
    save('numerical_floor_audit.json',dict(cases=floor_cases,
        note='Ratios use the frozen 1e-12 denominator floor. A zero/zero case is not evidence of predictive gain; this audit preserves degenerate cases separately from substantive improvements.'))
    decomposition=[]
    grouped=defaultdict(list)
    for row in analysis['forecast_by_unit']:
        # The same vector losses occur in both score rows; avoid double counting.
        if row['score']=='qkv_past':
            grouped[tuple(row[k] for k in ['id','seed','benchmark','split','layer','anchor','boundary','absolute_deletion'])].append(row)
    for key, rows in grouped.items():
        f={k:statistics.mean(x['forecast_loss'][k] for x in rows) for k in rows[0]['forecast_loss']}
        r={k:statistics.mean(x['residual_forecast_loss'][k] for x in rows) for k in rows[0]['residual_forecast_loss']}
        decomposition.append(dict(zip(['id','seed','benchmark','split','layer','anchor','boundary','absolute_deletion'],key)) |
            dict(available_units=len(rows),forecast_MSE=f,residual_forecast_MSE=r,
                full_oracle_vs_persistence=f['future_motion_oracle']/max(f['persistence'],1e-12),
                residual_oracle_vs_carry=r['future_motion_oracle']/max(r['carry_current_bias'],1e-12),
                qkv_vs_persistence=f['qkv_past']/max(f['persistence'],1e-12),
                q_vs_persistence=f['q_past']/max(f['persistence'],1e-12),
                qkv_vs_q=f['qkv_past']/max(f['q_past'],1e-12)))
    save('prediction_decomposition.json',dict(cells=decomposition,
        note='Evaluation-only actual-future-motion derivatives distinguish direction-forecast failure from finite-step linearization error. They never select support or trigger treatment. Losses are averaged within question/seed/layer/anchor/physical budget; unavailable units remain explicit.'))

    physical=[]
    for v in receipts:
        w=v['work'];rt=v['runtime']
        assert w['global_kept_tiles']==w['global_eligible_tiles'] and w['native_local_skipped_tiles']==0
        assert w['overall_eligible_rectangles']==w['global_eligible_tiles']+w['native_local_eligible_rectangles']
        assert all(x['current_QKV'] and x['skipped']==0 and x['protected_canvas_skipped']==0 for x in rt['physical_calls'])
        physical.append(dict(receipt=v['_relative_path'],job=v['_job'],id=v['id'],seed=v['seed'],label=v['label'],
            GLOBAL_eligible=w['global_eligible_tiles'],GLOBAL_kept=w['global_kept_tiles'],GLOBAL_skipped=0,
            LOCAL_eligible=w['native_local_eligible_rectangles'],LOCAL_kept=w['native_local_eligible_rectangles'],LOCAL_skipped=0,
            overall_eligible=w['overall_eligible_rectangles'],model_forwards=v['model_forwards'],ordinary_denoising_calls=v['N'],native_commits=v['C'],
            diagnostic_QK_rectangles=rt['diagnostic_QK_rectangles'],diagnostic_PV_rectangles=rt['diagnostic_PV_rectangles'],
            diagnostic_value_projection_key_rows=rt['diagnostic_value_projection_key_rows'],operator_seconds=rt['operator_only_seconds'],
            pending_future=rt['pending_future']))
    resource_totals=dict(counted_model_forwards=sum(x['model_executions']+x['startup_forwards'] for x in resources),
        worker_reported_allocation_seconds=sum(x['allocation_seconds'] for x in resources),
        exact_launcher_allocation_seconds=sum(x['launcher']['allocated_worker_seconds'] for x in resources if 'launcher' in x),
        full_launch_to_exit_upper_bound_seconds=next(x['observed_exit']['launch_to_observed_exit_upper_bound_seconds'] for x in resources if 'observed_exit' in x),
        completed_requests=len(receipts),counterfactual_diagnostic_QK_rectangles=sum(x['diagnostic_QK_rectangles'] for x in physical),
        counterfactual_diagnostic_PV_rectangles=sum(x['diagnostic_PV_rectangles'] for x in physical),
        diagnostic_value_projection_key_rows=sum(x['diagnostic_value_projection_key_rows'] for x in physical),
        operator_only_seconds=sum(x['operator_seconds'] for x in physical),
        native_sparse_applications=0,refreshes=0,held_mask_native_calls=0,timing_qualified=False,
        note='Forward totals include startup, graph/capture warmup, initialization, warm requests, qualification repeats and all completed study requests. The FULL parent exited without wait-status; its allocation has a measured worker duration and a 15-second-polled launch-to-exit upper bound. Physical denoising attention denominators exclude dense prefill/commit; those ordinary forwards are counted separately. Audit/operator nested times are not added to request time.')
    save('physical_and_resources.json',dict(totals=resource_totals,jobs=resources,requests=physical))
    quals=[x for v in receipts for x in v['runtime']['GQA_qualification']]
    for resource in resources:
        assert resource['model_executions']==resource['initialization_engine_forwards']+sum(v['model_forwards'] for v in receipts if v['_job']==resource['job'])
    for path in (ROOT/'executions/a1').glob('*/operators/*/future_evaluation.json'):
        assert (path.parent/'bank_choices.json').stat().st_mtime_ns <= path.stat().st_mtime_ns
    save('engineering.json',dict(passed=True,source_pins=len(protocol['sources']),external_source_pins=len(protocol['external_sources']),
        qualified_cells=qualified,GQA_checks=len(quals),maximum_relative_GQA_error=max(x['relative_operator_error'] for x in quals),
        assertions=['native actual argmax equality','unchanged complete output/RNG/forward counts across common fresh branches',
        'exact first-canvas logits/native state/RNG fingerprints','same current Q/K/V route fingerprints',
        'live current tail K/V equality and within-canvas fixed prefix hashes','zero extra model forwards or diagnostic RNG draws',
        'bank choices written before separate future evaluation','all engine forwards reconcile to initialization plus request forwards'],
        limits=['No restored-output assembly qualification','No compiled acceptance-count claim','No sparse native treatment in A1',
        'All-kept three-layer GQA/current-operand paths qualified; a native deleting path needs separate A2 engineering qualification','No clean speed measurement']))
    decisions=[dict(d,status='no_go_to_A2',failed_gates=(['development_identification'] if len(d['development_identifying_questions'])<2 else [])+
        (['replicated_confirmation_prediction_and_ranking'] if len(d['confirmation_replicated_questions'])<2 else [])) for d in analysis['score_location_decisions']]
    save('method_decisions.json',dict(stage='A1_complete',A2_launched=False,stage_BC_launched=False,decisions=decisions,
        margin='No exhaustive oracle; not advanced by these response prediction tests',validity_reuse='Norm-only track remains parked',
        online_policy='No deployable timing policy is qualified by reference-relative landmarks',
        causal='AA/SA/AS/SS and sparse task quality remain unmeasured because no score/location passed A1'))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for ax,benchmark in zip(axes,['aime26','humaneval','longbench_v2']):
        for score,offset in [('q_past',-.12),('qkv_past',.12)]:
            g=[x for x in analysis['aggregate_prediction'] if x['benchmark']==benchmark and x['score']==score and 'forecast_vs_bias_carry_dense_forecast' in x['metrics']]
            vals=[x['metrics']['forecast_vs_bias_carry_dense_forecast']['geometric_ratio'] for x in g]
            ax.plot([x['layer']+offset for x in g],vals,'o-',label=score)
        ax.axhline(1,color='gray',linewidth=1);ax.axhline(.9,color='gray',linestyle=':',linewidth=1)
        ax.set_title(benchmark);ax.set_xticks([5,17,29]);ax.set_xlabel('GLOBAL layer');ax.set_yscale('log')
    axes[0].set_ylabel('Forecast MSE / dense-motion bias-carry MSE')
    axes[-1].legend(fontsize=8);fig.suptitle('A1 late-anchor descriptive question-cluster forecast ratios');fig.tight_layout()
    fig.savefig(out/'forecast_by_layer.png',dpi=180);fig.savefig(out/'forecast_by_layer.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,score in zip(axes,['q_past','qkv_past']):
        for benchmark in ['aime26','humaneval','longbench_v2']:
            g=[x for x in analysis['aggregate_prediction'] if x['benchmark']==benchmark and x['score']==score and 'selected_motion_vs_current_error' in x['metrics']]
            ax.plot([x['layer'] for x in g],[x['metrics']['selected_motion_vs_current_error']['geometric_ratio'] for x in g],'o-',label=benchmark)
        ax.axhline(1,color='gray');ax.axhline(.9,color='gray',linestyle=':');ax.set_title(score);ax.set_xticks([5,17,29]);ax.set_xlabel('GLOBAL layer')
    axes[0].set_ylabel('Selected next attention motion / current-error selection');axes[-1].legend(fontsize=8)
    fig.suptitle('Operator ranking utility; lower motion is not native task improvement');fig.tight_layout()
    fig.savefig(out/'selection_by_layer.png',dpi=180);fig.savefig(out/'selection_by_layer.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4))
    for benchmark in ['aime26','humaneval','longbench_v2']:
        xs=[];ys=[]
        for layer in [5,17,29]:
            rows=[x for x in decomposition if x['benchmark']==benchmark and x['layer']==layer and x['anchor']=='late' and x['full_oracle_vs_persistence']>0]
            byid=defaultdict(list)
            for x in rows:byid[x['id']].append(math.log(x['full_oracle_vs_persistence']))
            if byid:
                xs.append(layer);ys.append(math.exp(statistics.mean(statistics.mean(v) for v in byid.values())))
        ax.plot(xs,ys,'o-',label=benchmark)
    ax.axhline(1,color='gray');ax.set_xticks([5,17,29]);ax.set_xlabel('GLOBAL layer')
    ax.set_ylabel('Actual-future derivative MSE / persistence MSE');ax.set_yscale('log')
    ax.set_title('Evaluation-only finite-step linearization check');ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'future_oracle_by_layer.png',dpi=180);fig.savefig(out/'future_oracle_by_layer.pdf');plt.close(fig)
    save('input_hashes.json',inputs)
    save('final_complete.json',dict(passed=True,A1_complete=True,A2_launched=False,protocol_sha256=sha(ROOT/'protocol.json'),
        source_checkpoint='81aae497',report_source_sha256=sha(Path(__file__)),consumed_inputs=len(inputs),unique_item_seed_cells=len(canonical),
        complete_stage_scope='A1 diagnostic leaf only; A2 blocked by prespecified scientific gates'))
    print(json.dumps(dict(resources=resource_totals,coverage_groups=len(coverage),quality=quality)))


if __name__=='__main__':
    main()
