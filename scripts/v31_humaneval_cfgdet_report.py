"""Data-derived corrected HumanEval reporting, with independent rescore gates."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from scripts import v31_humaneval_cfgdet_study as campaign
from scripts import v31_official as official
from scripts import v27_humaneval as he
from scripts.v31_score_humaneval import run_all

MODES,LABELS,REFERENCES=campaign.MODES,campaign.LABELS,campaign.REFERENCES
verify,read,write,sha=campaign.verify,campaign.read,campaign.write,campaign.sha
base,selected_paths,extract,work=campaign.base,campaign.selected_paths,campaign.extract,campaign.work
FIELDS=('output_hash','denoise_forwards','canvases','output_tokens','sampler_calls','native_interval_steps')


def maps(root,mode,pass_name,label,stage='runs',private=False):
    result={}
    for path in selected_paths(root,mode,pass_name,stage)[label]:
        file=path/f'private/run_{label}.private.jsonl' if private else path/'records.jsonl'
        for row in base.jsonl(file):
            key=(row['panel_seed'],row['index'])
            if key in result:raise RuntimeError('duplicate question/seed in selected population')
            result[key]=row
    return result


def qualification_equivalence(root):
    proof=[];cohort=read(root/'protocol.json')['cohort']
    expected={(seed,index) for seed in (42,43) for index in cohort}
    for mode in MODES:
        for label in LABELS:
            a,b=[maps(root,mode,part,label,'qualification') for part in ('clean','audit')]
            pa,pb=[maps(root,mode,part,label,'qualification',True) for part in ('clean','audit')]
            if set(a)!=expected or set(b)!=expected:raise RuntimeError('qualification population mismatch')
            differences={key:sum(a[i][key]!=b[i][key] for i in expected) for key in FIELDS}
            differences['private_completion']=sum(pa[i]['completion']!=pb[i]['completion'] for i in expected)
            proof.append(dict(mode=mode,label=label,questions=6,seeds=[42,43],records=12,
                differences=differences,passed=not any(differences.values())))
        for part in ('clean','audit'):
            a=maps(root,mode,part,'control_before','qualification')
            b=maps(root,mode,part,'control_after','qualification')
            differences={key:sum(a[i][key]!=b[i][key] for i in expected) for key in FIELDS}
            proof.append(dict(mode=mode,pass_name=part,label='v31_bookends',differences=differences,
                              passed=not any(differences.values())))
    return proof


def score(study, mode, pass_name, paths):
    dest = study/f'private/scoring/{mode}/{pass_name}.json'
    if dest.exists():
        return read(dest)
    pool = study/f'private/pools/{mode}'
    gold = read(pool/'gold.json')
    pools = official.Pools(pool, ('thinking',))
    jobs, identities, reasons = [], [], Counter()
    fingerprints = {}
    for label in LABELS:
        records = []
        for path in paths[label]:
            f = path/f'private/run_{label}.private.jsonl'
            fingerprints[str(f.relative_to(study))] = sha(f)
            records.extend(base.jsonl(f))
        if len(records) != 164 or {r['index'] for r in records} != set(range(164)):
            raise RuntimeError('official scoring coverage mismatch: '+label)
        for r in records:
            _, binding, verified = official.bind(r, pools, False)
            if not verified or r['id'] != f"humaneval/{r['index']}" or r['panel_seed'] != 42:
                raise RuntimeError('official input binding failed')
            problem = gold[r['id']]
            code, why = extract(r['completion'], problem['entry_point'], mode == 'thinking_on')
            reasons[label+'|'+why] += 1
            jobs.append((problem, code))
            identities.append((label, str(r['index']), official.finish_class(r['finish_reason'])))
    results, timeouts = run_all(jobs, he)
    scores = defaultdict(dict)
    for (label, index, finish), (passed, outcome) in zip(identities, results):
        scores[label][index] = dict(passed=bool(passed), pass_and_finished=bool(passed) and finish == 'finished', finish=finish, outcome=outcome)
    result = dict(scores=dict(scores), extraction=dict(reasons), timeouts_first_pass=timeouts,
        input_sha256=fingerprints, protocol_sha256=sha(study/'protocol.json'))
    write(dest, result, True)
    return result


def summarize(study):
    verify(study)
    results = {}
    for mode in MODES:
        scores = {}
        paths = {}
        for pass_name in ('audit', 'clean'):
            paths[pass_name] = selected_paths(study, mode, pass_name)
            scores[pass_name] = score(study, mode, pass_name, paths[pass_name])
        summaries, arrays = {}, {}
        for label in LABELS:
            clean = {r['index']: r for p in paths['clean'][label] for r in base.jsonl(p/'records.jsonl')}
            audit = {r['index']: r for p in paths['audit'][label] for r in base.jsonl(p/'records.jsonl')}
            if set(clean) != set(range(164)) or set(audit) != set(clean):
                raise RuntimeError('full164 record coverage')
            values = np.array([[scores['clean']['scores'][label][str(i)]['passed'], clean[i]['denoise_forwards'], clean[i]['canvases'],
                clean[i]['output_tokens'], clean[i]['wall_s'], clean[i]['decode_s'], clean[i]['prefill_s']] for i in range(164)], dtype=float)
            arrays[label] = values
            a = [audit[i]['receipts']['adapter'] for i in range(164)]
            required = ('global_eligible_tiles', 'global_kept_tiles', 'native_local_eligible_rectangles', 'native_local_skipped_tiles')
            totals = {k: sum(x[k] for x in a) for k in required}
            e, k, local = totals['global_eligible_tiles'], totals['global_kept_tiles'], totals['native_local_eligible_rectangles']
            if not all(r.get('native_interval_trace_available') is True and
                       sum(r['native_interval_steps'])==r['denoise_forwards'] for r in audit.values()):
                raise RuntimeError('required native interval telemetry absent')
            steps = [int(n) for r in audit.values() for n in r['native_interval_steps']]
            item = dict(questions=164, correct=int(values[:, 0].sum()), accuracy=values[:, 0].mean(),
                audit_correct=sum(x['passed'] for x in scores['audit']['scores'][label].values()),
                pass_and_finished=sum(x['pass_and_finished'] for x in scores['clean']['scores'][label].values()),
                N_mean=values[:, 1].mean(), N_median=float(np.median(values[:, 1])), N_p90=float(np.quantile(values[:, 1], .9)),
                N_p95=float(np.quantile(values[:, 1], .95)), N_max=int(values[:, 1].max()), C_mean=values[:, 2].mean(),
                N_per_C=values[:, 1].sum()/values[:, 2].sum(), T_mean=values[:, 3].mean(),
                W_mean_s=values[:, 4].mean(), W_median_s=float(np.median(values[:, 4])), W_p90_s=float(np.quantile(values[:, 4], .9)),
                S_mean_s=values[:, 5].mean(), P_mean_s=values[:, 6].mean(), amortized_S_per_N_ms=1000*values[:, 5].sum()/values[:, 1].sum(),
                generation_caps=sum(x['finish'] == 'capped' for x in scores['clean']['scores'][label].values()),
                audited_canvas_steps_available=len(steps), audited_cap_canvases=sum(x >= 48 for x in steps),
                global_sparsity=1-k/e, local_sparsity=totals['native_local_skipped_tiles']/local,
                overall_sparsity=(e-k)/(e+local), physical_totals=totals,
                clean_audit_hash_differences=sum(clean[i]['output_hash'] != audit[i]['output_hash'] for i in range(164)),
                clean_audit_N_differences=sum(clean[i]['denoise_forwards'] != audit[i]['denoise_forwards'] for i in range(164)),
                clean_audit_interval_differences=sum(clean[i]['native_interval_steps'] != audit[i]['native_interval_steps'] for i in range(164)),
                clean_audit_score_differences=sum(scores['clean']['scores'][label][str(i)]['passed'] != scores['audit']['scores'][label][str(i)]['passed'] for i in range(164)))
            if label not in REFERENCES:
                item['first_canvas'] = work(list(audit.values()))
            extra = Counter()
            timing = Counter()
            for x in a:
                extra.update(x.get('value_refinement_work', {}))
                if x.get('value_audit_timings'):
                    timing.update(x['value_audit_timings'])
                if 'value_anchor_module_cuda_elapsed_ms' in x:
                    timing['anchor_module_cuda_elapsed_ms'] += x['value_anchor_module_cuda_elapsed_ms']
            item['refinement_work'] = dict(extra)
            item['audit_selector_times_ms'] = dict(timing)
            summaries[label] = item
        rng = np.random.default_rng(1729)
        indices = rng.integers(0, 164, size=(20000, 164))
        contrasts = {}
        for label in LABELS:
            if label == 'control_before':
                continue
            ref, val = arrays['control_before'], arrays[label]
            delta = (val[:, 0]-ref[:, 0])*100
            draws = delta[indices].mean(1)
            speed = ref[indices, 4].mean(1)/val[indices, 4].mean(1)
            nd = (val[:, 1]-ref[:, 1])[indices].mean(1)
            contrasts[label] = dict(accuracy_delta_pp=float(delta.mean()), accuracy_delta_ci95=np.quantile(draws, [.025, .975]).tolist(),
                W_speedup=float(ref[:, 4].mean()/val[:, 4].mean()), W_speedup_ci95=np.quantile(speed, [.025, .975]).tolist(),
                N_delta=float((val[:, 1]-ref[:, 1]).mean()), N_delta_ci95=np.quantile(nd, [.025, .975]).tolist(),
                candidate_only_correct=int(((val[:, 0] == 1) & (ref[:, 0] == 0)).sum()),
                control_only_correct=int(((val[:, 0] == 0) & (ref[:, 0] == 1)).sum()))
        results[mode] = dict(methods=summaries, paired_vs_v31=contrasts,
            threshold=read(study/f'calibration/{mode}/full164_threshold.json'))
    receipts = [read(p) for p in study.rglob('receipt.json') if '/private/' not in str(p)]
    report = dict(status='complete', protocol_sha256=sha(study/'protocol.json'), modes=results,
        gpu_reservation_s=sum(r['gpu_reservation_s'] for r in receipts),
        limitations=['primary seed42; six-question seed43 confirmation is descriptive; calibration overlaps prompts',
            'deterministic model/helper compiler policy disables horizontal combo fusion; native graph modes remain',
            'V2/V3a Gaussian32 all GLOBAL; swaps/residual modify GLOBAL layer29 only',
            'attention-consumer sparsity from matched audit; extra selector/PV work is separate',
            'Q128/KV64 geometry rectangles, not a count of CUDA CTAs',
            'S/N is amortized across native trajectory, not a controlled per-forward price'])
    protocol=read(study/'protocol.json')
    report['observing_fa4_no_overlap']=protocol.get('observing_fa4_no_overlap',False)
    report['shared_references']=protocol.get('shared_references')
    if protocol.get('observing_fa4_no_overlap'):
        report['limitations'].append('mass-observing FA4 disables intra-warp-group overlap; plain paths stay native; its cost is included in measured latency')
    if protocol.get('shared_references'):
        report['limitations'].append('dense FULL/PIECEWISE/all-kept controls are the same completed common-source runs used for K256; they are not independent additional trials or additional GPU charges')
    write(study/'summary.json', report, True)
    lines = [f"# HumanEval164 K{read(study/'protocol.json')['prefix_budget_tokens']} seed42", '', 'Clean native accuracy and request latency; separate audit sparsity. Full inputs, generated text, gold and token IDs remain private.', '']
    for mode, result in results.items():
        t = result['threshold']
        lines += [f'## {mode}', '', f"V2 threshold: {t['threshold']:.12g}; full164 first-canvas matching: {t['matching']}.", '',
            '| Method | Correct/164 | N/problem | C/problem | N/C | T | S (s) | W (s) | GLOBAL sparsity | Overall sparsity |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label in LABELS:
            m = result['methods'][label]
            lines.append(f"| {label} | {m['correct']}/164 | {m['N_mean']:.3f} | {m['C_mean']:.3f} | {m['N_per_C']:.3f} | {m['T_mean']:.2f} | {m['S_mean_s']:.5f} | {m['W_mean_s']:.5f} | {100*m['global_sparsity']:.3f}% | {100*m['overall_sparsity']:.3f}% |")
        lines += ['', 'Paired question-bootstrap intervals, technical bookend results, audit/clean drift, caps, physical totals and selector overhead are in summary.json.', '']
    lines += ['## Limits', '', *['- '+s for s in report['limitations']], '', f"Charged GPU reservation: {report['gpu_reservation_s']:.3f} seconds. Older LongBench campaigns remain paused."]
    (study/'summary.md').write_text('\n'.join(lines)+'\n')
    bound = ['protocol.json', 'spec.md', 'summary.json', 'summary.md', 'qualification/scorer_gate.json']
    write(study/'native_complete.json', dict(status='native_and_primary_scoring_complete', protocol_sha256=sha(study/'protocol.json'),
        files_sha256={n: sha(study/n) for n in bound}, source_commit=read(study/'protocol.json')['source_commit'],
        expected_unique_questions=164, thinking_modes=MODES, passes=['audit', 'clean'],
        gpu_reservation_s=report['gpu_reservation_s']), True)
    write(study/'state.json', dict(stage='primary_scored_pending_independent_verification', active_attempt=None, gpu_reservation_s=report['gpu_reservation_s']))
    print(dict(event='primary_scored_pending_independent_verification', gpu_reservation_s=report['gpu_reservation_s']), flush=True)


def primary_equivalence(root):
    proof=[]
    for mode in MODES:
        for label in LABELS:
            a,b=[maps(root,mode,part,label) for part in ('clean','audit')]
            pa,pb=[maps(root,mode,part,label,private=True) for part in ('clean','audit')]
            expected={(42,i) for i in range(164)}
            if set(a)!=expected or set(b)!=expected:raise RuntimeError('primary164 population mismatch')
            differences={key:sum(a[i][key]!=b[i][key] for i in expected) for key in FIELDS}
            differences['private_completion']=sum(pa[i]['completion']!=pb[i]['completion'] for i in expected)
            proof.append(dict(mode=mode,label=label,questions=164,differences=differences,
                              passed=not any(differences.values())))
        for part in ('clean','audit'):
            a=maps(root,mode,part,'control_before');b=maps(root,mode,part,'control_after')
            differences={key:sum(a[i][key]!=b[i][key] for i in a) for key in FIELDS}
            proof.append(dict(mode=mode,pass_name=part,label='v31_bookends',differences=differences,
                              passed=not any(differences.values())))
    return proof


def score_qualification(root):
    """Score both predeclared seeds only after work-only thresholds are frozen."""
    dest=root/'private/scoring/qualification.json'
    if dest.exists():return read(dest)
    jobs=[];identities=[]
    for mode in MODES:
        gold=read(root/f'private/pools/{mode}/gold.json')
        pool=official.Pools(root/f'private/pools/{mode}',('thinking',))
        for part in ('clean','audit'):
            for label in LABELS:
                for (seed,index),r in maps(root,mode,part,label,'qualification',True).items():
                    _,_,okay=official.bind(r,pool,False)
                    if not okay:raise RuntimeError('qualification scorer binding failed')
                    code,why=extract(r['completion'],gold[r['id']]['entry_point'],mode=='thinking_on')
                    jobs.append((gold[r['id']],code))
                    identities.append(dict(mode=mode,pass_name=part,label=label,seed=seed,index=index,
                                           extractor=why,finish=official.finish_class(r['finish_reason'])))
    results,timeouts=run_all(jobs,he)
    rows=[dict(identity,passed=bool(result[0]),outcome=result[1]) for identity,result in zip(identities,results)]
    result=dict(records=rows,timeouts_first_pass=timeouts,protocol_sha256=sha(root/'protocol.json'),
                interpretation='six examined questions per seed; seed43 is descriptive, not seed-robust confirmation')
    write(dest,result,True);return result


def make_report(root,dest):
    verify(root)
    proof=primary_equivalence(root)
    proof_file=root/'receipts/full164_audit_clean_equivalence.json'
    if not proof_file.exists():write(proof_file,dict(comparisons=proof),True)
    elif read(proof_file)!=dict(comparisons=proof):raise RuntimeError('population equivalence proof drift')
    if not all(x['passed'] for x in proof):
        raise RuntimeError('full164 reproducibility failed; keep passes separate and withhold speedup claims')
    if not (root/'native_complete.json').exists():summarize(root)
    else:
        for rel,digest in read(root/'native_complete.json')['files_sha256'].items():
            if sha(root/rel)!=digest:raise RuntimeError('native summary identity drift')
    qualification=score_qualification(root)
    from scripts.v31_humaneval_cfgdet_verify import verify as independent_verify,rescore
    checked=independent_verify(root)
    rescored=rescore(root)
    if dest.exists():raise RuntimeError('new immutable analysis directory required')
    dest.mkdir(parents=True)
    write(dest/'independent_verification.json',checked,True)
    write(dest/'independent_rescoring.json',rescored,True)
    report=read(root/'summary.json');p=read(root/'protocol.json')
    items=[]
    for mode in MODES:
        for part in ('clean','audit'):
            scores=read(root/f'private/scoring/{mode}/{part}.json')['scores']
            for label in LABELS:
                for (_,index),r in maps(root,mode,part,label).items():
                    a=(r.get('receipts') or {}).get('adapter') or {}
                    record=dict(mode=mode,pass_name=part,method=label,index=index,seed=42,
                        prefix_budget_tokens=p['prefix_budget_tokens'] if label not in REFERENCES else None,
                        passed=scores[label][str(index)]['passed'],finish=scores[label][str(index)]['finish'],
                        N=r['denoise_forwards'],C=r['canvases'],T=r['output_tokens'],
                        S_s=r['decode_s'],W_s=r['wall_s'],amortized_S_per_N_ms=1000*r['decode_s']/r['denoise_forwards'],
                        native_interval_steps=r['native_interval_steps'],prompt_tokens=r['prompt_tokens'],
                        output_hash=r['output_hash'],measured_cuda_captures=r['cuda_graph_captures'])
                    for field in ('global_eligible_tiles','global_kept_tiles','native_local_eligible_rectangles',
                                  'native_local_skipped_tiles'):
                        record[field]=a.get(field) if part=='audit' else None
                    items.append(record)
    with (dest/'per_item_metrics.jsonl').open('x') as f:
        for row in items:f.write(json.dumps(row,sort_keys=True)+'\n')
    extra=[]
    for mode in MODES:
        for label in LABELS:
            rows=[r for r in qualification['records'] if r['mode']==mode and r['pass_name']=='clean' and
                  r['label']==label and r['seed']==43]
            extra.append(dict(mode=mode,label=label,seed=43,questions=len(rows),correct=sum(r['passed'] for r in rows)))
    write(dest/'seed43_descriptive.json',dict(results=extra,questions_previously_examined=True),True)
    write(dest/'summary.json',dict(**report,numerical_policy=p['numerical_policy_sha256'],
        audit_clean_equivalence=proof,independent_prefix_replication=read(root/'receipts/independent_prefix_replication.json'),
        additional_seed=extra,independent_rescoring=rescored),True)
    shutil_copy=root/'summary.md'
    (dest/'summary.md').write_text(shutil_copy.read_text()+
        '\nAudit/clean output, interval and private-completion comparisons pass for all164 questions per method. '
        'Independent engine prefix replication passes. Independent official rescores agree. '
        'Seed43 covers six examined questions per method/mode and remains descriptive.\n')
    table=['% Qualified deterministic compiler; native FULL/PIECEWISE consumers.','\\begin{tabular}{llrrrrrr}',
        '\\toprule','Mode & Method & Correct & $N$ & $N/C$ & $W$ (s) & Speedup & GLOBAL sparsity \\\\', '\\midrule']
    for mode in MODES:
        reference=report['modes'][mode]['methods']['control_before']
        for label in LABELS:
            if label=='control_after':continue
            m=report['modes'][mode]['methods'][label]
            name=label.replace('_','\\_')
            table.append(f"{mode.replace('_',' ')} & {name} & {m['correct']}/164 & {m['N_mean']:.2f} & {m['N_per_C']:.2f} & {m['W_mean_s']:.4f} & {reference['W_mean_s']/m['W_mean_s']:.3f} & {100*m['global_sparsity']:.2f}\\% \\\\")
    table+=['\\bottomrule','\\end{tabular}']
    (dest/'paper_table.tex').write_text('\n'.join(table)+'\n')
    paragraph=(f'We evaluate all164 HumanEval tasks with seed42, thinking enabled and disabled, '
        f'and a selectable prefix budget of {p["prefix_budget_tokens"]} tokens '
        f'({p["prefix_budget_tokens"]//64} KV64 blocks per query-head/Q128 tile); canvas/boundary tiles and LOCAL attention remain dense. '
        'V2 uses a work-calibrated online threshold. Native sampling and stopping formulas are preserved. '
        'Deterministic Inductor options apply explicitly to the model and unchanged native helper functions; '
        'horizontal combo fusion is disabled, with native FULL/PIECEWISE graph modes retained for matched controls. '
        'We separate clean request timing from audit accounting and verify matching outputs and denoising intervals '
        'before associating their metrics. Independent engine prefixes and official rescoring validate reproducibility. '
        'Question-bootstrap intervals condition on the observed engines; six examined tasks at seed43 are descriptive. '
        'Sparsity counts denoising GLOBAL and native-window LOCAL Q128/KV64 rectangles, excluding prefill/commits '
        'and additional selector/PV computation; complete selector/correction cost remains in measured request latency.')
    (dest/'paper_methods.txt').write_text(paragraph+'\n')
    write(dest/'analysis_complete.json',dict(status='complete',protocol_sha256=sha(root/'protocol.json'),
        analysis_source_sha256=sha(Path(__file__)),files_sha256={f.name:sha(f) for f in dest.iterdir() if f.is_file()}),True)
    write(root/'final_complete.json',dict(status='complete',protocol_sha256=sha(root/'protocol.json'),
        native_complete_sha256=sha(root/'native_complete.json'),analysis_complete_sha256=sha(dest/'analysis_complete.json'),
        expected_unique_questions=164,seed=42,additional_seed=43,additional_seed_questions=6,
        thinking_modes=MODES,passes=['audit','clean'],official_rescore_mismatches=0),True)
    write(root/'state.json',dict(stage='complete',active_attempt=None,resumable=True,
                               final_complete_sha256=sha(root/'final_complete.json')))


