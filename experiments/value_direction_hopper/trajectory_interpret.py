"""Explicit research questions, with quantitative bounds on interpretation."""
from collections import defaultdict
import json
import numpy as np

from .experiment import atomic


def avg(values):
    return float(np.mean(values)) if values else None


def fmt(value, digits=3):
    return 'unavailable' if value is None else f'{value:.{digits}f}'


def corr(rows,x,y):
    values=[(r[x],r[y]) for r in rows if r.get(x) is not None and r.get(y) is not None]
    if len(values)<3:return dict(n=len(values),pearson=None)
    a,b=np.asarray(values).T
    return dict(n=len(values),pearson=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None)


def interpret(root,summary,samples,steps,aligned,paired,historical,audit):
    adaptive=[r for r in summary if r['regime']=='adaptive']
    indexed={(r['regime'],r['method']):r for r in summary}
    first=[r for r in aligned if r['regime']=='adaptive' and r['alignment']=='step' and r['step']==1]
    allsame=[r for r in aligned if r['regime']=='adaptive' and r['alignment']=='step']
    diagnostics={}
    for method in ('native_dense','kernel_dense','blasst_capped','kernel_gaussian32','kernel_blasst'):
        ss=[r for r in samples if r['regime']=='adaptive' and r['method']==method]
        rr=[r for r in steps if r['regime']=='adaptive' and r['method']==method]
        dd=[r for r in allsame if r['method']==method and r['baseline']=='kernel_dense']
        ff=[r for r in first if r['method']==method and r['baseline']=='kernel_dense']
        if not ss:continue
        joined=[]
        lookup={(r['id'],r['step']):r for r in dd}
        for r in rr:
            if (r['id'],r['step']) in lookup:
                d=lookup[r['id'],r['step']]
                joined.append(dict(r,confidence_loss=-d['confidence_delta'],disagreement=d['disagreement']))
        per_prompt=defaultdict(list)
        for r in joined:per_prompt[r['id']].append(r)
        prompt_stats=[]
        for rows in per_prompt.values():
            prompt_stats.append({k:avg([r[k] for r in rows if r[k] is not None])
                                 for k in ('attention_error','confidence_loss','disagreement','accepted')})
        total_instability=sum(r['unstable'] for r in rr if r['step']>1)
        diag=dict(n=len(ss),
            first_step_disagreement=avg([r['disagreement'] for r in ff]),
            first_step_confidence_delta=avg([r['confidence_delta'] for r in ff]),
            first_step_same_token_confidence_delta=avg([r['same_token_confidence_delta'] for r in ff if r['same_token_confidence_delta'] is not None]),
            aligned_same_token_confidence_delta=avg([r['same_token_confidence_delta'] for r in dd if r['same_token_confidence_delta'] is not None]),
            delayed_same_token=sum(r['delayed_same_token'] for r in dd),delayed_changed_token=sum(r['delayed_changed_token'] for r in dd),
            post_output_instability_fraction=sum(r['unstable_after_output'] for r in rr if r['step']>1)/max(1,total_instability),
            mean_unique_unstable=avg([r['unique_unstable_positions'] for r in ss]),
            top8_instability_share=avg([r['top8_instability_share'] for r in ss]),
            mean_accepted_first=avg([r['accepted'] for r in rr if r['step']==1]),
            mean_accepted_last=avg([r['accepted'] for r in rr if r['terminated']]),
            mean_confidence_first=avg([r['confidence_mean'] for r in rr if r['step']==1]),
            mean_confidence_last=avg([r['confidence_mean'] for r in rr if r['terminated']]),
            accepted_nonargmax_total=sum(r['accepted_nonargmax'] for r in ss),
            accepted_nonargmax_per_step=sum(r['accepted_nonargmax'] for r in ss)/sum(r['steps'] for r in ss),
            reached_48=sum(r['steps']==48 for r in ss),
            only_outside_blocks_stop=sum(r['only_outside_blocks_stop'] for r in ss),
            output_region_candidates=sum(r['first_output_region_stop'] is not None for r in ss),
            output_region_early_score=avg([r['output_region_stop_score'] for r in ss if r['output_region_stop_score'] is not None]),
            final_score_same_candidates=avg([r['score'] for r in ss if r['output_region_stop_score'] is not None]),
            output_region_early_worse=sum(r['output_region_stop_score']<r['score']-1e-6 for r in ss if r['output_region_stop_score'] is not None),
            prompt_error_confidence=corr(prompt_stats,'attention_error','confidence_loss'),
            prompt_error_disagreement=corr(prompt_stats,'attention_error','disagreement'),
            prompt_error_acceptance=corr(prompt_stats,'attention_error','accepted'),
            prompt_error_steps=corr(ss,'attention_error','steps'))
        diagnostics[method]=diag
    fixed={}
    for method in ('kernel_dense','kernel_gaussian32','kernel_blasst'):
        records=[r for r in samples if r['regime']=='fixed512' and r['method']==method]
        if not records:continue
        withstop=[r for r in records if r['early_stop_score'] is not None]
        fixed[method]=dict(n=len(records),would_stop_n=len(withstop),
            earlier_score=avg([r['early_stop_score'] for r in withstop]),
            final_score_same_examples=avg([r['score'] for r in withstop]),
            later_better=sum(r['score']>r['early_stop_score']+1e-6 for r in withstop),
            later_worse=sum(r['score']<r['early_stop_score']-1e-6 for r in withstop),
            mean_first_would_stop=avg([r['first_would_stop'] for r in withstop]))
        fixed[method]['accepted_nonargmax_per_step']=sum(r['accepted_nonargmax'] for r in records)/sum(r['steps'] for r in records)
        fixed[method]['checkpoints']={}
        for step in (4,8,16,32,48,128,256,512):
            at=[r for r in steps if r['regime']=='fixed512' and r['method']==method and r['step']==step]
            fixed[method]['checkpoints'][str(step)]=dict(n=len(at),score=avg([r['draft_score'] for r in at]),
                confidence=avg([r['confidence_mean'] for r in at]),entropy=avg([r['entropy_mean'] for r in at]))
    atomic(root/'interpretation.json',dict(diagnostics=diagnostics,fixed=fixed,
        correlation_warning='Prompt-level Pearson correlations, observational; shared-state operator errors but diverged generation states',
        interim=not audit['complete']))
    dense=historical['native_dense'];value=historical['kernel_gaussian32']
    step_ratio=value['original_steps']/dense['original_steps']
    time_ratio=value['original_wall_seconds']/dense['original_wall_seconds']
    lines=['','## Interpretation of the eleven questions','',
        ('These findings are provisional until collection and audit finish.' if not audit['complete'] else
         'All conclusions below use the completed diagnostic run; sample exposure and numerical controls remain relevant.'),'',
        f"The archived Gaussian32 run has {step_ratio:.3f} times as many decoder calls and {time_ratio:.3f} times the generation wall time. "
        f"Wall time divided by calls is {value['original_wall_seconds']/value['original_steps']*1000:.2f} ms versus "
        f"{dense['original_wall_seconds']/dense['original_steps']*1000:.2f} ms for native dense. "
        f"The exact accounting identity is {step_ratio:.3f} x {time_ratio/step_ratio:.3f} = {time_ratio:.3f}. "
        'This supports iteration amplification as a major cost, but wall/call includes prompt encoding and fixed overhead, so it does not establish a causal per-step kernel saving.','',
        '1. **Why more iterations?** The native stopping rule is conjunctive: stable argmax over the entire canvas and low mean processed entropy. '
        'Acceptance count is not an exit condition, and the final output is the argmax canvas rather than a permanently committed sequence. '
        'Lower acceptance can affect subsequent inputs through renoising, but it cannot by itself establish why the loop failed to stop. '
        'The following counts separate stable-but-insufficiently-confident steps from confident-but-unstable steps; other nonterminal steps may fail both.','',
        '|Method|Stable, not confident|Confident, not stable|Both fail|Criterion passed|Mean steps|', '|---|---:|---:|---:|---:|---:|']
    timing_lines=['','## Clean archived timing decomposition','',
        'A descriptive least-squares fit wall_seconds = intercept + slope x realized_steps separates fixed overhead from iteration cost. '
        'This fit is across prompts, not a randomized timing intervention. It is independent of the diagnostic instrumentation.','',
        '|Method|Intercept ms|Added-step slope ms|R squared|','|---|---:|---:|---:|']
    for method,h in historical.items():
        fit=h['descriptive_wall_regression']
        timing_lines.append(f"|{method}|{1000*fit['intercept_seconds']:.2f}|{1000*fit['seconds_per_added_step']:.2f}|{fit['r_squared']:.4f}|")
    vf=value['descriptive_wall_regression'];df=dense['descriptive_wall_regression']
    extra=(value['original_steps']-dense['original_steps'])*vf['seconds_per_added_step']
    slowdown=value['original_wall_seconds']-dense['original_wall_seconds']
    timing_lines+=['',f"Using the sparse fitted slope, the additional iterations account for {extra:.2f}s of the {slowdown:.2f}s observed wall-time increase "
        f"({100*extra/slowdown:.1f}%). The remaining fitted difference combines per-step and fixed costs. "
        'This is a model-based accounting decomposition, not proof that eliminating those steps preserves the same output or yields exactly that speedup. '
        'The Gaussian32 fitted marginal iteration cost is higher than native dense in this archive, so the premise that sparse decoder steps are cheaper is unsupported.']
    for r in adaptive:
        lines.append(f"|{r['method']}|{r['stable_not_confident_steps']}|{r['confident_not_stable_steps']}|{r.get('both_stopping_conditions_fail','pending regeneration')}|{r.get('stopping_criterion_passed','pending regeneration')}|{r['steps_per_sample']:.2f}|")
    lines+=['','2. **Lower confidence or changed predictions?** At equal steps, compare to the unpruned kernel control. '
        'The first call starts from identical randomized canvas inputs; later comparisons include accumulated state divergence. '
        'A different token from dense is not automatically a wrong answer. Counts below separate rejection of a matching dense token from rejection where predictions differ.','',
        '|Method|First-step top-1 disagreement|First-step confidence delta|Same-token confidence delta, aligned steps|Delayed same-token positions|Delayed differing-token positions|',
        '|---|---:|---:|---:|---:|---:|']
    for method,d in diagnostics.items():
        if d['first_step_disagreement'] is not None:
            lines.append(f"|{method}|{100*d['first_step_disagreement']:.2f}%|{fmt(d['first_step_confidence_delta'])}|{fmt(d['aligned_same_token_confidence_delta'])}|{d['delayed_same_token']}|{d['delayed_changed_token']}|")
    g=diagnostics.get('kernel_gaussian32')
    if g:
        delayed=g['delayed_same_token']+g['delayed_changed_token']
        lines+=['',f"For Gaussian32, {100*g['delayed_same_token']/max(1,delayed):.1f}% of these delayed position observations still predict the dense token. "
            f"First-step confidence also drops by {100*(g['first_step_same_token_confidence_delta'] or 0):.2f}pp when the predicted token matches. "
            'This supports a confidence component, alongside the nonzero first-step prediction disagreement. '
            'It does not identify the causal fraction of extra iterations attributable to either component. '
            f"Mean confidence at the final observed step is {fmt(g['mean_confidence_last'],5)}, versus {fmt(g['mean_confidence_first'],5)} initially; "
            f"{g['reached_48']}/{g['n']} prompts still reach the48-step cap."]
    lines+=['','3. **Few persistent tokens or widespread degradation?** Initial instability is excluded from concentration statistics. '
        'Repeated unstable positions count how many canvas locations change after the first prediction. The top-eight share summarizes concentration. '
        'Post-output positions are beyond the current predicted EOS or official output budget; these still affect stopping.','',
        '|Method|Mean repeated unstable positions /256|Top8 share of changes|Changes beyond output|Reached adaptive cap /N|',
        '|---|---:|---:|---:|---:|']
    for method,d in diagnostics.items():
        lines.append(f"|{method}|{fmt(d['mean_unique_unstable'])}|{100*d['top8_instability_share']:.2f}%|{100*d['post_output_instability_fraction']:.2f}%|{d['reached_48']}/{d['n']}|")
    lines+=['','A diagnostic restricted to the current returned-output span (predicted EOS or official budget) asks whether that span is stable/confident while the full canvas fails. '
        'This is computed from saved traces, without changing generation. Its first qualifying draft is scored against the final draft; a loss directly cautions against declaring this stopping rule safe.','',
        '|Method|Steps blocked only outside current output|Prompts with candidate|Candidate score|Final score, same prompts|Candidate worse prompts|',
        '|---|---:|---:|---:|---:|---:|']
    for method,d in diagnostics.items():
        lines.append(f"|{method}|{d['only_outside_blocks_stop']}|{d['output_region_candidates']}|{fmt(d['output_region_early_score'])}|{fmt(d['final_score_same_candidates'])}|{d['output_region_early_worse']}|")
    lines+=['','4. **When does divergence begin?** The first-step table directly tests onset before earlier sparse decisions can affect the canvas. '
        'Later curves and the raw per-position traces distinguish onset from persistence.','',
        '5. **Are early steps more sensitive?** Step curves show association, not a sensitivity experiment: states, temperature and the population of still-running prompts all change. '
        'A controlled early-versus-late routing intervention would be required to establish causal sensitivity. No dynamic sparsity intervention was made.','',
        '6. **BLASST comparison.** Compare the capped and calibrated rows separately. The capped path is restricted to the official lambda range; '
        'the matched-sparsity path retains the previously calibrated above-one extension. Achieved whole/global/local sparsity appears in the main table. '
        'A difference from capped BLASST at lower achieved sparsity is not a matched-budget algorithm comparison.','',
        '7. **Does attention error predict delayed progress?** The table uses prompt-averaged observations, avoiding treating each correlated layer/step as an independent example. '
        'Correlations are descriptive and do not establish mediation or causation.','',
        '|Method|Error vs confidence loss r (N)|Error vs token disagreement r (N)|Error vs accepted positions r (N)|Error vs steps r (N)|',
        '|---|---:|---:|---:|---:|']
    for method,d in diagnostics.items():
        fields=('prompt_error_confidence','prompt_error_disagreement','prompt_error_acceptance','prompt_error_steps')
        cells=[f"{fmt(d[k]['pearson'])} ({d[k]['n']})" for k in fields]
        lines.append('|'+method+'|'+'|'.join(cells)+'|')
    if g and g['prompt_error_steps']['pearson'] is not None and g['prompt_error_steps']['pearson']<=0:
        lines+=['','The sampled output-error metric does **not** show the hypothesized positive cross-prompt association with step count for Gaussian32. '
            'Do not use this metric alone to rank decoding risk. Task difficulty, attention-output norm, query coverage and changed trajectory states can confound this association.']
    lines+=['','8. **What happens at exactly512 steps?** These controls compare all-retained and pruned execution under identical kernel/mask conventions. '
        'The native temperature schedule is stretched to512 for both. Longer sampling can itself change predictions; similar final accuracy would support, but not prove, a stopping-policy explanation.','',
        '|Method|Steps/example|Completed|Sparsity|Accuracy|Delta vs kernel dense pp [95% CI]|',
        '|---|---:|---:|---:|---:|---|']
    for method in ('kernel_dense','kernel_gaussian32','kernel_blasst'):
        r=indexed.get(('fixed512',method))
        if not r:continue
        p=next((p for p in paired if p['regime']=='fixed512' and p['first']==method and p['second']=='kernel_dense'),None)
        delta='--' if p is None else f"{100*p['mean']:+.3f} [{100*p['lower']:+.3f}, {100*p['upper']:+.3f}]"
        lines.append(f"|{method}|512|{r['n']}|{100*r['whole_sparsity']:.2f}%|{100*r['accuracy']:.2f}%|{delta}|")
    fixed_comparison=next((p for p in paired if p['regime']=='fixed512' and p['first']=='kernel_gaussian32' and p['second']=='kernel_dense'),None)
    if audit['complete'] and fixed_comparison:
        p=fixed_comparison
        if p['upper']<0:
            lines+=['','The fixed-step result supports CaseB in this setup: an accuracy loss remains when realized iterations are equal. '
                'Stopping sensitivity alone does not explain that loss; its measured magnitude and interval are given above.']
        elif p['lower']>0:
            lines+=['','Gaussian32 scores higher in this fixed-budget comparison. This does not validate an earlier stopping rule or establish a general sparse advantage; '
                'the long annealing schedule and previously examined cohort limit that conclusion.']
        else:
            lines+=['','The paired interval includes zero. The fixed-budget accuracy difference is inconclusive at this sample size; '
                'failure to detect a difference is not an equivalence result and does not establish CaseA.']
    lines+=['','9. **Does evidence support denser early steps?** Early prediction differences make an early-density intervention a motivated next test. '
        'They do not show that this schedule will restore accuracy or reduce total latency. Persistent late instability may require attention beyond the first few steps.','',
        '10. **Could decoding stop earlier?** The fixed runs record when the native criterion would first stop while continuing to512. '
        'This provides answer-score evidence in addition to confidence. It still does not validate a newly chosen stopping policy on fresh examples.','',
        '|Method|Ever met stopper /N|Mean first stop|Score then|Final score on same prompts|Later better|Later worse|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method,d in fixed.items():
        lines.append(f"|{method}|{d['would_stop_n']}/{d['n']}|{fmt(d['mean_first_would_stop'])}|{fmt(d['earlier_score'])}|{fmt(d['final_score_same_examples'])}|{d['later_better']}|{d['later_worse']}|")
    lines+=['','The raw traces also record sampled proposals. Accepted proposals can differ from argmax even after a high-confidence draft; '
        'continued stochastic sampling can therefore perturb the next canvas. Per-step CSVs count accepted nonargmax proposals and their probability-based expected count. '
        'Their association with later errors is descriptive; this experiment does not intervene on sampling randomness.','',
        '|Method|Adaptive accepted nonargmax proposals /step|Fixed512 accepted nonargmax proposals /step|',
        '|---|---:|---:|']
    for method,d in diagnostics.items():
        noise_rate=fixed.get(method,{}).get('accepted_nonargmax_per_step')
        lines.append(f"|{method}|{fmt(d['accepted_nonargmax_per_step'])}|{fmt(noise_rate)}|")
    lines+=['','11. **Next iteration.** Use the paired fixed-step accuracy interval and the stopping decomposition together. '
        'If matched fixed-step accuracy remains lower, prioritize preservation of prediction trajectories over changing the stopping threshold. '
        'If accuracy is close and instability is concentrated after the returned answer, test a carefully defined output-relevant stopping criterion with held-out accuracy checks. '
        'A dense-early ablation is justified as a test when first-step disagreement is substantial; no schedule or stopping change has been validated here.','',
        'Causal limit: these measurements can identify which native stopping conditions fail and whether errors persist at equal iteration budgets. '
        'They do not prove that reducing attention-output error alone will fix generation. The all-retained kernel control is essential because the original native dense comparison also includes mask and arithmetic differences.']
    with (root/'report.md').open('a') as f:f.write('\n'.join(lines+timing_lines)+'\n')
