"""Report narrative and work/information accounting, only after final audit."""
import json
import math
import os
import re
import shlex
from collections import Counter
from pathlib import Path


def historical_evidence_root(root, contract):
    """Resolve a reduced bundle's history only through its pinned parent contract."""
    if not contract.get('user_authorized_scope_reduction'):
        return root
    from .protocol import sha
    plan = json.loads((root/'iteration_plan.json').read_text())
    source = Path(plan['original_contract'])
    expected = contract['original_contract_sha256']
    if (plan['original_contract_sha256'] != expected
            or contract['sources'].get(str(source)) != expected
            or sha(source.read_bytes()) != expected):
        raise ValueError('historical contract is not pinned to this report')
    return source.parent


def historical_artifact_links(root, history):
    names = ('screening_pruning.json', 'screening_findings.md',
        'mechanism_diagnosis.md', 'mass_exact_screening_findings.md',
        'guarded_ranking_screen_summary.csv', 'sol_distribution_diagnosis.csv',
        'execution_notes.md', 'failures.jsonl')
    return ', '.join(f'[{name}]({Path(os.path.relpath(history/name, root)).as_posix()})'
        for name in names if (history/name).is_file())


def report_failure_rows(root, history):
    """Keep parent attempts distinct from current-bundle attempts, without loss."""
    roots = [history] if root.resolve() == history.resolve() else [history, root]
    return [dict(bundle=str(directory), **row) for directory in roots
        for row in execution_failure_rows(directory)]


def reproduction_command(root, history):
    """Regenerate the audited bundle, not an incomplete superseded parent."""
    prefix = "CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. python -m "
    if root.resolve() != history.resolve():
        return (prefix + 'experiments.diffusion_gemma_value_aware.efficient_iteration '
            + 'report --root ' + shlex.quote(str(history)))
    return (prefix + 'experiments.diffusion_gemma_value_aware.report '
        + '--output ' + shlex.quote(str(root)))


def dense_trec_format_rows(root, setup):
    """Expose first-line scoring limitations without cleaning or rescoring text."""
    from .run import shard_path
    from .protocol import sha
    result = []
    for row in setup['final']:
        if row['benchmark'] != 'longbench' or row['task'] != 'trec':
            continue
        path = shard_path(root, 'final', 'dense', row['id'])
        data = path.read_bytes()
        shard = json.loads(data)
        # Exactly the inherited scorer's preprocessing, not an alternate parser.
        first_line = shard['prediction'].lstrip('\n').split('\n')[0]
        result.append(dict(id=row['id'], condition='dense', first_line=first_line,
            first_line_is_thought=first_line == 'thought',
            stored_score=shard['score'], source=str(path), source_sha256=sha(data)))
    return result


def trec_format_note(records):
    if not records:
        return ''
    affected = sum(r['first_line_is_thought'] for r in records)
    if not affected:
        return ''
    return (f'**TREC formatting limitation:** {affected}/{len(records)} dense '
        'predictions have literal `thought` as their first line. The inherited '
        'official-style TREC preprocessing scores only that line, not the label '
        'on a later line. The zero TREC baseline therefore cannot be interpreted '
        'as a clean test of classification ability, and its contribution to the '
        'five-task macro is uninformative about sparsity robustness. Canonical '
        'scores, prompts and output parsing remain unchanged; no post-hoc '
        'cleaning/rescoring is substituted. See [raw-shard format evidence]'
        '(dense_trec_format_audit.json).\n\n')


def threshold_text(row):
    if row.get('beta') is not None:return f"β={row['beta']:.6f}"
    if not row.get('thresholds'):return 'none / ranking budget'
    parts=[]
    for kind in ('local','global'):
        entry=row['thresholds'][kind]
        if 'log_scale' in entry:
            value=f"exp({entry['log_scale']:.6g})/L"
            if entry.get('cap_one'):value=f'min(1, {value})'
        else:value=f"exp({entry['log_threshold']:.6g})"
        if entry.get('unattainable') and entry.get('cap_one'):value='1 (unattainable)'
        parts.append(f'{kind}: {value}')
    return '; '.join(parts)


def calibration_policy_row(policy,source,final_conditions):
    """Selection-only sources and repaired cap-one labels remain explicit."""
    p=policy
    return dict(benchmark=p['benchmark'],method=p['name'],target=p['target'],
        used_by_final_condition=Path(source).stem in final_conditions,
        target_metric='pv_omission' if p['config']['method'] in ('compensate','zero_pv') else 'physical_sparsity',
        local_measured=p['measured']['local'],global_measured=p['measured']['global'],
        misses=p.get('target_miss',{k:abs(v-p['target'])>.02 for k,v in p['measured'].items()}),
        cap1_unattainable=p.get('cap_one_unattainable') or p.get('independent_cap1_repair',{}).get('fixed_at_one'),
        threshold_policy=p['policy'],source=source)


def execution_failure_rows(root):
    """Historical failed attempts, never inferred missing/failed final samples."""
    path=root/'failures.jsonl'
    if not path.exists():return []
    groups={}
    with path.open() as handle:
        for line in handle:
            if not line.strip():continue
            failure=json.loads(line)
            trace=failure.get('traceback','')
            last=trace.strip().splitlines()[-1] if trace.strip() else str(failure.get('error','unspecified'))
            exception=re.match(r'^([\w.]+(?:Error|Exception)):',last)
            kind=exception.group(1) if exception else 'recorded error'
            key=(failure.get('stage','unspecified'),failure.get('condition','unspecified'),kind)
            entry=groups.setdefault(key,dict(attempts=0,samples=set(),other_gpu_pids=set()))
            entry['attempts']+=1
            if failure.get('id') is not None:entry['samples'].add(failure['id'])
            entry['other_gpu_pids'].update(re.findall(r'Process (\d+) has [\d.]+ GiB memory in use',trace))
    return [dict(stage=stage,condition=condition,error_type=kind,failed_attempts=value['attempts'],
        distinct_sample_ids=len(value['samples']),reported_other_gpu_pids=sorted(value['other_gpu_pids'],key=int))
        for (stage,condition,kind),value in sorted(groups.items())]


def compatibility_rows():
    return [
        dict(method='Original / aggressive BLASST',decision='After token QK and block maximum',metadata='None beyond m / valid counts',
             extra='Per-row gap + all-row reduction; aggressive variant tracks maximum of all preceding blocks',
             information='Streaming; no future logits or current dense output',omission='Softmax and PV, NOT QK'),
        dict(method='Value gap / mass bound / mass×value / output risk',decision='After token QK and block maximum; pre-block retained m and Z',
             metadata='Pooled norm per block/native KV head + full-KV reference RMS',extra='Scalar risk per query row + all-row reduction',
             information='Streaming once metadata/global reference reduction is ready',omission='Softmax and PV, NOT QK'),
        dict(method='Token aligned',decision='After token QK + each token log norm',metadata='One norm/log norm per KV token/native head',
             extra='Add to each logit before maximum; retained softmax still uses original logits',information='Streaming with token metadata',omission='Softmax and PV, NOT QK'),
        dict(method='Centered running / previous output',decision='After token QK + mass bound',metadata='Block mean vector and radius; optional previous-step Q×d output per attention head/layer',
             extra='O(Bq d) distance per candidate; running output warmup; invalidate stale previous canvas/step',information='Streaming or prior-step data only; no current dense output',omission='Softmax and PV, NOT QK'),
        dict(method='Exact-mass refinement',decision='After exact block exp/sum + retained pre-block Z',metadata='None for mass-only; centered metadata for combined variant',
             extra='Block softmax must be evaluated even when deleted',information='Streaming exact ONLINE mass, not exact final mass',omission='PV only; physical deletion does not imply softmax was skipped'),
        dict(method='Mean compensation / zero-PV control',decision='Risk after QK; exact exp/sum retained for every block',metadata='Block mean and radius for compensation',
             extra='Mean compensation adds O(Bq d) scaled-vector accumulation; zero control omits it',information='Streaming; exact denominator retained',omission='Full PV replaced, no physical deletion or softmax skipping'),
        dict(method='QK top-k',decision='All eligible block maxima available',metadata='Per-query-tile/head vector of block scores',
             extra='Global selection and a further pass or retained intermediates',information='Future block information; diagnostic, not one-pass streaming',omission='Mask opportunity only; QK already computed'),
        dict(method='True mass / contribution top-k or top-p',decision='All logits and final normalization; contribution also needs PV',metadata='Per-block exact masses or contributions',
             extra='Global selection; top-p accumulates nonnegative mass/norm',information='Nondeployable diagnostic reference in this experiment, not a task-optimal oracle',omission='No compute-saving claim from computing the selection itself'),
        dict(method='Sol mean-QK, Gaussian or ranking',decision='Mean Q and all eligible mean K proxies',metadata='Block means; optional RMS value scalar',
             extra='Whole eligible-row mean/variance or global ranking; log value factor',information='Pre-QK proxy possible, but full population reduction/selection required',omission='Potential QK+softmax+PV mask; current emulation computes all diagnostics'),
    ]


def write_report(root,setup,contract,rows,per_task,comparisons,correlations):
    from .report import table,csv_write
    from .report_metrics import work_opportunities
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write
    audit=json.loads((root/'audit.json').read_text());assert audit['complete']
    history=historical_evidence_root(root,contract)
    selected=[r for r in rows if r['split'] in ('full','heldout24')]
    main=[dict(benchmark=r['benchmark'],split=r['split'],method=r['condition'],pooling=r['pooling'],n=r['count'],
        target=None if r['target'] is None else 100*r['target'],threshold=threshold_text(r),
        score=100*r['accuracy'],dense_delta_pp=100*r['delta'],physical=100*r['overall_physical_sparsity'],
        global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'],
        target_metric='pv_omission' if r['method'] in ('compensate','zero_pv') else 'physical_sparsity',
        pv_omission=100*r['overall_pv_omission'],global_pv=100*r['global_pv_omission'],local_pv=100*r['local_pv_omission'],
        retained_mass=100*r['overall_denominator_mass'],exact_PV_mass=100*r['overall_mass'],
        output_error=r['overall_relative_error'],agreement=100*r['token_agreement']) for r in selected]
    work=[dict(benchmark=r['benchmark'],split=r['split'],method=r['condition'],eligible=r['overall_eligible'],
        deleted=r['overall_skipped'],softmax_skipped=r['overall_softmax_skipped'],PV_omitted=r['overall_pv_omitted'],
        mean_compensated=r['overall_compensated'],prefix_s=100*r['overall_prefix_sparsity'],canvas_s=100*r['overall_canvas_sparsity'],
        mixed_boundary_s=100*r['overall_boundary_sparsity'],mass_selection_gap=r['overall_mass_selection_gap'],
        **work_opportunities(r)) for r in selected]
    csv_write(root/'main_ablation.csv',main);csv_write(root/'execution_work.csv',work)
    compatibility=compatibility_rows();csv_write(root/'flashattention_compatibility.csv',compatibility)
    task_setup=[]
    for (benchmark,task),count in sorted(Counter((r['benchmark'],r['task']) for r in setup['final']).items()):
        prompts=[r for r in setup['final'] if r['benchmark']==benchmark and r['task']==task]
        length=[len(r['prompt_tokens']) if isinstance(r['prompt_tokens'],list) else r['prompt_tokens'] for r in prompts]
        task_setup.append(dict(benchmark=benchmark,task=task,final=count,
            calibration=sum(r['benchmark']==benchmark and r['task']==task for r in setup['calibration']),
            development=sum(r['benchmark']==benchmark and r['task']==task for r in setup['development']),
            prompt_tokens_min=min(length),prompt_tokens_max=max(length),budgets=sorted({r['generation_budget'] for r in prompts})))
    policies=[]
    for source in contract['sources']:
        if 'verified_policies/' not in source:continue
        p=json.loads(Path(source).read_text())
        policies.append(calibration_policy_row(p,source,contract['conditions']))
    csv_write(root/'threshold_calibration_audit.csv',policies)
    text='# Value-aware BLASST on DiffusionGemma — 128×64\n\n'
    text+=f"Completed and audited: {audit['completed']} final sample-condition shards, {len(contract['conditions'])} conditions. This is a reference-mask quality experiment on one H100, not a kernel benchmark.\n\n"
    text+='## Setup and data separation\n\n'
    text+=table(task_setup,['benchmark','task','final','calibration','development','prompt_tokens_min','prompt_tokens_max','budgets'])+'\n\n'
    text+='AIME26 uses all30 problems, with the original six calibration IDs2,8,14,20,23,30 also included in full30. Heldout24 is the confirmatory subset. LongBench reuses the exact controlled50 prompts; calibration10 and development5 are disjoint deterministic next examples, never final examples. Task scorers/prompts/budgets and source hashes are pinned in `setup.json`.\n\n'
    text+=f"BF16 model revision `{setup['revision']}`; thinking off, native canvas256, maximum48 denoising steps, seeds reset per manifest row. Requested temperature0 leaves the native0.4–0.8 schedule enabled: this is seeded native decoding, **not greedy decoding**. Prompts, budgets and unrelated denoising settings are audited against dense. Matching dense calibration results are reused.\n\n"
    text+='The measured scope is all30 **decoder denoising-attention layers** (five global layers5,11,17,23,29 and25 local layers), all16 query heads, with native GQA expansion. Encoder/prefill remains shared dense and is outside the routing denominator. “Whole” therefore means whole measured decoder attention, not end-to-end model FLOPs. Both prefix and canvas can be skipped. A mixed boundary tile is counted exactly once, separately from full-prefix and full-canvas tiles. Structurally impossible tiles are excluded; partial/mandatory-support tiles remain in the eligible count.\n\n'
    text+='## Selection and threshold rules\n\n'
    if history != root:
        text+='Earlier screening, refinement and failure records remain in the parent bundle; they are historical evidence, not extra completed conditions in this reduced final comparison. '
        text+=historical_artifact_links(root,history)+'\n\n'
        if (root/'screening_findings.md').exists():
            text+='The completed reduced calibration screen is described separately in [screening findings](screening_findings.md); those are tuning results, not the held-out final comparison.\n\n'
    text+='Initial pooling selection minimized calibration-only mean relative attention-output error over both benchmarks, both attention types and matched50/75% budgets. All five norm poolings were screened: max, mean, RMS and p95 of token norms, plus norm of the vector mean (not interchangeable with mean of norms). Separate LongBench development and six AIME calibration problems support iteration; no heldout scores select thresholds/candidates. The frozen decision and all provenance are in `candidate_decision.json` and `final_contract.json`.\n\n'
    text+='New risks use a separate scalar log threshold for local/global layers, shared across their heads/layers/steps, verified on sparse calibration trajectories. Dense-state physical-risk quantiles provide proposals, not achieved-sparsity claims. Each tile is removed only if **every valid query row** has risk strictly below threshold. First support is retained. New criteria use pre-block retained softmax state. The aggressive BLASST reference deliberately preserves its previous maximum over all preceding blocks, including skipped ones; this is material for λ>1.\n\n'
    text+='[BLASST Algorithm1](https://arxiv.org/html/2512.12087v3#S3) instead compares against the updated maximum. Our implementation-level analysis gives identical strict skip votes for λ≤1, but not for λ>1: the aggressive reference is the tested repository extension, not an unmodified paper algorithm above1. See `literature_verification.md` for the primary-source check.\n\n'
    text+='BLASST reuses the existing exponential sparsity/length fit and physical-margin correction, with λ(L)=exp(logκ)/L; L is the valid KV length measured for that attention call, not raw text length. AIME aggressive calibration starts from the exact existing refined policy. Original BLASST caps λ at1. An unattainable attention type uses exactλ1 independently; its failure never forces the other type to1. Sparse verification traces and calibration boundary measurements are in `threshold_calibration_audit.csv`; misses without a boundary proof are only “unattained in the tested range.”\n\n'
    text+='**Mechanism-attribution limitation:** comparisons with legacy BLASST evaluate whole routing policies, not a value-only intervention. They also change the running-maximum convention and the threshold parameterization (BLASST’s inverse-length rule versus new scalar risk thresholds). Near-constant token-value norms mean that token-aligned routing gains alone do not demonstrate a useful value signal. The mass-only versus mass×value control shares the retained-state convention and isolates the added value factor more closely, with independently calibrated thresholds. A claim specifically attributing a value-gap gain to V would require an otherwise identical retained-state, same-threshold-family no-value control; matched overall sparsity alone does not establish that attribution.\n\n'
    text+='Value metadata is layer/native-KV-head/block-specific and refreshed when V changes. Complete unchanged prefix blocks can reuse it; canvas and mixed blocks refresh. The shared reference scale is RMS norm over all currently valid KV tokens, requiring a metadata reduction before routing. Running centered risk uses preceding retained output; its refined version uses only the previous denoising-step attention output, with canvas/shape/step-gap invalidation and running-state warmup. Exact-mass refinements evaluate exact **online** block mass, not current dense final-output information. Mean compensation preserves exact denominator/mass and substitutes the block mean; zero-PV is its denominator-preserving control.\n\n'
    text+='Sol uses native-scaled mean-Q/mean-K tile proxies; the value version adds log RMS value norm. Standardization spans the whole eligible prefix+canvas population. Fixed Gaussian β values−0.674490,0,0.674490,1.281552 are never recalibrated. Same-proxy ranking isolates selection-budget effects. No approximate dropped-tile correction is used. Diagnostic top-p sums nonnegative attention masses or contribution norms; it never accumulates raw signed logits.\n\n'
    text+='All final Sol/diagnostic nonempty repairs use their **own declared proxy/ranking signal** restricted to each empty row’s structural support, with stable KV-index ties. The historical shared screen used an exact-mass fallback: at Sol Gaussian90% it repaired30.33% of AIME calibration rows and1.11% of LongBench rows. Those historical results are labeled as such and must not be interpreted as pure proxy routing. The corrected screen is `guarded_ranking_screen_summary.csv`, collected during the same refinement dense replay without retuning β. QK top-k had no fallback events in the audited initial calibration sample.\n\n'
    text+='## Main ablation results\n\nAll percentage columns below are percentages; output error is a ratio. Target denotes physical deletion except explicitly labeled compensation/zero-PV targets, which denote PV omission. Full thresholds, confidence intervals and exact count sums are retained in CSV/JSON.\n\n'
    for benchmark,split in [('aime26','heldout24'),('aime26','full'),('longbench','full')]:
        text+=f'### {benchmark} — {split}\n\n'
        text+=table([r for r in main if r['benchmark']==benchmark and r['split']==split],
            ['method','pooling','n','target','threshold','score','dense_delta_pp','physical','global_s','local_s','retained_mass','exact_PV_mass','output_error','agreement'])+'\n\n'
    replacements=[r for r in main if r['target_metric']=='pv_omission']
    if replacements:
        text+='### PV-replacement budgets — separate from physical sparsity\n\n'
        text+='Percentages below count omitted full-PV tiles / eligible tiles, with the same count weighting as physical sparsity. Compensation and zero-PV retain exact softmax denominators: physical deletion is zero, and retained denominator mass alone cannot measure approximation quality.\n\n'
        text+=table(replacements,['benchmark','split','method','n','target','physical','pv_omission',
            'global_pv','local_pv','exact_PV_mass','output_error','score','agreement'])+'\n\n'
    text+='## LongBench per-task scores before macro aggregation\n\n'
    task_rows=[dict(task=r['task'],condition=r['condition'],n=r['count'],score=100*r['accuracy'],dense_delta_pp=100*r['delta'],
        actual_sparsity=100*r['overall_physical_sparsity']) for r in per_task if r['benchmark']=='longbench']
    text+=table(task_rows,['task','condition','n','score','dense_delta_pp','actual_sparsity'])+'\n\n'
    text+='The LongBench aggregate is an equal-task macro score over normal task-specific metrics; it is not a common binary exact-match score. Paired prompt-bootstrap intervals use20,000 draws and seed42; they describe uncertainty on this small selected prompt set, not broad benchmark/generalization guarantees.\n\n'
    format_rows = dense_trec_format_rows(root, setup)
    _write(root/'dense_trec_format_audit.json', format_rows)
    text+=trec_format_note(format_rows)
    text+='## Actual-sparsity comparisons and empirical conclusions\n\n'
    text+='![Frozen streaming candidates and BLASST references on the confirmatory subsets](figures/primary_accuracy.png)\n\n'
    text+='This focused figure uses AIME heldout24 and LongBench50. Full-data figures separately label AIME full30, which includes calibration6. Sol Gaussian/ranking and other diagnostics remain separate curves in the complete figures; they are not merged into a streaming-candidate frontier.\n\n'
    text+=table(comparisons,['benchmark','split','candidate','reference','candidate_sparsity','reference_sparsity','comparable_within_three_points','score_delta','paired_ci95'])+'\n\n'
    text+='These comparisons choose the nearest observed BLASST point within three percentage points of actual physical sparsity. Outside that tolerance the score delta is intentionally omitted; no target-based or interpolated advantage is claimed. A positive point estimate alone is not a demonstrated winner. AIME comparisons use heldout24. Additional paired-seed results, if completed, are reported separately, without retuning.\n\n'
    for benchmark in ('aime26','longbench'):
        split='heldout24' if benchmark=='aime26' else 'full'
        dense=next(r for r in rows if r['benchmark']==benchmark and r['split']==split and r['method']=='dense')
        refs=[r for r in rows if r['benchmark']==benchmark and r['split']==split and r['method']=='blasst' and r['accuracy']>=dense['accuracy']]
        if refs:
            best=max(refs,key=lambda r:r['overall_physical_sparsity'])
            text+=f"- {benchmark}: BLASST `{best['condition']}` matches/exceeds the dense point score at {100*best['overall_physical_sparsity']:.2f}% measured deletion ({100*best['accuracy']:.2f} vs dense {100*dense['accuracy']:.2f}). This is score preservation on this finite subset, not proven equivalence.\n"
        else:text+=f'- {benchmark}: no evaluated BLASST setting matches the dense point score on {split}; safe sparsity is not established by these points.\n'
        matches=[r for r in comparisons if r['benchmark']==benchmark and r['comparable_within_three_points'] and not r['candidate'].startswith('sol_')]
        gains=[r for r in matches if r['score_delta']>0]
        text+=f"  New BLASST-based criteria have {len(gains)} positive accuracy point estimates among {len(matches)} nearby actual-sparsity comparisons. "
        text+=('No accuracy advantage over the matched BLASST points was observed.\n' if not gains else 'Inspect paired intervals and repeated seeds before interpreting these as a stronger frontier.\n')
    text+='\n### Sol Gaussian target mismatch and local/global behavior\n\n'
    sol=[dict(benchmark=r['benchmark'],method=r['condition'],target=100*r['target'],beta=r['beta'],
        overall=100*r['overall_physical_sparsity'],global_s=100*r['global_physical_sparsity'],local_s=100*r['local_physical_sparsity'])
        for r in rows if r['split']=='full' and r['method']=='sol' and r['mode']=='gaussian']
    text+=table(sol,['benchmark','method','target','beta','overall','global_s','local_s'])+'\n\n'
    text+='Calibration distribution diagnostics (`sol_distribution_diagnosis.csv`, `mechanism_diagnosis.md`) are descriptive correlated observations, not independent-sample normality tests. The installed DiffusionGemma applies scale-free V RMS normalization, making token-norm poolings nearly constant; norm-of-mean and dispersion can still vary. The screened RMS-value Sol proxy therefore often differs by an almost constant shift removed by standardization. Use the final table above for end-to-end achieved sparsity, not the earlier sampled-state percentages.\n\n'
    text+='### Mass, agreement and downstream quality\n\n'+table(correlations,['benchmark','split','scope','metric','n','pearson_with_accuracy'])+'\n\n'
    text+='These are descriptive correlations across dependent configurations, separated for deletion and PV replacement. They cannot establish causality or make retained mass a sufficient quality criterion. `summary.json` also includes sample10th/50th/90th percentiles of mass, output error and sparsity; aggregate mass remains valid-query-row weighted.\n\n'
    text+='## Routing diagnostics and execution-work accounting\n\n'
    text+=table(work,['benchmark','split','method','eligible','deleted','softmax_skipped','PV_omitted','mean_compensated','selection_requires_all_block_PV','selection_aware_full_PV_omission_tiles','prefix_s','canvas_s','mixed_boundary_s','mass_selection_gap'])+'\n\n'
    text+='Physical sparsity is sum(deleted eligible tiles)/sum(eligible tiles), never an unweighted average of step/layer/sample percentages. Complete deletion means the block is removed from the attention operator; BLASST still computes QK to decide. Exact-mass deletion already computed block softmax, so its softmax-skipped count is zero. Compensation has zero physical deletion and zero softmax skipping, while reporting PV replacement separately. None of these counts is measured speedup.\n\n'
    text+='True-mass/contribution diagnostic ranking also requires block softmax before selection, so its softmax-omission count is zero. Contribution ranking additionally requires every block’s PV contribution: its **selection-aware PV work opportunity is zero**, even when the final masked operator deletes blocks. `PV_omitted` describes the masked operator, while the selection-aware columns explicitly account for information already computed to choose that mask. All are algorithmic opportunities/requirements, not actual compute savings of this diagnostic-heavy emulation.\n\n'
    text+='Mass is evaluated against dense softmax for the **same Q/K state along each execution trajectory**, not the diverged dense-generation state. Denominator mass is100% for compensation/zero-PV; exact-PV mass identifies positions still using their original value contribution. FP32 relative output error is sqrt(sum squared error / sum squared dense output). It compares the corresponding mathematical operator, separate from BF16 rounding. Token agreement uses matching/compared generated IDs at aligned positions before and after the first divergence, with missing/extra positions disagreeing; full-sequence match is separately reported. EOS IDs are retained by the adapter. Empty/length-limited outputs remain scored and are counted as anomalies, not silently discarded.\n\n'
    text+='`per_layer_head_step.csv.gz` contains exact routing buckets; `routing_marginals.csv` aggregates by layer, head or step without double counting within each view. The same-budget mass comparator is an unconstrained ranking diagnostic, not necessarily structurally feasible or task-optimal. “Wrongly replaced” and “unnecessarily retained” count disagreements with that ranking at equal per-head/query-tile budgets; they are not labels of downstream correctness. Region sums and nonempty-row rescue counts are audited.\n\n'
    text+='## FlashAttention compatibility and costs\n\n'
    text+='FlashAttention’s tiled exact-attention design motivates examining what information is available before each block update. Its memory-traffic benefits do not by themselves validate a new sparse rule. [FlashAttention](https://arxiv.org/abs/2205.14135)\n\n'
    text+='Non-matmul work, communication and occupancy also matter, so scalar/vector bookkeeping can erase a theoretical arithmetic saving. [FlashAttention-2](https://arxiv.org/abs/2307.08691)\n\n'
    text+='The table below is an implementation-level compatibility assessment of **our emulated operators**, not a measured kernel result. The reference threshold family is adapted from [BLASST](https://arxiv.org/abs/2512.12087); λ>1 is explicitly the repository extension.\n\n'
    text+=table(compatibility,['method','decision','metadata','extra','information','omission'])+'\n\n'
    text+='For Bq=128, Bk=64 and value dimension d, retained dense PV costs O(Bq·Bk·d) arithmetic. A scaled-mean contribution or centered distance adds O(Bq·d) per candidate. Constructing norm summaries/means/radii costs O(K·d) for changed V; p95 additionally needs within-block selection/sorting. Block scalar metadata is O(Hkv·ceil(K/64)), means are O(Hkv·ceil(K/64)·d), and token-aligned norms O(Hkv·K). Previous-step output storage is O(layers·Hq·Q·d), FP32 in this reference. Global reference RMS needs another metadata reduction before routing; Sol population standardization/ranking also needs a reduction/selection phase. Prefix cache equality checks in this audit implementation are not free and would need write/version notifications in a deployment.\n\n'
    text+='## Failures, uncertainty, next iteration\n\n'
    from .pruning import evidence as pruning_evidence,report_rows as pruning_rows
    pruning,pruning_sources=pruning_evidence(history)
    if pruning:
        if any(contract['sources'].get(path)!=digest for path,digest in pruning_sources.items()):
            raise ValueError('screening pruning is absent from or differs from frozen provenance')
        text+='### Pruned development configurations\n\n'
        text+='The following screening stops used calibration evidence only, before heldout evaluation. Completed checkpoints and partial higher-sparsity shards were preserved; partial runs are **not completed final conditions**. They are disclosed as failed or deprioritized development settings, not silently omitted successes. Exact decisions and policy hashes are in `screening_pruning.json` and its linked evidence.\n\n'
        text+=table(pruning_rows(pruning),['candidate','decision_time','stopped_scope','aime_cal_score','lb_cal_score'])+'\n\n'
    text+='The original smoke caught a BF16 matmul-shape parity error; it was corrected before any retained calibration shard. A future-logit underflow in preceding-output diagnostics was likewise fixed before screening. Initial cap-one policy selection had a cross-type fallback error; old policies/traces were preserved and the final freeze blocks unresolved reviews. Full provenance and repairs remain in `execution_notes.md`, `policy_revisions/`, and `policy_review_issues.json`. Any per-condition errors remain in `failures.jsonl`; completion here requires every contracted final shard, not a partial success table.\n\n'
    failures=report_failure_rows(root,history)
    csv_write(root/'execution_failure_summary.csv',failures)
    if failures:
        text+='### Preserved execution errors\n\n'
        text+='These counts describe historical failed **attempts**, including calibration attempts later retried from checkpoints. They are not counts of missing final outputs: the completed final-shard audit is a separate requirement. Other GPU PIDs are only those reported by the CUDA exception, not an inferred process identity or cause for every failure. Logs and recovery chronology remain preserved.\n\n'
        text+=table(failures,['bundle','stage','condition','error_type','failed_attempts','distinct_sample_ids','reported_other_gpu_pids'])+'\n\n'
    anomalies=[dict(benchmark=r['benchmark'],split=r['split'],method=r['condition'],empty=r['empty_outputs'],length_terminated=r['length_terminated'])
        for r in selected if r['empty_outputs'] or r['length_terminated']]
    text+=table(anomalies,['benchmark','split','method','empty','length_terminated'])+'\n\n' if anomalies else 'No empty or length-terminated final outputs were observed.\n\n'
    text+='Small heldout samples limit power; no setting is called universally safe. Calibration target attainment is separate from correct selection and downstream quality. The next justified revision should follow measured failures: mass-bound looseness, worst-row inflation, stale output estimates, or accumulated denoising error. Current diagnostics cannot justify a kernel-speedup claim or guarantee that an observed point-score gain generalizes.\n\n'
    text+='## Reproduction\n\n'
    if history != root:
        text+='See the [reduced-run instructions](RUNBOOK.md). The superseded parent sweep is not required for this report. '
    else:
        text+='See `experiments/diffusion_gemma_value_aware/README.md` for the full workflow. '
    text+='From the repository root, using the pinned Python environment and completed shards:\n\n'
    text+='```bash\n'+reproduction_command(root,history)+'\n```\n\n'
    text+='This regenerates this bundle’s report, tables, plots and audit without model inference. Runtime health checks are every900 seconds. Source/config hashes, thresholds, raw generations and all calibration traces remain preserved.\n'
    (root/'report.md').write_text(text)
    _write(root/'report_manifest.json',dict(audit_complete=True,contract_conditions=len(contract['conditions']),
        final_shards=audit['completed'],scope='decoder denoising attention; no speedup claims',heldout_used_for_tuning=False))
