"""Canonical synthesis and explicit completion evidence, from completed shards.

No inference, selection, or fitting occurs here. Failed advancement gates are
reported as negative findings, not silently omitted or treated as speedups.
"""
import argparse
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .config import ROOT
from .run import digest,sample_key
from .report import report as screen_report,csv_write
from .online_report import report as online_report
from .refinement_report import analyze_refinement
from .validation import validation_report
from .work_volume import analyze
from .performance import summarize as cost_report
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write


def read(path):return json.loads(path.read_text())
def csv_rows(path):return list(csv.DictReader(path.open()))


def snapshot_audit(root):
    results=[]
    for stage in (root,root/'online_v1',root/'refinement_v1'):
        manifest=read(stage/'source_snapshot'/'manifest.json')
        for name,sha in manifest['source_sha256'].items():assert digest(stage/'source_snapshot'/name)==sha
        assert manifest['executed_fingerprint']==read(stage/'smoke.json')['fingerprint']
        results.append(dict(stage=str(stage),files=len(manifest['source_sha256']),fingerprint=manifest['executed_fingerprint']))
    cost=root/'reference_cost_v1';overlay=read(cost/'source_snapshot.json');setup=read(cost/'setup.json')
    import hashlib
    assert hashlib.sha256(overlay['performance_source'].encode()).hexdigest()==setup['performance_source_sha256']
    assert digest(overlay['inference_source_manifest'])==overlay['inference_source_manifest_sha256']
    validation=root/'heldout_quality_v1';overlay=read(validation/'source_snapshot.json')
    assert overlay['executed_fingerprint']==read(validation/'smoke.json')['fingerprint']
    assert digest(overlay['inference_source_manifest'])==overlay['inference_source_manifest_sha256']
    for item in overlay['files'].values():assert hashlib.sha256(item['source'].encode()).hexdigest()==item['sha256']
    return results


def audit_records(data,layer_types):
    records=data['records'];expected=set(range(len(layer_types)))
    assert {r['layer'] for r in records}==expected
    for layer in expected:
        rs=[r for r in records if r['layer']==layer]
        assert {r['head'] for r in rs}==set(range(16))
        kind='local' if layer_types[layer]=='sliding_attention' else 'global'
        assert all(r['attention_type']==kind for r in rs)
        calls=[sum(r['calls'] for r in rs if r['head']==h) for h in range(16)]
        assert len(set(calls))==1
    for r in records:
        assert 0<=r['skipped']<=r['eligible']
        assert sum(r[k+'_eligible'] for k in ('prefix','canvas','mixed'))==r['eligible']
        assert sum(r[k+'_skipped'] for k in ('prefix','canvas','mixed'))==r['skipped']
    assert sum(r['eligible'] for r in records)==data['costs']['eligible_tiles']
    assert sum(r['eligible']-r['skipped'] for r in records)==data['costs']['retained_tiles']
    assert data['metadata']['sampling']['native_temperature_schedule']
    assert data['executed_qk_saved']==data['executed_pv_saved']==0


def requirements(root,tests):
    """Every section of the objective has an explicit evidence row and scope."""
    rows=[]
    def add(key,requirement,evidence,scope='verified',limitation=''):
        paths=[root/p if not p.startswith(('experiments/','tests/')) else Path(p) for p in evidence]
        for p in paths:assert p.exists(),p
        rows.append(dict(id=key,requirement=requirement,status=scope,limitation=limitation,
            evidence=[dict(path=str(p),sha256=digest(p)) for p in paths]))
    add('1_prior_evidence','Reuse prior oracle/controlled manifests, model, formatting, scoring, budgets and native stochastic schedule',
        ['protocol.json','manifest_audit.json','source_snapshot/manifest.json','experiments/diffusion_gemma_preqk/run.py'])
    add('2_components','Independent position, history, KV/query neighborhood, value, allocation, refresh and exploration controls',
        ['experiments/diffusion_gemma_preqk/config.py','screen_summary.csv','stage2_selection.json','refinement_v1/freeze.json'])
    add('2_state_alignment','Request/canvas/cache/layer/head/query/KV state alignment, GQA, structural/partial tiles and row coverage',
        ['experiments/diffusion_gemma_preqk/state.py','experiments/diffusion_gemma_preqk/summaries.py','test_results.xml','smoke.json','refinement_v1/smoke.json'])
    add('2_information_contract','Masks precede current full QK; own sparse observations are distinct from paid dense-history reference; diagnostics preserve RNG/output',
        ['experiments/diffusion_gemma_preqk/routing.py','experiments/diffusion_gemma_preqk/online.py','test_results.xml','online_v1/smoke.json','heldout_quality_v1/smoke.json'])
    for category in ('beginning','prefix_end','canvas_end','diagonal'):
        add('3A_'+category,'Independent '+category+' protection with equal-count random controls, inside retention budget',
            ['screen_summary.csv','screen_paired_ablations.csv','test_results.xml'])
    add('3B_history','Last important mask/mass, binary importance frequency and EMA; consecutive history, observed-only updates, age/counts, paid refresh/exploration and estimated coverage correction',
        ['screen_rank_correlations.csv','screen_summary.csv','online_v1/freeze.json','online_v1/cost_breakdown.csv','experiments/diffusion_gemma_preqk/state.py'])
    add('3C_neighborhood','KV and historical neighboring-query smoothing tested separately; mean and peak-sensitive variants; whole physical-tile decisions',
        ['screen_summary.csv','screen_paired_ablations.csv','test_results.xml'])
    add('3D_value','RMS/max norm multipliers vs same unweighted history; cached summaries and deletion-risk diagnostic',
        ['screen_deletion_risk.csv','screen_audit.json','experiments/diffusion_gemma_preqk/summaries.py','development_findings.md'],
        limitation='Prefix values are scanned once at cache initialization and charged; canvas norms attach to production. No online finalist uses V norms.')
    add('4_stage1','Same-state matched-budget screening against QK/BLASST/mass/contribution on development only',
        ['screen_audit.json','screen_summary.csv','screen_per_prompt.csv','raw_integrity.json'],
        limitation='Consecutive first-canvas steps0–7; all-head/query metrics, raw masks/ranks saved for query tile0. Not an all-step predictability claim.')
    add('4_stage2','Own-history full denoising vs paid dense observations; isolated sparse perturbations after downstream failure',
        ['online_v1/audit.json','online_v1/summary.csv','refinement_v1/refinement_audit.json','refinement_v1/findings.md'])
    add('4_stage3','One ranking-fixed coarse allocation; exact per-forward/type/head/query total retention, same-state shadow, total-rollout accounting',
        ['refinement_v1/allocation_same_state.csv','refinement_v1/allocation_paired_deltas.csv','refinement_v1/trajectory_work.csv','refinement_v1/regression_audit.json'],
        limitation='Full allocated trajectory used more total retained work, so its final-score gain is NOT established at matched total rollout cost. Allocation rejected.')
    add('4_stage4_quality','Freeze before reserved additional IDs and paired seeds; uncertainty clustered by prompt; no held-out tuning',
        ['heldout_quality_v1/freeze.json','heldout_quality_v1/validation_audit.json','heldout_quality_v1/summary.csv','test_results.xml'],
        limitation='LongBench quality-only confirmation. AIME failed the development quality gate and was not expanded.')
    add('4_stage4_expansion','Expand RULER/LiveCodeBench only for finalists',
        ['heldout_quality_v1/freeze.json','reference_cost_v1/summary.json','development_findings.md'],scope='conditional_closed',
        limitation='No net-cost deployment finalist; neither benchmark is rerun in this pre-QK study.')
    add('5_quality','Task scores, paired deltas/CI, output lengths, budget hits and denoising steps',
        ['online_v1/per_task.csv','refinement_v1/per_example.csv','heldout_quality_v1/per_task.csv','heldout_quality_v1/per_example.csv'])
    add('5_physical_work','Sum skipped/eligible physical tiles overall/local/global/prefix/canvas; init/refresh/probes; actual QK/PV execution',
        ['online_v1/cost_breakdown.csv','refinement_v1/summary.csv','heldout_quality_v1/trajectory_work.csv','physical_work.csv'],
        limitation='Mixed prefix/canvas boundary tiles are a separate disjoint region; no executed QK/PV tile avoidance in dense emulation.')
    add('5_preservation','Retained mass, relative output error, mean and tail query error; distinguish original dense vs sparse state',
        ['screen_summary.csv','online_v1/summary.csv','heldout_quality_v1/summary.csv'],
        limitation='p95 summaries are means of head-call/query-tile p95s, not pooled population percentiles.')
    add('5_routing_quality','Same-budget important overlap/recall, rank correlation, newly missed blocks and rediscovery',
        ['screen_summary.csv','screen_rank_correlations.csv','screen_rediscovery_summary.csv','temporal_diagnostic_audit.json'],
        limitation='Temporal labels are early dense-trajectory diagnostics; rediscovery delays are conditional with loss-of-importance and censoring reported separately.')
    add('5_heterogeneity','Step/layer/head/query-position/region breakdowns with coverage; matched-prompt temporal contrasts',
        ['screen_summary.csv','screen_matched_prompt_steps.csv','online_v1/routing_breakdowns.csv','heldout_quality_v1/routing_breakdowns.csv'],
        limitation='Online full-rollout query tiles are aggregated within head calls; explicit query-position breakdowns come from the screen.')
    add('5_efficiency','State, collection/observation/selection spans, paid init/refresh/probe work, attention override and full-budget end-to-end timing',
        ['reference_cost_v1/components.csv','reference_cost_v1/timings.csv','reference_cost_v1/audit.json','online_v1/cost_breakdown.csv','backend_compatibility.json'],
        limitation='Nested capped host/CUDA-stream spans, not isolated kernel timing; two development cost probes. No compatible sparse backend was integrated.')
    add('6_decisions','Hypothesis-driven iterations; explicit retain/reject decisions; unsuccessful results preserved; no Cartesian online search or held-out retuning',
        ['iteration_log.md','development_findings.md','stage2_selection.json','refinement_v1/selection.json','heldout_quality_v1/freeze.json'])
    add('7_reproducibility','Resumable commands, frozen manifests, raw/aggregate results, tests and failed-case transparency',
        ['experiments/diffusion_gemma_preqk/README.md','test_results.xml','raw_integrity.json','heldout_quality_v1/source_snapshot.json'])
    assert tests>=47
    return rows


def build(root=ROOT):
    # Rebuild measured aggregates first; stale complete flags are insufficient.
    screen_report(root)
    online_report(root/'online_v1');analyze(root/'online_v1')
    analyze_refinement(root/'refinement_v1')
    validation_report(root/'heldout_quality_v1');cost_report(root/'reference_cost_v1')
    from .backend_audit import audit as backend_audit
    backend_audit(root)
    protocol=read(root/'protocol.json');layer_types=read(Path(protocol['model'])/'config.json')['text_config']['layer_types']
    junit=ET.parse(root/'test_results.xml').getroot();suites=list(junit.iter('testsuite'))
    assert suites and all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
    test_count=sum(int(s.get('tests',0)) for s in suites)
    test_names={t.get('name') for t in junit.iter('testcase')}
    required_tests={'test_geometry_gqa_partial_tiles_and_dense_no_rng_change','test_row_rescue_is_geometry_only',
        'test_state_resets_step_cache_and_length','test_online_current_logits_cannot_change_history_mask',
        'test_online_diagnostics_do_not_affect_rng_masks_or_outputs','test_online_count_weighting_and_refresh_denominator',
        'test_direct_online_diagnostic_matches_block_contribution_oracle','test_heldout_quality_bootstrap_clusters_storage_replicates',
        'test_paired_allocation_balances_each_type_head_and_query_row','test_isolated_sparse_call_fires_only_once_across_canvases',
        'test_raw_trace_audit_checks_masks_budgets_and_cache_alignment'}
    assert required_tests<=test_names
    assert read(root/'screen_audit.json')['complete']
    assert read(root/'heldout_quality_v1/validation_audit.json')['complete']
    assert read(root/'reference_cost_v1/audit.json')['complete']
    snapshots=snapshot_audit(root);raw=[]
    for path in sorted((root/'screen/shards').glob('*.json')):
        shard=read(path);cache=protocol['cached_dense'][shard['id']];assert digest(cache['path'])==cache['sha256']
        dense=read(Path(cache['path']));assert shard['completion_tokens']==dense['completion_tokens']
        assert all(shard[k]==dense[k] for k in ('seed','generation_budget','prompt_hash'))
        raw.append(dict(path=str(path),sha256=digest(path)))
        raw.extend(dict(path=r['path'],sha256=digest(r['path'])) for r in shard['records'])
    conditions=[];work=[]
    for stage in ('online_v1','refinement_v1','heldout_quality_v1'):
        summary=read(root/stage/'summary.json');assert summary['audit']['complete']
        conditions.extend(dict(stage=stage,**r) for r in summary['conditions'])
        work.extend(dict(stage=stage,**r) for r in csv_rows(root/stage/'trajectory_work.csv'))
        for item in summary['audit']['source_shards']:
            assert digest(item['path'])==item['sha256'];data=read(Path(item['path']));audit_records(data,layer_types);raw.append(item)
    held=read(root/'heldout_quality_v1/freeze.json')
    from .validation import reserved_rows
    assert held['samples']==reserved_rows(protocol,{e['benchmark'] for e in held['conditions'].values()})
    for path,sha in held['evidence'].items():assert digest(path)==sha,'advancement evidence was changed after held-out freeze'
    dev_ids={r['id'] for r in protocol['development']};dev_hashes={r['prompt_hash'] for r in protocol['development']}
    assert all(r['prompt_id'] not in dev_ids and r['prompt_hash'] not in dev_hashes for r in held['samples'])
    assert {r['seed'] for r in held['samples']}=={314159,271828}
    held_cache=read(root/'heldout_quality_v1/dense_cache.json')
    assert set(held_cache)=={r['id'] for r in held['samples']}
    for item in held_cache.values():
        assert digest(item['path'])==item['sha256'];audit_records(read(Path(item['path'])),layer_types);raw.append(item)
    raw.extend(dict(path=str(p),sha256=digest(p)) for p in sorted((root/'reference_cost_v1').glob('timings/*/*.json')))
    _write(root/'raw_integrity.json',dict(artifacts=raw,snapshots=snapshots,tests=test_count,
        development_dense_token_parity=True,heldout_ids_disjoint=True,paired_seed_clustering=True))
    csv_write(root/'physical_work.csv',work);csv_write(root/'quality_results.csv',conditions)
    screens=csv_rows(root/'screen_summary.csv')
    ablations=[r for r in screens if r['dimension']=='overall' and r['target_sparsity']=='0.5']
    csv_write(root/'compact_ablations.csv',ablations)
    plot(root,conditions)
    write_report(root,protocol,conditions,ablations,work,test_count)
    source_paths=list(Path(__file__).parent.glob('*.py'))+[Path('tests/test_diffusion_gemma_preqk.py'),
        Path('tests/test_diffusion_gemma_oracle.py'),Path('experiments/diffusion_attention_threshold_modeling/routing.py')]
    sources={str(p):dict(sha256=digest(p),source=p.read_text()) for p in sorted(source_paths)}
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _fingerprint
    source_path=root/'report_sources'/(_fingerprint(sources)+'.json');_write(source_path,sources)
    req=requirements(root,test_count)
    req.append(dict(id='7_final_report',requirement='Five research answers, compact ablations, quality curves, temporal/heterogeneity/routing diagnostics, cost and recommendation',
        status='verified',limitation='Negative deployment conclusion; small-subset quality evidence is not a universal guarantee.',
        evidence=[dict(path=str(root/p),sha256=digest(root/p)) for p in ('report.md','quality_sparsity.png','compact_ablations.csv','quality_results.csv')]+[dict(path=str(source_path),sha256=digest(source_path))]))
    _write(root/'completion_audit.json',dict(complete=True,requirements=req,tests=test_count,
        stages={'screen':16,'online':64,'refinement':44,'heldout_sparse':20,'heldout_dense':20,'reference_timing':12},
        executed_sparse_speedup=False,deployment_recommendation='dense',heldout_retuning=False,
        conditional_exclusions=['AIME held-out: development quality gate failed','RULER/LiveCodeBench expansion: no net-cost finalist']))


def plot(root,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,benchmark in zip(axes,('longbench','aime24')):
        rows=[r for r in conditions if r['benchmark']==benchmark and r['stage']!='heldout_quality_v1' and r['history_source']=='sparse' and r['predictor']!='last_mass_control']
        ax.scatter([0],[rows[0]['dense_score']*100],c='black',marker='*',s=100,label='dense')
        for r in rows:
            label=r['predictor'].replace('protect_','+ ').replace('paired_layer_shift','layer allocation')+f" s{r['target_sparsity']*100:g}"
            ax.scatter([r['overall_sparsity']*100],[r['score']*100],label=label,marker='x' if 'isolated' in r['predictor'] else 'o')
        ax.set(xlabel='Measured skipped / eligible tiles (%)',ylabel='Task-macro score (%)',title=benchmark+' — development')
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
    fig.suptitle('Full native denoising; initialization/refresh included. Points do not imply matched TOTAL work.',fontsize=10)
    fig.tight_layout();fig.savefig(root/'quality_sparsity.png',dpi=180);plt.close(fig)


def write_report(root,protocol,conditions,ablations,work,test_count):
    held=next(r for r in conditions if r['stage']=='heldout_quality_v1')
    costs=read(root/'reference_cost_v1/summary.json')['conditions'];cost=next(r for r in costs if r['method']!='dense')
    validation=read(root/'heldout_quality_v1/validation_audit.json')['decisions'][0]
    lines=['# Pre-QK routing on DiffusionGemma — final study report','',
        '## Conclusion','',
        'Previous-step mass is a useful pre-current-QK ranking signal, but the tested implementation does not provide net inference savings. '
        'Keep dense execution for deployment. Extra history, smoothing and value-norm weighting did not reliably help. '
        'Small positional or allocation gains in local diagnostics did not justify their full-rollout quality/work tradeoffs.',
        '',f"Frozen LongBench validation: score {held['score']:.2%} vs dense {held['dense_score']:.2%}; paired Δ {held['accuracy_delta']*100:+.2f} pp, "
        f"95% prompt bootstrap [{held['accuracy_delta_ci_low']*100:+.2f}, {held['accuracy_delta_ci_high']*100:+.2f}] pp. "
        f"The predeclared tolerance is 3 percentage points. Point estimate within tolerance: {validation['point_within_tolerance']}; "
        f"entire bootstrap interval within tolerance: {validation['bootstrap_interval_within_tolerance']}. "
        'This is quality-only confirmation, not a deployment recommendation or universal noninferiority guarantee.',
        '',f"Measured reference end-to-end time is {cost['reference_e2e_time_ratio_vs_dense']:.3f}× dense on two development cost probes. "
        'The backend computes full QK and PV: physical masks establish logical work opportunities, not measured sparse-kernel speedup.',
        '', '## Setup and correctness contract','',
        f"Pinned revision: `{protocol['revision']}`; BF16, one H100, 30 decoder layers (25 local / 5 global), 16 query heads with native GQA. "
        'Physical tiles are 64×64 across prefix and canvas. Structural impossibility/padding is excluded. '
        'Prefix-only, canvas-only and boundary-straddling mixed tiles have disjoint denominators.',
        '', 'Development: 10 LongBench prompts (2 each qasper, hotpotqa, gov_report, trec, passage_retrieval_en), '
        'plus 6 AIME2024 prompts (3 each AIME I/II), using the existing oracle subset and verified dense caches. '
        'Held-out: 10 other reserved LongBench IDs (2/task), each with seeds314159 and271828:20 dense +20 sparse generations. '
        'IDs, prompt hashes, exact prompts/budgets and source lineage are in protocol.json and heldout_quality_v1/freeze.json. '
        'Held-out means disjoint from this study’s development prompts, not a claim about model-training contamination.',
        '', 'Native canvas256 and denoising schedule are unchanged: requested temperature0 selects the adapter’s native stochastic schedule, '
        'not greedy decoding. Thinking=False; native confidence/entropy stopping and official cached generation budgets are preserved. '
        'First-canvas consecutive steps0–7 were collected for screening; step0 is initialization and is excluded from temporal screening comparisons. '
        'Later online quality runs use complete native denoising, including changing sequence lengths.',
        '', 'Online last-mass selection uses only permitted cached observations and geometry before current full QK. '
        'Dense initialization and every-fourth-step refresh are paid; one oldest-tile exploration stays within the retention quota. '
        'Unobserved blocks retain stale estimates with age/counts, not zero importance. Sparse-softmax observations are rescaled by estimated previous coverage '
        '(clamped0.001–1), not by current dense diagnostic mass. This is a heuristic bias correction, not guaranteed dense-mass recovery. '
        'Dense-reference history observes dense attention on its own sparse trajectory, not replay of original dense generation.',
        '', f"{test_count} regression tests pass. CUDA smokes establish exact no-skip dense parity and diagnostic-on/off token parity; "
        'all development screen generations match their cached dense tokens. Raw metric checks cover all query tiles; raw mask/rank traces are saved for query tile0. '
        'The final audit additionally checks all30 layers/16heads, region/count sums, source hashes, native scheduling and paired prompt identities.',
        '', '## Independent same-state ablations','',
        'Target50 below. Errors are sqrt(sum squared output error / sum squared dense output), with exact joint post-mask renormalization. '
        'All methods use the same local retention quota. Complete targets40/50 for AIME and50/75 for LongBench, recalls and tails are in screen_summary.csv.',
        'The oracle_ labels mean privileged current-state ranking references, not globally optimal joint-deletion masks. The proxy is mean-Q/K ranked at the same quota, not recalibrated Gaussian thresholding.',
        '', '| Predictor | LongBench error | AIME error |', '|---|---:|---:|']
    lookup={(r['benchmark'],r['method']):r for r in ablations}
    for name in ('proxy','last_mass','last_mask','frequency','ema_mass','peak','protect_beginning','random_beginning','protect_prefix_end','random_prefix_end',
                 'protect_canvas_end','random_canvas_end','protect_diagonal','random_diagonal','kv_mean','kv_max','query_mean','query_max','value_rms','value_max',
                 'oracle_qk','oracle_blasst','oracle_mass','oracle_contribution'):
        a=lookup['longbench',name+'_s50'];b=lookup['aime24',name+'_s50']
        lines.append(f"| {name} | {float(a['relative_error']):.5f} | {float(b['relative_error']):.5f} |")
    lines+=['', 'Last mass at target50 actually skips49.46% on LongBench and45.24% on AIME after structural/rounding effects. '
        'It beats the current mean-Q/K proxy on all10/6 prompts. Local LongBench error remains .1495 vs mass oracle .0937; global .0985 vs .0837. '
        'Rank correlation (~.92–.94) is strong but does not imply low deletion risk.',
        '', 'EMA and frequency worsen prediction; the frequency signal counts observed importance above1/eligible_count, not mere retention. '
        'KV/query smoothing worsens preservation. Peak pooling has a small LongBench aggregate benefit but worse mass/mean/tail error, so it is not advanced. '
        'Previous important-mask reuse duplicates last-mass selections in the dense-history screen while adding a sort.',
        '', 'Value RMS variation is only about0.004–0.010%, consistent with native v_norm; multiplication adds negligible useful information. '
        'Prefix norm acquisition scans V once at cache initialization and is charged; canvas norms attach to production. No online finalist scans V for norms. '
        'Single-block deletion-risk rankings correlate strongly with mass (.964–.990), less identically with contribution (.933–.961), '
        'and weakly with norms (−.003 to .082). They include cancellation/renormalization but do not characterize joint deletion interactions.',
        '', '## Full-rollout sparsity–quality results','',
        'Table includes paid initialization/refresh; history modes are distinguished. Sparse-state mass/error diagnostics are not original-dense-trajectory measurements. '
        'Token agreement compares every positional token ID even after divergence, with missing/extra positions counted as disagreement.',
        '', '| Stage | Benchmark | Predictor / history | Actual | Global | Local | Mass | Token agreement | Score | Δ dense (pp) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in conditions:
        if r['stage']=='refinement_v1' and r['predictor']=='last_mass_control':continue
        lines.append(f"| {r['stage']} | {r['benchmark']} | {r['predictor']} / {r['history_source']} s{r['target_sparsity']*100:g} | "
            f"{r['overall_sparsity']:.2%} | {r['global_sparsity']:.2%} | {r['local_sparsity']:.2%} | {r['overall_retained_mass']:.4f} | "
            f"{r['token_agreement']:.2%} | {r['score']:.2%} | {r['accuracy_delta']*100:+.2f} |")
    lines+=['','### Frozen held-out task breakdown','',
        '| Task | Prompts / generations | Dense score | Router score | Δ (pp) | Token agreement |',
        '|---|---:|---:|---:|---:|---:|']
    for r in csv_rows(root/'heldout_quality_v1/per_task.csv'):
        lines.append(f"| {r['task']} | {r['prompts']} / {r['generations']} | {float(r['dense_score']):.2%} | {float(r['score']):.2%} | {float(r['accuracy_delta'])*100:+.2f} | {float(r['token_agreement']):.2%} |")
    lines+=['', 'The held-out macro gain is concentrated in HotpotQA and gov_report; the other three tasks have unchanged scores. '
        'Dense scores are zero on these tiny HotpotQA/TREC subsets, and retrieval is already at the ceiling. With only two prompts/task, '
        'the favorable bootstrap interval is limited evidence for this frozen subset, not broad benchmark superiority. '
        'All gov_report generations hit the512-token budget: their low token agreement dominates the token-count-weighted aggregate while ROUGE improves. '
        'Retained mass and token agreement are therefore diagnostics, not stand-alone guarantees of task accuracy.']
    lines+=['','![Development quality vs measured physical sparsity](quality_sparsity.png)','',
        'LongBench diagonal protection improves early same-state output but needs58% more denoising steps than dense, '
        'leaving retained tile volume at.988× dense. AIME beginning protection drops to3/6 vs dense5/6. '
        'Plain AIME s50 gives4/6 with either observation source; s40 gives3/6 with either source. '
        'These six-prompt results are nonmonotonic and do not establish a universal density/accuracy relationship.',
        '', 'Exactly one first-reuse perturbation per request also hurts AIME: global layer5 gives3/6 and local layer16 gives4/6. '
        'Overall sparsity is only~0.01%. Repeated sparse calls are not required for damage in these cases. '
        'Later dense denoising can still amplify a single perturbation, and changed stopping/step counts can change native stochastic draw alignment. '
        'This does not rule out error accumulation/amplification, uniquely establish a failure mechanism, or justify a large sensitive-layer lookup table.',
        '', '## Allocation and total work','',
        'One fixed last-mass ranking was tested with paired +1/−1 tile transfers: five high/low-error local layers and one high/low-error global layer. '
        'Joint clipping preserves exact per-forward/type/head/query retention, including init/refresh and geometric rescues. '
        'Same-state uniform shadow masks improve mean prompt relative error by.00168 (95% CI improvement .00092–.00242), 9/10 prompts. '
        'The effect is mainly local; the global interval includes zero and retained mass slightly falls. Preservation and mass are not interchangeable.',
        '', 'Full allocated score rises to38.72%, but steps rise to161 vs plain115/dense88 and retained volume becomes1.128× dense. '
        'Its accuracy gain is NOT a matched-total-work gain. Allocation is rejected despite the local diagnostic improvement.',
        '', '| Stage | Benchmark / condition | Steps / dense | Retained tiles / dense | Actual reference QK FLOPs / dense |',
        '|---|---|---:|---:|---:|']
    for r in work:
        if 'dense_reference' in r['condition'] or 'isolated_' in r['condition']:continue
        lines.append(f"| {r['stage']} | {r['condition']} | {float(r['denoising_step_ratio']):.3f} | {float(r['retained_tile_volume_ratio']):.3f} | {float(r['dense_emulation_decoder_qk_work_ratio']):.3f} |")
    lines+=['', 'Retained-volume ratios compare sums over complete paired trajectories, not averages of sample/layer sparsity percentages. '
        'They are potential logical tile work, not executed sparse work. Actual generation QK/PV shapes are fully executed; diagnostic PV is separately charged. '
        'Encoder/prefill and MLP/MoE are excluded from this FLOP table but included in the timing experiment.',
        '', '## Temporal, heterogeneity and efficiency findings','',
        'LongBench step1–7 coverage falls10→8→6→5→4→3→2 prompts; AIME retains6. On matched IDs, first-reuse error is higher than step2 '
        '(.2135→.0979 on8 LongBench prompts; .2179→.1465 on6 AIME). Late-step small-N averages are not population trends. '
        'Rediscovery reports distinguish successful rediscovery, lost importance and end-of-trace censoring; delays are conditional, not an uncensored mean.',
        '', 'Full step/layer/head/query-position/region data are in screen_summary.csv, screen_matched_prompt_steps.csv, '
        'screen_rank_correlations.csv, screen_rediscovery_summary.csv and each online stage’s routing_breakdowns.csv. '
        'Explicit query-position diagnostics come from the screen; online records aggregate query tiles within head calls. '
        'Local “beginning” means the beginning of the1023-token available sliding cache, not original BOS. '
        'Global layers are5,11,17,23,29. Error-tail summaries are means of per-head-call/tile p95s, not pooled percentiles.',
        '', 'Reference timing: two frozen LongBench prompt-length endpoints, dense/router, three full-budget repetitions each, diagnostics disabled, '
        'alternating order, excluded capped warmup and model load. All12 runs reproduce cached tokens. '
        f"Ratio is {cost['reference_e2e_time_ratio_vs_dense']:.3f}× dense; endpoint ratios2.870× and1.419×. Max router state2.739MiB, peak allocated56.71GiB. "
        'The first dense full-budget repeats have first-use overhead; raw ranges remain available and the predeclared median aggregation is unchanged. '
        'This is not a population timing claim.',
        '', 'Separate capped component spans expose costs of cache/state access, geometry/row rescue, prediction/selection, observations/history update, '
        'attention finishing and the full decoder override. Host and CUDA-stream spans are nested; do not add them or interpret them as isolated kernel execution. '
        'Paid init/refresh/exploration tile counts and diagnostic collection work are in cost_breakdown.csv. '
        'No compatible sparse backend was integrated; the implementation avoids zero executed tile QK/PV.',
        '', 'The repository does contain a Python two-pass physical reference in diffusion_attention_threshold_modeling/routing.py. '
        'It was inspected rather than silently overlooked: it computes FP32 QK/PV instead of the frozen native BF16 operations, '
        'fails bit-exact no-skip output equality in the saved small numerical fixture, and returns no block-mass/peak observations required by online history. '
        'It is not a validated drop-in backend for this frozen study. Integrating/revalidating it would be new backend work; '
        'no GPU latency claim is made for that helper. See backend_compatibility.json for the source hash and exact scope.',
        '', '## Answers to the five research questions','',
        '1. **Can pre-QK signals approximate useful selections?** Yes on early development states: previous-step mass strongly outperforms the current mean-Q/K proxy, '
        'but does not recover the mass/contribution oracle or guarantee downstream preservation.',
        '2. **Which components independently help?** No robust reason to retain frequency, EMA, KV/query smoothing, peak safeguards or value multipliers. '
        'Some position protections help at fixed local budgets, but not consistently across density, benchmark, tails and rollout cost.',
        '3. **Do benefits survive online history/full rollouts?** Only partially. Own-history observations add error relative to paid dense observations; '
        'AIME quality fails even with the latter and even under a single-call perturbation. Development LongBench plain ranking is the strongest simple quality/cost compromise, '
        'not a net-cost success.',
        '4. **Does allocation improve quality beyond ranking?** It modestly improves same-state preservation at fixed instantaneous total retention. '
        'Full-run quality improves with more total work, so no matched-total-cost quality advantage is established. Reject the tested allocation.',
        f"5. **Held-out robustness and net savings?** LongBench held-out Δ is {held['accuracy_delta']*100:+.2f} pp, CI[{held['accuracy_delta_ci_low']*100:+.2f},{held['accuracy_delta_ci_high']*100:+.2f}] pp. "
        f"Bootstrap interval within the3-point tolerance: {validation['bootstrap_interval_within_tolerance']}. "
        'AIME was not advanced after development failure. No net savings: reference execution is slower and no sparse-kernel speedup was measured. '
        'RULER/LiveCodeBench were not expanded because there is no net-cost deployment finalist.',
        '', '## Reproduction, anomalies and recommendation','',
        'Use the dedicated prepare/screen/online/refinement/validation/performance commands in experiments/diffusion_gemma_preqk/README.md. '
        'Run `PYTHONPATH=src python -m experiments.diffusion_gemma_preqk.final_report` after the saved regression XML exists to regenerate this bundle from completed artifacts. '
        'Frozen source overlays, inference fingerprints, prompts/seeds, raw generations, mask/diagnostic traces, timing shards and negative cases are preserved. '
        'completion_audit.json maps each requirement to inspected evidence; raw_integrity.json records raw hashes.',
        '', 'One reporting anomaly was repaired without inference reruns: value_rms/value_max raw rank keys originally held norm side information, '
        'overwriting weighted-score logs but not actual masks/errors. Weighted ranks were reconstructed as saved last_mass×norm. '
        'The initial mislabeled CSV is retained under report_revisions/. There were no final generation failures; development-only and conditional exclusions are explicit.',
        '', '**Recommendation:** retain dense deployment. If later engineering a compatible sparse kernel, plain last-step mass is the simplest research starting point, '
        'with paid initialization/refresh/exploration and honest observation-bias accounting. This study does not recommend adding all proposed components '
        'or continuing a complexity search to force a positive result.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);build(p.parse_args().root)
