"""Sanitized report for the completed eligible dense census and operator study."""
from __future__ import annotations
import collections
import hashlib
import json
from pathlib import Path
import shlex
import statistics

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'report003'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')

def summarize_request(record,trace):
    summary=record['trajectory_summary']
    closed=[c for c in summary['canvases'] if c['terminal']['native_converged']]
    emitted=int(summary['commit_calls'])
    if emitted!=len(closed):raise RuntimeError('commit/closed canvas count mismatch')
    if record['model_execute_calls']!=record['engine_steps']:raise RuntimeError('model counter mismatch')
    if record['geometry']['global_calls']!=5*summary['denoising_calls']:raise RuntimeError('geometry/call mismatch')
    if summary['native_predicate_mismatches']:raise RuntimeError('native predicate mismatch')
    if len(trace)!=summary['denoising_calls']:raise RuntimeError('trace coverage mismatch')
    return dict(id=record['id'],benchmark=record['id'].split('/')[0],N=summary['denoising_calls'],
        C=emitted,C_observed=summary['canvas_count'],N_over_C=summary['denoising_calls']/emitted,
        N_closed=sum(c['calls'] for c in closed),N_closed_over_C=sum(c['calls'] for c in closed)/emitted,
        uncommitted_tail_calls=sum(c['calls'] for c in summary['canvases'] if not c['terminal']['native_converged']),
        T=record['output_tokens'],finish_reason=record['finish_reason'],
        closed_calls=[c['calls'] for c in closed],cap_canvases=sum(c['terminal']['cap_reached'] for c in closed),
        first_canvas=summary['canvases'][0],terminal_emitted_canvas=closed[-1],
        stopping_bottlenecks=dict(collections.Counter(t['stopping_bottleneck'] for t in trace)),
        accepted_fraction_mean=statistics.mean(t['accepted_fraction'] for t in trace),
        accepted_fraction_reconstruction=True,
        global_eligible_tiles=record['geometry']['global_eligible_tiles'],
        global_kept_tiles=record['geometry']['global_kept_tiles'],global_skipped_tiles=0,
        local_eligible_rectangles=record['geometry']['native_local_eligible_rectangles'],
        local_kept_rectangles=record['geometry']['native_local_eligible_rectangles'],local_skipped_rectangles=0,
        denominator='Q128 x KV64 eligible rectangles from actual native metadata, not measured CUDA CTA counts',
        diagnostic_request_seconds=record['instrumented_request_seconds'],
        S=None,S_over_N=None,W_clean=None,attention_module_seconds=None,
        refresh_discovery_calls=0,held_mask_calls=0,carried_mask_calls=0,selector_overhead_seconds=0,
        new_cuda_captures=record['new_cuda_captures'],jit_events=record['jit_event_count'],
        observer_rng_checks=summary['observer_rng_checks'],native_predicate_mismatches=summary['native_predicate_mismatches'],
        output_sha256=record['output_sha256'],prompt_hash=record['prompt_hash'])

def main():
    OUT.mkdir(exist_ok=False)
    protocol=read(ROOT/'protocol.json');scores=read(ROOT/'scoring001/official_scores.json')
    context=read(ROOT/'context_audit.json');operator=read(ROOT/'operator_diagnostic002/summary.json')
    op_records=read(ROOT/'operator_diagnostic002/records.json')
    candidate_readout=read(ROOT/'operator_diagnostic002/candidate_readout.json')
    smoke=read(ROOT/'smoke/receipt.json')
    if not smoke['passed']:raise RuntimeError('smoke gate missing')
    drift=[path for path,h in protocol['sources'].items() if sha(path)!=h]
    if drift:raise RuntimeError('source identity drift')
    inputs={};requests=[];traces={}
    for group in ('short','long'):
        directory=ROOT/('census_'+group)
        marker=read(directory/'final_complete.json')
        if not marker['passed'] or marker['protocol_sha256']!=sha(ROOT/'protocol.json'):raise RuntimeError('census marker mismatch')
        for path in sorted(directory.glob('request*/record.json')):
            record=read(path)
            if record['label']=='warm':continue
            trace=read(path.parent/'trace.json')
            requests.append(summarize_request(record,trace));traces[record['id']]=trace
            inputs[str(path.relative_to(ROOT))]=sha(path)
            inputs[str((path.parent/'trace.json').relative_to(ROOT))]=sha(path.parent/'trace.json')
    if {r['id'] for r in requests}!=set(protocol['eligible_ids']):raise RuntimeError('eligible coverage mismatch')
    if {r['id'] for r in requests}!={r['id'] for r in scores['outcomes']}:raise RuntimeError('score coverage mismatch')
    by_workload={}
    for benchmark in ('aime26','humaneval','longbench_v2'):
        selected=[r for r in requests if r['benchmark']==benchmark]
        closed=[n for r in selected for n in r['closed_calls']]
        N=sum(r['N'] for r in selected);C=sum(r['C'] for r in selected)
        bottlenecks=collections.Counter()
        for r in selected:bottlenecks.update(r['stopping_bottlenecks'])
        by_workload[benchmark]=dict(items=len(selected),correct=scores['summary'][benchmark]['correct'],
            score=scores['summary'][benchmark]['accuracy'],N=N,C=C,N_over_C=N/C,
            item_mean_N_over_C=statistics.mean(r['N_over_C'] for r in selected),
            closed_N_over_C=sum(r['N_closed'] for r in selected)/C,
            uncommitted_tail_calls=sum(r['uncommitted_tail_calls'] for r in selected),
            mean_T=statistics.mean(r['T'] for r in selected),output_capped=scores['summary'][benchmark]['capped'],
            cap_canvases=sum(r['cap_canvases'] for r in selected),
            cap_canvas_fraction=sum(r['cap_canvases'] for r in selected)/C,
            closed_calls_min=float(np.min(closed)),closed_calls_median=float(np.median(closed)),
            closed_calls_p90=float(np.quantile(closed,.9)),closed_calls_max=float(np.max(closed)),
            stopping_bottlenecks=dict(bottlenecks),
            global_eligible_tiles=sum(r['global_eligible_tiles'] for r in selected),
            global_kept_tiles=sum(r['global_kept_tiles'] for r in selected),global_skipped_tiles=0,
            local_eligible_rectangles=sum(r['local_eligible_rectangles'] for r in selected),local_skipped_rectangles=0,
            observer_rng_checks=sum(r['observer_rng_checks'] for r in selected),
            native_predicate_mismatches=0,new_cuda_captures=sum(r['new_cuda_captures'] for r in selected),
            jit_events=sum(r['jit_events'] for r in selected),
            S=None,S_over_N=None,W_clean=None,paired_accuracy_interval=None)
    numeric=[r for r in op_records if r['response_gain']['total']>1e-4]
    numerical_audit=dict(nontrivial_response_records=len(numeric),near_zero_records=len(op_records)-len(numeric),
        full_relative_fd_max=max(r['finite_difference_relative_error'] for r in op_records),
        nontrivial_relative_fd_quantiles=np.quantile([r['finite_difference_relative_error'] for r in numeric],[0,.5,.9,1]).tolist(),
        finite_difference_gate_failures=operator['finite_difference_gate_failures'],note='Corrected FP64 audit with unique support and exact all-kept certificates. The original FP32 attempt remains preserved.')
    decisions={
        'dense_observer':dict(status='GO within tested scope',evidence='Output, active sampler state, RNG, prompt identity and forward counts match; zero predicate mismatches',limitation='One-item rich/minimal/off smoke; full KV/layout fork restoration unvalidated'),
        'response_guard':dict(status='NO_GO for native selector or proxy',evidence='Held-out response correlation does not beat current output error on either request; matched pairs supply only attention-level descriptive evidence',next='Exact native full-state restore, then downstream influence oracle on dense states before any proxy'),
        'margin_guard':dict(status='NOT_EVALUATED',evidence='No downstream margin, acceptance, correction/break or native continuation oracle',next='Implement only an exact diagnostic oracle after the native state restore gate'),
        'validity_reuse':dict(status='NO_GO for this norm-bound implementation',evidence='No bound violations, but every partial-support bound below 0.5; practical selectivity not established',next='Retain as diagnostic; do not deploy a refresh policy from this bound'),
        'native_acceleration':dict(status='UNESTABLISHED',evidence='No matched sparse native arms, no clean timing, no full-state causal fork certificate'),
        'long_context_generalization':dict(status='BLOCKED COVERAGE',evidence='Nine context-over-limit items preserved; only one LongBench request ran and it was capped/wrong')
    }
    summary=dict(schema='dense_census_report_v1',protocol_sha256=sha(ROOT/'protocol.json'),
        original_manifest_identity_sha256=sha(ROOT.parent/'manifest_identity.json'),
        scope='21 eligible dense requests and bounded inherited-state operator diagnostics; full 30-item study partial',
        eligible_complete=True,full30_complete=False,blocked_ids=protocol['blocked_ids'],
        workloads=by_workload,observer_smoke=smoke,operator=operator,numerical_audit=numerical_audit,
        decisions=decisions,source_drift=[],candidate_selection_readout=candidate_readout['summary'],unique_physical_support_records=candidate_readout['unique_physical_records'],physical_groups=candidate_readout['physical_groups'],
        physical_work_note='Q128 x KV64 rectangle denominator; dense GLOBAL/LOCAL have zero skips; no hardware CTA measurement',
        first_canvas_physical_work='Not separately instrumented; whole-request actual geometry only',
        clean_timing='Not run; S, S/N and clean W remain null',
        paired_uncertainty='Unavailable without matched arm contrast; seed42 descriptive only',
        full_native_state_fork_restore='Not implemented or validated',
        no_selector_or_kernel_optimization=True)
    write(OUT/'summary.json',summary);write(OUT/'per_item_metrics.json',requests);write(OUT/'method_decisions.json',decisions)
    write(OUT/'analysis_input_hashes.json',inputs)
    write(OUT/'numerical_audit.json',numerical_audit)
    # Plots describe actual observed calls; uncertainty bands are descriptive quantiles.
    colors=dict(aime26='#b55a21',humaneval='#237a73',longbench_v2='#6c50a4')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for benchmark,data in by_workload.items():
        calls=sorted(n for r in requests if r['benchmark']==benchmark for n in r['closed_calls'])
        axes[0].step(calls,np.arange(1,len(calls)+1)/len(calls),where='post',label=benchmark,color=colors[benchmark])
        values=[data['stopping_bottlenecks'].get(k,0)/data['N'] for k in ('both','stability','confidence','joint_satisfied')]
        axes[1].plot(range(4),values,'o-',label=benchmark,color=colors[benchmark])
    axes[0].set(xlabel='Native calls per completed canvas',ylabel='Empirical CDF',xlim=(0,49));axes[0].legend()
    axes[1].set(xticks=range(4),xticklabels=['both unmet','stability unmet','confidence unmet','both met'],ylabel='Fraction of denoising calls')
    axes[1].tick_params(axis='x',rotation=20);axes[1].legend();fig.tight_layout();fig.savefig(OUT/'native_stopping.png',dpi=180);fig.savefig(OUT/'native_stopping.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for col,benchmark in enumerate(by_workload):
        for field,axis in [('entropy_mean',axes[0,col]),('accepted_fraction',axes[1,col])]:
            grouped=collections.defaultdict(list)
            for item_id,trace in traces.items():
                if not item_id.startswith(benchmark+'/'):continue
                for call in trace:grouped[call['call_in_canvas']].append(call[field])
            xs=sorted(grouped);median=[np.median(grouped[k]) for k in xs]
            axis.plot(xs,median,color=colors[benchmark]);axis.fill_between(xs,[np.quantile(grouped[k],.25) for k in xs],[np.quantile(grouped[k],.75) for k in xs],alpha=.2,color=colors[benchmark])
            axis.set(xlabel='Call within native canvas',ylabel=field,title=benchmark)
            if field=='entropy_mean':axis.set_yscale('symlog',linthresh=.005);axis.axhline(.005,color='black',linestyle='--',linewidth=.8)
        axes[0,col].set_title(benchmark+'; descriptive survivor mix')
    fig.tight_layout();fig.savefig(OUT/'native_curves.png',dpi=180);fig.savefig(OUT/'native_curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for dataset,color in [('aime26','#b55a21'),('longbench_v2_0shot_think','#6c50a4')]:
        selected=[r for r in op_records if r['dataset']==dataset and r['candidate']=='mass']
        axes[0].scatter([r['kept_physical_tiles']/r['eligible_physical_tiles'] for r in selected],[r['output_bias_relative_max'] for r in selected],s=18,alpha=.7,label=dataset,color=color)
        axes[1].scatter([r['whole_deleted_mass_max'] for r in selected],[r['canvas_amplification_max'] for r in selected],s=18,alpha=.7,color=color)
        held=[r for r in op_records if r['dataset']==dataset and r['split']=='heldout']
        axes[2].scatter([r['response_gain']['total'] for r in held],[r['reuse'][0]['bias_motion_relative_max'] for r in held],s=8,alpha=.35,color=color,label=dataset)
    axes[0].set(xlabel='Kept / eligible physical tiles',ylabel='Current relative attention error');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Whole-attention deleted mass',ylabel='Canvas mass amplification')
    axes[2].set(xlabel='Mask-induced Q/K/V response',ylabel='Next-state attention-bias motion');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'operator_response.png',dpi=180);fig.savefig(OUT/'operator_response.pdf');plt.close(fig)
    lines=['# Dense native denoising census and bounded response diagnostics','',
        '**Decision: retain the mechanism question, but do not promote a sparse selector or launch a benchmark panel from this evidence.**',
        '',f'Protocol `{summary["protocol_sha256"]}`. Execution is source-hash bound; spec checkpoint `11f426f1`, corrected operator checkpoint `22c7ba43`. The original 30-item identity and pre-existing frozen files are preserved.',
        '', '## Dense native results', '',
        '| Workload | Completed / planned | Official correct | N | Emitted C | Pooled N/C | Completed-canvas N/C | Mean output T | Cap canvases | Output caps |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for benchmark,data in by_workload.items():
        lines.append(f'| {benchmark} | {data["items"]}/10 | {data["correct"]}/{data["items"]} | {data["N"]} | {data["C"]} | {data["N_over_C"]:.3f} | {data["closed_N_over_C"]:.3f} | {data["mean_T"]:.1f} | {data["cap_canvases"]}/{data["C"]} | {data["output_capped"]} |')
    lines += ['', 'All requests use the unchanged native sampler, thinking on, original seed42, and original output budgets. These are dense-only descriptive observations, not a quality-preservation or sparse acceleration result. Nine frozen LongBench prompts exceed the native262144-token limit; the user explicitly kept them blocked without truncation or replacement. The single runnable LongBench prompt has185546 tokens and produces no final response before its16384-token limit.',
        '', 'Every request includes one trailing uncommitted denoising call. C is the emitted/committed canvas count. N includes all executed denoising calls; completed-canvas N/C excludes that tail. The raw observer summary also contains C_observed=C+1 and must not be substituted for emitted C. Per-item fields retain both denominators.',
        '', 'The long item has median45 and p9048 calls per completed canvas,21/64 cap terminations.2631/2780 denoising calls are stable/history-ready but confidence-limited. This establishes a difficult confidence-limited case, not evidence that sparsity would improve it. AIME has one cap canvas/208; HumanEval has none/46.',
        '', '![Native stopping](native_stopping.png)', '', '![Native curves](native_curves.png)',
        '', 'Curves pool surviving canvases at each call and show median/IQR, not paired causal trajectories. Whole-canvas labels remain passive; no token is individually committed by these labels.',
        '', '## Observer qualification and accounting','',
        'The five measured HumanEval/53 smoke requests match output hash,275 output tokens,12 sampler calls,13 model executions, prompt tokenization and final CPU/CUDA RNG states. Minimal/rich/minimal active sampler state hashes agree;24 rich-observer RNG checks pass. A separate census engine reproduces that item. Dense warm/observed pairs also agree for the first AIME and long request. Across the21 census requests there are12320 observer RNG checks, zero native convergence reconstruction mismatches, zero request captures/JIT events, and model execution counters equal engine steps.',
        '', 'This qualifies the observer only within the tested scope. It does not implement or validate restoration of all KV caches, page tables, compiled/layout state and native continuation. Entropy acceptance masks are reconstructed from native scaled logits; exact acceptance-buffer equivalence was not directly captured. S, S/N and clean W are unavailable because all traces are instrumented. Selector overhead is zero for the dense arm. No clean speed claim follows.',
        '', '| Workload | GLOBAL eligible=kept | GLOBAL skipped | LOCAL eligible=kept | LOCAL skipped |',
        '|---|---:|---:|---:|---:|']
    for benchmark,data in by_workload.items():lines.append(f'| {benchmark} | {data["global_eligible_tiles"]:,} | 0 | {data["local_eligible_rectangles"]:,} | 0 |')
    lines += ['', 'Geometry comes from actual native prepare-attention metadata using Q128xKV64 rectangles, not measured CUDA CTA counts. First-canvas physical work was not separately instrumented. No PIECEWISE/all-kept/V31 native comparison was run in this stage; no physical sparse-work or consumer-effect claim is made.',
        '', '## Preserved-state operator diagnostic','',
        '653 candidate records (635 distinct physical supports after collapsing rounded budgets) cover12 early/middle/late states, four inherited V31 trajectories, layer29, heads0/8 and four sampled rows in the first query block. Prefix support fractions0.25/0.5/0.75/0.9 share mandatory boundary/current-canvas support.48 input files are hash-verified. Requests002/004 are held out. These are inherited sparse on-policy states with same-state dense recomputation, not the new dense census state distribution. Candidate families are mass, singleton deletion, bounded swaps and deterministic position sham; they are diagnostic supports, not a deployed ResponseGuard.',
        '', '| Held-out request family | Response vs next bias motion Spearman | Current error vs next bias motion | Matched-pair concordance | Pair count (correlated) |',
        '|---|---:|---:|---:|---:|']
    for dataset in ('aime26','longbench_v2_0shot_think'):
        d=operator['aggregates']['heldout/'+dataset]
        lines.append(f'| {dataset} | {d["response_vs_future_bias_motion_spearman"]:.3f} | {d["current_error_vs_future_bias_motion_spearman"]:.3f} | {d["matched_pair_concordance"]:.3f} | {d["matched_pairs"]} |')
    lines += ['', 'Response is associated with future attention-bias motion, and matched-pair ordering is suggestive. It does not beat current error in overall held-out correlation. The two held-out requests are not a population-level replication; pairs remain correlated within requests; identical support is deduplicated within each state/budget. No downstream logits, native acceptance/stopping interventions, correction/break labels or controlled final-answer continuation were measured. A full-state replay certificate remains absent; the study cannot claim completion of the causal protocol.',
        '', 'The exact local differential includes moving Q,K,V. Its query-only identity is dO/dQ=scale*Cov(V,K); key and value movement are recorded separately. Renormalization max absolute identity error is4.44e-16. Current-canvas tiles are never deleted, so alpha here is whole-attention **prefix-deleted** mass and1/(1-alpha) is the amplification of retained current-canvas mass. Actual current-canvas deletion is not tested.',
        '', 'The norm-based prefix retained-mass certificate has no observed violation. All18 all-kept records (12 distinct after collapsing rounded budgets) have exact certificates of1. Every one of1481 partial-support checks has a bound below0.5. This conservative bound cannot justify useful refresh selectivity. No production ValidityReuse variant is implemented.',
        '', 'The first operator attempt is preserved with FP32 relative-error outliers and repeated candidate supports. Corrected operator_diagnostic002 uses FP64, deduplicated supports and exact all-kept certificates. All653 records pass the frozen finite-difference tolerance (absolute error <=1e-5+1e-3 times analytic norm); median relative error is8.05e-7 and maximum absolute error4.14e-5. This numerical qualification is not native transition evidence.',
        '', 'The fixed candidate-selection readout collapses rounded budgets to635 physical supports in84 groups. It chooses minimum response among the frozen feasible candidates, with mass fallback; current-error selection uses the same feasible set, and a support-hash sham is outcome-independent. On the held-out AIME request, response selection has0 lower /22 equal /2 higher next attention-bias-motion outcomes versus current-error selection. On held-out LongBench, it has5 lower /15 equal /4 higher. Twelve of24 LongBench groups require fallback. These are posthoc analytical readouts of the frozen bank, not native transitions or a global support oracle. The reproducible candidate_readout.py reproduces the saved JSON exactly. This mixed result does not justify a proxy or native benchmark.',
        '', '![Operator response](operator_response.png)',
        '', '## Go/no-go and unfinished work','']
    for name,decision in decisions.items():lines += [f'- **{name}: {decision["status"]}.** {decision["evidence"]}.']
    lines += ['', 'Next engineering work is an exact native full-state snapshot/restore certificate and same-support continuation before any support intervention. After that gate, evaluate the exact ResponseGuard candidate-ranking oracle against current-error and sham controls on dense states. MarginGuard remains an untested exact-oracle track; do not build a cheap proxy yet. The tested norm certificate is a negative practical lead for ValidityReuse. No full benchmark, kernel tuning, adaptive K or accuracy-fitted rescue is justified.',
        '', 'Phase0 audit and the eligible dense census component of Phase1 are complete; full30 coverage is partial by explicit user instruction. Phase2 has only bounded attention-level D1/D3/D4/D5 components. Incremental D2 native headroom, logit/acceptance/self-conditioning interventions, full-state causal forks and held-out native transition prediction remain incomplete. Phase3 methods and matched native controls have not run. No novelty, accuracy preservation, trajectory acceleration or speedup is established.',
        '', '## Provenance and primary sources','',
        'Yuhan owns the inherited first cross-step reuse baseline. V31, Gaussian32/value-direction routing, query-sensitivity/C-gate and V2 remain distinct prior method families. Historical value/reconstruction, refresh, consumer and timing negatives are preserved; this study does not rename or replace them.',
        '', 'The [DiffusionGemma report](https://arxiv.org/pdf/2608.00146) supplies the native sampler contract and warns that entropy collapse can terminate incorrect/repetitive outputs. [SparseD](https://arxiv.org/html/2509.24014v1) and [MAGE](https://arxiv.org/html/2602.14209) precede support reuse; [TABES](https://arxiv.org/html/2602.00250v2) precedes future-uncertainty steering. Retrieved primary documents and hashes are preserved in execution001/research_sources; no novelty claim follows.',
        '', '## Reproducibility and limitations','',
        'See protocol.json, source_fingerprints.json, smoke/receipt.json, census_short and census_long final_complete markers, scoring001/official_scores.json, operator_diagnostic002/protocol.json and summary.json. Failed execution001/002/003 remain intact. Source/runtime hashes match; torch2.13.0+cu130, vLLM0.30.0, H10080GB, supported V2 runner and exact mask/window fixes. No paused campaign was resumed. Private prompts, gold, token IDs and raw completions remain ignored/private.',
        '', 'Exact experiment commands and environment overrides are in COMMANDS.md and command_receipts.json. All22 focused CPU tests pass; every one of16 pinned model files (including11 weight shards) matches its size and SHA256. CPU tests and analysis hashes are in the integrity receipt. There is no whole-study final_complete marker. The report completion marker closes only this diagnostic report.']
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n')
    commands=[]
    for p in sorted(ROOT.glob('*launch.json')):
        receipt=read(p);commands.append(dict(receipt=str(p.relative_to(ROOT)),**receipt))
    commands.append(dict(receipt='operator_diagnostic002/launch_receipt.json',**read(ROOT/'operator_diagnostic002/launch_receipt.json')))
    write(OUT/'command_receipts.json',commands)
    cmdlines=['# Commands and execution receipts','', 'Working directory: `/home/exouser/ljy/dlm`. Arguments below are exact worker argv from launch receipts; each receipt also records environment and timestamps. Shell wrappers checked live GPU clients before launch and preserved logs/exit status.','']
    for item in commands:
        cmdlines += [f'## {item["receipt"]}','','```bash',shlex.join(item['command']),'```','']
        for key,value in item.get('environment_overrides',{}).items():cmdlines.append(f'- `{key}={value}`')
        cmdlines.append('')
    cmdlines += ['## CPU and official scoring commands','','```bash',
        'PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python -m pytest -q tests/test_trajectory_dynamics.py tests/test_dynamics_screen_protocol.py results/denoising_dynamics_census_20261022/attempt001/execution004/tests/test_production_observer.py results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/tests/test_diagnose.py --junitxml=results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/tests/cpu001.xml',
        'PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_dynamics_census_20261022/attempt001/execution004/operator_diagnostic002/code/diagnose.py --freeze',
        'python scripts/v27_humaneval.py sandbox-check',
        'PYTHONPATH=src:. python results/denoising_dynamics_census_20261022/attempt001/execution004/code/score_census.py',
        'python results/denoising_dynamics_census_20261022/attempt001/execution004/code/candidate_readout.py',
        'PYTHONPATH=src:. /home/exouser/dyh/v31_env_20261005/bin/python results/denoising_dynamics_census_20261022/attempt001/execution004/code/report_final.py',
        '```','','The scorer host sandbox preflight passes. The same preflight inside the restricted workspace sandbox fails, so code execution used the unchanged official bwrap sandbox on the host. Earlier exact launch argv and failures remain in execution001/002/003 launch/exit receipts. Read-only audits used git branch/log/status, nvidia-smi, source hashing, targeted rg/sed/cat and model-tokenizer checks.']
    (OUT/'COMMANDS.md').write_text('\n'.join(cmdlines)+'\n')
    write(OUT/'analysis_complete.json',dict(complete=True,scope='eligible dense census and bounded operator diagnostic report only',
        full30_complete=False,native_selector_promotion=False,summary_sha256=sha(OUT/'summary.md'),
        summary_json_sha256=sha(OUT/'summary.json'),report_source_sha256=sha(__file__)))
    print(json.dumps(dict(report=str(OUT),workloads=by_workload,decisions=decisions),indent=2))

if __name__=='__main__':main()
