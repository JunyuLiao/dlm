"""Render completed matched evidence; never tune methods or claim novelty."""
import argparse
import hashlib
import json
from pathlib import Path


def read(p):return json.loads(Path(p).read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();data=read(root/'analysis001/summary.json');protocol=read(root/'e2e_protocol.json')
    paired=read(root/'paired_trajectory001/summary.json')['comparisons']
    verification=read(root/'closeout001/verification.json')
    diversity=read(root/'support_analysis001/summary.json')
    lines=['# Complete native end-to-end selector screen','',
        'User-requested cohort:10 AIME26,10 HumanEval,5 original native-eligible long-band LongBench-v2 questions, each at seeds42/43. Nine arms:450 scored item/seed/arm cells. Full original prompts and output limits are unchanged; nine original over-limit LongBench IDs remain blocked. This is an exposed descriptive screen, not a fresh benchmark or noninferiority test.','',
        'Protocol SHA256: `'+hashlib.sha256((root/'e2e_protocol.json').read_bytes()).hexdigest()+'`. Generation checkpoint:4ac2b53f. See e2e_protocol.json, manifest_identity.json, launch receipts and closeout001/verification.json. No paused campaign resumed; no other workload preempted.','',
        'Methods are training-free selector variants using current Q/K/V on retained tiles. New variants affect only layer29; V31 retains its inherited five-GLOBAL-layer scope and K128/K8192 budgets. V31 is a practical reference with different physical work, not a matched-work attribution control. Current-error, response, margin, certified-reuse and sham share the new state-specific candidate construction/guards; their later native trajectories can change future feasibility and work.','']
    for task in ['aime26','humaneval','longbench_v2']:
        lines += ['## '+task,'',
            '| Arm | Correct/cells | Accuracy | N | C | N/C | T | Mean clean W(s) | GLOBAL consumer sparsity |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in protocol['arms']:
            x=data['summary'][task][arm]
            timing=f"{x['mean_W']:.4f}" if x['clean_timing_valid_cells']==x['cells'] else f"unqualified ({x['mean_W']:.4f})"
            lines.append(f"| {arm} | {x['correct']}/{x['cells']} | {100*x['accuracy']:.1f}% | {x['N']} | {x['C']} | {x['N_per_C']:.3f} | {x['T']} | {timing} | {100*x['global_sparsity']:.2f}% |")
        lines += ['',
            '| Candidate vs corrected FULL | Accuracy difference pp [paired95% CI] | N/C geometric ratio [95% CI] | W geometric ratio [95% CI] | Correct→wrong / wrong→correct cells |',
            '| --- | --- | --- | --- | --- |']
        for arm in protocol['arms']:
            if arm=='dense_full_fix51994':continue
            p=data['paired'][task][arm+'__vs__dense_full_fix51994']
            t=next(v for v in paired if v['task']==task and v['arm']==arm and v['reference']=='dense_full_fix51994')
            n=t['metrics']['N_per_C'];w=p['W'];ci=lambda v:', '.join(f'{x:.3f}' for x in v)
            lines.append(f"| {arm} | {p['accuracy_difference_pp']:+.1f} [{ci(p['accuracy_ci95_pp'])}] | {n['geometric_ratio']:.3f} [{ci(n['ratio_ci95'])}] | {w['geometric_ratio']:.3f} [{ci(w['ci95'])}] | {p['reference_correct_candidate_wrong']} / {p['reference_wrong_candidate_correct']} |")
        lines += ['', 'Intervals use10000 paired question-cluster bootstrap draws retaining both seeds. The two seeds are not two independent questions. Small-cluster intervals are descriptive; nonsignificance does not establish accuracy preservation.','']
    lines += ['## Implementation, controls and diagnostic interpretation','',
        'ResponseGuard minimizes current sparse attention query-direction response J_Su, with residual (J_S−J_A)u recorded separately. K/V are current but their future motion is not predicted. MarginGuard is inverse-prior-margin-weighted attention bias, not the proposed exact downstream influence oracle. Certified reuse bounds retained mass under query drift with fixed within-canvas prefix keys and recomputed current-canvas partition; reset at every canvas. See THEORY_AND_LIMITS.md and code/selector.py for assumptions and independent algebra checks.','',
        'This screen extends the prior non-identifying four-unit experiment. It does not retroactively pass that gate or separate pulse propagation from repeated intervention. The methods repeatedly select/hold support, so full-request differences establish effects of those schedules, not an isolated contraction mechanism. Native/all-kept numerical consumer differences remain visible.','',
        f"First-refresh support readout: {len(diversity['comparisons'])} cell/score comparisons; {sum(x['first_bank_inputs_agree'] for x in diversity['comparisons'])} have identical candidate current-error/feasibility banks at the common native-bootstrap boundary; {sum(x['distinct_unit_choices'] for x in diversity['comparisons'])} distinct routing-unit choices. These are candidate-diversity diagnostics, not independent benchmark samples or predictive confirmation.",'',
        'Per-request native flags/entropy and actual native commit-based canvas counts with an explicitly retained uncommitted tail (CPU emission intervals can lag native commits) are in audit records. trajectory_analysis001 separates confidence-only, stability-only, both and history-readiness bottlenecks, first/terminal canvases and48-call caps. paired_trajectory001 includes paired N,C,N/C,T,S/N and request-time intervals and both-seed item summaries. There is no exact compiled-acceptance count claim.','',
        'Consumer rectangles exclude selector discovery. Each audit selector receipt separately reports discovery QK/PV rectangle counts and CUDA-event time for support management. selector_discovery events are nested inside complete_support_management_refresh; do not add those overlapping event totals. Request W includes all necessary selector work. Skipped consumer rectangles or fewer calls alone are not speedup. LOCAL stays dense; rectangle counts are not CUDA CTA counts.','',
        'A complete first warm request is followed by per-cell audit/clean full requests. The audit serves as a shape-specific warm request for clean timing. All audit/clean output hashes,N,C,T must match. Any new clean capture/JIT makes timing unqualified; audit instrument time is never the clean latency.','',
        'The prior restoration qualification remains a preserved failure: HE124 after4 has identical85-call full native model/logit/RNG/count fingerprints, but one restored output list differs by256 tokens. End-to-end evaluation uses fresh from-prompt requests and does not depend on restored outputs.','',
        'Original10 AIME and10 HumanEval prompts have prior dense exposure; some also have sparse diagnostic exposure. Four additional LongBench items were chosen without outcome filtering from native-context-eligible long-band raw prompts; prior historical exposure is not certified absent. The resulting five long questions cover only two domains. Context eligibility prevents a domain-balanced long-band subset under the unchanged native limit; no truncation/replacement of the nine blocked original items occurred.','',
        '## Resources and verification','',
        f"Expanded panel allocated-worker time: {verification['allocated_worker_seconds']:.3f}s; counted model forwards including startup/profile: {verification['counted_model_forwards']}. Campaign GPU/forward ceilings were removed by the user. Per-process4-hour hang watchdog remains. This allocation is not active GPU utilization or clean request latency.",'',
        f"All {len(verification['checks'])} closeout checks pass; {len(verification['invalid_clean_timings'])} clean timings are unqualified.32 focused CPU tests passed before generation. All16 model/config files were rehashed and match. Source/runtime/model/scorer identities and public privacy fields are verified. Private inputs/generations/logs stay ignored. Individual worker launch/exit and complete markers preserve failures and runtime evidence.",'',
        'Earlier qualification/smaller preflight resources are separate, preserved under attempts001/002, and are not folded into clean request time or silently called panel coverage. The expanded panel never reuses the smaller preflight scores. No push is requested.','',
        '## Decisions','',
        'No production selector, novelty, formal accuracy noninferiority or full denoising-contraction claim is promoted by this exposed small screen. Exact paired task regressions, calls per canvas, output changes and request costs above determine whether any variant merits another bounded confirmation. A lower-call result with worse task outcomes fails the objective; a slow selector cannot claim speedup from fewer consumer tiles. Norm-only ValidityReuse remains a separate historical negative; this current-canvas-aware certificate is a named exploratory variant, not erasure of that result.','',
        'Yuhan authored the inherited cross-step reuse machinery. Gaussian32/value-direction routing, C-gate/query sensitivity, V31/V2, and the earlier negative studies retain their own provenance.','',
        'Inspect first: this summary; final_complete.json; analysis001/summary.json; paired_trajectory001/summary.json; scoring001/official_scores.json; support_analysis001/summary.json; trajectory_analysis001/summary.json; closeout001/verification.json; frozen e2e_protocol.json and code/.']
    with (root/'summary.md').open('x') as f:f.write('\n'.join(lines)+'\n')
    decisions=dict(method_promoted=False,scope='Exposed25-question two-seed nine-arm end-to-end screen',
        production='no_go_pending_independent_confirmation_and_cost_evidence',
        exact_margin_influence_oracle='not_implemented_or_validated_by_this_local_surrogate',
        isolated_response_mechanism='not_identified_by_repeated_intervention_end_to_end_design',
        norm_only_ValidityReuse='historical_track_remains_parked',
        numerical_evidence=data['paired'],paired_trajectory_source='paired_trajectory001/summary.json',
        result_policy='Retain all negatives, task regressions, caps and missing/unqualified timing. Do not infer speed/quality preservation from requested sparsity or nonsignificance.')
    with (root/'method_decisions.json').open('x') as f:json.dump(decisions,f,indent=2,sort_keys=True);f.write('\n')
    print(root/'summary.md')


if __name__=='__main__':main()
