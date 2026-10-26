"""Build the bounded handoff from complete, source-bound public records."""
from pathlib import Path
from collections import defaultdict
import json,hashlib

R=Path(__file__).resolve().parents[1];C=R/'panel/attempt004';B=R/'boundary/attempt001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False)
def main():
    inputs={}
    def read(p):inputs[str(p.relative_to(R))]=sha(p);return json.loads(p.read_text())
    assert read(R/'verification002/summary.json')['passed']
    s=read(C/'analysis001/summary.json');p=read(C/'paired_readout001/summary.json');b=read(B/'analysis001/summary.json');d=read(C/'diagnostics001/summary.json');resources=read(R/'closeout001/resources.json')
    protocol=read(C/'protocol.json');manifest=read(C/'manifest_identity.json');binding=read(C/'manifest_binding.json')
    arms=protocol['arms'];short=dict(zip(arms,['FULL','PW','A','CE','sham','high','low']))
    by={}
    for arm in arms:
        for i in range(24):
            v=read(C/f'runs/panel/{arm}/audit{i:03d}.json');by[(v['id'],v['seed'],arm)]=v
    decisions=[]
    for arm in arms[3:]:
        replicated=[];local_A_rep=[]
        for m in manifest['identities']:
            useful=True;local=True
            for seed in [42,43]:
                v=by[(m['id'],seed,arm)];a=by[(m['id'],seed,arms[2])];ce=by[(m['id'],seed,arms[3])];sh=by[(m['id'],seed,arms[4])]
                exposed=all(x['runtime']['pulse_calls']==1 for x in [v,ce,sh])
                useful=useful and exposed and v['remaining_calls']<ce['remaining_calls'] and v['remaining_calls']<sh['remaining_calls']
                local=local and bool(v['runtime']['pulse_calls']) and v['remaining_calls']<a['remaining_calls']
            if useful:replicated.append(m['id'])
            if local:local_A_rep.append(m['id'])
        rows=[v for (identity,seed,ar),v in by.items() if ar==arm]
        baseline=[by[(v['id'],v['seed'],arms[2])] for v in rows]
        transitions=[x for x in s['comparisons'] if x['method']==arm and x['cohort']=='policy_all' and x['stratum']=='all']
        decisions.append(dict(arm=arm,advance=False,decision='no_go_for_policy_advancement',
            replicated_remaining_call_lead_over_CE_and_sham_questions=replicated,
            replicated_local_call_lead_over_A_questions=local_A_rep,
            complete_request_N_change_vs_A=sum(v['N'] for v in rows)-sum(v['N'] for v in baseline),
            observed_correct_to_wrong_vs_declared_controls=sum(x['correct_to_wrong'] for x in transitions),
            exposed_seed_cells=sum(v['runtime']['pulse_calls'] for v in rows),
            correct_A_exposed_seed_cells=sum(v['runtime']['pulse_calls'] and a['correct'] for v,a in zip(rows,baseline)),
            reason='No replicated useful remaining-call advantage over both CE and sham on two questions; no complete-request work/quality improvement over A; clean policy latency is higher. Sparse correct-reference exposure is one seed cell.'))
    machine=dict(decisions=decisions,engineering_pass=True,boundary_observation='supports_RNG_position_coupling_within_measured_scope_not_exclusive_mediation',
        old_Response_and_A1_gates='remain_failed_not_reinterpreted',ValidityReuse='parked_no_hold_or_carry_test',
        exhaustive_MarginGuard='not_justified',Stage_BC='not_launched_not_authorized_by_this_screen',kernel_optimization='not_performed_not_justified_by_scientific_gates',
        no_accuracy_noninferiority_or_novelty_claim=True,inputs=inputs)
    save(R/'closeout001/method_decisions.json',machine)
    lines=['# Commit-boundary diagnostic and actual online single-pulse panel', '',
        '**Complete bounded study; no-go for advancing these pulse selectors.** Native sparse pulses can change stopping, but neither signed-bias directional rule provides a replicated useful advantage over current-error and sham. All four policies are slower than matched all-kept. The boundary observation supports RNG-position coupling within the measured scope; it does not establish exclusive mediation or expected quality harm.', '',
        '## Identity and completed scope', '',
        'Branch `denoising_trajectory_research`; start `748e22b328e167cd64c64a46afda9bec1ff06d32`. Study date is 2026-10-26 HKT; execution receipts use Unix UTC. Task-owned final publication commit is reported in the final handoff/git log, avoiding a self-referential commit hash in its own contents.', '',
        '|Leaf|Generation source|Protocol SHA256|Completion|', '|---|---|---|---|']
    for name,root in [('B: boundary',B),('C: online panel',C)]:
        z=read(root/'binding_receipt.json');lines.append(f"|{name}|`{z['source_checkpoint']}`|`{z['protocol_sha256']}`|{str(root.relative_to(R))}/execution_complete.json|")
    lines += ['',f"Cohort checkpoint `{binding['cohort_checkpoint']}`; manifest SHA256 `{binding['manifest_sha256']}`. Generation and cohort freezes precede treatments. All596 B and601 C source pins,16 model/config/tokenizer files,7 installed runtime files and the official gold binding reverify in [verification002](verification002/summary.json).", '',
        'A independently corrects historical reporting without changing frozen inputs. B completed24 fresh native requests,3 qualified common states and10 distinct historical deleting supports. C completed48 baseline-profiling requests,21 engineering audit/clean pairs and168 actual panel audit/clean pairs (12questions ×2seeds ×7arms). No scientific condition in B was used to decide whether C would run.', '',
        'No campaign resource ceiling, restored continuation, RNG realignment, hidden/suffix model forward, forced continuation, stale K/V, compensation, kernel campaign, paused-campaign resumption, Stage B/C or push. Native output caps8192short/16384long and48-call canvas cap are unchanged. Retained support consumes current Q/K/V; LOCAL/protected/prefill/commit stay dense.', '',
        '## A. Corrected historical evidence', '',
        'All10 old target-canvas native argmax tensors match their A control. Five earlier stops changed complete output/final RNG; five unchanged stops reproduced them. Official rescoring of34 historical control/deleting outputs matches all saved quality/extraction labels. Canonical entropy slack is0.005−entropy. All700 canonical R work rows verify;420 stale raw overall/phase fields remain untouched and explicitly noncanonical. Published R aggregates and failed R/A1/H conclusions do not change. Deleting GQA maximum is0.003998036; control-inclusive maximum0.004263655. Signed-zero SC byte differences are not numerical differences. H\'s old gate omitted complete work/correct-reference coverage and is not reused.', '',
        'See [CPU addendum](cpu_addendum001/summary.md), [work audit](cpu_addendum001/canonical_work_verification.json) and [official rescoring](cpu_addendum001/official_rescore_verification.json).', '',
        '## B. Native commit boundary', '',
        'Observation reproduces old full output,N,C,T,final RNG and forward counts for all tested pulses; three common states and repeated/bracketing A controls qualify. All10 terminal argmax canvases, emitted cumulative prefixes, commit inputs and committed active K/V across all30 layers agree numerically. Reset SC agrees numerically (zero), positions/input metadata agree, and all measured states are finite. CPU signed-zero canonicalization touches copies only.', '',
        'In all five earlier-stop cases, the next native RNG state/random canvas and next model input differ. Five same-stop cases reproduce them. Returned scaled commit logits can differ because native step/temperature differs; these observed logits are not a persistent SC value after reset. Captured cache/model-layer outputs agree at commit. The evidence supports RNG-position coupling, while unmeasured opaque state and mediation exclusivity remain outside qualification. No RNG is manipulated.', '',
        '|Historical state / support|Remaining A→S|Complete N A→S|T A→S|Task result|', '|---|---:|---:|---:|---|']
    for x in b['cases']:lines.append(f"|{x['id']} seed{x['seed']} / {x['label'].split('_',1)[1]}|{x['A_remaining']}→{x['remaining']}|{x['A_N']}→{x['N']}|{x['A_T']}→{x['T']}|wrong→wrong|")
    lines += ['', 'All ten deleting requests are wrong/output-capped. The15→12local gain with767→2963total calls is a failure of useful complete-request acceleration. These are selected historical states, not independent quality-preservation evidence. B timing is instrumented audit timing, not clean policy latency. See [boundary summary](boundary/attempt001/summary.md), [phase comparisons](boundary/attempt001/analysis001/summary.json) and [metadata addendum](boundary/attempt001/metadata_addendum001.json).', '',
        '## C. Cohort, policy and opportunity coverage', '',
        'The24-question baseline-only roster was frozen before profiling. A deterministic rule selects2targeted+2coverage questions/task, with prospective fallback within that roster. Only AIME26/2 seed42 meets the strongest correct-FULL/A and≥3remaining-call criterion; no HumanEval/LongBench candidate meets it. No later recruitment or sparse-outcome selection occurred. All AIME identities were exposed in R; HumanEval/LongBench are outside known R/A1/H exposure, with generic historical exposure not certified absent. The original nine over-limit LongBench identities remain blocked; these four belong to a new eligible manifest.', '',
        '|Task|Targeted identities|Coverage identities|', '|---|---|---|']
    for task in ['aime26','humaneval','longbench_v2']:
        ids=lambda st:', '.join(x['id'] for x in manifest['identities'] if x['benchmark']==task and x['stratum']==st)
        lines.append(f'|{task}|{ids("targeted")}|{ids("coverage")}|')
    lines += ['', 'All seeds42/43 are paired. Only layer29, heads0/8, queryblocks0/1 can delete Q128×KV64 wholly-prefix rectangles; each deletes max(1,floor(.05B)). Native bootstrap1/canvas then all-kept layer29 is the common consumer. CE/sham eligibility is independent of a directional pair. High/low use the original error/mass-matched signed-bias pair and otherwise abstain. Only the first native opportunity is considered. Selection is online from current operands in the clean request; no prerecorded map is loaded.', '',
        '|Task|Anchor cells /8|CE/sham eligible|Directional pairs|Four distinct supports|', '|---|---:|---:|---:|---:|']
    for x in p['bank_coverage']:lines.append(f"|{x['task']}|{x['anchor_cells']}|{x['feasible_four_unit_cells']}|{x['direction_pair_cells']}|{x['four_distinct_support_cells']}|")
    lines += ['', 'There are8 distinct anchor item/seed states on5questions,7directional pairs and30exposed method×seed records. CE/sham expose8each; high/low7each. Sixteen cells lack an anchor; directions additionally abstain on LongBench structured-data seed42 because no directional pair exists. At HumanEval/145 seed42, CE and sham alias:29distinct support×cell records, not30independent effects. Only ONE correct-A seed cell is exposed (AIME26/2 seed42). No-pulse equality in other correct cases is not sparse quality-preservation evidence.', '',
        'Every unit has≥32feasible alternatives. High/low match error/mass but do not match CE\'s current error in any of the7pairs. Varied-unit cosine spans−0.297to0.456; composite cosine spans−0.204to0.780. All varied units are head8; do not call this a head-independent geometry result. The present current signed-bias score is distinct from failed A1 full/residual-response forecasts. Finite scalar-error matching does not equate position-level perturbations.', '',
        '## Complete-request quality, trajectory and clean latency', '',
        'Each task has4independent question clusters and8seed cells; this is a targeted/exposed diagnostic sample, not a representative benchmark or noninferiority test. N,C,T below are arithmetic totals; N/C is the arithmetic mean of request N/C. C is actual native commits. N includes the native uncommitted tail (one call/request in all168panel rows), retained in accounting. W/S/SN are means of synchronized clean requests after warmup; W starts at tokenized engine request admission, S after prefill, and S/N is an amortized request cost including native loop/commit work. Startup/scoring/audit costs are separate.', '',
        '|Task|Arm|Correct /8|N|C|T|Mean N/C|W sec|S sec|S/N ms|', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for x in s['aggregates']:
        if x['stratum']!='all':continue
        rr=[v for v in s['rows'] if v['benchmark']==x['task'] and v['arm']==x['arm']]
        lines.append(f"|{x['task']}|{short[x['arm']]}|{x['correct']}|{x['N']}|{x['C']}|{x['T']}|{x['mean_N_per_C']:.4f}|{x['mean_clean_W']:.4f}|{sum(v['S'] for v in rr)/8:.4f}|{sum(v['S_per_N'] for v in rr)/8*1000:.3f}|")
    lines += ['', 'All168timings qualify: zero new CUDA captures/JIT and exact audit/clean output,N,C,T,RNG,forward/support equality. FULL/PW output/state counters agree24/24; A/pulse common prestate/current operands and all abstentions qualify. Timing intervals require both seeds valid in both arms; none excluded. Intervals are descriptive question-cluster bootstrap20,000draws, not system-level timing replication. Arm order is fixed; one clean measurement/cell.', '',
        'All four pulse policies match A official quality, with0correct→wrong and0wrong→correct transitions versus A/CE/sham. Against FULL/PW, all show3wrong→correct AIME transitions and0regressions; the identical improvement is already present in A. Thus it is a consumer/trajectory effect, not a sparse-selector accuracy gain. Targeted/coverage quality is respectively: A/pulses AIME3/4and2/4; HumanEval2/4and4/4; LongBench0/4and4/4. FULL AIME is2/4and0/4; other task strata match A. All exact transitions/strata/common-eligible/exposed summaries remain in [analysis001](panel/attempt004/analysis001/summary.json).', '',
        'Output-token caps (FULL→A/pulses): AIME5→3,HumanEval1→1,LongBench3→2. Missing final responses/extraction failures: AIME3→2,HumanEval1→1,LongBench4→2.48-call canvas-cap events:0short,LongBench20→3. Native stop/confidence/cap-coincidence flags remain separate in per-call records. Wrong/capped cases and missing final responses block an unqualified accuracy-preservation claim.', '',
        '|Policy / A|AIME paired W ratio [95%CI]|HumanEval|LongBench|', '|---|---|---|---|']
    for arm in arms[3:]:
        vals=[]
        for task in ['aime26','humaneval','longbench_v2']:
            e=next(x for x in p['primary_timing'] if x['task']==task and x['stratum']=='all' and x['method']==arm and x['reference']==arms[2])['estimates']['W']
            vals.append(f"{e['point']:.4f} [{e['CI95'][0]:.4f},{e['CI95'][1]:.4f}]")
        lines.append('|'+short[arm]+'|'+'|'.join(vals)+'|')
    lines += ['', 'All policies are slower than A. Arithmetic mean LongBench W is lower than FULL, but A is already27.6249sec versus FULL34.2285sec before deleting any tile. Its paired W ratio is0.8748 [0.4961,1.3162]; this does not establish a sparse speedup. No selector improves complete N on AIME/LongBench; HumanEval totals increase by13(CE),1(sham),8(high),12(low). All C totals equal A. Paired N/C ratios are1on AIME/LongBench and slightly greater than1on HumanEval. No complete-request utility gate passes.', '',
        '## Local outcomes at fixed physical budgets', '',
        '|Anchor item/seed|Canvas / after call|Prefix B / deletion per unit|A remaining|CE|sham|high|low|A correct|', '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for z in p['bank_cells']:
        if z['anchor'] is None:continue
        a=by[(z['id'],z['seed'],arms[2])];ce=by[(z['id'],z['seed'],arms[3])];bank=ce['runtime']['candidate_bank'];values=[]
        for arm in arms[3:]:
            v=by[(z['id'],z['seed'],arm)];values.append(str(v['remaining_calls']) if v['runtime']['pulse_calls'] else 'abstain')
        an=z['anchor'];lines.append(f"|{z['id']} /{z['seed']}|{an['canvas']} /{an['after_completed_calls']}|{bank['blocks']} /{bank['drop_count']}|{a['remaining_calls']}|"+'|'.join(values)+f"|{a['correct']}|")
    lines += ['', 'AIME outputs and N/C/T all reproduce A despite distinct supports. HumanEval/145 seed42: all target canvases agree, low4→2 versus CE/sham/high4→3, yet every policy increases total N554→562 and remains wrong/output-capped. Seed43: CE33→38,sham33→26,high33→33,low33→37; sham alone reduces total N554→547 while retaining the same wrong complete output. CE/low change target content and remain wrong. Directional ranking is mixed across seeds; neither direction beats both CE/sham on any question in both seeds.', '',
        'LongBench seed43: CE/sham7→6 yet total N/output match A; high/low7→7. Seed42directions abstain and CE/sham2→2. Both LongBench exposed references are wrong/capped. The B association between earlier stopping and changed whole output is therefore case-specific: earlier stopping can later recouple. No RNG mediation exclusivity follows.', '',
        'At pulse/next ordinary same-canvas calls, native winners do not change in the30exposed records, although position-wise signed margins/entropy and later states can differ. Worst recorded margin erosion is9.49794scaled-logit units. Actual winner changes at later terminal states are separate. These are reference-winner metrics, not reasoning-token correctness. Numeric logit/entropy/margin curves are collected only in the target window, not the entire request; no exact compiled acceptance counts are claimed. SC/history byte differences alone are not numerical SC proofs. See [position/content diagnostics](panel/attempt004/diagnostics001/summary.json).', '',
        '## Physical work and overhead', '',
        'Physical counts are Q128×KV64 rectangles in native denoising consumers, not FLOPs; dense prefill/commit remain separate. GLOBAL/LOCAL eligible/kept/skipped and denominators are in every cell. LOCAL/protected skips are zero. The overall denominator is GLOBAL+LOCAL eligible rectangles. All189qualification/panel work rows independently verify. Different complete trajectories can change eligible work even at the same total N.', '',
        '|Task / arm|GLOBAL eligible|GLOBAL skipped|LOCAL eligible|Overall skipped fraction|Discovery QK / PV|', '|---|---:|---:|---:|---:|---:|']
    for x in s['aggregates']:
        if x['stratum']!='all' or x['arm'] not in [arms[2],*arms[3:]]:continue
        w=x['physical'];skip=w['global_eligible']-w['global_kept'];fraction=skip/(w['global_eligible']+w['local_eligible'])
        lines.append(f"|{x['task']} /{short[x['arm']]}|{w['global_eligible']}|{skip}|{w['local_eligible']}|{fraction:.9g}|{w.get('discovery_QK_rectangles',0)} /{w.get('discovery_PV_rectangles',0)}|")
    lines += ['', 'Kept=eligible−skipped; LOCAL kept equals eligible. Discovery value projections/bias inner products and audit-only exact GQA QK/PV are separately itemized in analysis001. Each reached opportunity builds the bank once, including failed-direction banks; no held/carried calls or cross-step reuse is tested. Audit selector seconds include optional GQA work and are NOT an isolated clean selector-overhead estimate. Clean W charges the complete online policy, including required discovery and anchor logic. Nested costs are not added twice. Tiny physical deletion alone implies no speed benefit.', '',
        '## Engineering, resources, failures and decision', '',
        f"23focused CPU tests pass. B24fresh requests and C21qualification+168panel pairs pass;16model/config files,7runtime sources,official scorer/gold and596/601source closures reverify. Resources: {resources['complete_native_requests']}complete native requests including{resources['warm_requests']}warmups;{resources['engine_execute_forwards']}engine executions (including{resources['initialization_engine_forwards']}initialization executions) +{resources['startup_model_calls_outside_execute']}startup calls outside execute ={resources['explicit_total_model_calls']}explicit model calls. {resources['allocated_GPU_worker_seconds']:.3f}allocated worker seconds ={resources['allocated_GPU_worker_hours']:.6f}hours; one H100worker at a time, no overlapping allocations. Allocation includes startup/warmup/qualification/scoring/cleanup and CPU hashing inside workers, at10secparent-poll resolution; it is not clean latency.", '',
        'Preserved failures: panel001/002 were superseded before GPU launch; only their identified waiting controllers were stopped. Panel003 failed at import because select.py shadowed stdlib select: one failed allocated process, zero model calls. Panel004 repairs the import and passes23tests. CPU failures include a legacy work-field alias, a verifier interpreter wrapper, final runtime-package path lookup, and relay001 stopping at upstream panel003 failure. New named revisions preserve each failure; no GPU rerun was used to repair reporting. All18successful native workers exit0, one pre-model import worker exits1. No native/unrelated worker was killed. Final H100 is idle0MiB,0compute clients.', '',
        '**Go:** accept the qualified boundary observation and complete exploratory measurements for independent review. **No-go:** advance CE, high or low as useful stopping/latency policies; build an exhaustive MarginGuard oracle; rescue failed A1 forecasts with kernels; infer reuse, AA/SA/AS/SS interaction, population noninferiority, novelty, or Stage B/C readiness. The sham is a control, not a promoted method. Current norm-only ValidityReuse stays parked. These negatives concern this scope/rule, not every layer, budget or response mechanism.', '',
        'The most informative next step is independent review, followed—only under a new authorization/protocol—by a dense-only opportunity audit with adequate correctly answered, slow, reachable anchors on independent questions and both seeds before another selector study. Do not expand this completed matrix. No V31 rerun was authorized in this seven-arm protocol; historical V31/Response results remain separate and source-matched. Yuhan\'s inherited cross-step reuse, Gaussian32/value routing, C-gate, V31/V2, Response and these signed-bias pulse variants retain distinct provenance.', '',
        '## Review and reproduction', '',
        'Read first: this summary; [method decisions](closeout001/method_decisions.json); [paired cells/intervals](panel/attempt004/paired_readout001/summary.json); [boundary comparisons](boundary/attempt001/analysis001/summary.json); [verification](verification002/summary.json); [resources](closeout001/resources.json). [COMMANDS.md](COMMANDS.md) identifies canonical and failed/superseded commands. Public per-call/cell traces contain hashes, flags and numeric metrics only; prompts/gold/token IDs/tensors/raw generations/private paths stay ignored. Full public file inventory and final completion marker are written after publication/privacy verification. Existing AGENTS/HANDOFF/DECISIONS edits are preserved; only task-owned files/document deltas are committed, with no push.', '']
    (R/'summary.md').write_text('\n'.join(lines))
    save(R/'closeout001/handoff_inputs.json',dict(inputs=inputs,reader_sha256=sha(Path(__file__))))
    save(C/'final_complete.json',dict(passed=True,scientific_advancement=False,protocol_sha256=sha(C/'protocol.json'),manifest_sha256=sha(C/'manifest_identity.json'),source_checkpoint=binding['source_checkpoint'],audit_clean_pairs=168,timing_valid=168,fairness_sha256=sha(C/'fairness001/summary.json'),paired_readout_sha256=sha(C/'paired_readout001/summary.json'),verification_sha256=sha(R/'verification002/summary.json')))
    print('Complete bounded handoff generated; four policy advancement decisions are no-go.')
if __name__=='__main__':main()
