"""Verified combined handoff; no generation, selector changes or GPU work."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

STUDY=Path(__file__).resolve().parents[1];REPO=STUDY.parents[1]
R=STUDY/'track_r/attempt001';H=STUDY/'track_h/attempt001'
START='3a1e6e9c903b3152012e84ac105854fbb09451ee'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False)


def prose(text):
    # Formatting only: scientific identifiers, values and JSON stay untouched.
    pairs={
        '70-question,two-seed,five-arm':'70 questions, two seeds, five arms',
        'model16file/runtime7file':'16 model and 7 runtime file',
        'all700':'all 700','All700':'All 700','700audit/clean':'700 audit/clean',
        '700complete':'700 complete','30AIME':'30 AIME','40HE':'40 HE','20AIME':'20 AIME','30HE':'30 HE','AIME18':'AIME 18',
        '50independent':'50 independent','50000question':'50000 question','50000 question-cluster':'50000 question-cluster',
        '24-cell':'24-cell','200-cell':'200-cell','31focused':'31 focused','31focusedCPUtests':'31 focused CPU tests',
        '24frozen':'24 frozen','200supplemental':'200 supplemental','8fresh':'8 fresh','8control':'8 control',
        '8repeated':'8 repeated','8controlcells':'8 control cells','3development':'3 development','3/8banks':'3/8 banks',
        '10deleting':'10 deleting','40GQA':'40 GQA','3-call':'3-call','40GQAchecks':'40 GQA checks',
        'R5ebd':'R 5ebd','H2480':'H 2480','seeds42/43':'seeds 42/43','fivearms':'five arms',
        'versusFULL':'versus FULL','versus21':'versus 21','versus15':'versus 15','current-error22':'current-error 22',
        'current-error59':'current-error 59','sham21':'sham 21','sham58':'sham 58','FULL21':'FULL 21','FULL58':'FULL 58',
        'HEResponse':'HE Response','AIME21':'AIME 21','HE56':'HE 56','AIME/40HE':'AIME / 40 HE',
        'R:1464complete':'R: 1464 complete','H:56requests':'H: 56 requests','240741model':'240741 model','85312forwards':'85312 forwards',
        '13560.813allocated':'13560.813 allocated','5460.291seconds':'5460.291 seconds','19serial':'19 serial','19warmups':'19 warmups',
        'noStageB/C':'no Stage B/C','NoStageB/C':'No Stage B/C','StageB/C':'Stage B/C','normValidityReuse':'norm-only ValidityReuse',
        'local15':'local 15','Local15':'Local 15','local15':'local 15','→12/13calls':'→12/13 calls',
        '→2963/2945total':'→2963/2945 total','→2963/2945totalcalls':'→2963/2945 total calls',
        'Noquality':'No quality','orclean':'or clean','Noquality-preserving':'No quality-preserving',
        'isdead':'is dead','isestablished':'is established','preservenegatives':'preserve negatives',
        'stayparked':'stay parked','noaccuracy':'no accuracy','no confirmationpulse':'no confirmation pulse',
        'confirmationpulse':'confirmation pulse','confirmation banks':'confirmation banks',
        'NoStage':'No Stage','noconfirmation':'no confirmation','deletingrequests':'deleting requests',
        'FourLongBench':'Four LongBench','4LongBench':'4 LongBench','2seeds':'2 seeds',
        'A1reporting':'A1 reporting','Noclearquality/workgain':'No clear quality/work gain',
        'No primary quality/work\ngate':'No primary quality/work\ngate','generation5ebd':'generation 5ebd','generation2480':'generation 2480',
        'bothtracks':'both tracks','completeaudit/clean':'complete audit/clean','workerhours':'worker hours',
        'notabenchmarknoninferiorityclaim':'not a benchmark noninferiority claim','andusefulglobalgain':'and useful global gain',
        'deletingreference':'deleting reference','werewrong':'were wrong','notlaunched':'not launched',
        'Source5ebd':'Source 5ebd','source5ebd':'source 5ebd','source2480':'source 2480',
    }
    for old,new in pairs.items():text=text.replace(old,new)
    return text


def main():
    started=time.time();out=STUDY/'closeout001';out.mkdir(exist_ok=False)
    assert read(R/'final_complete.json')['passed'] and read(H/'final_complete.json')['passed']
    r=read(R/'analysis001/summary.json');h=read(H/'analysis001/summary.json');effects=read(H/'closeout001/native_pulse_effects.json')
    rr=read(R/'closeout002/resources.json');hr=read(H/'closeout001/resources.json')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip();assert branch=='denoising_trajectory_research'
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    pins={};sources={}
    for name,root in [('R',R),('H',H)]:
        p=read(root/'protocol.json');assert all(sha(REPO/f)==v for f,v in p['sources'].items())
        assert read(root/'final_complete.json')['protocol_sha256']==sha(root/'protocol.json')
        sources[name]=dict(protocol_sha256=sha(root/'protocol.json'),source_checkpoint=read(root/'binding_receipt.json')['source_checkpoint'],source_pins=len(p['sources']),manifest_sha256=sha(root/'manifest_identity.json'))
        pins[name]=True
    runtime=[]
    for filename in ['external_sources.json','scorer_sources.json']:
        values=read(H/'private'/filename)
        for path,expected in values.items():
            assert sha(Path(path))==expected
            runtime.append(dict(logical_file=Path(path).name,sha256=expected))
    # Seal proves H scientific choices preceded inspection of new R outcomes.
    prep=STUDY/'track_h/preparation001';seal=read(prep/'source_seal001.json');seals=[]
    for relative,expected in seal['sources'].items():
        original=STUDY/relative;assert sha(original)==expected
        destination=H/original.relative_to(prep)
        if destination.name=='README.md':
            text=original.read_text().replace('# Prepared Track H contract — not launched or frozen for execution yet','# Track H: later-canvas signed-bias stopping-headroom diagnostic')
            text=text.replace('This source is preparation only. A separate execution attempt/source/protocol\ncheckpoint, tests, binding and R-readout hash must precede H GPU generation.',
                'This execution source requires its own committed source/protocol checkpoint,\nCPU tests, binding and R-readout hash before H GPU generation.')
            assert destination.read_text()==text
        else:assert destination.read_bytes()==original.read_bytes()
        seals.append(str(destination.relative_to(STUDY)))
    tests=[]
    for path in [R/'tests/prelaunch001.xml',R/'tests/inherited_selector001.xml',H/'tests/prelaunch002.xml']:
        suite=ET.parse(path).getroot().find('testsuite');v={k:int(suite.attrib[k]) for k in ['tests','errors','failures','skipped']}
        assert v['errors']==v['failures']==v['skipped']==0
        tests.append(dict(file=str(path.relative_to(STUDY)),sha256=sha(path),**v))
    assert sum(x['tests'] for x in tests)==31
    assert len(r['rows'])==700 and len({(x['arm'],x['id'],x['seed']) for x in r['rows']})==700
    assert all(x['timing_valid'] for x in r['rows'])
    for x in r['rows']:
        clean=read(R/x['clean_path']);assert clean['cuda_capture_events']==clean['request_jit_events']==0
        assert clean['protocol_sha256']==sources['R']['protocol_sha256']
    assert read(R/'legacy_reproduction_receipt.json')['passed']
    assert read(STUDY/'behavior_analysis001/summary.json')['legacy_checks_passed']==200
    qualified=list((H/'executions/references').glob('cell???_qualification.json'));assert len(qualified)==8 and all(read(x)['passed'] for x in qualified)
    deleting_checks=[x for x in h['GQA_checks'] if x['deleting']];assert len(deleting_checks)==40 and max(x['error'] for x in h['GQA_checks'])<=.03
    physical_pulses=[];after_controls=0
    for path in (H/'executions/development').glob('*/receipt.json'):
        v=read(path)
        if v['label']=='warm':continue
        i=int(v['label'][4:7]);control=read(H/f'executions/references/cell{i:03d}_A_observed/receipt.json')
        assert v['pre_intervention_native_state']==control['pre_intervention_native_state']
        if v['runtime']['pulse_calls']:
            assert v['runtime']['pulse_calls']==1
            work=v['runtime']['physical_calls'];assert all(x['LOCAL_skipped']==x['protected_skipped']==0 and x['current_QKV'] for x in work)
            physical_pulses.append(dict(id=v['id'],seed=v['seed'],label=v['label'],deleted=sum(x['skipped'] for x in work),correct=v['correct'],T=v['T'],answer_parsed=v['answer_parsed'],no_final_response=v['no_final_response']))
        if v['label'].endswith('_A_after'):
            assert all(v[k]==control[k] for k in ['output_sha256','N','C','T','final_cpu_rng','final_cuda_rng','native_window_fingerprints'])
            after_controls+=1
    assert len(physical_pulses)==10 and after_controls==3
    assert not (H/'executions/confirmation').exists() and not h['development_decision']['confirmation_authorized']
    assert sha(REPO/'AGENTS.md')==sha(STUDY/'private/AGENTS.md'),'preserve initial AGENTS edit'
    own_prefix=(STUDY/'private/running_handoff_prefix.txt').read_text();handoff=(REPO/'HANDOFF.md').read_text();assert handoff.startswith(own_prefix)
    assert handoff[len(own_prefix):]==(STUDY/'private/HANDOFF.md').read_text(),'preserve initial handoff body'
    assert (REPO/'docs/DECISIONS.md').read_bytes()==(STUDY/'private/DECISIONS.md').read_bytes(),'preserve initial decisions before own append'
    resources=dict(complete_native_requests=rr['complete_native_requests']+hr['complete_native_requests'],
        explicit_model_forwards=rr['explicit_total_model_forwards']+hr['explicit_total_model_forwards'],warm_requests=rr['warm_requests']+hr['warm_requests'],
        GPU_workers=rr['GPU_workers']+hr['GPU_jobs'],allocated_GPU_worker_seconds=rr['allocated_gpu_worker_seconds']+hr['allocated_GPU_worker_seconds'],
        GPU_failures=0,resource_ceilings=None,CPU_reporting_failures=1,relay_preflight_failures=1)
    resources['allocated_GPU_worker_hours']=resources['allocated_GPU_worker_seconds']/3600
    save(out/'resources.json',dict(**resources,Track_R=rr,Track_H=hr,no_nested_timing_addition=True))
    verification=dict(passed=True,branch=branch,starting_HEAD=START,HEAD_before_publication=head,sources=sources,all_source_pins_pass=pins,
        external_runtime_and_scorer_rechecks=runtime,H_pre_R_outcome_scientific_seal_files=seals,focused_CPU_tests=tests,
        R_cells=700,R_clean_capture_JIT_qualified=700,legacy_frozen_gate_cells=24,legacy_supplemental_cells=200,
        H_control_cells_qualified=8,H_identifying_bank_cells=3,H_native_anchor_cells=5,H_actual_deleting_requests=len(physical_pulses),
        H_deleting_GQA_checks=len(deleting_checks),H_max_GQA_relative_error=max(x['error'] for x in h['GQA_checks']),H_post_pulse_controls_exact=after_controls,
        H_actual_pulse_physical_receipts=physical_pulses,H_total_deleted_rectangles=sum(x['deleted'] for x in physical_pulses),
        H_correct_reference_deleting_cells=0,H_confirmatory_deleting_cells=0,preexisting_AGENTS_unchanged=True,preexisting_HANDOFF_body_preserved=True,preexisting_DECISIONS_preserved=True,
        restored_branches_used=False,exact_acceptance_count_claim=False,Stage_BC_launched=False,production_kernel_changes=False,pushed=False,
        no_GPU_model_reruns_for_reporting_repair=True)
    save(out/'verification.json',verification)
    decisions=dict(A1_qonly_QKV_forecasting='no_go; reporting corrections preserve the negative result and missing component-matched controls',
        unchanged_Response=dict(legacy_reproduction='pass24frozen and200supplemental cells',AIME_confirmation='earlier quality/work advantage not reproduced',HumanEval_confirmation='quality regression and uncertain work reduction',clean_latency='slower than dense',promotion='no_go'),
        signed_bias_pulse=dict(engineering='pass',identification='3development cells;0confirmation banks',local_native_headroom=True,
            score_advantage_over_current_error=False,high_rule='no_go',low_rule='no_go',development_gate=False,confirmation='not launched',
            quality_preservation='unqualified; all deleting requests start from wrong development references and score wrong',
            end_to_end_gain='not established; local improvements can increase total work',promotion='no_go for this tested scope'),
        MarginGuard_exact_oracle='parked; no qualified predictive decision lead',ValidityReuse_norm_only='parked',
        broader_screen='No Stage B/C or production/kernel promotion',
        next_independent_review='Audit saved native pulse/commit/RNG fingerprints to explain local canvas acceleration versus longer full requests before proposing another selector. Do not rerun or redesign automatically.')
    save(out/'method_decisions.json',decisions)
    failed=read(STUDY/'reporting001/failure.json');save(R/'closeout001/failure.json',dict(**failed,source_sha256=sha(STUDY/'reporting001/close_track_r.py')))
    save(out/'failures.json',dict(GPU_failures=[],CPU_reporting=[failed],relay_preflight=[dict(error_class='FileNotFoundError',reason='missing private log parent before relay was created',model_forwards=0)],
        no_failed_or_partial_native_shards=True,no_false_completion_of_confirmation=True))
    lines=['# Response behavior and native stopping headroom — completed gated studies','',
        'The original Response behavior reproduces exactly on legacy questions, but its quality/work advantage does not replicate on new-to-Response questions. The native single-pulse study finds local stopping headroom, without a useful directional-score or end-to-end gain. No accuracy-preservation, speedup or validated past-motion mechanism is claimed.','',
        '## Identity and execution','',f'Branch `{branch}`; starting HEAD `{START}`. R generation source `{sources["R"]["source_checkpoint"]}`, readout commit `e2f5d47d`; H generation source `{sources["H"]["source_checkpoint"]}`. Final publication commit is the commit containing this handoff; generation source checkpoints stay distinct.',
        '',f'R protocol SHA256 `{sources["R"]["protocol_sha256"]}`; H protocol SHA256 `{sources["H"]["protocol_sha256"]}`. Exact manifest/exposure identities, model16file/runtime7file verification and scored-input hashes are public. No paused campaign resumed, GPU preempted, resource ceiling imposed or push performed. Native output caps8192(short)/16384(long),48-call canvas cap and sampler/RNG order remain fixed.',
        '', '## CPU reporting correction','',
        'The new addendum preserves A1’s negative Q-only/QKV forecast result against persistence, both frozen aggregation definitions and five individual10% crossings. Missing Q-only component-matched controls remain explicit. Selected error/mass-matched contrasts are19/185 and18/179. A1 deleted no native model support and its common first canvases had zero confidence-only delays. The299 prior receipt corrections do not change aggregate conclusions. FULL completion is recorded; the original parent wait status remains unavailable.',
        '', '## Track R: complete70-question,two-seed,five-arm panel','',
        'All700 audit/clean pairs match output,N,C,T and final CPU/CUDA RNG; all700 clean timings have zero new CUDA captures/JIT events. The24-cell source-matched legacy gate passes. A supplemental200-cell original-stratum check matches every output/count/support/lifecycle record. Question clusters are30AIME/40HumanEval, with two seeds each—not140 independent questions. The primary confirmation strata have20AIME/30HE questions; strict-no-A1 AIME18 is separate. All70 have some known historical exposure; new-to-Response is not universal benchmark non-exposure.','',
        '|Task / stratum|Arm|Official correct / seed-cells|Total N|Total C|Total T|Mean clean W(s)|',
        '|---|---|---:|---:|---:|---:|---:|']
    for a in r['aggregates']:
        if a['stratum'] not in ['all','new_to_response']:continue
        lines.append(f"|{a['benchmark']} / {a['stratum']}|{a['arm']}|{a['correct']}/{a['cells']}|{a['totals']['N']}|{a['totals']['C']}|{a['totals']['T']}|{a['means']['W']:.3f}|")
    lines+=['','Primary comparisons use50000 question-cluster bootstraps retaining both seeds; eight comparisons use simultaneous99.375% intervals. No noninferiority margin was declared.','',
        '|New-to-Response task|Comparator|Paired N ratio|Primary simultaneous N interval when applicable|Paired N/C|Paired T|Paired clean W|Correct→wrong / wrong→correct|',
        '|---|---|---:|---|---:|---:|---:|---|']
    for c in r['comparisons']:
        if c['stratum']!='new_to_response':continue
        v=c['ratios'];ci=v['N']['simultaneous_ci99375']
        lines.append(f"|{c['benchmark']}|{c['reference']}|{v['N']['estimate']:.4f}|{ci}|{v['N_per_C']['estimate']:.4f}|{v['T']['estimate']:.4f}|{v['W']['estimate']:.4f}|{c['correct_to_wrong']} / {c['wrong_to_correct']}|")
    lines+=['', 'Geometric question-weighted ratios and arithmetic totals differ. New AIME has more total calls/tokens under Response; new HumanEval’s small paired call reduction is uncertain and accompanies quality regressions. The legacy AIME15/20 versus FULL12/20 remains, but confirmation is21/40 versus21/40. HumanEval confirmation is56/60 versusFULL58/60,current-error59/60,sham58/60. No primary promotion gate passes.',
        '', 'R physical deletion is1.579%GLOBAL/0.619%overall on AIME and1.074%GLOBAL/0.201%overall onHE; LOCAL stays dense. Discovery QK/PV,counter refresh/hold work and nested timers are separately reported in track_r/attempt001/closeout002/. Native prefill/commit remain dense and outside the denoising rectangle denominator. No speed is inferred from these counts.',
        '', 'The frozen text rubric and sealed descriptive prefix addendum do not establish reduced distraction/redundancy. New AIME Response outputs are longer; repeated8-grams decrease slightly while duplicate lines and revision terms increase. Retrospective extractable candidates can be wrong and are not native stopping times. Every correctness transition is programmatically inspected and input-hash checked; HumanEval losses versusFULL include94/42(entry point),71/43(syntax),99/43(syntax),156/43(test),133/43(syntax), with69 improving on both seeds. Correct-in-both descriptions remain post-treatment subsets.',
        '', '## Track H: fresh later-canvas single-pulse diagnostic','',
        'All8 repeated-control cells qualify full output,state,RNG,counters and active operands. Five native anchors exist; only3development cells have qualifying directional banks. Confirmation has three missing anchors and one failed directional contrast, so0/4confirmatory banks. No identity replacement, truncation, later-canvas substitution or forced continuation. Each pulse deletes exactly5%floor prefix blocks in four units atlayer29(heads0/8,qblocks0/1,Q128×KV64); protected/current-canvas,LOCAL,prefill,commit stay dense.',
        '', 'Ten distinct deleting requests pass40GQA-route tests; maximum relative error versus explicit current-QKV masked attention is0.004264(bound0.03). Three post-pulse all-kept controls exactly reproduce. No restoration is used; old restored output assembly remains unqualified. High/low pairs match each other in current error/mass; their current-error comparisons do not. Sham aliases and unit-versus-whole bias angles are explicit.',
        '', '|Development question / seed|A remaining|Current-error|Sham|Direction-high|Direction-low|',
        '|---|---:|---:|---:|---:|---:|']
    for x in effects['coverage']:
        if not x['bank_qualified']:continue
        vals={y['arm']:y['remaining_calls'] for y in effects['effects'] if (y['id'],y['seed'])==(x['id'],x['seed'])}
        lines.append(f"|{x['id']} / {x['seed']}|{x['remaining_calls']}|{vals['current_error']}|{vals['support_sham']}|{vals['direction_high']}|{vals['direction_low']}|")
    lines+=['', 'The matched directional pair produces a3-call difference in one development state, but its better member ties current-error. Neither current-only rule beats both controls across both seeds of a development question. Development fails, so no confirmation pulse is launched. No AA/SA/AS/SS factorial, hold or production policy was run.',
        '', 'Local stopping gain does not ensure lower whole-request work:66ec…seed42 moves from15remaining calls to12(current-error) or13(direction), but totalN rises767→2963/2945 andT4546→16384. On672464…seed43, high reduces20→17remaining whileN rises2905→2993;low retains20 and2905. Every deleting request scores wrong and reaches the output cap. All deleting reference states were already wrong; therefore zero correct-reference deleting cases can support accuracy preservation. Native confidence/reference agreement is not correctness. Different stopping counts can legitimately change later native RNG offsets; no RNG compensation is introduced.',
        '', 'All H timing is diagnostic instrumentation and unqualified for clean policy latency. FULL and PIECEWISE references remain distinct from the common A trajectory; consumer substitutions are systematic treatments. See H summary for every reference/method N,C,T and quality, missing cells,physical work and signed native pulse traces.',
        '', '## Resource use, limitations and decision','',
        f"R:1464complete requests,240741model forwards,13560.813allocated GPU-worker seconds. H:56requests,85312forwards,5460.291seconds. Combined:{resources['complete_native_requests']}requests,{resources['explicit_model_forwards']}forwards,{resources['allocated_GPU_worker_hours']:.4f}GPU-worker hours across19serial workers, including19warmups,startup,qualification and cleanup. There are zero GPU-process failures. One relay-log preflight error and one derived CPU-reporting error were preserved and corrected with no GPU rerun. Parent allocation has15s polling resolution. S/N is amortized; nested timers are not added twice.",
        '', 'Decisions: no promotion of unchanged Response,Q-only/QKV past-motion forecasts,or eitherHdirection rule. Exact MarginGuard and norm-onlyValidityReuse remain parked. Local headroom and different future effects are demonstrated in the tested development states, but no useful score advantage,accuracy-preserving total-work reduction,confirmed held-out headroom or clean speedup is established. Results do not reject every layer,strength or sparse-attention mechanism.',
        '', 'Recommended next action: independent review of the saved native pulse/commit/RNG evidence explaining local acceleration versus longer complete requests, before designing another selector. No automatic Stage B/C,production selector,kernel optimization,or additional GPU experiment is authorized by this handoff.',
        '', '## Inspect first','',
        '- cpu_reporting_addendum001/summary.md and behavior_analysis001/summary.md',
        '- track_r/attempt001/summary.md,analysis001/summary.json,closeout002/resources.json',
        '- track_h/attempt001/summary.md,analysis001/summary.json,closeout001/native_pulse_effects.json',
        '- closeout001/verification.json,method_decisions.json,resources.json,publication_files.json',
        '- COMMANDS.md and every launch/exit receipt for exact reproduction arguments.',
        '', 'All new study files are task-owned. Existing AGENTS edits and pre-existing HANDOFF/DECISIONS bodies are preserved. Prompts,gold,token IDs,raw text,tensors,private paths and logs remain ignored. No push.']
    (STUDY/'summary.md').write_text(prose('\n'.join(lines)+'\n'))
    # Shared files receive only our prefix/append; index staging is a separate reviewed step.
    date=datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d')
    prefix=f'''## {date} HKT — Response behavior and stopping-headroom closeout

Completed study `results/response_behavior_and_stopping_headroom_20261025/`.
Verified generation commits:R5ebd3e66,H2480aa68;R readout commit e2f5d47d.
Branch denoising_trajectory_research. Starting HEAD3a1e6e9c.

R: all700audit/clean cells,30AIME/40HE,seeds42/43,five arms;24frozen and200
supplemental legacy checks pass. New-to-Response AIME21/40 versusFULL21/40,
current-error22/40,sham21/40;HE56/60 versusFULL58/60. No primary quality/work
gate passes. Clean latency versusFULL:4.059xAIME,2.590xHE. Original legacy AIME
lead reproduces exactly; it does not carry into confirmation.

H:8/8fresh controls qualify,3/8banks identify,10deleting requests and40GQA
checks pass. Native remaining calls can decrease, but neither directional rule
beats current-error/sham in replicated development. No confirmation pulse.
Local15→12/13calls becomes767→2963/2945total calls; all deleting requests
are wrong/output-capped. No quality preservation,global acceleration orspeedup.

31focused CPU tests pass;{resources['complete_native_requests']}complete requests,
{resources['explicit_model_forwards']}explicit model forwards,{resources['allocated_GPU_worker_hours']:.4f}allocated GPU-worker hours.
No GPU failure,paused-campaign resume,StageB/C,kernel campaign orpush.
Review `summary.md`, both track summaries and `closeout001/verification.json`.
Next: independently inspect native pulse/commit/RNG traces before any new selector.

The following pre-existing handoff content and edits remain preserved.

'''
    prefix=prose(prefix)
    (REPO/'HANDOFF.md').write_text(prefix+handoff[len(own_prefix):]);(STUDY/'private/running_handoff_prefix.txt').write_text(prefix)
    decision=f'''\n\n## {date} — Unchanged Response confirmation and signed-bias headroom (verified)\n\nStudy: `results/response_behavior_and_stopping_headroom_20261025/summary.md`.\nR protocol `{sources['R']['protocol_sha256']}`, source5ebd3e66:700complete\naudit/clean cells,all30AIME/40HE,two seeds. Legacy24/24and200/200checks pass.\nNew-to-Response AIME21/40 equalsFULL/sham andtrailscurrent-error22/40;\nHE56/60 trailsFULL/sham58/60andcurrent-error59/60. No primary promotion gate.\nPaired totalN versusFULL:1.1045AIME,0.9766HE;simultaneous intervals cross1.\nCleanW ratios4.0587and2.5901;no speedup ornoninferiority.\n\nH protocol `{sources['H']['protocol_sha256']}`, source2480aa68:\n8fresh-control cells qualify,3development banks,10deleting requests,40GQAchecks\n(maxerror0.004264). Direction-high/low fail development;confirmation notlaunched\nandhas0/4identifying banks. Local native headroom exists,butnotbeyondcurrent-error;\nlocal15→12/13calls accompanies767→2963/2945totalcalls andwrong/capped outputs.\nAll deleting reference cases werewrong;noaccuracy-preservation evidence.\nA1reporting corrections preservenegatives;MarginGuard/normValidityReuse stayparked.\nNoStageB/C,production optimization,resumed pausedcampaign orpush. Seecomplete\nreceipts,failures,question-cluster intervals andwork accounting;do notpromote.\n'''
    ledger=f'''\n\n## {date} — Completed Response behavior replication and gated native headroom leaf\n\n- Source: `results/response_behavior_and_stopping_headroom_20261025/summary.md`.\n- R:30AIME/40HEquestions,seeds42/43,fivearms,700completeaudit/clean cells;\n  protocol `{sources['R']['protocol_sha256']}`, generation5ebd3e66.\n  Newstrata:AIMEResponse21/40,FULL21/40,CE22/40,sham21/40;\n  HEResponse56/60,FULL58/60,CE59/60,sham58/60. Noclearquality/workgain;slowercleanW.\n- H:4LongBench questions,8controlcells,gated3development interventioncells;\n  protocol `{sources['H']['protocol_sha256']}`, generation2480aa68.\n  Nativecurrent-QKVdeleting paths qualify. Localcalls canfall,butscoreadvantage\n  andusefulglobalgain do notconfirm;no confirmationpulse,0/4confirmation banks.\n- Completion:bothtracks `final_complete.json`;combinedverificationin`closeout001/`.\n  31focusedCPUtests;1520fullrequests,326053model forwards;5.2836allocatedGPU-workerhours.\n  Exploratory/targetednegativeandmixedresults,notabenchmarknoninferiorityclaim.\n'''
    decision=prose(decision);ledger=prose(ledger)
    for path,text in [('docs/DECISIONS.md',decision),('docs/RESULTS_LEDGER.md',ledger)]:
        with (REPO/path).open('a') as f:f.write(text)
    save(STUDY/'private/shared_own_changes.json',{'HANDOFF.md':dict(mode='prepend',text=prefix),'docs/DECISIONS.md':dict(mode='append',text=decision),'docs/RESULTS_LEDGER.md':dict(mode='append',text=ledger)})
    (STUDY/'README.md').write_text('# Response behavior replication and stopping headroom\n\nCompleted protocol-identified studies; read [summary.md](summary.md) first.\nTrack R completes700audit/clean cells. Track H completes the gated native\nsingle-pulse leaf;both current rules fail,no confirmation pulse.\nA1reporting corrections preserve previous negatives. Noquality-preserving\nefficiency orclean speedup isestablished. NoStageB/C orproduction optimization.\n')
    save(out/'completion_metadata.json',dict(completed_UTC=datetime.now(timezone.utc).isoformat(),completed_HKT=datetime.now(timezone(timedelta(hours=8))).isoformat(),CPU_closeout_seconds=time.time()-started,new_model_forwards=0))
    print(json.dumps(dict(passed=True,resources=resources,source_pins=sources,focused_tests=31,post_pulse_controls=after_controls)))


if __name__=='__main__':main()
