"""Independent post-run integrity, official-score recheck, and cost decomposition.

Reads immutable generation/report artifacts; creates a separately source-bound
analysis attempt. Private prompts, answers and completions never enter output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
from scripts import v31_v2_local_threshold_sweep as base
from scripts import v31_official as official
from scripts.v31_vllm_paired_bench import request_seed

LABELS=['control_before','control_after','full_swap8','residual_direction_guard',
        'dense_full_fix51994','dense_piecewise','allkept_fa4']


def check(condition,message):
    if not condition:raise RuntimeError(message)


def verify(study,output):
    check(not output.exists(),'preserve the existing verification attempt')
    protocol=base.read(study/'protocol.json');done=base.read(study/'final_complete.json')
    check(done['protocol_sha256']==base.sha(study/'protocol.json'),'final protocol binding')
    for rel,digest in done['reports'].items():
        check(base.sha(study/rel)==digest,'final report binding')
        if rel.endswith('final_complete.json'):
            for name,value in base.read(study/rel)['files_sha256'].items():
                check(base.sha((study/rel).parent/name)==value,'reported numeric artifact binding')
    base.verify(study,protocol)
    pool=study/'private/pools/longbench_v2'
    rowinfo=base.read(pool/'longbench_v2_0shot_think_rowinfo.json')
    meta=rowinfo['rows']
    gold=base.read(pool/'longbench_v2_0shot_think_gold.json')
    cells=base.read(study/'private/schedules/longbench_v2/cells.json')
    planned={c['index']:c for c in cells}
    final_response,parser_source=official.load_final_response()
    proof={};records={};scores={};bindings={};phase={}
    source_hash=base.sha(study/'receipts/source_fingerprints.json')
    for pass_name in ('clean','audit'):
        markers=sorted((study/'runs'/pass_name).glob('*/shard*/attempt*/attempt_complete.json'))
        check(len(markers)==64,'all64 engine shards per pass required')
        rr={label:[] for label in LABELS};judges={label:{} for label in LABELS}
        reservation=0.;checks=0;raw_trials=0;raw_captures=0;oracle=0
        for marker in markers:
            leaf=marker.parent;receipt=base.read(marker)
            check(receipt['status']=='complete' and not receipt['errors'],'successful structural receipt required')
            check(receipt['protocol_sha256']==base.sha(study/'protocol.json'),'engine protocol binding')
            for rel,digest in receipt['files_sha256'].items():check(base.sha(leaf/rel)==digest,'engine record hash')
            reservation+=receipt['gpu_reservation_s']
            raw=base.jsonl(leaf/'records.jsonl');raw_trials+=len(raw)
            raw_captures+=sum(r['cuda_graph_captures'] for r in raw)
            for folder in (leaf/'exports').iterdir():
                label=folder.name;check(label in rr,'unexpected arm')
                public=base.jsonl(folder/'records.jsonl')
                private=base.jsonl(folder/f'private/run_{label}.private.jsonl')
                check(len(public)==len(private),'public/private coverage')
                for r,t in zip(public,private):
                    key=base.key(r);i=r['index'];c=planned[i]
                    check(key not in judges[label],'duplicate scored question')
                    check(r['panel_seed']==42 and r['repeat']==0 and r['rng_seed']==request_seed(31,c,0),'native RNG binding')
                    check(r['frozen_source_identity_sha256']==source_hash,'generation source binding')
                    check(r['sampler_calls']==r['denoise_forwards']+r['canvases'],'native sampler count')
                    check(r['engine_steps']==r['prefill_steps']+r['sampler_calls'],'native engine count')
                    check(r['canvases']==math.ceil(r['output_tokens']/256),'native canvas count')
                    check(r['cuda_graph_captures']==0,'measured graph capture')
                    check(r['prompt_tokens']==meta[i]['prompt_token_count'] and r['budget']==16384,'input/output budget')
                    check(r['prompt_sha256']==meta[i]['prompt_sha256']==t['prompt_sha256'],'private prompt binding')
                    check(t['id']==meta[i]['id']==c['id'] and t['index']==i,'private identity')
                    check(t['rng_seed']==r['rng_seed'] and t['panel_seed']==42,'private RNG binding')
                    check(r['manifest_sha256']==t['manifest_sha256']==rowinfo['manifest_sha256'],'manifest binding')
                    # Recompute each official decision directly from the retained private completion.
                    predicted=official.lb_extract_answer(final_response(t['completion'],bool(meta[i]['thinking'])))
                    judges[label][key]=predicted==gold[t['id']]
                    adapter=(r.get('receipts') or {}).get('adapter') or {}
                    if pass_name=='clean':
                        check(not r['value_audit'] and r['value_clean_timing'],'clean instrumentation disabled')
                        check(r['timer_policy']=='request_and_prefill_boundaries','clean timer')
                        check(abs(r['wall_s']-r['prefill_s']-r['decode_s'])<2.1e-5,'clean time partition')
                    else:
                        check(r['value_audit'] and not r['value_clean_timing'],'separate audit')
                        check(adapter['native_local_skipped_tiles']==0,'LOCAL remains dense')
                        ge,gk=adapter['global_eligible_tiles'],adapter['global_kept_tiles']
                        le=adapter['native_local_eligible_rectangles']
                        check(0<=gk<=ge and adapter['overall_eligible_rectangles']==ge+le,'physical denominator')
                        check(adapter['overall_skipped_rectangles']==ge-gk,'physical skipped work')
                        check(adapter['global_calls']==5*r['denoise_forwards'],'GLOBAL layer coverage')
                        check(adapter['native_local_calls_audited']==25*r['denoise_forwards'],'LOCAL layer coverage')
                        work=adapter.get('value_refinement_work',{})
                        if label in ('full_swap8','residual_direction_guard'):
                            check(work['budget_errors']==0 and adapter['value_refinement_layers']==[29],'value scope/budget')
                        else:check('value_refinement' not in adapter,'unaltered reference')
                        if label in ('dense_full_fix51994','dense_piecewise','allkept_fa4'):check(gk==ge,'dense/all-kept work')
                    check(adapter.get('value_refinement_work',{}).get('native_same_state_oracle_checks',0)==0,'qualification oracle absent in full run')
                    rr[label].append(r);checks+=1
        expected={f"{c['dataset']}|{c['index']}|42|0" for c in cells}
        for label in LABELS:
            check(set(judges[label])==expected and len(rr[label])==503,'full503 coverage')
            score=base.read(study/f'scoring/{pass_name}/{label}/official.official.json')[label]
            check(score==judges[label],'independent official score mismatch')
        report=base.read(study/f'reports/{pass_name}/summary.json')['summaries']
        for label in LABELS+['current_v31_control']:
            rows=rr[label] if label in rr else rr['control_before']+rr['control_after']
            bits=list(judges[label].values()) if label in judges else list(judges['control_before'].values())+list(judges['control_after'].values())
            for name,field in [('mean_N','denoise_forwards'),('mean_C','canvases'),('mean_T','output_tokens')]:
                check(math.isclose(report[label][name],statistics.mean(r[field] for r in rows),abs_tol=1e-9),'aggregate '+name)
            check(math.isclose(report[label]['score_points'],100*statistics.mean(bits),abs_tol=1e-9),'aggregate official accuracy')
            if pass_name=='clean':
                for name,field in [('mean_S','decode_s'),('mean_W','wall_s')]:
                    check(math.isclose(report[label][name],statistics.mean(r[field] for r in rows),abs_tol=1e-9),'aggregate '+name)
        check(raw_captures==0,'warmup/measured raw capture evidence')
        records[pass_name]=rr;scores[pass_name]=judges
        proof[pass_name]=dict(engine_shards=len(markers),scored_trials=checks,raw_trials_including_explicit_warmups=raw_trials,
            all_rows_independently_rescored=True,measured_graph_captures=raw_captures,gpu_reservation_s=reservation)
        bindings[pass_name]={str(m.relative_to(study)):base.sha(m) for m in markers}
    clean=base.read(study/'reports/clean/summary.json')['summaries']
    decomposition={}
    for reference in ('current_v31_control','dense_full_fix51994'):
        for arm in ('full_swap8','residual_direction_guard'):
            a,b=clean[reference],clean[arm]
            ratio=b['mean_S']/a['mean_S'];calls=b['mean_N']/a['mean_N'];cost=b['S_per_N']/a['S_per_N']
            check(math.isclose(ratio,calls*cost,rel_tol=1e-12),'S=N times amortized cost identity')
            decomposition[arm+'_vs_'+reference]=dict(generation_time_ratio=ratio,calls_ratio=calls,
                amortized_cost_ratio=cost,generation_speedup=1/ratio,request_speedup=a['mean_W']/b['mean_W'],
                interpretation='algebraic decomposition, not causal attribution')
    qualification=sum(base.read(p)['gpu_reservation_s'] for p in (study/'qualification').glob('**/attempt_complete.json'))
    output.mkdir(parents=True)
    base.write(output/'config.json',dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        verifier_sha256=base.sha(__file__),protocol_sha256=base.sha(study/'protocol.json'),
        final_complete_sha256=base.sha(study/'final_complete.json'),completion_bindings=bindings),True)
    base.write(output/'summary.json',dict(status='passed',passes=proof,qualification_gpu_reservation_s=qualification,
        total_gpu_reservation_s=qualification+sum(v['gpu_reservation_s'] for v in proof.values()),
        decomposition=decomposition,official_parser_source=parser_source,
        reservation_interpretation='includes process/model startup and warmup; not clean request latency'),True)
    base.write(output/'final_complete.json',dict(status='complete',summary_sha256=base.sha(output/'summary.json'),
        config_sha256=base.sha(output/'config.json')),True)
    print(json.dumps({'status':'passed','verified_scored_trials':sum(x['scored_trials'] for x in proof.values()),'questions':503}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.study.resolve(),a.output.resolve())
