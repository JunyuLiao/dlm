"""Finalize only the bounded investigation; independently retain parent pause."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from scripts.v31_value_pause_analysis import read, lines, sha, require, CURRENT
from scripts.v31_vllm_paired_bench import request_seed
from scripts import v31_official as official


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def run(root):
    require(not (root/'final_complete.json').exists(),'preserve completion')
    pause=read(CURRENT/'PAUSED_USER_REQUEST.json')
    require(pause['status']=='paused_user_request' and pause['completed_engine_jobs']==17,'parent pause changed')
    require(not (CURRENT/'final_complete.json').exists(),'parent must remain incomplete')
    require(len(list((CURRENT/'runs').glob('**/attempt_complete.json')))==17,'parent queue advanced')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    require(not gpu,'GPU must be idle after bounded work')
    tests=ET.parse(root/'investigation_tests.xml').getroot()
    suites=list(tests.findall('testsuite'))
    require(sum(int(s.attrib['tests']) for s in suites)==4,'four analysis/dispatch tests required')
    require(not any(int(s.attrib.get(k,0)) for s in suites for k in ('failures','errors','skipped')),'tests failed/skipped')
    profile=read(root/'cost_profile/attempt003/summary.json')
    require(sum(r['graph_parity']['units'] for r in profile['results'])==64,'graph parity coverage')
    require(not any(v for r in profile['results'] for k,v in r['graph_parity'].items() if k!='units'),'graph parity failure')
    integrity=read(root/'input_integrity.json');require(integrity['status']=='passed','input integrity')
    cross=root/'native_cross/attempt001';trace=root/'native_trace/attempt001'
    final,parser=official.load_final_response();pool=CURRENT/'private/pools/longbench_v2'
    pools=official.Pools(pool,fields=('thinking',));gold=read(pool/'longbench_v2_0shot_think_gold.json')
    trial_count=0;measured=0;audited=0;binding={}
    jobs=[]
    for rel,digest in read(cross/'final_complete.json')['engines'].items():
        marker=cross/rel;require(sha(marker)==digest,'cross completion changed')
        jobs.append((cross,marker,4,read(marker.parent/'scored.json')))
    jobs.append((trace,trace/'final_complete.json',4,read(root/'trace_analysis/attempt001/summary.json')['trials']))
    require(len(jobs)==5,'bounded engine count')
    for study,marker,warm,scores in jobs:
        proof=read(marker);require(proof['status']=='complete' and not proof['errors'],'engine incomplete')
        require(proof['protocol_sha256']==sha(study/'protocol.json'),'engine protocol changed')
        for rel,digest in proof['files_sha256'].items():require(sha(marker.parent/rel)==digest,'engine artifact changed')
        source=sha(study/'receipts/source_fingerprints.json')
        rr=lines(marker.parent/'records.jsonl');pp=lines(marker.parent/'private/generation.private.jsonl')
        require(len(rr)==len(pp)==warm+len(scores),'native coverage')
        for i,(r,t) in enumerate(zip(rr,pp)):
            require(r['frozen_source_identity_sha256']==source,'source identity')
            require(r['rng_seed']==request_seed(31,dict(dataset=r['dataset'],index=r['index'],seed=r['panel_seed']),r['repeat']),'RNG seed')
            for field in official.COMPARE_FIELDS:require(r[field]==t[field],'public/private binding')
            meta,_,verified=official.bind(t,pools);require(verified,'official binding')
            score=int(official.lb_extract_answer(final(t['completion'],bool(meta['thinking'])))==gold[t['id']])
            if i>=warm:
                require(score==scores[i-warm]['score'],'independent diagnostic rescore')
                measured+=1
            require(r['sampler_calls']==r['denoise_forwards']+r['canvases'],'sampler count')
            require(r['canvases']==math.ceil(r['output_tokens']/256),'canvas count')
            require(r['cuda_graph_captures']==0 and r['cuda_matmul_allow_tf32'] is False,'capture/precision')
            require(r['local_kv_budget'] is None,'LOCAL budget')
            a=(r.get('receipts') or {}).get('adapter') or {}
            if r['value_audit']:
                audited+=1
                require(a['native_local_skipped_tiles']==0,'LOCAL work')
                require(a['global_calls']==5*r['denoise_forwards'],'GLOBAL coverage')
                require(a['native_local_calls_audited']==25*r['denoise_forwards'],'LOCAL coverage')
                require(not a.get('order_errors') and not a.get('value_nonfinite_logits_calls') and not a.get('value_nonfinite_attention_calls'),'nonfinite/order')
                if a.get('value_refinement'):
                    require(a['value_refinement_layers']==[29] and a['value_refinement_prefix_budget']==128,'scope/budget')
                    require(a['value_refinement_work']['budget_errors']==0,'fixed budget')
            trial_count+=1
        binding[str(marker.relative_to(root))]=sha(marker)
    require(trial_count==120 and measured==100 and audited==64,'diagnostic trial accounting')
    cross_s=read(cross/'final_complete.json')['gpu_reservation_s'];trace_s=read(trace/'receipt.json')['gpu_reservation_s']
    conservative=read(trace/'protocol.json')['prior_diagnostic_gpu_s']+trace_s
    require(conservative<3600,'diagnostic reservation exceeded target')
    write(root/'gpu_accounting.json',dict(profile_success_measured_s=profile['gpu_reservation_s'],cross_s=cross_s,trace_s=trace_s,
        successful_measured_s=profile['gpu_reservation_s']+cross_s+trace_s,conservative_total_s=conservative,
        failed_profile_upper_bounds_s=[120,90],profile_and_launch_allowance_s=220,
        measured_requests=100,explicit_warmups=20,implicit_warmups=5,
        note='Reservation includes native startup/warmup. Conservative prior allowance includes failed profiler attempts and process overhead. Excludes the already-running full503 worker allowed to finish during pause finalization; that belongs to its original panel.'))
    write(root/'receipt.json',dict(status='passed',utc=datetime.now(timezone.utc).isoformat(),GPU_compute_processes=[],
        full503_status='paused_user_request',parent_engine_jobs=17,parent_complete_marker_exists=False,
        parent_protocol_sha256=sha(CURRENT/'protocol.json'),parent_pause_sha256=sha(CURRENT/'PAUSED_USER_REQUEST.json'),
        independently_scored_diagnostic_measured_requests=measured,validated_raw_records=trial_count,audit_records=audited,
        CPU_tests=4,saved_state_graph_parity_units=64,engine_bindings=binding,official_parser=parser,
        verifier_sha256=sha(__file__),verified_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    important=['PLAN.md','TRACE_GATE_ADDENDUM.md','summary.md','CPU_FINDINGS.md','CROSSED_FINDINGS.md','TRACE_FINDINGS.md',
        'input_integrity.json','reference_source_comparison.json','extra_work_accounting.json','cohort_metadata.json','investigation_tests.xml',
        'cpu_analysis/attempt001/summary.json','cpu_analysis/attempt001/config.json','cpu_analysis/attempt001/final_complete.json',
        'cost_analysis/attempt001/summary.json','cost_analysis/attempt001/final_complete.json',
        'cost_profile/attempt003/summary.json','cost_profile/attempt003/final_complete.json',
        'cross_analysis/attempt001/summary.json','cross_analysis/attempt001/final_complete.json',
        'trace_analysis/attempt001/summary.json','trace_analysis/attempt001/final_complete.json','gpu_accounting.json','receipt.json']
    write(root/'final_complete.json',dict(status='complete_bounded_investigation',parent_full503='paused_incomplete',
        decision='no_go_current_methods_full503',files_sha256={rel:sha(root/rel) for rel in important}))
    print(dict(status='complete_bounded_investigation',GPU='idle',full503='paused',conservative_GPU_minutes=conservative/60),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
