"""CPU-only draft contracts and committed-source inventory for peer reproduction.

This does not import, deploy, build or run a peer implementation. The root must
review the protocol and freeze a new committed source before any execution.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,re,subprocess
from pathlib import Path

JUNYU='b890ff49488474c5d476df44019597dc045a5969'
HAOWEI='6f7279c1625f7fa53bacb233b96fb69a385df8c1'
ARMS=('hf_sdpa_native','hf_peer_gaussian32_allretained','hf_peer_gaussian32_global')
GLOBAL_LAYERS=(5,11,17,23,29)
PUBLIC_PATHS=('experiments/value_direction_hopper','experiments/diffusion_gemma_jl_output_aware',
 'experiments/diffusion_gemma_solattn_blasst_multibench',
 'experiments/diffusion_gemma_value_aware_followup/protocol.py',
 'experiments/diffusion_gemma_ruler8k_jl.py','src/dllm',
 'tests/test_value_direction_hopper.py','tests/test_value_direction_hopper_report.py')

def require(condition,message):
    if not condition:raise ValueError(message)

def draft_spec():
    return dict(schema='v30_peer_reproduction_draft_v1',status='root_review_required',
      execution_enabled=False,formal_launch_enabled=False,performance_claim_allowed=False,
      attribution='Junyu Gaussian32 retained-state router; cooperation candidate',
      peer_source_commit=JUNYU,haowei_review_commit=HAOWEI,
      protocol_id='v30_hf_global_gaussian32_extended_draft001',
      fidelity='Peer value-risk/kernel unchanged; GLOBAL-only scope and thinking/full8192 protocol are explicitly new',
      arms=list(ARMS),seeds=list(range(31001,31009)),
      qualification=dict(per_arm_longest_prompt=True,full_budget=True,performance_evidence=False),
      pilot=dict(datasets={'aime26':dict(items=3),'longbench_v2_32k':dict(items=3)},
                 selection='Predeclare manifest before any candidate generation; no answer-based selection',
                 timed_cells=48,timed_requests=144),
      full_panel=dict(datasets={'aime26':dict(items=30),'longbench_v2':dict(items=59)},
                      timed_cells=712,timed_requests=2136,launch_enabled=False),
      settings=dict(generation_budget=8192,thinking=True,canvas_length=256,max_denoising_steps=48,
                    native_temperature_schedule=[.8,.4],confidence_threshold=.005,
                    stability_threshold=1,entropy_bound=.1,cpu_threads=1),
      method=dict(rank=32,projection_seed=1729,query_sensitivity=None,temporal=False,
                  risk_state='current_qk_sequential_retained',
                  output='retained original BF16 V',precision='tf32x3_register',projection='fused',
                  tma_for_supported_unmasked_calls=True,global_layers=list(GLOBAL_LAYERS),local='original_dense'),
      thresholds=dict(status='not_bound',source='Existing immutable peer policy only after byte/source/scope qualification; otherwise new independent calibration',
                      identical_across_seeds=True,final_score_retuning=False,
                      allretained_log_threshold='negative_infinity'),
      fairness=dict(same_host_per_cell=True,same_prompt_tokens=True,same_sampling=True,
                    same_source_and_build=True,fresh_per_condition_engine=True,
                    default_rng_equivalence_claim=False,local_mask_oracle_required=True,
                    original_peer_scope='LOCAL and GLOBAL; separate protocol if reproduced',
                    reference='HF/SDPA diagnostic only; official vLLM FA4 strong-reference port not yet implemented'),
      gates=['source_dependency_closure','own_toolchain_headers_and_bridge_pin','cpu_import_tests_cuda_hidden',
             'sm90_kernel_full_output_oracle','scope_and_local_mask_oracle','allretained_zero_skip',
             'native_sampling_and_stopping','full_budget_multicanvas_accounting',
             'timed_jit_capture_zero','full_private_scorer_proof','root_freeze_and_resource_barriers'],
      reporting=dict(public='Numeric per-request/worker evidence and question-cluster CIs only',
                     private_only=['prompts','tokens','gold','completion_text','private_paths','rng_states'],
                     spans=['W_including_binding_setup_cleanup','decode_S','actual_N_C_P','outer_reserved_GPU_seconds'],
                     counters=['global_routes','local_routes_zero','eligible_and_skipped_tiles','projected_tokens',
                               'reused_tokens','output_length','caps','jit_and_graph_capture_deltas']))

def validate_spec(spec):
    require(spec==draft_spec(),'Draft immutable mathematical/protocol fields changed; root must declare a new protocol')
    require(len(spec['seeds'])==8 and len(set(spec['seeds']))==8,'Eight distinct seeds required')
    for key in ('pilot','full_panel'):
        items=sum(x['items'] for x in spec[key]['datasets'].values())
        require(spec[key]['timed_cells']==items*8 and spec[key]['timed_requests']==items*8*len(ARMS),'Complete multi-question/seed/arm inventory required')
    return spec

def git(repo,*args):return subprocess.check_output(['git','-c','safe.directory=*',*args],cwd=repo)

def source_inventory(repo,commit=JUNYU):
    require(re.fullmatch('[0-9a-f]{40}',commit) is not None and commit==JUNYU,'Exact reviewed peer commit required')
    require(git(repo,'cat-file','-t',commit).decode().strip()=='commit','Committed source unavailable')
    paths=git(repo,'ls-tree','-r','--name-only',commit,*PUBLIC_PATHS).decode().splitlines()
    require(bool(paths),'Source closure absent')
    files={};syntax_errors=[]
    for path in paths:
        raw=git(repo,'show',commit+':'+path)
        if path.endswith('.py'):
            try:ast.parse(raw,filename=path)
            except SyntaxError as error:syntax_errors.append(dict(path=path,line=error.lineno,reason=error.msg))
        files[path]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return dict(schema='v30_peer_source_plan_v1',peer_commit=commit,files=files,
                source_ast_checked=True,source_ast_passed=not syntax_errors,syntax_errors=syntax_errors,peer_code_executed=False,
                closure_status='candidate roots; transitive imports must be qualified in isolated CUDA-hidden own environment',
                missing_build_inputs=['Pinned TensorRT-LLM headers and source commit','Complete CUTLASS header manifest',
                                      'Toolchain include/runtime/CCCL closure','New kernel and matching Torch bridge'],
                archive_only_to_private_own_root=True)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-draft',type=Path);parser.add_argument('--source-inventory',type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args(argv)
    spec=validate_spec(draft_spec())
    if args.write_draft:
        with args.write_draft.open('x',encoding='utf-8') as out:json.dump(spec,out,indent=2);out.write('\n')
    if args.source_inventory:
        inventory=source_inventory(args.repo)
        with args.source_inventory.open('x',encoding='utf-8') as out:json.dump(inventory,out,indent=2);out.write('\n')
    print(json.dumps(dict(draft_valid=True,execution_enabled=False,gpu_started=False,peer_code_executed=False,seeds=8,pilot_timed_requests=144,full_timed_requests=2136)))

if __name__=='__main__':main()
