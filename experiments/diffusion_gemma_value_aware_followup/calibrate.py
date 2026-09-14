"""Resumable sparse-trajectory calibration for all six families and controls.

Only the six designated calibration samples in each benchmark set thresholds.
The existing inverse-L BLASST fit and empirical-rank scalar-risk refinement
are reused. Original BLASST has independent, actually verified lambda=1 bounds.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append, _fingerprint
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_value_aware.policy_search import next_policy, select_point
from experiments.diffusion_gemma_value_aware.rank_calibration import next_policy as rank_next
from experiments.diffusion_gemma_value_aware.repair_original import scale_upper
from .engine import contract
from .execution import cached, provenance
from .policies import (METHODS, BENCHMARKS, KINDS, inputs, calibration_rows,
    starting_policy, distributions, import_aime_policy, measurements,
    audit_point, publish, audit_policy)
from .protocol import ROOT, MODEL, REVISION, TARGETS, prepare


def smoke_dense(adapter, root, row, execution):
    # The old cache's AIME dense-parity shortcut is for the full official budget,
    # not this16-token smoke. Cache a native short reference explicitly instead.
    from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _install_dense, _set_context, _request
    path = shard_path(root, 'sparse_smoke', 'native_dense16', row['id'])
    identity = dict(fingerprint=execution['fingerprint'],
        **{k:row[k] for k in ('id','prompt_hash','seed','generation_budget')})
    if path.exists():
        data=json.loads(path.read_text())
        if data['identity']!=identity:
            raise ValueError('short native smoke reference mismatch')
        return data
    binding=_install_dense(adapter)
    try:
        _set_context(binding,row);native=adapter.generate(_request(row))
    finally:
        binding.close()
    if isinstance(row['prompt_tokens'],list) and native.prompt_tokens!=row['prompt_tokens']:
        raise ValueError('native smoke prompt token mismatch')
    data=dict(identity=identity,completion_tokens=native.completion_tokens,
        generation_metadata=native.metadata)
    frozen_write(path,data);return data


def smoke(adapter, root, setup, execution, configs, names):
    """Every family gets finite/physical-accounting CUDA coverage before fitting."""
    identity = dict(fingerprint=execution['fingerprint'], configs={n:configs[n] for n in names},
        provenance={n:provenance(configs[n]) for n in names})
    path = root/'sparse_smoke'/f'{_fingerprint(identity)[:16]}.json'
    if path.exists():
        previous = json.loads(path.read_text())
        if previous['identity'] != identity:
            raise ValueError('sparse smoke mismatch')
        if previous['passed']:
            return previous
    examples = [next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
        max((r for r in setup['calibration'] if r['benchmark']=='longbench_v2'),key=lambda r:len(r['prompt_tokens']))]
    tests = []; failures = []
    for example in examples:
        row = dict(example, generation_budget=16)
        try:
            dense = smoke_dense(adapter, root, row, execution)
        except Exception:
            error=dict(stage='sparse_smoke_dense',id=row['id'],traceback=traceback.format_exc())
            failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
            continue
        for name in names:
            for unpruned in (True, False):
                try:
                    config = configs[name]
                    threshold = -100. if unpruned else (0. if config['method'] in ('blasst','value','aligned') else -1.)
                    policy = {kind:dict(log_threshold=threshold) for kind in KINDS}
                    case = f'{name}_{"unpruned" if unpruned else "sparse"}_16'
                    result = cached(adapter, root, row, 'sparse_smoke', case, config, policy, execution)
                    if unpruned and result['completion_tokens'] != dense['completion_tokens']:
                        raise ValueError(f'{name}: unpruned dense token parity failed')
                    records = result['records']
                    if unpruned and any(r['pv_omitted'] for r in records):
                        raise ValueError(f'{name}: unpruned smoke omitted PV')
                    for key in ('native_canvas_length','thinking','sampling','denoising_configuration'):
                        if result['generation_metadata'].get(key) != dense['generation_metadata'].get(key):
                            raise ValueError(f'{name}: unrelated decoding setting changed: {key}')
                    if config['method'] in ('compensate','zero_pv'):
                        if any(r['skipped'] or r['denominator_mass_sum'] != r['rows'] for r in records):
                            raise ValueError('PV approximation was incorrectly counted as physical deletion')
                    tests.append(dict(id=row['id'], name=name, unpruned=unpruned,passed=True,
                        dense_parity=True if unpruned else None, finite_calls=result['finite_calls'],
                        skipped=sum(r['skipped'] for r in records),
                        pv_omitted=sum(r['pv_omitted'] for r in records),
                        prefix_skipped=sum(r['prefix_skipped'] for r in records),
                        canvas_skipped=sum(r['canvas_skipped'] for r in records)))
                except Exception:
                    error=dict(stage='sparse_smoke',id=row['id'],name=name,unpruned=unpruned,
                        traceback=traceback.format_exc())
                    failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
                    tests.append(dict(id=row['id'],name=name,unpruned=unpruned,passed=False))
                finally:
                    import torch
                    torch.cuda.empty_cache()
        print('sparse family smoke checked', row['id'], flush=True)
    verified=smoke_verified(tests,names,[r['id'] for r in examples])
    result = dict(passed=len(verified)==len(names), identity=identity, tests=tests,
        verified_methods=verified,failures=failures)
    if path.exists():
        old=json.loads(path.read_text())
        frozen_write(path.with_name(path.stem+'_'+_fingerprint(old)[:16]+'.json'),old)
    _write(path,result);return result


def smoke_verified(tests,names,ids):
    expected={(identity,unpruned) for identity in ids for unpruned in (True,False)}
    verified=[]
    for name in names:
        rows=[r for r in tests if r['name']==name]
        if (len(rows)==len(expected) and all(r['passed'] for r in rows)
                and {(r['id'],r['unpruned']) for r in rows}==expected):
            verified.append(name)
    return verified


def run_group(adapter, root, rows, condition, config, policy, execution):
    if any(r['split']!='calibration' for r in rows):
        raise ValueError('sparse calibration cannot execute final/development examples')
    outputs=[]; sources={}; failures=[]
    import torch
    for row in sorted(rows,key=lambda r:(r['generation_budget'],r['id'])):
        try:
            result=cached(adapter,root,row,'calibration',condition,config,policy,execution)
            path=shard_path(root,'calibration',condition,row['id'])
            # Historical calibration reuse is explicit in import_aime_policy,
            # not a final-cache shortcut in this new sparse verification path.
            if not path.exists():
                raise ValueError('new calibration point lacks its own raw shard')
            sources[row['id']]=dict(path=str(path),sha256=sha(path.read_bytes()),
                fingerprint=execution['fingerprint'],stage='calibration')
            outputs.append(result)
        except Exception:
            error=dict(stage='calibration',condition=condition,id=row['id'],traceback=traceback.format_exc())
            failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
        finally:
            torch.cuda.empty_cache()
    return outputs,sources,failures


def fitting_done(observations, target, name, upper):
    if not observations:
        return False
    best, fixed = select_point(observations,target,name=='blasst_original',upper)
    # A verified impossible local target must not prevent a feasible global
    # target being fitted; conversely do not chase a known capped boundary.
    return all(fixed[k] or abs(best['achieved'][k]-target)<=.02 for k in KINDS)


def calibrate_one(adapter,root,setup,execution,proposals,name,config,target,benchmark,max_rounds):
    rows=calibration_rows(setup,benchmark);upper=scale_upper(rows)
    final=root/'verified_policies'/benchmark/f'{name}_s{round(target*100)}.json'
    if final.exists():
        result=json.loads(final.read_text());audit_policy(result,rows,config,execution)
        if (result['name'],result['benchmark'],result['target'])!=(name,benchmark,target):
            raise ValueError('cached policy path/identity mismatch')
        return result
    trace_path=root/'calibration_traces'/benchmark/f'{name}_s{round(target*100)}.json'
    observations=json.loads(trace_path.read_text()) if trace_path.exists() else []
    if not observations:
        imported=import_aime_policy(root,rows,name,config,target,execution)
        if imported:
            observations=[imported];_write(trace_path,observations)
    for point in observations:
        audit_point(point,rows,config,execution)
    values=search_source=None
    if not name.startswith('blasst_'):
        values,search_source=distributions(root,rows,name,config,execution)
    def update():
        return rank_next(observations,target,values) if values is not None else next_policy(
            observations,target,name=='blasst_original',upper)
    policy=update() if observations else starting_policy(proposals,benchmark,name,config,target)
    start=max((p['iteration'] for p in observations),default=-1)+1
    for iteration in range(start,start+max_rounds):
        if fitting_done(observations,target,name,upper):
            break
        # Avoid a redundant deterministic replay of an already observed policy.
        if any(p['policy']==policy for p in observations):
            break
        condition=f'{name}_s{round(target*100)}/{benchmark}/{_fingerprint(policy)[:16]}'
        outputs,sources,failures=run_group(adapter,root,rows,condition,config,policy,execution)
        if failures or len(outputs)!=len(rows):
            raise RuntimeError('incomplete calibration point; saved successes remain resumable')
        metrics,achieved=measurements(outputs,config)
        point=dict(iteration=iteration,policy=deepcopy(policy),achieved=achieved,metrics=metrics,
            error=max(abs(v-target) for v in achieved.values()),sources=sources,
            source_condition=condition,source_ids=[r['id'] for r in rows])
        observations.append(point);_write(trace_path,observations)
        print('calibrated point',name,benchmark,target,achieved,flush=True)
        policy=update()
    if not observations:
        raise ValueError('cannot publish an unverified threshold')
    result=publish(root,benchmark,name,config,target,observations,rows,execution,upper,search_source)
    audit_policy(result,rows,config,execution)
    return result


def execute(root,names,targets,max_rounds):
    setup=prepare(root);execution=contract(root);proposals,configs=inputs(root,execution)
    if not names or len(names)!=len(set(names)) or not set(names)<=set(configs):
        raise ValueError('unknown/empty/duplicated method selection')
    if not targets or len(targets)!=len(set(targets)) or not set(targets)<=set(TARGETS) or max_rounds<1:
        raise ValueError('invalid target selection/round count')
    protocol=dict(fingerprint=execution['fingerprint'],names=names,targets=targets,max_rounds=max_rounds,
        max_rounds_semantics='new sparse verification points per resumed invocation; completed points never repeated',
        source_hashes={str(p):sha(p.read_bytes()) for p in
            (Path(__file__),Path(__file__).with_name('policies.py'),Path(__file__).with_name('execution.py'))},
        configs={n:configs[n] for n in names},provenance={n:provenance(configs[n]) for n in names},
        threshold_proposals_sha256=sha((root/'threshold_proposals.json').read_bytes()),heldout_used=False)
    tag=_fingerprint(protocol)[:16];frozen_write(root/'calibration_jobs'/f'{tag}.json',protocol)
    from dllm.models import create_adapter
    adapter=create_adapter('diffusion_gemma',MODEL,device='cuda',precision='bfloat16',revision=REVISION).load()
    smoke_result=smoke(adapter,root,setup,execution,configs,names)
    verified=set(smoke_result['verified_methods'])
    failures=[dict(stage='sparse_smoke',name=name,error='incomplete or failed family smoke; independent families continue')
        for name in names if name not in verified];complete=[]
    # All families at50 first; short-budget v2 before AIME within a family.
    for target in targets:
        for name in names:
            if name not in verified:
                continue
            for benchmark in BENCHMARKS:
                try:
                    policy=calibrate_one(adapter,root,setup,execution,proposals,name,configs[name],target,benchmark,max_rounds)
                    complete.append(dict(name=name,benchmark=benchmark,target=target,
                        measured=policy['measured'],within_two_points=policy['within_two_points']))
                except Exception:
                    error=dict(stage='calibration_policy',name=name,benchmark=benchmark,target=target,
                        traceback=traceback.format_exc())
                    failures.append(error);_append(root/'failures.jsonl',error);print(error,flush=True)
    result=dict(complete=not failures and len(complete)==len(names)*len(targets)*2,
        completed_policies=complete,failures=failures,protocol=protocol,finished=time.time())
    _write(root/'calibration_jobs'/f'{tag}_audit.json',result)
    if not result['complete']:
        raise RuntimeError('some calibration groups failed; independent groups continued')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT)
    p.add_argument('--methods',nargs='+',default=list(METHODS))
    p.add_argument('--targets',nargs='+',type=float,default=[.5])
    p.add_argument('--max-rounds',type=int,default=3)
    args=p.parse_args();execute(args.output,args.methods,args.targets,args.max_rounds)


if __name__=='__main__':
    main()
