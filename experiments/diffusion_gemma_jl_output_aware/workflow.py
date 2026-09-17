"""One resumable GPU worker; all primary directions remain in final evaluation."""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch

from dllm.models import create_adapter
from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware_followup.evidence import check_sources
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from .protocol import ROOT, MODEL, REVISION, prepare, execution
from .config import BASELINES, PROJECTED, METHODS, TARGETS, SENSITIVITY_SEEDS
from .runner import cached, import_aime
from .validation import smoke
from .observer_validation import validate_observer
from .diagnostics import SnapshotObserver
from .calibration import rows_for, fit_one, audit_policy
from .screen import source_index, shared_screen
from .test_gate import require as require_tests


def failure(root, stage, **details):
    error = dict(stage=stage, **details, traceback=traceback.format_exc())
    _append(root/'failures.jsonl', error)
    print(error, flush=True)
    torch.cuda.empty_cache()
    return error


def tokens(row):
    value = row['prompt_tokens']
    return len(value) if isinstance(value, list) else int(value)


def phase(root, stage, **details):
    _write(root/'phase.json', dict(stage=stage, started=time.time(), **details))


def freeze_conditions(root, setup, contract):
    common = {**contract['sources'], **setup['source_hashes']}
    common.update(require_tests(root,contract)['sources'])
    for name in ('workflow.py', 'validation.py', 'observer_validation.py', 'calibration.py', 'screen.py'):
        path = Path(__file__).with_name(name)
        common[str(path)] = sha(path.read_bytes())
    for path in (root/'execution_contract.json', root/'setup.json', root/'calibration_contract.json',
                 root/'validation'/contract['fingerprint']/'smoke.json',
                 root/'validation'/contract['fingerprint']/'observer.json', root/'tensor_validation.json'):
        common[str(path)] = sha(path.read_bytes())
    check_sources(common)
    result = {}
    for name, target in [('dense', 0.)]+[(n,t) for t in TARGETS for n in METHODS]:
        label = 'dense' if name == 'dense' else f'{name}_s{int(target*100)}'
        config = {} if name == 'dense' else {**BASELINES, **PROJECTED}[name]
        thresholds, policy_sources, unavailable = {}, {}, []
        for benchmark in ('aime26', 'longbench_v2'):
            if name == 'dense':
                thresholds[benchmark] = None; continue
            path = root/'policies'/benchmark/f'{label}.json'
            if not path.exists():
                unavailable.append(benchmark); continue
            p = json.loads(path.read_text()); audit_policy(root, p, setup, contract)
            if p['config'] != config or p['target'] != target:
                raise ValueError('Policy identity mismatch')
            thresholds[benchmark] = p['policy']
            policy_sources[str(path)] = sha(path.read_bytes())
        out = dict(fingerprint=contract['fingerprint'], name=name, target=target, config=config,
            thresholds=thresholds, unavailable_benchmarks=unavailable,
            sources={**common, **policy_sources}, target_metric='physical_sparsity',
            expected_per_benchmark=dict(aime26=30, longbench_v2=50))
        frozen_write(root/'final_configs'/f'{label}.json', out)
        result[label] = out
    return result


def development(adapter, root, setup, contract, conditions):
    aime = rows_for(setup, 'aime26')[:2]
    # Development generation uses full final-style budgets, not fitting caps.
    original = {r['id']:r for r in setup['calibration']}
    aime = [original[r['id']] for r in aime]
    domains = {}
    for r in setup['development']:
        domains.setdefault(r['task'], r)
    seed_rows = aime+list(domains.values())
    small_rows = [aime[0], max(setup['development'], key=tokens)]
    failures = []
    for row in seed_rows:
        try:
            cached(adapter, root, row, 'development', 'dense', 'dense', {}, None, contract)
        except Exception:
            failures.append(failure(root, 'development_dense', id=row['id']))
    for label, condition in conditions.items():
        name = condition['name']
        if name == 'dense':
            continue
        primary = name.startswith('jl_')
        rows = seed_rows if primary else small_rows
        seeds = (None, *SENSITIVITY_SEEDS) if primary else (None,)
        for projection_seed in seeds:
            config = dict(condition['config'])
            candidate, tag = name, label
            if projection_seed is not None:
                config['projection_seed'] = projection_seed
                candidate = 'seedcheck/'+name+f'/seed{projection_seed}'
                tag = label+f'/seed{projection_seed}'
            for row in rows:
                if row['benchmark'] not in condition['thresholds']:
                    continue
                try:
                    cached(adapter, root, row, 'development', tag, candidate, config,
                        condition['thresholds'][row['benchmark']], contract)
                except Exception:
                    failures.append(failure(root, 'development', condition=tag, id=row['id']))
    frozen_write(root/'development_policy.json', dict(
        selection='No accuracy- or operator-error-based method exclusions; all requested candidates proceed when execution is valid',
        primary_seed_fixed=1729, alternate_projection_seeds=list(SENSITIVITY_SEEDS),
        seed_check_ids=[r['id'] for r in seed_rows], other_control_check_ids=[r['id'] for r in small_rows],
        thresholds='Primary calibrated thresholds reused unchanged across alternate projection seeds; report resulting sparsity changes',
        failures=failures, final_scores_used=False))
    return failures


def work(root):
    torch.backends.cuda.matmul.allow_tf32 = False
    setup = prepare(root)
    contract = execution(root, freeze=True)
    require_tests(root, contract)
    phase(root, 'load_model')
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    _write(root/'runtime.json', dict(model=adapter.model.config.to_dict(), generation=adapter.model.generation_config.to_dict(),
        runtime=contract['runtime'], fingerprint=contract['fingerprint']))
    phase(root, 'validation')
    validated = smoke(adapter, root, setup, contract)
    validate_observer(adapter, root, setup, contract, validated)
    failures = []
    phase(root, 'dense', expected=80)
    for row in sorted(setup['final'], key=lambda r:(tokens(r),r['id'])):
        try:
            if import_aime(root,row,'dense','dense',{},None,contract) is None:
                cached(adapter,root,row,'dense','dense','dense',{},None,contract)
        except Exception:
            failures.append(failure(root,'dense',id=row['id']))
    rows = rows_for(setup,'aime26')+rows_for(setup,'longbench_v2')
    frozen_write(root/'calibration_execution_manifest.json',rows)
    fitting = dict(fingerprint=contract['fingerprint'], calibration_budget=512,
        calibration_ids=[r['id'] for r in rows], final_scores_used=False, max_verified_points=3,
        source_hashes={str(p):sha(p.read_bytes()) for p in
            (Path(__file__).with_name('calibration.py'), Path(__file__).with_name('screen.py'))},
        rule='Historical scalar local/global convention, sparse-trajectory empirical rank or inverse-length search; AIME baseline policies reused when identical')
    frozen_write(root/'calibration_contract.json',fitting)
    phase(root,'dense_calibration_shared_states',expected=12)
    calibration_outputs=[]
    for index,row in enumerate(rows):
        try:
            observer=SnapshotObserver(root,row,index%6)
            out=cached(adapter,root,row,'dense_calibration','dense','dense',{},None,contract,observer=observer)
            if not out.get('shared_state_sources'):
                raise ValueError('Dense calibration cache lacks shared QKV evidence')
            calibration_outputs.append(out)
        except Exception:
            failures.append(failure(root,'dense_calibration',id=row['id']))
    if len(calibration_outputs)!=12:
        raise RuntimeError('Shared calibration incomplete; successful dense outputs remain cached')
    sources=source_index(root,calibration_outputs)
    phase(root,'shared_calibration_screen',expected=len(sources))
    shared_screen(root,sources,contract)
    configs={**BASELINES,**PROJECTED}
    phase(root,'calibration')
    for target in TARGETS:
        for name in METHODS:
            for benchmark in ('longbench_v2','aime26'):
                try:
                    fit_one(adapter,root,setup,contract,benchmark,name,configs[name],target)
                except Exception:
                    failures.append(failure(root,'calibration',benchmark=benchmark,name=name,target=target))
    conditions=freeze_conditions(root,setup,contract)
    phase(root,'development_generation_checks')
    failures.extend(development(adapter,root,setup,contract,conditions))
    phase(root,'shared_operator_diagnostics')
    try:
        from .shared_analysis import analyze
        analyze(root,sources,contract)
    except Exception:
        failures.append(failure(root,'shared_operator_diagnostics'))
    for label,condition in conditions.items():
        if condition['name']=='dense':
            continue
        phase(root,'final',condition=label,expected=80)
        for row in sorted(setup['final'],key=lambda r:(tokens(r),r['id'])):
            if row['benchmark'] not in condition['thresholds']:
                continue
            try:
                threshold=condition['thresholds'][row['benchmark']]
                if import_aime(root,row,label,condition['name'],condition['config'],threshold,contract) is None:
                    cached(adapter,root,row,'final',label,condition['name'],condition['config'],threshold,contract)
            except Exception:
                failures.append(failure(root,'final',condition=label,id=row['id']))
    del adapter
    torch.cuda.empty_cache()
    phase(root,'report')
    from .report import regenerate
    audit=regenerate(root)
    if audit['complete']:
        from .verify import verify
        verify(root)
    _write(root/'run_terminal.json',dict(complete=audit['complete'],completed=audit['completed'],
        expected=audit['expected'],failed_attempts=len(failures),finished=time.time()))


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
        with (root/'run.log').open('a',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','work','--output',str(root)],
                stdout=log,stderr=subprocess.STDOUT)
            _write(root/'job.json',dict(pid=child.pid,supervisor_pid=os.getpid(),started=time.time()))
            while True:
                try:
                    code=child.wait(timeout=900);break
                except subprocess.TimeoutExpired:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
                    status={}
                    for key in ('phase','progress'):
                        path=root/f'{key}.json'
                        status[key]=json.loads(path.read_text()) if path.exists() else None
                    _append(root/'monitor.jsonl',dict(pid=child.pid,time=time.time(),alive=True,gpu=gpu.stdout.strip(),
                        gpu_error=gpu.stderr.strip(),disk_free_bytes=shutil.disk_usage(root).free,**status))
            _write(root/'supervisor_terminal.json',dict(exit_code=code,pid=child.pid,finished=time.time()))
            if code:
                raise SystemExit(code)


def launch(root):
    prepare(root)
    gpu_idle()
    with (root/'supervisor.log').open('a',buffering=1) as log:
        child=subprocess.Popen([sys.executable,'-u','-m',__package__+'.workflow','supervise','--output',str(root)],
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(supervisor_pid=child.pid,status='launched; verify job.json/GPU worker')))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','launch','supervise','work','report'))
    parser.add_argument('--output',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.command=='prepare':
        prepare(args.output)
    elif args.command=='report':
        from .report import regenerate
        print(json.dumps(regenerate(args.output)))
    else:
        globals()[args.command](args.output)


if __name__=='__main__':
    main()
