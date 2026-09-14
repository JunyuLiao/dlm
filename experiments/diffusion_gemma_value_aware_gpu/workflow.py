"""One-H100 resumable1040-result sweep with a persistent15-minute monitor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from experiments.diffusion_gemma_value_aware.protocol import frozen_write, sha
from experiments.diffusion_gemma_value_aware.run import shard_path
from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _write, _append
from experiments.diffusion_gemma_value_aware_followup.run import gpu_idle
from experiments.diffusion_gemma_value_aware_followup.engine import check_result
from experiments.diffusion_gemma_value_aware_followup.protocol import MODEL, REVISION
from .protocol import ROOT, METHODS, TARGETS, prepare, execution
from .runner import smoke, dense, cached
from .calibration import prepare_policies, condition


def failure(root, stage, **identity):
    error = dict(stage=stage, **identity, traceback=traceback.format_exc())
    _append(root/'failures.jsonl', error)
    print(error, flush=True)
    return error


def semantic_policy(config, policy):
    result = {}
    for kind, entry in policy.items():
        if entry.get('cap_one') and entry.get('unattainable'):
            result[kind] = dict(log_threshold=0.)
        elif 'log_scale' in entry:
            result[kind] = dict(log_scale=entry['log_scale'], cap_one=bool(entry.get('cap_one')))
        else:
            value = entry['log_threshold']
            result[kind] = dict(log_threshold=min(0., value) if entry.get('cap_one') else value)
    return json.dumps(dict(config=config, policy=result), sort_keys=True)


def equivalent(root, row, label, selected, available, contract):
    wanted = semantic_policy(selected['config'], selected['thresholds'][row['benchmark']])
    for name, c in available.items():
        if name in ('dense', label):
            continue
        if semantic_policy(c['config'], c['thresholds'][row['benchmark']]) != wanted:
            continue
        source = shard_path(root, 'final', name, row['id'])
        if not source.exists():
            continue
        out = cached(None, root, row, 'final', name, c['config'], c['thresholds'][row['benchmark']], contract)
        out = dict(out, thresholds=selected['thresholds'][row['benchmark']],
            equivalent_source=dict(path=str(source), sha256=sha(source.read_bytes())),
            equivalence='Identical accelerated operator and numerical thresholds; only target/source annotations differ')
        check_result(out, row, contract['fingerprint'], selected['config'], selected['thresholds'][row['benchmark']])
        path = shard_path(root, 'final', label, row['id'])
        _write(path, out)
        _append(root/'completed.jsonl', dict(stage='final', condition=label, id=row['id'], path=str(path),
            finished=time.time(), backend='equivalent_accelerated_operator', equivalent_source=str(source)))
        return out
    return None


def freeze_conditions(root, setup, contract):
    smoke_path = root/'validation'/'smoke.json'
    proof = json.loads(smoke_path.read_text())
    if not proof['passed'] or proof['fingerprint'] != contract['fingerprint'] or len(proof['cases']) != 26:
        raise ValueError('Matching complete CUDA validation required')
    examples = [next(r for r in setup['calibration'] if r['benchmark']=='aime26'),
                max((r for r in setup['calibration'] if r['benchmark']=='longbench_v2'), key=lambda r:len(r['prompt_tokens']))]
    labels = {'dense'} | {f'{n}_s50' for n in METHODS} | {f'{n}_s75_proposal' for n in METHODS}
    if {(c['id'],c['condition']) for c in proof['cases']} != {(r['id'],n) for r in examples for n in labels}:
        raise ValueError('Missing or duplicated real-model method/benchmark validation')
    if any(not c['passed'] or not c['reference_masks_and_diagnostics_checked'] or not c['prefix_eligible']
           or not c['canvas_eligible'] or not c['finite_calls'] for c in proof['cases']):
        raise ValueError('Incomplete CUDA mask/region/finite coverage')
    if any(not c['exact_generation_parity'] for c in proof['cases'] if c['condition'] in ('dense','value_s50','mass_s50')):
        raise ValueError('Required independent generated-token parity missing')
    paths = [root/'execution_contract.json', root/'setup.json', smoke_path,
             Path(__file__), Path(__file__).with_name('calibration.py')]
    paths += [Path('experiments/diffusion_gemma_value_aware_followup')/n for n in ('calibrate.py','policies.py','evidence.py','protocol.py')]
    paths += [Path('experiments/diffusion_gemma_value_aware')/n for n in ('policy_search.py','rank_calibration.py','calibration.py')]
    common = {**contract['sources'], **setup['source_hashes'], **proof['sources'], **{str(p):sha(p.read_bytes()) for p in paths}}
    result = {}
    requested = [('dense', 'dense', 0.)]+[(f'{n}_s{int(t*100)}', n, t) for t in TARGETS for n in METHODS]
    for label, name, target in requested:
        try:
            c = condition(root, setup, contract, name, target)
            sources = dict(common)
            for source in c['policy_sources'].values():
                sources[source['path']] = source['sha256']
            c.update(fingerprint=contract['fingerprint'], sources=sources,
                     expected_per_benchmark=dict(aime26=30, longbench_v2=50))
            frozen_write(root/'final_configs'/f'{label}.json', c)
            result[label] = c
        except Exception:
            failure(root, 'freeze_condition', condition=label)
    return result


def work(root):
    import torch
    from dllm.models import create_adapter
    setup, contract = prepare(root), execution(root)
    adapter = create_adapter('diffusion_gemma', MODEL, device='cuda', precision='bfloat16', revision=REVISION).load()
    smoke(adapter, root, setup, contract)
    contract = execution(root, freeze=True)
    _write(root/'runtime.json', dict(model=adapter.model.config.to_dict(), generation=adapter.model.generation_config.to_dict(),
                                   runtime=contract['runtime'], fingerprint=contract['fingerprint']))
    _write(root/'phase.json', dict(stage='dense_baselines', expected=80, started=time.time()))
    errors = []
    for row in sorted(setup['final'], key=lambda r:(r['generation_budget'], r['id'])):
        try:
            dense(adapter, root, row, contract)
        except Exception:
            errors.append(failure(root, 'dense', id=row['id']))
            torch.cuda.empty_cache()
    _write(root/'phase.json', dict(stage='calibration75_and_import50', started=time.time()))
    errors.extend(prepare_policies(adapter, root, setup, contract))
    available = freeze_conditions(root, setup, contract)
    # Each configuration is independently frozen; failures in one policy do
    # not discard or prevent unrelated requested conditions.
    for label, c in available.items():
        if label == 'dense':
            continue
        _write(root/'phase.json', dict(stage='final', condition=label, expected=80, started=time.time()))
        for row in sorted(setup['final'], key=lambda r:(r['generation_budget'], r['id'])):
            try:
                cached(None, root, row, 'dense', 'dense', {}, None, contract)
                path = shard_path(root, 'final', label, row['id'])
                if path.exists():
                    cached(None, root, row, 'final', label, c['config'], c['thresholds'][row['benchmark']], contract)
                elif equivalent(root, row, label, c, available, contract) is None:
                    cached(adapter, root, row, 'final', label, c['config'], c['thresholds'][row['benchmark']], contract)
            except Exception:
                errors.append(failure(root, 'final', condition=label, id=row['id']))
                torch.cuda.empty_cache()
    del adapter
    torch.cuda.empty_cache()
    _write(root/'phase.json', dict(stage='report', started=time.time()))
    from .report import regenerate
    audited = regenerate(root)
    _write(root/'run_terminal.json', dict(complete=audited['complete'], completed=audited['completed'],
        expected=1040, failed_attempts=len(errors), finished=time.time()))
    if not audited['complete']:
        raise RuntimeError('Incomplete sweep; independent results preserved, resume only missing work')


def supervise(root):
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
        command = [sys.executable, '-u', '-m', __package__+'.workflow', 'work', '--output', str(root)]
        with (root/'run.log').open('a', buffering=1) as log:
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            _write(root/'job.json', dict(pid=worker.pid, supervisor_pid=os.getpid(), command=command, started=time.time()))
            while True:
                try:
                    code = worker.wait(timeout=900)
                    break
                except subprocess.TimeoutExpired:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
                    progress_path = root/'progress.json'
                    phase_path = root/'phase.json'
                    _append(root/'monitor.jsonl', dict(time=time.time(), pid=worker.pid, alive=True,
                        gpu=gpu.stdout.strip(), gpu_error=gpu.stderr.strip(), disk_free_bytes=shutil.disk_usage(root).free,
                        phase=json.loads(phase_path.read_text()) if phase_path.exists() else None,
                        progress=json.loads(progress_path.read_text()) if progress_path.exists() else None))
            _write(root/'supervisor_terminal.json', dict(exit_code=code, pid=worker.pid, finished=time.time(), command=command))
            if code:
                raise SystemExit(code)


def launch(root):
    root.mkdir(exist_ok=True, parents=True)
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        gpu_idle()
    with (root/'supervisor.log').open('a', buffering=1) as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', __package__+'.workflow', 'supervise', '--output', str(root)],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps(dict(supervisor_pid=process.pid, status='launched; verify job.json and actual GPU process')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('launch','supervise','work','report'))
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == 'report':
        from .report import regenerate
        print(json.dumps({k:v for k,v in regenerate(args.output).items() if k not in ('sources','artifacts')}))
    else:
        globals()[args.command](args.output)
