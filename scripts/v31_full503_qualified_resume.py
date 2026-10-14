"""Resume the unchanged frozen study after qualifying the dense control."""
import argparse
import contextlib
import json
from pathlib import Path
import subprocess
import time

from scripts.v31_full503_refresh_study import PY, read, write, verify


def external_evaluator_running():
    # A live CPU-side evaluator can be loading its next model while nvidia-smi
    # briefly reports an empty device. Do not mistake that gap for availability.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc/'cmdline').read_bytes().split(b'\0')
            if any(arg.endswith(b'run_aime_eval.py') for arg in args):
                return True
        except OSError:
            pass
    return False


def resource_busy():
    return external_evaluator_running() or bool(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', type=Path, required=True)
    args = p.parse_args()
    study = args.study.resolve()
    root = Path(__file__).resolve().parents[1]
    protocol = read(study/'specs/protocol.json')
    verify(study, protocol)
    last = 0
    idle_since = None
    while True:
        if resource_busy():
            idle_since = None
        elif idle_since is None:
            idle_since = time.monotonic()
        elif time.monotonic()-idle_since >= 60:
            break
        if time.monotonic()-last >= 900:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used', '--format=csv,noheader'], text=True).strip()
            write(study/'progress.json', dict(status='waiting_for_idle_H100', gpu=gpu, unix=time.time()))
            with (study/'monitor.jsonl').open('a') as f:
                f.write(json.dumps(dict(event='waiting_for_idle_H100', gpu=gpu, unix=time.time()))+'\n')
            print(json.dumps(dict(event='waiting_for_idle_H100', gpu=gpu)), flush=True)
            last = time.monotonic()
        time.sleep(5)
    qualification = study/'qualification/dense_mode_resolution.json'
    if not qualification.exists():
        for n in range(1, 1000):
            output = study/f'qualification/causal_alias_full_attempt{n:03d}'
            log = study/f'causal_alias_full_attempt{n:03d}.log'
            if not output.exists() and not log.exists():
                break
        write(study/'progress.json', dict(status='qualifying_causal_buffer', attempt=n, unix=time.time()))
        with log.open('w') as stream:
            result = subprocess.run([PY, str(root/'scripts/v31_dense_causal_qualification.py'),
                '--study', str(study), '--output', str(output)], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            write(study/'progress.json', dict(status='causal_qualification_failed', log=str(log.relative_to(study)), unix=time.time()))
            raise RuntimeError('causal qualification failed; preserved diagnostic requires inspection')
        receipt = read(output/'receipt.json')
        if receipt['status'] != 'complete' or receipt['alias_errors'] or receipt['dtype_errors']:
            raise RuntimeError('persistent causal metadata did not validate')
        actual = [json.loads(l) for l in (output/'records.jsonl').read_text().splitlines()]
        repeated = [json.loads(l) for l in (study/'attempt002/qualification/dense_parity_mem085/dense_full_fix51994/records.jsonl').read_text().splitlines()]
        fields = ('index', 'output_hash', 'denoise_forwards', 'output_tokens', 'canvases')
        if len(actual) != 2 or any(a[k] != b[k] for a, b in zip(actual, repeated) for k in fields):
            raise RuntimeError('within-FULL native repetition failed')
        write(qualification, dict(status='complete', source_commit=protocol['source_commit'],
            causal_alias_receipt=str((output/'receipt.json').relative_to(study)),
            diagnostic_source_sha256=receipt['diagnostic_sha256'],
            persistent_int32_causal_alias_validated=True, within_FULL_output_N_T_C_mismatches=0,
            compared_memory_utilizations=[0.80, 0.85],
            cross_mode_output_equality_required=False,
            interpretation='Both fixed native modes valid; FULL and PIECEWISE can have different free-running trajectories. Current FULL repeats exactly; causal metadata aliases the persistent int32 buffer in capture and runtime. No selector or sampler settings changed.',
            reference='docs/V31_BASELINE_ROOT_CAUSE_20261003.md; fixed FULL versus PIECEWISE and documented host/cache-dependent divergence',
            generation_source_unchanged=True, clean_timing='paused'), True)
    verify(study, protocol)
    n = len(list(study.glob('coordinator_attempt*.log')))+1
    log = study/f'coordinator_attempt{n:03d}.log'
    print(json.dumps(dict(event='resume', frozen_source_commit=protocol['source_commit'], log=str(log.relative_to(study)))), flush=True)
    from scripts import v31_full503_refresh_study as runner
    quiet_since = None

    def guarded_idle():
        nonlocal quiet_since
        if resource_busy():
            quiet_since = None
            return False
        if quiet_since is None:
            quiet_since = time.monotonic()
            return False
        if time.monotonic()-quiet_since < 60:
            return False
        quiet_since = None
        return True

    # Only resource admission changes. Frozen native worker sources, inputs,
    # sampler settings and attention implementations remain unchanged.
    runner.gpu_idle = guarded_idle
    with log.open('w') as stream:
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            runner.run(study)


if __name__ == '__main__':
    main()
