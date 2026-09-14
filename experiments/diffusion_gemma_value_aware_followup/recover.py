"""Bounded recovery after the current calibration/development queue finishes.

Reuses the original calibration, development, GPU lock and 900-second monitor.
Completed shards are never deleted; no thresholds or model settings are edited.
One recovery attempt is made, with failures preserved for subsequent diagnosis.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from experiments.diffusion_gemma_solattn_blasst_multibench.runner import _append, _write
from . import jobs
from .policies import METHODS, BENCHMARKS
from .protocol import ROOT, sha
from .workflow import process_identity, same_live_process, stage_command


def missing_methods(root):
    # This selects work, NOT a scientific policy. Existing files are fully
    # audited by calibrate/development; corruption is never silently erased.
    return [name for name in METHODS if any(not (
        root/'verified_policies'/benchmark/f'{name}_s50.json').exists()
        for benchmark in BENCHMARKS)]


def run_recovery(root):
    failed = []
    names = missing_methods(root)
    if names:
        try:
            jobs.supervise(root, 'calibrate', ['--methods', *names, '--targets', '0.5'])
        except (Exception, SystemExit):
            failed.append(dict(stage='recovery_calibration', methods=names, traceback=traceback.format_exc()))
    # The ordinary development run reuses every completed matching output and
    # regenerates the full all-family audit, even if its previous run was partial.
    try:
        with patch.object(jobs, 'worker_args', stage_command):
            jobs.supervise(root, 'development', [])
    except (Exception, SystemExit):
        failed.append(dict(stage='recovery_development', traceback=traceback.format_exc()))
    for failure in failed:
        _append(root/'failures.jsonl', failure)
    return dict(retried_methods=names, failures=failed, remaining_missing=missing_methods(root))


def supervise(root):
    path = root/'recovery_job.json'
    state = json.loads(path.read_text())
    if state['source_sha256'] != sha(Path(__file__).read_bytes()):
        raise ValueError('recovery source changed after launch')
    with (root/'recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state.update(pid=os.getpid(), status='waiting')
        _write(path, state)
        dependency = state['after_identity']
        while same_live_process(dependency, process_identity(dependency['pid'])):
            _append(root/'recovery_monitor.jsonl', dict(time=time.time(), pid=os.getpid(),
                status='waiting_on_original_queue', dependency=dependency))
            time.sleep(900)
        state.update(status='running', started=time.time())
        _write(path, state)
        try:
            result = run_recovery(root)
        except (Exception, SystemExit):
            result = dict(failures=[dict(stage='recovery_supervisor', traceback=traceback.format_exc())])
            _append(root/'failures.jsonl', result['failures'][0])
        complete = not result['failures'] and not result.get('remaining_missing')
        state.update(result, status='complete' if complete else 'finished_with_failures', finished=time.time())
        _write(path, state)
        if not complete:
            raise SystemExit(1)


def launch(root, dependency_pid):
    identity = process_identity(dependency_pid)
    queued = json.loads((root/'queued_workflow.json').read_text())
    if (not same_live_process(identity, identity) or queued.get('pid') != dependency_pid
            or 'diffusion_gemma_value_aware_followup.workflow supervise' not in identity['command']):
        raise ValueError('recovery must wait on this experiment’s actual live queue')
    path = root/'recovery_job.json'
    with (root/'recovery.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            previous = json.loads(path.read_text())
            live = process_identity(previous.get('pid', 0))
            if same_live_process(live, live):
                raise ValueError('recovery is already live')
            _write(root/'recovery_history'/f'{sha(path.read_bytes())}.json', previous)
        _write(path, dict(after_identity=identity, created=time.time(), status='launching',
            source_sha256=sha(Path(__file__).read_bytes()),
            reason='Resume incomplete calibration after observed external-process GPU contention; reuse complete outputs'))
    with (root/'recovery.log').open('a', buffering=1) as log:
        process = subprocess.Popen([sys.executable, '-u', '-m', __package__+'.recover',
            'supervise', '--output', str(root)], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    print(json.dumps(dict(recovery_pid=process.pid, after_identity=identity)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('launch', 'supervise'))
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--after-queue', type=int)
    args = parser.parse_args()
    if args.command == 'launch':
        if not args.after_queue:
            raise ValueError('explicit live queue dependency required')
        launch(args.output, args.after_queue)
    else:
        supervise(args.output)


if __name__ == '__main__':
    main()
