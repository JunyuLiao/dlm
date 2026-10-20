"""Sequential continuation of the two frozen HumanEval panels; no method changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, immutable=False):
    path = Path(path)
    if immutable:
        with path.open('x') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
    else:
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        temporary.replace(path)


def process_identity(pid):
    """A recycled PID and a zombie must not hold or advance the queue."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except (OSError, IndexError):
        return None


def require_complete(study, protocol_sha):
    if sha(study / 'protocol.json') != protocol_sha:
        raise RuntimeError('frozen panel protocol changed')
    marker = read(study / 'final_complete.json')
    if marker['status'] != 'complete' or marker['protocol_sha256'] != protocol_sha:
        raise RuntimeError('panel lacks a qualified final completion marker')
    if marker['official_rescore_mismatches'] != 0:
        raise RuntimeError('official rescoring mismatch')
    analysis = study / 'analysis/attempt001'
    seal = read(analysis / 'analysis_complete.json')
    if sha(analysis / 'analysis_complete.json') != marker['analysis_complete_sha256']:
        raise RuntimeError('final analysis marker changed')
    for name, digest in seal['files_sha256'].items():
        if sha(analysis / name) != digest:
            raise RuntimeError('completed analysis artifact changed')


def snapshot(queue, config, stage):
    value = dict(stage=stage, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                 queue_protocol_sha256=sha(queue / 'protocol.json'), panels=[])
    device = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                             '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    value['gpu_status'] = dict(returncode=device.returncode, utilization_and_memory=device.stdout.strip())
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    value['gpu_processes'] = dict(returncode=processes.returncode, processes=processes.stdout.strip())
    for panel in config['panels']:
        study = Path(panel['study'])
        state = read(study / 'state.json')
        item = dict(budget=panel['budget'], state=state)
        active = state.get('active_attempt')
        if active:
            records = study / active / 'records.jsonl'
            item['saved_request_records'] = sum(1 for _ in records.open()) if records.exists() else 0
        value['panels'].append(item)
    with (queue / 'progress.jsonl').open('a') as handle:
        handle.write(json.dumps(value, sort_keys=True) + '\n')
    write(queue / 'state.json', value)
    return time.monotonic()


def command_plan(study, index, owns_coordinators):
    if (study / 'final_complete.json').exists():
        commands = []
    elif index or owns_coordinators:
        commands = [['-m', 'scripts.v31_humaneval_cfgdet_study', 'run', '--study', str(study)]]
    else:
        raise RuntimeError('attached K256 coordinator ended without qualified completion')
    if not (study / 'independent_rescoring.json').exists():
        commands.append(['-m', 'scripts.v31_humaneval_cfgdet_verify', '--study', str(study), '--rescore'])
    return commands


def run(queue):
    config = read(queue / 'protocol.json')
    if sha(__file__) != config['controller_sha256']:
        raise RuntimeError('queue controller source changed')
    if (queue / 'final_complete.json').exists():
        raise RuntimeError('completed queue must not be relaunched')
    owns = config.get('owns_coordinators', False)
    interval = config.get('monitor_interval_s', 1800)
    if interval not in (900, 1800):
        raise RuntimeError('unsupported monitor interval')
    if not owns:
        last = snapshot(queue, config, 'waiting_for_attached_k256')
        while process_identity(config['attached_pid']) == config['attached_start_ticks']:
            if time.monotonic() - last >= interval:
                last = snapshot(queue, config, 'waiting_for_attached_k256')
            time.sleep(5)
    for index, panel in enumerate(config['panels']):
        study = Path(panel['study'])
        if sha(study / 'protocol.json') != panel['protocol_sha256']:
            raise RuntimeError('queued panel protocol changed')
        if (study / 'final_complete.json').exists():
            require_complete(study, panel['protocol_sha256'])
        # In owned mode both coordinators are service children. Each coordinator
        # verifies and skips complete leaves and starts a fresh immutable attempt
        # for interrupted work. The reference-owner checks still gate K128.
        commands = command_plan(study, index, owns)
        for number, command in enumerate(commands):
            stage = f'k{panel["budget"]}_command{number + 1:02d}'
            snapshot(queue, config, stage)
            with (queue / f'{stage}.log').open('x') as log:
                process = subprocess.Popen([config['python'], *command], cwd=config['root'], stdout=log, stderr=subprocess.STDOUT)
                write(queue / f'{stage}_process.json', dict(pid=process.pid, command=[config['python'], *command]), True)
                last = time.monotonic()
                while process.poll() is None:
                    if time.monotonic() - last >= interval:
                        last = snapshot(queue, config, stage)
                    time.sleep(5)
                if process.returncode:
                    raise RuntimeError(f'{stage} failed ({process.returncode}); preserve attempts and inspect its log')
        require_complete(study, panel['protocol_sha256'])
        verified = read(study / 'independent_verification.json')
        rescored = read(study / 'independent_rescoring.json')
        if verified['status'] != 'passed' or rescored['mismatches'] or rescored['qualification_mismatches']:
            raise RuntimeError('independent panel verification failed')
        snapshot(queue, config, f'k{panel["budget"]}_complete')
    write(queue / 'final_complete.json', dict(status='complete', queue_protocol_sha256=sha(queue / 'protocol.json')),
          True)
    snapshot(queue, config, 'complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--queue', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.queue.absolute())
    except Exception as error:
        write(args.queue / 'failure.json', dict(status='stopped', error=f'{type(error).__name__}: {error}',
              resumable=True), True)
        raise
