"""One immutable stage process, resource receipt and finite wall timeout."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--max-seconds', type=float, required=True)
    p.add_argument('--gpu-uuid')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    limit = min(a.max_seconds, a.deadline-start)
    if limit <= 0 or not command:
        raise ValueError('no remaining time or command')
    gpu_lock = None
    if a.gpu_uuid:
        import fcntl
        gpu_lock = open('/tmp/v20_' + a.gpu_uuid + '.lock', 'w')
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        actual = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid',
                                         '--format=csv,noheader'], text=True).strip()
        if actual != a.gpu_uuid:
            raise ValueError('physical GPU identity drift')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU is occupied; never kill or overlap another worker')
    record = dict(start_epoch=start, pid=os.getpid(), command=command,
                  cwd=os.getcwd(), maximum_seconds=limit, status='started', gpu_uuid=a.gpu_uuid)
    with a.receipt.open('x') as f:
        json.dump(record, f, indent=2)
    child = None
    try:
        child = subprocess.Popen(command, start_new_session=True)
        record['child_pid'] = child.pid
        try:
            record['returncode'] = child.wait(timeout=limit)
            record['status'] = 'complete' if record['returncode'] == 0 else 'failed'
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            record.update(status='deadline_stop', returncode=child.returncode)
    except BaseException as exc:
        record.update(status='supervisor_error', error=repr(exc))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        raise
    finally:
        record['end_epoch'] = time.time()
        record['charged_process_seconds'] = record['end_epoch']-start
        temp = a.receipt.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2)+'\n')
        temp.replace(a.receipt)
    raise SystemExit(record.get('returncode', 1))


if __name__ == '__main__':
    main()
