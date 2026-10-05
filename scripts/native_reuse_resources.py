"""Summarize this task's GNU-time receipts without reading model outputs."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def duration(value):
    result = 0.
    for part in value.split(':'):
        result = result * 60 + float(part)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(set(args.root.glob('preflight/gpu_*.log')) |
                   set(args.root.glob('preflight/fresh_qualify_*.log')) |
                   set(args.root.glob('runs/*/process.log')))
    records = []
    for path in paths:
        content = path.read_text(errors='replace')
        def match(pattern):
            found = re.findall(pattern, content)
            return found[-1] if found else None
        elapsed = match(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)')
        user = match(r'User time \(seconds\): ([\d.]+)')
        system = match(r'System time \(seconds\): ([\d.]+)')
        exit_code = match(r'Exit status: (\d+)')
        records.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            wall_seconds=None if elapsed is None else duration(elapsed),
            cpu_user_seconds=None if user is None else float(user),
            cpu_system_seconds=None if system is None else float(system),
            exit_code=None if exit_code is None else int(exit_code),
            complete=exit_code is not None,
            accounting='Whole process wall charged, including loading/JIT/failure; not GPU kernel active time'))
    payload = dict(schema='numerical_reuse_resource_v1', records=records,
        completed_gpu_process_seconds=sum(r['wall_seconds'] or 0 for r in records),
        measured_cpu_seconds=sum((r['cpu_user_seconds'] or 0)+(r['cpu_system_seconds'] or 0) for r in records),
        incomplete_logs=[r['path'] for r in records if not r['complete']],
        excluded_costs='CPU build: first nvcc about70s wall estimate, bridge13.06s measured; shell/CPU analysis not fully metered. Do not call measured_cpu_seconds exhaustive.',
        weights='Existing single read-only model snapshot; no weights copied or downloaded')
    args.output.write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('completed_gpu_process_seconds','measured_cpu_seconds','incomplete_logs')}))


if __name__ == '__main__':
    main()
