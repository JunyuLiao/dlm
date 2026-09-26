"""CPU stage gate and optional bounded launcher for frozen v18 evaluation.

The budget file is fixed before evaluation and includes historical usage. Both
host ledgers must be available locally before advancing beyond RULER initial.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from scripts.v13_seed_runs import execution_key
from scripts.v18_evaluate import logical_protocol_digest


def events(paths):
    return [json.loads(line) for path in paths if Path(path).exists()
            for line in Path(path).read_text().splitlines() if line.strip()]


def completed_keys(protocol, ledgers):
    planned = {execution_key(e): e for e in protocol['schedule']}
    if len(planned) != len(protocol['schedule']):
        raise ValueError('duplicate planned execution')
    found = set()
    for event in events(ledgers):
        if event.get('event') != 'run':
            continue
        key = event.get('execution_key')
        if key not in planned or key in found:
            raise ValueError('foreign or duplicate evaluation execution')
        spec = planned[key]
        host = protocol['block_assignments'][str(spec['block'])]
        if (event.get('host'), event.get('gpu_uuid')) != (host['host'], host['gpu_uuid']):
            raise ValueError('execution differs from frozen host/GPU block assignment')
        if any(event.get(k) != spec[k] for k in ('arm', 'id', 'seed', 'role', 'repeat', 'cell_id', 'block')):
            raise ValueError('execution differs from frozen schedule')
        found.add(key)
    return found


def gpu_seconds_by_host(paths, now):
    totals = {}
    for path in paths:
        pending = None
        for e in events([path]):
            if e.get('event') == 'start':
                if pending is not None:
                    # A crashed worker has no end row. Charge its full interval
                    # until the next start as a conservative upper bound.
                    totals[pending['host']] = totals.get(pending['host'], 0.0) + max(0.0, e['when'] - pending['when'])
                pending = e
            elif e.get('event') == 'worker_end':
                if pending is None or pending.get('host') != e.get('host'):
                    raise ValueError('worker_end without matching start')
                seconds = e.get('gpu_process_seconds')
                if type(seconds) not in (int, float) or seconds < 0:
                    raise ValueError('invalid GPU process duration')
                totals[e['host']] = totals.get(e['host'], 0.0) + seconds
                pending = None
        if pending is not None:
            totals[pending['host']] = totals.get(pending['host'], 0.0) + max(0.0, now - pending['when'])
    return totals


def initial_gate(ruler, ruler_ledgers, found):
    """AIME may follow complete initial blocks or a clean soft stop on both hosts."""
    initial = [e for e in ruler['schedule'] if e['block'] < ruler['initial_prefix_blocks']]
    events_by_host = {}
    for e in events(ruler_ledgers):
        if e.get('event') == 'worker_end':
            events_by_host[e.get('host')] = e
    for host in ruler['assigned_hosts']:
        host_name = host['host']
        blocks = {e['block'] for e in initial
                  if ruler['block_assignments'][str(e['block'])]['host'] == host_name}
        incomplete = False
        for block in blocks:
            entries = [e for e in initial if e['block'] == block]
            count = sum(execution_key(e) in found for e in entries)
            if 0 < count < len(entries):
                raise ValueError('partial initial block cannot authorize AIME')
            incomplete |= count == 0
        if incomplete and events_by_host.get(host_name, {}).get('status') != 'stopped_before_block':
            return False
    return True


def stage_plan(stage, ruler, aime, ruler_ledgers, aime_ledgers, budget, *, host, now=None):
    now = time.time() if now is None else now
    for protocol in (ruler, aime):
        if protocol.get('status') != 'frozen' or logical_protocol_digest(protocol) != protocol['logical_protocol_sha256']:
            raise ValueError('frozen logical protocol changed')
    r_done = completed_keys(ruler, ruler_ledgers)
    a_done = completed_keys(aime, aime_ledgers)
    initial = {execution_key(e) for e in ruler['schedule'] if e['block'] < ruler['initial_prefix_blocks']}
    all_aime = {execution_key(e) for e in aime['schedule']}
    gate_ready = initial_gate(ruler, ruler_ledgers, r_done)
    if stage in ('aime', 'remainder') and not gate_ready:
        raise ValueError('both hosts must finish or cleanly soft-stop frozen RULER initial blocks before AIME')
    if stage == 'remainder' and not all_aime <= a_done:
        raise ValueError('both hosts must finish all frozen AIME blocks before RULER remainder')
    protocol = aime if stage == 'aime' else ruler
    if host not in {h['host'] for h in protocol['assigned_hosts']}:
        raise ValueError('this host is absent from frozen assignment')
    accounted = gpu_seconds_by_host([*ruler_ledgers, *aime_ledgers], now)
    gpu_remaining = budget['gpu_cap_s_by_host'][host] - budget['baseline_gpu_s_by_host'][host] - accounted.get(host, 0.0)
    requests_used = budget['baseline_requests'] + len(r_done) + len(a_done)
    if (set(budget['request_cap_by_host']) != {h['host'] for h in protocol['assigned_hosts']} or
            sum(budget['request_cap_by_host'].values()) > budget['request_cap']):
        raise ValueError('host request reservations do not fit global cap')
    host_used = (budget['baseline_requests_by_host'][host] +
                 sum(e['host'] == host for e in events([*ruler_ledgers, *aime_ledgers]) if e.get('event') == 'run'))
    requests_remaining = min(budget['request_cap'] - requests_used,
                             budget['request_cap_by_host'][host] - host_used)
    guard = budget['block_guard_s']
    evaluation_deadline = budget['deadline_epoch'] - 60 * budget['scoring_minutes_reserved']
    deadline = min(evaluation_deadline, budget['initial_soft_deadline_epoch']) if stage == 'initial' else evaluation_deadline
    if any(type(v) not in (int, float) for v in (gpu_remaining, requests_remaining, guard, deadline)) or guard <= 0:
        raise ValueError('invalid frozen budget')
    done = a_done if stage == 'aime' else r_done
    selected = [e for e in protocol['schedule']
                if protocol['block_assignments'][str(e['block'])]['host'] == host and
                (stage in ('aime', 'remainder') or
                 (e['block'] < ruler['initial_prefix_blocks']) == (stage == 'initial'))]
    outstanding = sorted({e['block'] for e in selected if execution_key(e) not in done})
    next_block_requests = (sum(execution_key(e) not in done for e in selected if e['block'] == outstanding[0])
                           if outstanding else 0)
    allowed = bool(outstanding and gpu_remaining > guard and requests_remaining >= next_block_requests
                   and now + guard < deadline)
    return dict(stage=stage, host=host, allowed=allowed,
                gpu_remaining_s=gpu_remaining, requests_remaining=requests_remaining,
                deadline_epoch=deadline, block_guard_s=guard,
                ruler_initial_complete=initial <= r_done, ruler_initial_gate_ready=gate_ready,
                aime_complete=all_aime <= a_done,
                host_stage_complete=not outstanding, next_block_requests=next_block_requests,
                protocol_id=protocol['protocol_id'],
                eval_stage={'initial': 'initial', 'aime': 'all', 'remainder': 'all'}[stage])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('initial', 'aime', 'remainder'), required=True)
    for name in ('ruler-protocol', 'aime-protocol', 'budget', 'private', 'ledger', 'lock'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker-cwd', type=Path, default=Path.cwd(),
                        help='Immutable code root; defaults to the current directory')
    parser.add_argument('--ruler-ledger', type=Path, nargs='+', required=True)
    parser.add_argument('--aime-ledger', type=Path, nargs='+', required=True)
    parser.add_argument('--execute', action='store_true', help='Launch the explicitly selected frozen stage')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    ruler = json.loads(args.ruler_protocol.read_text())
    aime = json.loads(args.aime_protocol.read_text())
    budget = json.loads(args.budget.read_text())
    ledger_set = args.ruler_ledger if args.stage != 'aime' else args.aime_ledger
    if args.ledger not in ledger_set:
        parser.error('--ledger must be one of the stage ledger paths')
    plan = stage_plan(args.stage, ruler, aime, args.ruler_ledger, args.aime_ledger,
                      budget, host=socket.gethostname())
    print(json.dumps(plan, sort_keys=True), flush=True)
    if not args.execute:
        return
    if not plan['allowed']:
        raise SystemExit('budget/deadline guard does not permit another complete block')
    protocol_path = args.aime_protocol if args.stage == 'aime' else args.ruler_protocol
    timeout_s = max(1, int(min(plan['deadline_epoch'] - time.time(), plan['gpu_remaining_s'])))
    worker_cwd = args.worker_cwd.resolve()
    if not (worker_cwd / 'scripts/v18_evaluate.py').is_file():
        parser.error('--worker-cwd must contain the pinned evaluation source')
    worker_env = dict(os.environ)
    worker_env['PYTHONPATH'] = os.pathsep.join((str(worker_cwd / 'src'), str(worker_cwd),
                                               worker_env.get('PYTHONPATH', '')))
    try:
        subprocess.run([sys.executable, '-m', 'scripts.v18_evaluate', 'run',
                    '--protocol', str(protocol_path), '--private', str(args.private),
                    '--ledger', str(args.ledger), '--lock', str(args.lock),
                    '--stage', plan['eval_stage'], '--deadline-epoch', str(plan['deadline_epoch']),
                    '--gpu-budget-s', str(plan['gpu_remaining_s']),
                    '--remaining-requests', str(plan['requests_remaining']),
                    '--block-guard-s', str(plan['block_guard_s']), '--timeout', str(args.timeout)],
                   check=True, timeout=timeout_s, cwd=worker_cwd, env=worker_env)
    except subprocess.TimeoutExpired:
        # The worker start has no matching end; the next plan conservatively
        # charges elapsed time until its next invocation.
        raise SystemExit('hard wall/GPU timeout; preserve partial first outputs and ledger')


if __name__ == '__main__':
    main()
